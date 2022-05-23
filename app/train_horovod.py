import os
import zipfile
import psutil
import pandas as pd
import numpy as np
import imageio as io
from tqdm import tqdm
import time

import torch
import torch.optim as optim

from torch.utils.data import Dataset
from torchsummary import summary

from sklearn import preprocessing

import models

import Kylberg
import torch.nn as nn
import torch.nn.functional as F

import horovod.torch as hvd

class Texture_CNN:
    def __init__(self, model, version, device='gpu'):
        
        self.gpu = False
        if device == 'gpu': self.gpu = True
        
        if model == 'tcnn2':
            self.model = models.tcnn2(version)
        elif model == 'tcnn3':
            self.model = models.tcnn3(version)
        else:
            raise NameError('No such model:', model)
        
        if self.gpu: 
            self.device = torch.device("cuda")
            self.model = self.model.cuda()
        else: 
            self.device = torch.device("cpu")
            self.model.cpu()
        
        self.model.to(device)
            
        self.model.double()
        self.version = version
        self.checkpoint_path = 'models/' + version + '/' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time())) + '/'
    
    def fit(self, dataset, batch_size, epochs=10, show_summary=False):
        
        hvd.init()
        
        #train dataloader
        self.train_ds=dataset(train=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_ds, 
                                                                        num_replicas=hvd.size(), 
                                                                        rank=hvd.rank())
        self.train_ds_loader=torch.utils.data.DataLoader(self.train_ds, 
                                                         batch_size=batch_size, 
                                                         sampler=train_sampler)

        #testn dataloader
        self.test_ds=dataset(train=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_ds, 
                                                                       num_replicas=hvd.size(), 
                                                                       rank=hvd.rank())
        self.test_ds_loader=torch.utils.data.DataLoader(self.test_ds,
                                                        batch_size=batch_size,
                                                        sampler=test_sampler)
        
        if show_summary:
            
            #summary(self.model, tuple([1] + list(np.array(self.train_ds[0][0]).shape)))
            summary(self.model, (1, 256, 256), dtypes=[torch.double])

        self.loss_fn=nn.CrossEntropyLoss()
        self.optimizer=optim.SGD(self.model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0001)
        self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=self.model.named_parameters())
        
        # Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)

        self.train_losses=[]
        self.train_accuracies=[]
        self.valid_losses=[]
        self.valid_accuracies=[]
        self.ts=[]
        self.start=time.time()

        self.num_epoch=20

        for epoch in tqdm(range(1,self.num_epoch+1)):
            train_loss=0.0
            train_acc=0.0
            valid_loss=0.0
            valid_acc=0.0

            
            self.model.train()
            for img,lbl in tqdm(self.train_ds_loader):
                if self.gpu:
                    img=img.cuda()
                    lbl=lbl.cuda()
                    
                self.model.to(self.device)

                img = torch.reshape(img, [-1, 1, 256, 256])
                ###
                self.optimizer.zero_grad()
                predict=self.model(img)
                
                loss=self.loss_fn(predict,lbl)
                loss.backward()
                self.optimizer.step()
                
                train_loss+=loss.item()*img.size(0)
                
                batch_predictions = predict.cpu().detach().numpy()
                predicted_classes = np.array([np.argmax(batch_predictions[i]) for i in range(batch_size)])
                train_acc += np.sum(predicted_classes == lbl.cpu().numpy())

            self.model.eval()
            for img,lbl in tqdm(self.test_ds_loader):
                if self.gpu:
                    img=img.cuda()
                    lbl=lbl.cuda()
                    
                self.model.to(self.device)

                img = torch.reshape(img, [-1, 1, 256, 256])
                predict=self.model(img)
                
                loss=self.loss_fn(predict,lbl)
                valid_loss+=loss.item()*img.size(0)
                
                batch_predictions = predict.cpu().detach().numpy()
                predicted_classes = np.array([np.argmax(batch_predictions[i]) for i in range(batch_size)])
                valid_acc += np.sum(predicted_classes == lbl.cpu().numpy())

            train_loss /= len(self.train_ds_loader.sampler) 
            train_acc /= len(self.train_ds) 
            valid_loss /= len(self.test_ds_loader.sampler)
            valid_acc /= len(self.test_ds) 

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.valid_losses.append(valid_loss)
            self.valid_accuracies.append(valid_acc)
            self.ts.append(time.time()-self.start)
            
            epoch_save_path = self.checkpoint_path + 'epoch_' + str(epoch) + '/'
            
            if not os.path.exists(epoch_save_path):
                os.makedirs(epoch_save_path)
                
            EPOCH = epoch
            PATH = epoch_save_path + "model.pt"
            LOSS = valid_loss

            torch.save({
                        'epoch': EPOCH,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': LOSS,
                        }, PATH)

            print('Epoch:{} Train Loss:{:.4f} valid Loss:{:.4f}'.format(epoch,train_loss,valid_loss))
            print('Epoch:{} Train Accuracy:{:.4f} valid Accuracy:{:.4f}'.format(epoch,train_acc,valid_acc))

            np.save(self.checkpoint_path + 'train_losses.npy', np.array(self.train_losses))
            np.save(self.checkpoint_path + 'train_acc.npy', np.array(self.train_accuracies))
            np.save(self.checkpoint_path + 'valid_losses.npy', np.array(self.valid_losses))
            np.save(self.checkpoint_path + 'valid_acc.npy', np.array(self.valid_accuracies))
            np.save(self.checkpoint_path + 'ts.npy', np.array(self.ts))
            
            
def main():
    TCNN = Texture_CNN(model='tcnn3', version='v3.0', device='cpu')
    dataset = Kylberg.KylbergDataset
    
    max_ram_load = 0.75
    batch_size = int((psutil.virtual_memory().total * max_ram_load) / 82000000)
    
    TCNN.fit(dataset, batch_size=batch_size, show_summary=True)
    
if __name__ == '__main__':
    main()