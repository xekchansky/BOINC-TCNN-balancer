import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchsummary import summary
from tqdm import tqdm

import Kylberg
import models


class TextureCNN:
    def __init__(self, model, version, device='gpu'):

        self.num_epoch = None
        self.start = None
        self.ts = None
        self.valid_accuracies = None
        self.valid_losses = None
        self.train_accuracies = None
        self.train_losses = None
        self.optimizer = None
        self.loss_fn = None
        self.test_ds_loader = None
        self.test_ds = None
        self.train_ds_loader = None
        self.train_ds = None
        self.gpu = False

        if device == 'gpu': 
            self.gpu = True

        if model == 'tcnn2':
            self.model = models.tcnn2()
        elif model == 'tcnn3':
            self.model = models.tcnn3()
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
        self.checkpoint_path = 'models/' + version + '/' + time.strftime("%Y-%m-%d-%H:%M:%S",
                                                                         time.localtime(time.time())) + '/'

    def fit(self, dataset, batch_size, epochs=10, show_summary=False):
        self.train_ds = dataset(train=True)
        self.train_ds_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)

        #self.test_ds = dataset(train=False)
        #self.test_ds_loader = torch.utils.data.DataLoader(self.test_ds, batch_size=batch_size, shuffle=False)

        if show_summary:
            # summary(self.model, tuple([1] + list(np.array(self.train_ds[0][0]).shape)))
            summary(self.model, (1, 256, 256), dtypes=[torch.double])

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

        self.train_losses = []
        self.train_accuracies = []
        self.valid_losses = []
        self.valid_accuracies = []
        self.ts = []
        self.start = time.time()

        self.num_epoch = epochs

        for epoch in tqdm(range(1, self.num_epoch + 1)):
            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0

            self.model.train()
            loop = 0
            for img, lbl in tqdm(self.train_ds_loader):
                if self.gpu:
                    img = img.cuda()
                    lbl = lbl.cuda()

                self.model.to(self.device)

                img = torch.reshape(img, [-1, 1, 256, 256])
                ###
                self.optimizer.zero_grad()
                predict = self.model(img)

                loss = self.loss_fn(predict, lbl)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * img.size(0)

                batch_predictions = predict.cpu().detach().numpy()
                predicted_classes = np.array([np.argmax(batch_predictions[i]) for i in range(batch_size)])
                train_acc += np.sum(predicted_classes == lbl.cpu().numpy())

                loop += 1
                print('loop_loss=', loss.item() * img.size(0))
                print('norm_agg_loss=', train_loss / loop)
                print('acc=', train_acc / (loop * batch_size))

            '''
            self.model.eval()
            for img, lbl in tqdm(self.test_ds_loader):
                if self.gpu:
                    img = img.cuda()
                    lbl = lbl.cuda()

                self.model.to(self.device)

                img = torch.reshape(img, [-1, 1, 256, 256])
                predict = self.model(img)

                loss = self.loss_fn(predict, lbl)
                valid_loss += loss.item() * img.size(0)

                batch_predictions = predict.cpu().detach().numpy()
                predicted_classes = np.array([np.argmax(batch_predictions[i]) for i in range(batch_size)])
                valid_acc += np.sum(predicted_classes == lbl.cpu().numpy())
            '''


            train_loss /= len(self.train_ds_loader.sampler)
            train_acc /= len(self.train_ds)
            #valid_loss /= len(self.test_ds_loader.sampler)
            #valid_acc /= len(self.test_ds)

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.valid_losses.append(valid_loss)
            self.valid_accuracies.append(valid_acc)
            self.ts.append(time.time() - self.start)

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

            print('Epoch:{} Train Loss:{:.4f} valid Loss:{:.4f}'.format(epoch, train_loss, valid_loss))
            print('Epoch:{} Train Accuracy:{:.4f} valid Accuracy:{:.4f}'.format(epoch, train_acc, valid_acc))

            np.save(self.checkpoint_path + 'train_losses.npy', np.array(self.train_losses))
            np.save(self.checkpoint_path + 'train_acc.npy', np.array(self.train_accuracies))
            np.save(self.checkpoint_path + 'valid_losses.npy', np.array(self.valid_losses))
            np.save(self.checkpoint_path + 'valid_acc.npy', np.array(self.valid_accuracies))
            np.save(self.checkpoint_path + 'ts.npy', np.array(self.ts))


def main():
    TCNN = TextureCNN(model='tcnn3', version='v3.0', device='cpu')
    dataset = Kylberg.KylbergDataset
    TCNN.fit(dataset, batch_size=50, show_summary=True)


if __name__ == '__main__':
    main()
