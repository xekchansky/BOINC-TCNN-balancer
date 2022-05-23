import os
import zipfile
import pandas as pd
import numpy as np
import imageio as io
import random
from tqdm import tqdm

import torch

from torch.utils.data import Dataset

from sklearn import preprocessing

class KylbergDataset(Dataset):
    def __init__(self, data_path='data', train=True):
        self.data = self.make_dataframe(data_path=data_path, train=train)
        self.labels = sorted(list(set(np.array(self.data).T[5])))
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.labels)
        self.target = self.label_encoder.transform(np.array(self.data).T[5])

    def __len__(self):
        return len(self.data) // 12

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx * 12 + random.randint(0, 11) #return random rotation for given image part
        try:
            image = np.array(io.imread(self.data[idx][0]))
        except IndexError:
            print("ERROR INDEX_ERROR", idx)
            idx = 228
            image = np.array(io.imread(self.data[idx][0]))
        sample = [np.double((image/255.0 - 0.5)), self.target[idx]]
        return sample
            
    def make_dataframe(self, data_path='data', return_df=False, train=True):
        data = []
        parts_in_validation = set(['040', '039', '038', '037'])
        for folder in os.listdir(data_path):
            target = folder.split('-')[0]
            for file in os.listdir(data_path+'/'+folder):
                path = '/'.join([data_path, folder, file])
                name = file.split('-')
                row = [path, name[1], name[2][1:], name[3], name[4][1:4], name[0]]
                if train != (row[2] in parts_in_validation):
                    data.append(row)
        if return_df: return pd.DataFrame(data, columns = ['path', 'original image', 'part', 'crop', 'rotation', 'label'])
        return sorted(data)

