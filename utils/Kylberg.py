import os

import imageio.v2 as io
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset


def make_dataframe(data_path='data', return_df=False, train=True):
    data = []
    for file in os.listdir(data_path):
        path = '/'.join([data_path, file])
        name = file.split('-')
        row = [path, name[0]]
        data.append(row)
    if return_df:
        return pd.DataFrame(data, columns=['path', 'original image', 'part', 'crop', 'rotation', 'label'])
    return sorted(data)


class KylbergDataset(Dataset):
    def __init__(self, data_path='data', train=True):
        self.data = make_dataframe(data_path=data_path, train=train)
        self.labels = sorted(list(set(np.array(self.data).T[1])))
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.labels)
        self.target = self.label_encoder.transform(np.array(self.data).T[1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            image = np.array(io.imread(self.data[idx][0]))
        except IndexError:
            print("ERROR INDEX_ERROR", idx)
            idx = 228
            image = np.array(io.imread(self.data[idx][0]))
        sample = [np.double((image / 255.0 - 0.5)), self.target[idx]]
        return sample
