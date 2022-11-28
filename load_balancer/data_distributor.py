import os
import random
from collections import Counter
import configparser

import boto3
from sklearn.model_selection import train_test_split


def download_dataset_table(path='dataset.txt', credentials_path='credentials.ini'):
    config = configparser.ConfigParser()
    config.read(credentials_path)
    s3 = boto3.client('s3',
                      endpoint_url='https://storage.yandexcloud.net',
                      aws_access_key_id=config['AWS']['access_key_id'],
                      aws_secret_access_key=config['AWS']['secret_access_key'])
    with open(path, 'wb') as f:
        s3.download_fileobj('modified-kylberg-dataset', 'dataset.txt', f)


def read_dataset_table(dataset_path='dataset.txt'):
    with open(dataset_path, 'r') as f:
        data = [line.strip() for line in f]
    target = [image.split('-')[0] for image in data]
    return data, target


class DataDistributor:
    def __init__(self, members_estimate=7, replication_factor=2, seed=42, dataset_path='dataset.txt'):
        if replication_factor > members_estimate:
            print('Data partition size > dataset size, setting partition size to dataset size')
            replication_factor = members_estimate
        self.seed = seed

        if not os.path.exists(dataset_path):
            download_dataset_table(dataset_path)
        data, target = read_dataset_table(dataset_path)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=0.2,
                                                                                random_state=seed)
        train_data_size = len(self.X_train)
        if replication_factor == members_estimate:
            self.part_size = train_data_size
        else:
            self.part_size = train_data_size * replication_factor // members_estimate + 1

        self.data_usage = {index: 0 for index in range(len(self.X_train))}

    def update_data_usage(self, indexes):
        for i in indexes:
            self.data_usage[i] += 1

    def get_data_part(self, size=None):
        if size is None:
            size = self.part_size
        res_size = 0
        res_indexes = []
        usages_count = Counter(self.data_usage.values())
        usages = sorted(usages_count.keys())
        least_used_indexes = [key[0] for key in sorted(self.data_usage.items(), key=lambda item: item[1])]
        usages_index = 0
        while res_size < size:
            usage = usages[usages_index]
            usages_index += 1
            usage_count = usages_count[usage]
            if res_size + usage_count <= size:
                res_indexes += least_used_indexes[:usage_count]
                least_used_indexes = least_used_indexes[usage_count:]
                res_size += usage_count
            else:
                left_size = size - res_size
                chosen = random.sample(least_used_indexes[:usage_count], k=left_size)
                res_indexes += chosen
                res_size += left_size
        self.update_data_usage(res_indexes)
        x_res = [self.X_train[i] for i in res_indexes]
        y_res = [self.y_train[i] for i in res_indexes]
        return x_res, y_res

    def get_test_data(self):
        return self.X_test, self.y_test
