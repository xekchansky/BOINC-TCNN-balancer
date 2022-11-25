import pathlib
import pickle
import sys
from time import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(1, str(pathlib.Path(__file__).parent.parent.resolve()))
from utils.Kylberg import KylbergDataset
from utils.models import tcnn3


class Group:
    def __init__(self, tensor, total_batch_size=0, participants_number=0, total_elapsed_time=0, completed=False):
        self.tensor = tensor
        self.total_batch_size = total_batch_size
        self.participants_number = participants_number
        self.total_elapsed_time = total_elapsed_time
        self.completed = completed

    def __add__(self, other):
        self.tensor = torch.add(self.tensor, other.tensor)
        self.total_batch_size += other.total_batch_size
        self.participants_number += other.participants_number
        self.total_elapsed_time += other.total_elapsed_time

    def update(self, other):
        self.tensor = torch.add(self.tensor, other.tensor)
        self.total_batch_size += other.total_batch_size
        self.participants_number += other.participants_number
        self.total_elapsed_time += other.total_elapsed_time

    def normalize(self):
        self.tensor = torch.div(self.tensor, self.total_batch_size)


def split_to_groups(tensor, num_members):
    tensor_len = tensor.size()[0]
    group_size = tensor_len // num_members
    print('group_size: ', group_size)
    groups = {}
    for i in range(num_members):
        left = i * group_size
        if i == num_members - 1:
            right = tensor_len
        else:
            right = left + group_size
        groups[i] = Group(tensor[left:right])
    return groups


class Horovod:
    def __init__(self, model, api, logger=None):
        self.model = model
        self.flatten_sizes = {}
        self.orig_sizes = {}
        for i, parameter in enumerate(model.parameters()):
            self.orig_sizes[i] = None
            self.flatten_sizes[i] = None

        self.incoming_groups = []  # (i, group)
        self.groups = None

        self.api = api
        self.logger = logger

    def synchronize(self, batch_size, elapsed_time, max_batch_size, members=None):
        if members is None:
            members = self.api.ready_nodes
        united_tensor = self.flatten_layers()
        torch.mul(united_tensor, batch_size)
        num_members = len(members)
        self.groups = split_to_groups(united_tensor, num_members)
        for i in self.groups.keys():
            self.groups[i].total_batch_size = batch_size
            self.groups[i].participants_number = 1
            self.groups[i].total_elapsed_time = elapsed_time

        start_index = num_members - 1 - self.api.id
        for i in range(2 * (num_members - 1)):
            group_index = (start_index + i) % num_members
            print('GROUP INDEX:', group_index)
            print('WAITING')
            print('needed:', min((i + 1), num_members), 'got:', self.groups[group_index].participants_number)
            while min((i + 1), num_members) > self.groups[group_index].participants_number:
                self.process_incoming_groups()
                num_members = len(self.api.ready_nodes)
            print('needed:', min((i + 1), num_members), 'got:', self.groups[group_index].participants_number)
            print('PREPARING MESSAGE')
            print('tensor_size = ', self.groups[group_index].tensor.size())
            msg = (group_index,
                   self.groups[group_index].tensor.tolist(),
                   self.groups[group_index].total_batch_size,
                   self.groups[group_index].participants_number,
                   self.groups[group_index].total_elapsed_time)
            msg = pickle.dumps(msg)
            print('SENDING ', len(msg))
            self.api.send_message(msg_type='SUBMIT', msg=msg, target_node=self.api.load_balancer)
            print('SENT\n')

        #  wait for last groups
        while not self.check_all_completed():
            self.process_incoming_groups()

        self.logger.info('check completion: %s', self.check_all_completed())
        print('all completed:', self.check_all_completed())
        flatten_layers = torch.cat([self.groups[i].tensor for i in self.groups.keys()], dim=-1)
        self.restore_layers(flatten_layers)
        return max_batch_size

    def add_incoming_group(self, msg):
        group_index, tensor_list, total_batch_size, participants_number, total_elapsed_time = pickle.loads(msg)
        print(f'Received group {group_index}')
        group = Group(torch.tensor(tensor_list).double(), total_batch_size, participants_number, total_elapsed_time)
        group.completed = self.check_group_completion(group)
        self.incoming_groups.append((group_index, group))

    def check_group_completion(self, group):
        members = self.api.ready_nodes
        num_members = len(members)
        if group.participants_number == num_members:
            return True
        else:
            return False

    def check_all_completed(self):
        for group in self.groups.values():
            if not group.completed:
                return False
        return True

    def process_incoming_groups(self):
        while len(self.incoming_groups):
            i, group = self.incoming_groups.pop()
            print(f'Processing group {i}')
            if group.completed:
                self.groups[i] = group
            else:
                self.groups[i].update(group)
                if self.check_group_completion(self.groups[i]):
                    self.groups[i].completed = True

    def flatten_layers(self):
        flatten_parts = []
        for i, parameter in enumerate(self.model.parameters()):
            flatten_parts.append(self.flatten(parameter.grad, i))
        return torch.cat(flatten_parts, dim=-1)

    def flatten(self, tensor, i):
        flatten_tensor = torch.flatten(tensor)

        if self.orig_sizes[i] is None:
            self.orig_sizes[i] = tensor.size()
            self.flatten_sizes[i] = flatten_tensor.size()

        return flatten_tensor

    def restore_layers(self, flatten_layers):
        prev_index = 0
        for i, parameter in enumerate(self.model.parameters()):
            index = prev_index + self.flatten_sizes[i][0]
            flatten_layer = flatten_layers[prev_index: index]
            parameter.grad = self.restore(flatten_layer, i)
            prev_index = index

    def restore(self, flatten_tensor, i):
        restored_tensor = flatten_tensor.resize_(self.orig_sizes[i])

        return restored_tensor


class HorovodTrain:
    def __init__(self, api, logger=None):
        self.api = api
        self.logger = logger

        self.model = tcnn3()
        self.model.double()

        self.model_size = 81.99 * 1024 * 1024  # 82Mb for 1 image in batch
        self.RAM_size = psutil.virtual_memory().total
        self.max_RAM_usage = 0.75
        # self.max_batch_size = self.RAM_size * self.max_RAM_usage // self.model_size
        # self.max_batch_size = self.get_max_batch_size()
        self.max_batch_size = 1
        ###

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        self.optimizer.zero_grad()
        self.dataset = KylbergDataset(data_path='data', train=True)
        self.train_ds_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.max_batch_size, shuffle=True)
        self.horovod = Horovod(self.model, api=api, logger=self.logger)

    def get_max_batch_size(self):
        available_ram_size = psutil.virtual_memory().available
        self.max_batch_size = int(available_ram_size // self.model_size)
        return self.max_batch_size

    def fit(self):
        print('fitting')
        self.train_epoch()

    def train_epoch(self):
        train_loss = 0.0
        train_acc = 0.0
        self.model.train()

        start = time()
        batch_size = self.max_batch_size
        for img, lbl in self.train_ds_loader:

            img = torch.reshape(img, [-1, 1, 256, 256])
            self.optimizer.zero_grad()

            predict = self.model(img)

            loss = self.loss_fn(predict, lbl)
            loss.backward()
            elapsed_time = time() - start
            print('syncing')
            batch_size = self.horovod.synchronize(batch_size=batch_size,
                                                  elapsed_time=elapsed_time,
                                                  max_batch_size=self.max_batch_size)
            print(f'batch size: {batch_size}')
            start = time()
            self.optimizer.step()

            train_loss += loss.item() * img.size(0)

            batch_predictions = predict.cpu().detach().numpy()
            predicted_classes = np.array([np.argmax(batch_predictions[i]) for i in range(batch_size)])
            train_acc += np.sum(predicted_classes == lbl.cpu().numpy())

        train_loss /= len(self.train_ds_loader.sampler)
        train_acc /= len(self.dataset)
