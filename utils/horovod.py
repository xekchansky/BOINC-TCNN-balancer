import pathlib
import pickle
import sys
from time import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

sys.path.insert(1, str(pathlib.Path(__file__).parent.parent.resolve()))
from utils.Kylberg import KylbergDataset
from utils.models import tcnn3
from utils.state_logger import StateLogger


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
    def __init__(self, model, api, logger=None, state_logger=None):
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
        self.state_logger = state_logger

    def synchronize(self, batch_size, elapsed_time, max_batch_size, members=None):
        if members is None:
            members = self.api.ready_nodes
        united_tensor = self.flatten_layers()
        united_tensor = torch.mul(united_tensor, batch_size)
        num_members = len(members)
        self.groups = split_to_groups(united_tensor, num_members)
        for i in self.groups.keys():
            self.groups[i].total_batch_size = batch_size
            self.groups[i].participants_number = 1
            self.groups[i].total_elapsed_time = elapsed_time

        start_index = num_members - 1 - self.api.id
        for i in range(2 * (num_members - 1)):
            group_index = (start_index + i) % num_members

            self.state_logger.wait()
            while min((i + 1), num_members) > self.groups[group_index].participants_number:
                self.process_incoming_groups()
                num_members = len(self.api.ready_nodes)

            self.state_logger.sync()
            msg = (group_index,
                   self.groups[group_index].tensor.tolist(),
                   self.groups[group_index].total_batch_size,
                   self.groups[group_index].participants_number,
                   self.groups[group_index].total_elapsed_time)
            msg = pickle.dumps(msg)
            self.api.send_message(msg_type='SUBMIT', msg=msg, target_node=self.api.load_balancer)

        #  wait for last groups
        self.state_logger.wait()
        while not self.check_all_completed():
            self.process_incoming_groups()
        self.state_logger.sync()

        self.logger.debug('check completion: %s', self.check_all_completed())
        for group in self.groups.values():
            group.normalize()
        flatten_layers = torch.cat([self.groups[i].tensor for i in self.groups.keys()], dim=-1)
        self.restore_layers(flatten_layers)
        return max_batch_size

    def add_incoming_group(self, msg):
        group_index, tensor_list, total_batch_size, participants_number, total_elapsed_time = pickle.loads(msg)
        group = Group(torch.tensor(tensor_list).float(), total_batch_size, participants_number, total_elapsed_time)
        group.completed = self.check_group_completion(group)
        self.incoming_groups.append((group_index, group))

    def check_group_completion(self, group):
        members = self.api.ready_nodes
        num_members = len(members)
        if group.participants_number == num_members:
            group.completed = True
            return True
        else:
            return False

    def check_all_completed(self):
        for group in self.groups.values():
            if not group.completed:
                if not self.check_group_completion(group):
                    return False
        return True

    def process_incoming_groups(self):
        while len(self.incoming_groups):
            i, group = self.incoming_groups.pop()
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
    def __init__(self, api, logger=None, init_model_path='initial_model.pth'):
        self.api = api
        self.logger = logger
        self.state_logger = StateLogger(logger=logger, send_period=60)
        self.model_logging_period = 60

        self.model = tcnn3()
        self.model.float()
        # self.model.load_state_dict(torch.load(init_model_path))

        self.model_size = 81.99 * 1024 * 1024  # 82Mb for 1 image in batch
        self.RAM_size = psutil.virtual_memory().total
        self.max_RAM_usage = 0.75
        # self.max_batch_size = self.RAM_size * self.max_RAM_usage // self.model_size
        # self.max_batch_size = self.get_max_batch_size()
        self.max_batch_size = 50
        ###

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        self.optimizer.zero_grad()
        self.dataset = KylbergDataset(data_path='data', train=True)
        self.train_ds_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.max_batch_size, shuffle=True)
        self.horovod = Horovod(self.model, api=api, logger=self.logger, state_logger=self.state_logger)

        checkpoint = torch.load(init_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def get_max_batch_size(self):
        available_ram_size = psutil.virtual_memory().available
        self.max_batch_size = int(available_ram_size // self.model_size)
        return self.max_batch_size

    def fit(self):
        while True:
            self.train_epoch()

    def train_epoch(self):
        self.state_logger.backward()
        train_loss = 0.0
        train_acc = 0.0
        self.model.train()

        start = time()
        batch_size = self.max_batch_size

        # variables for logging model state
        samples = 0
        processed_images = 0
        sub_loss = 0.0
        sub_acc = 0.0
        last_log_time = time()

        for img, lbl in tqdm(self.train_ds_loader):

            img = torch.reshape(img, [-1, 1, 256, 256])
            self.optimizer.zero_grad()

            predict = self.model(img)

            loss = self.loss_fn(predict, lbl)
            loss.backward()
            elapsed_time = time() - start

            self.state_logger.sync()
            batch_size = self.horovod.synchronize(batch_size=batch_size,
                                                  elapsed_time=elapsed_time,
                                                  max_batch_size=self.max_batch_size)
            self.state_logger.backward()
            start = time()
            self.optimizer.step()

            train_loss += loss.item() * img.size(0)

            batch_predictions = predict.cpu().detach().numpy()
            predicted_classes = np.array([np.argmax(batch_predictions[i]) for i in range(img.size(0))])
            correct_predictions = np.sum(predicted_classes == lbl.cpu().numpy())
            train_acc += correct_predictions

            samples += 1
            processed_images += img.size(0)
            sub_loss += loss.item() * img.size(0)
            sub_acc += correct_predictions

            if self.api.id == 0:
                if (time() - last_log_time) > self.model_logging_period:
                    msg = f'MODEL STATE: loss: {sub_loss/samples} acc: {sub_acc/processed_images}'
                    self.logger.info(msg)
                    print(msg)
                    samples = 0
                    processed_images = 0
                    sub_loss = 0.0
                    sub_acc = 0.0
                    last_log_time = time()

        train_loss /= len(self.train_ds_loader.sampler)
        train_acc /= len(self.dataset)

        print('EPOCH FINISHED')
        print(train_loss)
        print(train_acc)
        msg = f'MODEL EPOCH STATE: loss: {train_loss} acc: {train_acc}'
        self.logger.info(msg)
