import torch


class Group:
    def __init__(self, tensor):
        self.tensor = tensor
        self.total_batch_size = 0
        self.participants = 0
        self.completed = False

    def __add__(self, other):
        self.tensor = torch.add(self.tensor, other.tensor)
        self.total_batch_size += other.total_batch_size
        self.participants += other.participants

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
    def __init__(self, model=None):
        self.model = model
        self.flatten_sizes = {}
        self.orig_sizes = {}
        for i, parameter in enumerate(model.parameters()):
            self.orig_sizes[i] = None
            self.flatten_sizes[i] = None

        self.incoming_groups = []  # (i, group)
        self.groups = None

    def synchronize(self, batch_size, elapsed_time, members):
        united_tensor = self.flatten_layers()
        torch.mul(united_tensor, batch_size)
        num_members = len(members)
        self.groups = split_to_groups(united_tensor, num_members)
        for i in self.groups.keys():
            self.groups[i].total_batch_size = batch_size
            self.groups[i].participants = 1
        while True:
            self.process_incoming_groups(num_members)
            if self.check_complition():
                break

        flatten_layers = torch.cat([self.groups[i].tensor for i in self.groups.keys()], dim=-1)
        self.restore_layers(flatten_layers)

    def check_complition(self):
        for group in self.groups.values():
            if not group.completed:
                return False
        return True

    def process_incoming_groups(self, num_members):
        while len(self.incoming_groups):
            i, group = self.incoming_groups.pop()
            self.groups[i] += group
        for i in self.groups.keys():
            if self.groups[i].participants == num_members:
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
