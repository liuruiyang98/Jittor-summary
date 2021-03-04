import jittor as jt
from jittor import init
from jittor import nn

class SingleInputNet(nn.Module):

    def __init__(self):
        super(SingleInputNet, self).__init__()
        self.conv1 = nn.Conv(1, 10, 5)
        self.conv2 = nn.Conv(10, 20, 5)
        self.conv2_drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def execute(self, x):
        x = nn.relu(nn.max_pool2d(self.conv1(x), 2))
        x = nn.relu(nn.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(((- 1), 320))
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)

class MultipleOutputNet(nn.Module):
    def __init__(self):
        super(MultipleOutputNet, self).__init__()
        self.conv1 = nn.Conv(1, 10, 5)
        self.conv2 = nn.Conv(10, 20, 5)
        self.conv2_drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def execute(self, x):
        x = nn.relu(nn.max_pool2d(self.conv1(x), 2))
        x = nn.relu(nn.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(((- 1), 320))
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1), x

class MultipleInputNet(nn.Module):

    def __init__(self):
        super(MultipleInputNet, self).__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)
        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def execute(self, x1, x2):
        x1 = nn.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = nn.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = jt.contrib.concat((x1, x2), dim=0)
        return nn.log_softmax(x, dim=1)

class MultipleInputNetDifferentDtypes(nn.Module):

    def __init__(self):
        super(MultipleInputNetDifferentDtypes, self).__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)
        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def execute(self, x1, x2):
        x1 = nn.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.float()
        x2 = nn.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = jt.contrib.concat((x1, x2), dim=0)
        return nn.log_softmax(x, dim=1)

def main():
    # model = SingleInputNet()
    # x = jt.ones([1, 1, 28, 28])
    # y = model(x)
    # print (y.shape)

    # model = MultipleInputNet()
    # x1 = jt.ones([1, 1, 300])
    # x2 = jt.ones([1, 1, 300])
    # y = model(x1, x2)
    # print (y.shape)

    model = MultipleInputNetDifferentDtypes()
    x1 = jt.ones([1, 1, 300]).float()
    x2 = jt.ones([1, 1, 300]).int64()
    y = model(x1, x2)
    print (y.shape)

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    '''
    SingleInputNet
    21,840 total parameters.
    21,840 training parameters.

    MultipleInputNet
    31,120 total parameters.
    31,120 training parameters.

    MultipleInputNetDifferentDtypes
    31,120 total parameters.
    31,120 training parameters.
    '''


if __name__ == '__main__':
    main()

# from jittor.utils.pytorch_converter import convert

# pytorch_code="""

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SingleInputNet(nn.Module):
#     def __init__(self):
#         super(SingleInputNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d(0.3)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# class MultipleInputNet(nn.Module):
#     def __init__(self):
#         super(MultipleInputNet, self).__init__()
#         self.fc1a = nn.Linear(300, 50)
#         self.fc1b = nn.Linear(50, 10)

#         self.fc2a = nn.Linear(300, 50)
#         self.fc2b = nn.Linear(50, 10)

#     def forward(self, x1, x2):
#         x1 = F.relu(self.fc1a(x1))
#         x1 = self.fc1b(x1)
#         x2 = F.relu(self.fc2a(x2))
#         x2 = self.fc2b(x2)
#         x = torch.cat((x1, x2), 0)
#         return F.log_softmax(x, dim=1)

# class MultipleInputNetDifferentDtypes(nn.Module):
#     def __init__(self):
#         super(MultipleInputNetDifferentDtypes, self).__init__()
#         self.fc1a = nn.Linear(300, 50)
#         self.fc1b = nn.Linear(50, 10)

#         self.fc2a = nn.Linear(300, 50)
#         self.fc2b = nn.Linear(50, 10)

#     def forward(self, x1, x2):
#         x1 = F.relu(self.fc1a(x1))
#         x1 = self.fc1b(x1)
#         x2 = x2.type(torch.FloatTensor)
#         x2 = F.relu(self.fc2a(x2))
#         x2 = self.fc2b(x2)
#         # set x2 to FloatTensor
#         x = torch.cat((x1, x2), 0)
#         return F.log_softmax(x, dim=1)

# """

# jittor_code = convert(pytorch_code)
# print(jittor_code)