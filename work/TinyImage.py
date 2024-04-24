import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from util.Noise import gasuss_noise


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(500, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.flatten(self.dropout(x))
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class noiseNet(Net):
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = gasuss_noise(x, var=x.detach().abs().mean())
        x = self.pool2(self.conv2(x))
        x = self.flatten(self.dropout(x))
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x