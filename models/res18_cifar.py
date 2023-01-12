import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import ImageClassification_Basemodel

class RES18_CIFAR(ImageClassification_Basemodel):
    def __init__(self):
        super(RES18_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Sequential(resnet_block(32, 64),resnet_block(64, 64))
        self.conv3 = nn.Sequential(resnet_block(64, 64),resnet_block(64, 64))
        self.conv4 = nn.Sequential(resnet_block(64, 128),resnet_block(128, 128))
        self.conv5 = nn.Sequential(resnet_block(128, 128),resnet_block(128, 128))
        self.fc = nn.Linear(128*8*8, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 128*8*8)
        x = self.fc(x)
        return x

class resnet_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resnet_block, self).__init__()
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, 2, 0)
            self.bn0 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if hasattr(self, 'downsample'):
            x = self.downsample(x)
            x = self.bn0(x)
        pre = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + pre
        return x