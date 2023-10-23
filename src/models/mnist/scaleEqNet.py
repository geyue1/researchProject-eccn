# -* UTF-8 *-
'''
==============================================================
@Project -> File : eccn -> scaleEqNet.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/10/23 15:23
@Desc :

==============================================================
'''

import sys

import torch.nn as nn


from scaleEqNet.ScaleEqNet import ScaleConv, VectorMaxPool, VectorBatchNorm, Vector2Magnitude, Vector2Angle

sys.path.append('../')

#!/usr/bin/env python
#__author__ = "Anders U. Waldeland"
#__email__ = "anders@nr.no"

"""
A reproduction of the MNIST-classification network described in:
Rotation equivariant vector field networks (ICCV 2017)
Diego Marcos, Michele Volpi, Nikos Komodakis, Devis Tuia
https://arxiv.org/abs/1612.09346
https://github.com/dmarcosg/RotEqNet
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = ScaleConv(1, 12, [7, 7], 1, padding=3, mode=1)
        self.pool1 = VectorMaxPool(2)
        self.bn1 = VectorBatchNorm(12)

        self.conv2 = ScaleConv(12, 32, [7, 7], 1, padding=3, mode=2)
        self.pool2 = VectorMaxPool(2)
        self.bn2 = VectorBatchNorm(32)

        self.conv3 = ScaleConv(32, 48, [7, 7], 1, padding=3, mode=2)
        self.pool3 = VectorMaxPool(4)
        self.v2m = Vector2Magnitude()
        self.v2a = Vector2Angle()

        self.fc1 = nn.Conv2d(48, 256, 1)  # FC1
        self.fc1bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(256, 10, 1)  # FC2

        self.afc1 = nn.Conv2d(48, 48, 1)  # FC1
        self.afc1bn = nn.BatchNorm2d(48)
        self.afc1relu = nn.ReLU()
        self.adropout = nn.Dropout2d(0.7)
        self.afc2 = nn.Conv2d(48, 1, 1)  # FC2

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        xm = self.v2m(x)
        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)

        # xa = F.torch.cat(x,dim=1)
        xa = self.v2a(x)
        # xa = x[0]
        # xa = self.afc1(xa)
        # xa = self.relu(self.afc1bn(xa))
        xa = self.adropout(xa)
        xa = self.afc2(xa)
        xm = xm.view(xm.size()[0], xm.size()[1])
        xa = xa.view(xa.size()[0], xa.size()[1])

        return xm, xa


class Net_scalar(nn.Module):
    def __init__(self):
        super(Net_scalar, self).__init__()

        self.conv1 = ScaleConv(1, 12, [7, 7], 1, padding=3, mode=1)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(12)

        self.conv2 = ScaleConv(12, 32, [7, 7], 1, padding=3, mode=1)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = ScaleConv(32, 48, [7, 7], 1, padding=3, mode=1)
        self.pool3 = nn.MaxPool2d(4)

        self.v2m = Vector2Magnitude()
        self.v2a = Vector2Angle()

        self.fc1 = nn.Conv2d(48, 256, 1)  # FC1
        self.fc1bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(256, 10, 1)  # FC2

        self.afc1 = nn.Conv2d(48, 256, 1)  # FC1
        self.afc1bn = nn.BatchNorm2d(256)
        self.afc1relu = nn.ReLU()
        self.adropout = nn.Dropout2d(0.7)
        self.afc2 = nn.Conv2d(48, 1, 1)  # FC2

    def forward(self, x):
        x = self.conv1(x)
        x = self.v2m(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.v2m(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.v2m(x)
        x = self.pool3(x)

        xm = self.fc1(x)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)

        # xa = self.afc1(x)
        # xa = self.relu(self.afc1bn(xa))
        xa = self.adropout(x)
        xa = self.afc2(xa)
        xm = xm.view(xm.size()[0], xm.size()[1])
        xa = xa.view(xa.size()[0], xa.size()[1])

        return xm, xa


class Net_std(nn.Module):
    def __init__(self, filter_mult=3):
        super(Net_std, self).__init__()

        self.conv1 = nn.Conv2d(1, 12 * filter_mult, [7, 7], 1, padding=3)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(12 * filter_mult)

        self.conv2 = nn.Conv2d(12 * filter_mult, 32 * filter_mult, [7, 7], 1, padding=3)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(32 * filter_mult)

        self.conv3 = nn.Conv2d(32 * filter_mult, 48, [7, 7], 1, padding=3)
        self.pool3 = nn.MaxPool2d(4)

        self.fc1 = nn.Conv2d(48, 256, 1)  # FC1
        self.fc1bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(256, 10, 1)  # FC2

        # self.afc1 = nn.Conv2d(48*filter_mult, 256*filter_mult, 1)  # FC1
        # self.afc1bn = nn.BatchNorm2d(256*filter_mult)
        # self.afc1relu = nn.ReLU()
        self.adropout = nn.Dropout2d(0.7)
        self.afc2 = nn.Conv2d(48, 1, 1)  # FC2

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        xm = self.fc1(x)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)

        # xa = self.afc1(x)
        # xa = self.relu(self.afc1bn(xa))
        xa = self.adropout(x)
        xa = self.afc2(xa)
        xm = xm.view(xm.size()[0], xm.size()[1])
        xa = xa.view(xa.size()[0], xa.size()[1])

        return xm, xa


if __name__ == '__main__':
    net = Net()
    print(net)