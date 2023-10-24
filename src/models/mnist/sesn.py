# -* UTF-8 *-
'''
==============================================================
@Project -> File : eccn -> sesn.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/10/24 19:51
@Desc :

==============================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.impl.ses_conv import SESConv_Z2_H, SESConv_H_H, SESMaxProjection


class MNIST_SES_V(nn.Module):

    def __init__(self, pool_size=4, kernel_size=11, scales=[1.0], basis_type='A', dropout=0.7, **kwargs):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        self.main = nn.Sequential(
            SESConv_Z2_H(1, C1, kernel_size, 7, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C1),

            SESConv_H_H(C1, C2, 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C2),

            SESConv_H_H(C2, C3, 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size, padding=2),
            nn.BatchNorm2d(C3),
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * C3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def get_name(self):
        return MNIST_SES_V.__name__


if __name__ == '__main__':
    net = MNIST_SES_V()
    print(net)