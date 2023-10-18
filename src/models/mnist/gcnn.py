# -* UTF-8 *-
'''
==============================================================
@Project -> File : eccn -> gcnn.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/10/18 16:49
@Desc :

==============================================================
'''

from torch import nn
from groupy.gconv.chainer_gconv.p4_conv import P4ConvZ2

def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return (P4ConvZ2(in_channels, out_channels, kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),nn.ReLU())




class GCNN(nn.Module):
    def __int__(self):
        super().__init__()
        self.model = nn.Sequential()
        in_channels = 1
        out_channels = 20
        for i in range(6):
            self.model.append(conv_block(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=3))
            in_channels = out_channels

    def forward(self,x):
        return self.model(x)
