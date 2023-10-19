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
import torch.nn.functional as F
from groupy.gconv.pytorch_gconv import P4ConvZ2

from src.models.model import SuperNet


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return (P4ConvZ2(in_channels, out_channels, kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),nn.ReLU())


class BN_Block(SuperNet):
    def __init__(self,conv,bn=True,act=nn.ReLU()):
        super(BN_Block,self).__init__()
        self.conv = conv
        if bn:
            out_channels = self.conv.out_channels
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.act = act
    def forward(self,x):
        y = self.conv(x)
        if self.bn:
            y = self.bn(y)
        if self.act:
            y = self.act(y)
        return y
class Z2CNN(SuperNet):
    def __init__(self):
        super(Z2CNN, self).__init__()
        kernel_size = 3
        bn = True
        act = nn.ReLU()
        self.dr = 0.3
        self.layer_1 = BN_Block(
            conv = nn.Conv2d(in_channels=1,out_channels=20,kernel_size=kernel_size,stride=1,padding=0),
            bn = bn,
            act = act
        )
        self.layer_2 = BN_Block(
            conv=nn.Conv2d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act
        )
        self.layer_3 = BN_Block(
            conv=nn.Conv2d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act
        )
        self.layer_4 = BN_Block(
            conv=nn.Conv2d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act
        )
        self.layer_5 = BN_Block(
            conv=nn.Conv2d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act
        )
        self.layer_6 = BN_Block(
            conv=nn.Conv2d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act
        )
        self.layer_7 = nn.Conv2d(in_channels=20,out_channels=10  ,kernel_size=4,stride=1,padding=0)

        self.fc = nn.Linear(10, 10)

    def forward(self,x):
        y = self.layer_1(x)
        y = F.dropout(y,self.dr,training=True)

        y = self.layer_2(y)
        y = F.max_pool2d(y,kernel_size=2,stride=2,padding=0)

        y = self.layer_3(y)
        y = F.dropout(y, self.dr, training=True)

        y = self.layer_4(y)
        y = F.dropout(y, self.dr, training=True)

        y = self.layer_5(y)
        y = F.dropout(y, self.dr, training=True)

        y = self.layer_6(y)
        y = F.dropout(y, self.dr, training=True)

        y = self.layer_7(y)
        y = self.fc(y)

        return F.softmax(y)
    def get_name(self):
        return Z2CNN.__name__



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

    def get_name(self):
        return GCNN.__name__


if __name__ == '__main__':
    net = GCNN()
    print(net.get_name())
