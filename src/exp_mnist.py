# -* UTF-8 *-
'''
==============================================================
@Project -> File : eccn -> exp_mnist.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/10/19 18:55
@Desc :

==============================================================
'''
import os.path

from torch import optim, nn
from torchvision import transforms

from exp_models.mnist.sesn import MNIST_SES_V
from src.exp_models.mnist.mnist_xu import mnist_xu_28
from utils import mnist_data, get_device

from train import train
import utils



data_path = os.path.join("..","data","mnist")
device = get_device()
utils.log("exp_mnist").debug(device)
transform = transforms.Compose([
    transforms.ToTensor()
])
test_data,train_data = mnist_data(data_path=data_path,transform=transform,batch_size=128)
#net = MNIST_SES_V()
net = mnist_xu_28()
epoch_num = 5
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss().to(device)

train(net,
      device,
      epoch_num,
      lr,optimizer,loss_fn,train_data,test_data)