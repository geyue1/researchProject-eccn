# -* UTF-8 *-
'''
==============================================================
@Project -> File : eccn -> utils.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/10/18 21:01
@Desc :

==============================================================
'''
import os

import torch
from torchvision import datasets, transforms


def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    print('Device: {}'.format(device))
    return device

def mnist_data(data_path,transform,batch_size):

    train_data = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=True,transform=transform,),
        batch_size=batch_size, shuffle=True
    )

    test_data = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(data_path, train=False, download=True,transform=transform,),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    return train_data,test_data



if __name__ == '__main__':
    transform_ = transforms.Compose([
        transforms.ToTensor()
    ])
    path = os.path.join("..","data","mnist")
    print(path)
    mnist_data(path,transform_,128)
