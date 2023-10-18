# -* UTF-8 *-
'''
==============================================================
@Project -> File : eccn -> train.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/10/18 19:11
@Desc :

==============================================================
'''
import torch


def train(net,device=torch.device("cpu"),epoch_num,lr=0.1,optimizer,train_data):
    net.to(device)
    net.train()