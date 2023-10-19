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
import os.path

import torch
from torch import optim


def train(net,device,epoch_num,lr,optimizer,loss_fn,train_data,test_data):
    net.to(device)
    net.train()

    for epoch in range(epoch_num):
        train_loss = 0
        correct = 0
        total = 0
        for batch_id, (inputs, targets) in enumerate(train_data):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f"Train Epoch:{epoch + 1} Losss:{100. * train_loss / (total):.2f}% Acc:{100. * correct / total :.2f}%")

        test(net,device,epoch,loss_fn,test_data)

def test(net,device,epoch,loss_fn,test_data):
    test_loss = 0
    correct = 0
    total = 0
    global best_acc
    net.to(device)
    net.eval()

    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(test_data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f"Test Epoch:{epoch + 1} Losss:{100. * test_loss / (total):.2f}% Acc:{100. * correct / total :.2f}%")
        temp = correct / total
        if temp>best_acc:
            best_acc = temp
            print(f"******best_acc={best_acc}")
            torch.save(net.state_dict(), os.path.join("..","saved_models",net.get_name(),".pth"))
