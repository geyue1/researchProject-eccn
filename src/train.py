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
import logging
import os.path

import torch
from torch import optim

import utils

# log_file = os.path.join("..","logs")
# if not os.path.isdir(log_file):
#     os.makedirs(log_file)
# logging.basicConfig(filename=os.path.join("..","logs","train.log"), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = utils.log("train",level=logging.INFO)
best_acc = 0
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

        logger.info(f"Train Epoch:{epoch + 1} Losss:{100. * train_loss / (total):.2f}% Acc:{100. * correct / total :.2f}%")
        logger.info(f"train_loss={train_loss}")
        logger.info(f"train total={total}")
        logger.info(f"train size={len(train_data)}")
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

        logger.info(f"Test Epoch:{epoch + 1} Losss:{100. * test_loss / (total):.2f}% Acc:{100. * correct / total :.2f}%")
        temp = correct / total
        save_flag = True
        if save_flag and temp>best_acc:
            best_acc = temp
            logger.info(f"******best_acc={best_acc}")
            path = os.path.join("..","saved_models")
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(net.state_dict(), os.path.join("..","saved_models",net.get_name()+".pth"))
