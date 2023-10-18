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

import torch
def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    print('Device: {}'.format(device))
    return device

if __name__ == '__main__':
    get_device()
