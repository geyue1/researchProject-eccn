# -* UTF-8 *-
'''
==============================================================
@Project -> File : eccn -> model.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/10/19 16:21
@Desc :

==============================================================
'''

from torch import nn

class SuperNet(nn.Module):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return SuperNet.__name__
