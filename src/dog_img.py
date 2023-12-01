# -* UTF-8 *-
'''
==============================================================
@Project -> File : eccn -> dog_img.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/12/01 16:26
@Desc :

==============================================================
'''
import torch.nn
from PIL import Image
import os
from torchvision import transforms

path = os.path.join("C:\\Users\\yge\\Downloads","dog-png-22667.png")
img = Image.open(path)
img = img.resize((28,28))
img_t = transforms.ToTensor()(img)
print(img_t.shape)
conv = torch.nn.Conv2d(4,4,3)
y = conv(img_t)
print(y.shape)
img_2 = transforms.ToPILImage()(y)
img_2 = img_2.resize((537,800))
img_2.save(os.path.join("C:\\Users\\yge\\Downloads","dog-conv3.png"),"PNG")
