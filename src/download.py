# -* UTF-8 *-
'''
==============================================================
@Project -> File : eccn -> download.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/11/28 19:31
@Desc :

==============================================================
'''
import os

import torchvision
from torchvision import transforms
from PIL import Image
import utils


def download_():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_path = os.path.join("..", "data", "mnist")
    images_dir = os.path.join("..", "data", "images2")
    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)
    train_data,_ = utils.mnist_data(data_path,transform,1)
    count = 0
    for i, (image, label) in enumerate(train_data):
        count+=1
        image = image.view(image.size()[2],-1)
        image_pil = transforms.ToPILImage()(image)
        image_pil = image_pil.resize((128,128))
        image_path = os.path.join(images_dir, f'{i}.png')
        print(label)
        #torchvision.utils.save_image(image_pil, image_path)
        image_pil.save(image_path)
        if count==10:
            break

if __name__ == '__main__':
    download_()