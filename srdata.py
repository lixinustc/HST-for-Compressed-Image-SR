import os

import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import ToTensor
import cv2
import random
import glob


class Data_Train(data.Dataset):
    def __init__(self, patch_size=64):
        self.scale = 4
        self.patch_size = patch_size

        self.dir_hr = sorted(glob.glob('PATH_TO_CLIC/*') + glob.glob('PATH_TO_DF2K/*'))
        self.dir_lr = sorted(glob.glob('PATH_TO_CLIC_COMP/*') + glob.glob('PATH_TO_DF2K_COMP/*'))

    def __getitem__(self, idx):
        name_hr = self.dir_hr[idx]
        name_lr = self.dir_lr[idx]
        
        number_hr = os.path.basename(name_hr).split('.')[0]
        number_lr = os.path.basename(name_lr).split('.')[0]

        assert number_hr == number_lr

        hr = cv2.cvtColor(cv2.imread(name_hr), cv2.COLOR_BGR2RGB)
        lr = cv2.cvtColor(cv2.imread(name_lr), cv2.COLOR_BGR2RGB)
        hmax, wmax, _ = lr.shape

        crop_h = np.random.randint(0, hmax-self.patch_size)
        crop_w = np.random.randint(0, wmax-self.patch_size)

        hr = hr[crop_h*4:(crop_h+self.patch_size)*4, crop_w*4:(crop_w+self.patch_size)*4, ...]
        lr = lr[crop_h:crop_h+self.patch_size, crop_w:crop_w+self.patch_size, ...]

        mode = random.randint(0, 7)

        lr, hr = augment_img(lr, mode=mode), augment_img(hr, mode=mode)

        lr = ToTensor()(lr.copy())
        hr = ToTensor()(hr.copy())

        output = {'L': lr, 'H': hr, 'N': number_hr}
        return output

    def __len__(self):
        return len(self.dir_hr)


class Data_Test(data.Dataset):
    def __init__(self):

        self.dir_hr = 'PATH_TO_DIV2K_VALID'
        self.dir_lr = 'PATH_TO_DIV2K_VALID_COMP'
        self.name_hr = sorted(os.listdir(self.dir_hr))

    def __getitem__(self, idx):
        name = self.name_hr[idx]
        
        number = name.split('.')[0]

        hr = cv2.cvtColor(cv2.imread(os.path.join(self.dir_hr, name)), cv2.COLOR_BGR2RGB)
        lr = cv2.cvtColor(cv2.imread(os.path.join(self.dir_lr, number+'.jpg')), cv2.COLOR_BGR2RGB)

        lr = ToTensor()(lr)
        hr = ToTensor()(hr)
        output = {'L': lr, 'H': hr, 'N': number}
        return output

    def __len__(self):
        return len(self.name_hr)


def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))