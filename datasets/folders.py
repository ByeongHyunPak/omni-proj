import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob

from datasets import register

@register('erp-folder')
class erpFolder(Dataset):
    def __init__(self, root_path, repeat=1, first_k=None, cache='None'):
        self.repeat = repeat
        self.cache = cache

        self.imgs = []
        files = glob(f"{root_path}/*.png")
        if first_k is not None:
            files = files[:first_k]

        for file in files:
            file_num = file.split('/')[-1][:-len('.png')]
            img = f"{root_path}/{file_num}.png"

            if cache == 'none':
                self.imgs.append(img)

            elif cache == 'in_memory':
                self.imgs.append(transforms.ToTensor()(
                    Image.open(img).convert('RGB')))
        
    def __len__(self):
        return len(self.imgs) * self.repeat

    def __getitem__(self, idx):
        img = self.imgs[idx % len(self.imgs)]

        if self.cache == 'none':
            return transforms.ToTensor()(
                Image.open(img).convert('RGB'))

        elif self.cache == 'in_memory':
            return img
