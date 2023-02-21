import functools
import random
import math
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
import utils

@register('erp-downsample')
class erpDownsampleWrapper(Dataset):
    def __init__(self, dataset, scale_min=2, scale_max=4, 
                 inp_size=48, sample_q=2304):
        self.dataset = dataset
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.downsample = utils.erpDownsample(inp_size, sample_q)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # HR ERP image
        hr_erp_img = self.dataset[idx]

        # downscale ratio
        s1 = random.uniform(self.scale_min, self.scale_max) # erp2erp
        s2 = random.uniform(self.scale_min, self.scale_max) # erp2fis
        s3 = random.uniform(self.scale_min, self.scale_max) # erp2per

        # prepare data pairs
        erp2erp = self.downsample.erp2erp(hr_erp_img, s1)
        erp2fis = self.downsample.erp2fis(hr_erp_img, s2)
        erp2per = self.downsample.erp2per(hr_erp_img, s3)

        inps  = torch.stack([erp2erp['inp'], erp2fis['inp'], erp2per['inp']])
        grids = torch.stack([erp2erp['grid'], erp2fis['grid'], erp2per['grid']])
        cells = torch.stack([erp2erp['cell'], erp2fis['cell'], erp2per['cell']])
        gts   = torch.stack([erp2erp['gt'], erp2fis['gt'], erp2per['gt']])
        
        return {'inp': inps, 'grid': grids, 'cell': cells, 'gt': gts}

@register('fisheye-downsample')
class fisheyeDownsampleWrapper(Dataset):
    def __init__(self, dataset, scale_min=2, scale_max=4, 
                 inp_size=48, sample_q=2304):
        self.dataset = dataset
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.inp_size = inp_size
        self.sample_q = sample_q