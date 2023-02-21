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

@register('wrapper')
class wrapper(Dataset):
    def __init__(self, dataset, scale_min=2, scale_max=4, 
                 inp_size=48, sample_q=2304, crop_pos_fn='square'):
        self.dataset = dataset
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.inp_size = inp_size
        self.sample_q = sample_q

        if crop_pos_fn == 'square':
            self.crop_pos_fn = utils.square_crop_pos
        else:
            self.crop_pos_fn = utils.square_crop_pos
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # HR ERP image
        hr_erp_img = self.dataset[idx]
        H, W = hr_erp_img.shape[-2:]

        # HR images' shape
        hr_erp_shape = (H, W)
        hr_fish_shape = (H, H)
        hr_pers_shape = (H//2, H//2)

        # downscales
        s = random.uniform(self.scale_min, self.scale_max)

        # LR images' shape
        lr_erp_shape = (int(H/s), int(W/s))
        lr_fish_shape = (int(H/s), int(H/s))
        lr_pers_shape = (int((H//2)/s), int((W//2)/s))

        # prepare data pairs
        prepare = utils.prepare(hr_erp_img, self.inp_size, self.sample_q, self.crop_pos_fn)
        ### 1. lr erp -> hr pers
        erp2pers = prepare.erp2pers(*lr_erp_shape, *hr_pers_shape)
        ### 2. lr erp -> hr fish
        erp2fish = prepare.erp2fish(*lr_erp_shape, *hr_fish_shape)
        ### 3. lr fish -> hr pers
        fish2erp = prepare.fish2pers(*lr_fish_shape, *hr_pers_shape)
        # ### 4. lr fish -> hr erp
        fish2pers = prepare.fish2erp(*lr_fish_shape, *hr_erp_shape)
        # ### 5. lr pers -> hr erp
        pers2erp = prepare.pers2erp(*lr_pers_shape, *hr_erp_shape)
        # ### 6. lr pers -> hr fish
        pers2fish = prepare.pers2fish(*lr_pers_shape, *hr_fish_shape)

        inps = torch.stack([
            erp2pers['inp'],  erp2fish['inp'], fish2erp['inp'],
            fish2pers['inp'], pers2erp['inp'], pers2fish['inp']
        ], dim=0)
        
        grids = torch.stack([
            erp2pers['grid'],  erp2fish['grid'], fish2erp['grid'],
            fish2pers['grid'], pers2erp['grid'], pers2fish['grid']
        ], dim=0)

        cells = torch.stack([
            erp2pers['cell'],  erp2fish['cell'], fish2erp['cell'],
            fish2pers['cell'], pers2erp['cell'], pers2fish['cell']
        ], dim=0)

        gts = torch.stack([
            erp2pers['gt'],  erp2fish['gt'], fish2erp['gt'],
            fish2pers['gt'], pers2erp['gt'], pers2fish['gt']    
        ], dim=0)

        return {'inp': inps, 'grid': grids, 'cell': cells, 'gt': gts}