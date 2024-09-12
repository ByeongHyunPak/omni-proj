import random
import torch

from torch.utils.data import Dataset
from datasets import register

import utils

@register('erp-downsample')
class erpDownsampleWrapper(Dataset):
    def __init__(self, dataset, scale_min=2, scale_max=4, 
                 inp_size=48, sample_q=2304, ys=['erp', 'fis', 'per']):
        self.dataset = dataset
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.downsample = utils.erpDownsample(inp_size, sample_q)
        self.ys = ys
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # HR ERP image
        hr_erp_img = self.dataset[idx]

        # prepare data pairs
        downsample_fn = {
            'erp2erp': self.downsample.erp2erp,
            'erp2fis': self.downsample.erp2fis,
            'erp2per': self.downsample.erp2per,
        }; inps, grids, cells, gts = [], [], [], []
        for y in self.ys:
            data = downsample_fn[f'erp2{y}'](hr_erp_img,
                random.uniform(self.scale_min, self.scale_max))
            inps.append(data['inp'])
            grids.append(data['grid'])
            cells.append(data['cell'])
            gts.append(data['gt'])

        return {'inp' : torch.stack(inps), 
                'grid': torch.stack(grids), 
                'cell': torch.stack(cells), 
                'gt'  : torch.stack(gts)}

@register('fisheye-downsample')
class fisheyeDownsampleWrapper(Dataset):
    def __init__(self, dataset, scale_min=2, scale_max=4, 
                 inp_size=48, sample_q=2304):
        self.dataset = dataset
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.inp_size = inp_size
        self.sample_q = sample_q