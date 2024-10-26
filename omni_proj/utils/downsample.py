import random
import numpy as np
import torch
import torch.nn.functional as F
import utils

from torchvision import transforms

FOV = {'erp': 360, 'fis': 180, 'per': 90}

class erpDownsample():
    def __init__(self, inp_size, sample_q):
        self.inp_size = inp_size
        self.sample_q = sample_q
    
    def erp2erp(self, hr_erp_img, s):
        H, W = hr_erp_img.shape[-2:]

        HWy = (H, W)
        HWx = (round(H/s), round(W/s))

        # get hr gridy projected to lr
        gridy = utils.make_coord(HWy)
        gridy2x = gridy.clone()

        # get valid region on lr corresponding to hr
        gridx = utils.make_coord(HWx, flatten=False)
        maskx = torch.ones((1, *HWx))

        # randomly crop on valid region of lr
        x0, y0 = utils.crop_pos(maskx, self.inp_size)
        x1, y1 = x0 + self.inp_size, y0 + self.inp_size
        gridx2z = gridx[y0:y1, x0:x1, :]
        inp = F.grid_sample(hr_erp_img.unsqueeze(0),
            gridx2z.unsqueeze(0).flip(-1), mode='bicubic', 
            align_corners=False).clamp_(0, 1)[0]
        
        # get valid region on hr corresponding to lr crop
        y_min, x_min = gridx[y0, x0, :]
        y_max, x_max = gridx[y1, x1, :]

        yd, xd = (y_max + y_min) / 2, (x_max + x_min) / 2
        dy, dx = (y_max - y_min) / 2, (x_max - x_min) / 2

        gridy2x[:, 0] = (gridy2x[:, 0] - yd) / dy # Normalize to [-1, 1]
        gridy2x[:, 1] = (gridy2x[:, 1] - xd) / dx # Normalize to [-1, 1]

        cropy = torch.where(torch.abs(gridy2x) > 1, 0, 1)
        masky = cropy[:, 0] * cropy[:, 1]

        # sample query points
        sample_lst = np.random.choice(
            torch.nonzero(masky)[:, 0], self.sample_q, replace=False)
        gridy = gridy[sample_lst, :]

        gt = hr_erp_img.view(3, -1)[:, sample_lst].permute(1, 0)

        grid = gridy2x[sample_lst, :]

        cell = utils.shape_estimation(utils.make_cell(gridy, HWy))
        cell[:, [0, 2, 4, 6, 8]] *= HWx[0] # make cell independent to resolution
        cell[:, [1, 3, 5, 7, 9]] *= HWx[1] # make cell independent to resolution

        return {'inp': inp, 'grid': grid, 'cell': cell, 'gt': gt}

    def erp2fis(self, hr_erp_img, s):
        H, W = hr_erp_img.shape[-2:]

        HWy = (H, H)
        HWx = (round(H/s), round(W/s))
        
        # THE = random.uniform(-180, 180)
        # PHI = random.uniform(-90, 90)

        THE = random.uniform(-90, 90)
        PHI = 0

        FOVy = FOV['fis']
        FOVx = FOV['erp']

        # get hr gridy projected to lr
        gridy = utils.make_coord(HWy)
        gridy2x, masky = utils.gridy2x_erp2fis(
            gridy, HWy, HWx, THE, PHI, FOVy, FOVx)

        # get valid region on lr corresponding to hr
        gridx = utils.make_coord(HWx, flatten=False)
        gridx2y, maskx = utils.gridy2x_fis2erp(
            gridx, HWx, HWy, THE, PHI, FOVx, FOVy)
        maskx = maskx.view(1, *HWx)

        # randomly crop on valid region of lr
        x0, y0 = utils.crop_pos(maskx, self.inp_size)
        x1, y1 = x0 + self.inp_size, y0 + self.inp_size
        gridx2z = gridx[y0:y1, x0:x1, :]
        inp = F.grid_sample(hr_erp_img.unsqueeze(0),
            gridx2z.unsqueeze(0).flip(-1), mode='bicubic', 
            align_corners=False).clamp_(0, 1)[0]
        
        # get valid region on hr corresponding to lr crop
        y_min, x_min = gridx[y0, x0, :]
        y_max, x_max = gridx[y1, x1, :]

        yd, xd = (y_max + y_min) / 2, (x_max + x_min) / 2
        dy, dx = (y_max - y_min) / 2, (x_max - x_min) / 2

        gridy2x[:, 0] = (gridy2x[:, 0] - yd) / dy # Normalize to [-1, 1]
        gridy2x[:, 1] = (gridy2x[:, 1] - xd) / dx # Normalize to [-1, 1]

        cropy = torch.where(torch.abs(gridy2x) > 1, 0., 1.)
        masky = masky * cropy[:, 0] * cropy[:, 1]

        # sample query points
        sample_lst = np.random.choice(
            torch.nonzero(masky)[:, 0], self.sample_q, replace=False)
        gridy = gridy[sample_lst, :]
        gridy2z, _ = utils.gridy2x_erp2fis(
            gridy, HWy, (H, W), THE, PHI, FOVy, FOVx)

        gt = F.grid_sample(hr_erp_img.unsqueeze(0), 
            gridy2z.view(1, 1, self.sample_q, 2).flip(-1), mode='bicubic',
            align_corners=False).clamp_(0, 1)[0, :, 0, :].permute(1, 0)

        grid = gridy2x[sample_lst, :]

        cell = utils.celly2x_erp2fis(
            utils.make_cell(gridy, HWy), HWy, HWx, THE, PHI, FOVy, FOVx)
        cell[:, [0, 2, 4, 6, 8]] *= HWx[0] # make cell independent to resolution
        cell[:, [1, 3, 5, 7, 9]] *= HWx[1] # make cell independent to resolution
        
        return {'inp': inp, 'grid': grid, 'cell': cell, 'gt': gt}

    def erp2per(self, hr_erp_img, s):
        H, W = hr_erp_img.shape[-2:]

        HWy = (H//2, H//2)
        HWx = (round(H/s), round(W/s))
        
        # THE = random.uniform(-180, 180)
        # PHI = random.uniform(-90, 90)

        THE = random.uniform(-135, 135)
        PHI = random.uniform(-45, 45)

        FOVy = FOV['per']
        FOVx = FOV['erp']

        # get hr gridy projected to lr
        gridy = utils.make_coord(HWy)
        gridy2x, masky = utils.gridy2x_erp2per(
            gridy, HWy, HWx, THE, PHI, FOVy, FOVx)

        # get valid region on lr corresponding to hr
        gridx = utils.make_coord(HWx, flatten=False)
        gridx2y, maskx = utils.gridy2x_per2erp(
            gridx, HWx, HWy, THE, PHI, FOVx, FOVy)
        maskx = maskx.view(1, *HWx)

        # randomly crop on valid region of lr
        x0, y0 = utils.crop_pos(maskx, self.inp_size)
        x1, y1 = x0 + self.inp_size, y0 + self.inp_size
        gridx2z = gridx[y0:y1, x0:x1, :]
        inp = F.grid_sample(hr_erp_img.unsqueeze(0),
            gridx2z.unsqueeze(0).flip(-1), mode='bicubic', 
            align_corners=False).clamp_(0, 1)[0]

        # get valid region on hr corresponding to lr crop
        y_min, x_min = gridx[y0, x0, :]
        y_max, x_max = gridx[y1, x1, :]

        yd, xd = (y_max + y_min) / 2, (x_max + x_min) / 2
        dy, dx = (y_max - y_min) / 2, (x_max - x_min) / 2

        gridy2x[:, 0] = (gridy2x[:, 0] - yd) / dy # Normalize to [-1, 1]
        gridy2x[:, 1] = (gridy2x[:, 1] - xd) / dx # Normalize to [-1, 1]
        
        cropy = torch.where(torch.abs(gridy2x) > 1, 0., 1.)
        masky = masky * cropy[:, 0] * cropy[:, 1]

        # sample query points
        sample_lst = np.random.choice(
            torch.nonzero(masky)[:, 0], self.sample_q, replace=False)
        gridy = gridy[sample_lst, :]
        gridy2z, _ = utils.gridy2x_erp2per(
            gridy, HWy, (H, W), THE, PHI, FOVy, FOVx)

        gt = F.grid_sample(hr_erp_img.unsqueeze(0),
            gridy2z.view(1, 1, self.sample_q, 2).flip(-1), mode='bicubic',
            align_corners=False).clamp_(0, 1)[0, :, 0, :].permute(1, 0)
        
        grid = gridy2x[sample_lst, :]

        cell = utils.celly2x_erp2per(
            utils.make_cell(gridy, HWy), HWy, HWx, THE, PHI, FOVy, FOVx)
        cell[:, [0, 2, 4, 6, 8]] *= HWx[0] # make cell independent to resolution
        cell[:, [1, 3, 5, 7, 9]] *= HWx[1] # make cell independent to resolution

        return {'inp': inp, 'grid': grid, 'cell': cell, 'gt': gt}

class fisheyeDownsample():
    def __init__(self, inp_size, sample_q):
        self.inp_size = inp_size
        self.sample_q = sample_q
    
    def erp2erp(self, hr_erp_img, s):
        return 
    
    def erp2fis(self, hr_erp_img, s):
        return 

    def erp2per(self, hr_erp_img, s):
        return 