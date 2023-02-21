import random
import numpy as np
import torch
import torch.nn.functional as F
import utils

class prepare():
    def __init__(self, hr_erp_img, inp_size, sample_q, crop_pos_fn):
        self.hr_erp_img = hr_erp_img
        self.hr_erp_shape = hr_erp_img.shape[-2:]
        self.inp_size = inp_size
        self.sample_q = sample_q
        self.crop_pos_fn = crop_pos_fn
    
    def erp2pers(self, h, w, H, W):
        the_erp2pers = random.uniform(-90, 90)
        phi_erp2pers = random.uniform(-45, 45)

        # get hr gridy projected to lr
        gridy = utils.make_coord((H, W))
        gridy2x, masky = utils.gridy2x_erp2pers(gridy, 
            H, W, h, w, the_erp2pers, phi_erp2pers)

        # get valid region on lr corresponding to hr
        gridx = utils.make_coord((h, w), flatten=False)
        gridx2y, maskx = utils.gridy2x_pers2erp(gridx,
            h, w, H, W, the_erp2pers, phi_erp2pers)
        maskx = maskx.view(1, h, w)

        # randomly crop on valid region of lr
        x0, y0 = self.crop_pos_fn(maskx, self.inp_size)
        x1, y1 = x0 + self.inp_size, y0 + self.inp_size
        mask = maskx[:, y0:y1, x0:x1]
        gridx2z = gridx[y0:y1, x0:x1, :]
        inp = F.grid_sample(self.hr_erp_img.unsqueeze(0),
            gridx2z.unsqueeze(0).flip(-1), mode='bicubic', 
            align_corners=False).clamp_(0, 1)[0] * mask

        # get valid region on hr corresponding to lr crop
        crop_y_min, crop_x_min = gridx[y0, x0, :]
        crop_y_max, crop_x_max = gridx[y1, x1, :]

        yd, xd = (crop_y_max + crop_y_min) / 2, (crop_x_max + crop_x_min) / 2
        dy, dx = (crop_y_max - crop_y_min) / 2, (crop_x_max - crop_x_min) / 2

        gridy2x[:, 0] = (gridy2x[:, 0] - yd) / dy # Normalize to [-1, 1]
        gridy2x[:, 1] = (gridy2x[:, 1] - xd) / dx # Normalize to [-1, 1]
        
        cropy = torch.where(torch.abs(gridy2x) > 1., 0., 1.)
        masky = masky * cropy[:, 0] * cropy[:, 1]

        # sample query points
        sample_lst = np.random.choice(
            torch.nonzero(masky)[:, 0], self.sample_q, replace=False)
        
        gridy2z, _ = utils.gridy2x_erp2pers(gridy[sample_lst, :], 
            H, W, *self.hr_erp_shape, the_erp2pers, phi_erp2pers)
        gt = F.grid_sample(self.hr_erp_img.unsqueeze(0),
            gridy2z.view(1, 1, self.sample_q, 2).flip(-1), mode='bicubic',
            align_corners=False).clamp_(0, 1)[0, :, 0, :].permute(1, 0)
        
        grid = gridy2x[sample_lst, :]

        cell = utils.celly2x_erp2pers(
            utils.make_cell(gridy[sample_lst, :], (H, W)),
            H, W, h, w, the_erp2pers, phi_erp2pers)
        cell[:, [0, 2, 4, 6, 8]] *= h # make cell independent to resolution
        cell[:, [1, 3, 5, 7, 9]] *= w # make cell independent to resolution

        return {'inp': inp, 'mask': mask, 'grid': grid, 'cell': cell, 'gt': gt}

    def erp2fish(self, h, w, H, W):
        the_erp2fish = random.uniform(-90, 90)
        phi_erp2fish = 0

        # get hr gridy projected to lr
        gridy = utils.make_coord((H, W))
        gridy2x, masky = utils.gridy2x_erp2fish(gridy, 
            H, W, h, w, the_erp2fish, phi_erp2fish)

        # get valid region on lr corresponding to hr
        gridx = utils.make_coord((h, w), flatten=False)
        gridx2y, maskx = utils.gridy2x_fish2erp(gridx,
            h, w, H, W, the_erp2fish, phi_erp2fish)
        maskx = maskx.view(1, h, w)

        # randomly crop on valid region of lr
        x0, y0 = self.crop_pos_fn(maskx, self.inp_size)
        x1, y1 = x0 + self.inp_size, y0 + self.inp_size
        mask = maskx[:, y0:y1, x0:x1]
        gridx2z = gridx[y0:y1, x0:x1, :]
        inp = F.grid_sample(self.hr_erp_img.unsqueeze(0),
            gridx2z.unsqueeze(0).flip(-1), mode='bicubic', 
            align_corners=False).clamp_(0, 1)[0] * mask
        
        # get valid region on hr corresponding to lr crop
        crop_y_min, crop_x_min = gridx[y0, x0, :]
        crop_y_max, crop_x_max = gridx[y1, x1, :]

        yd, xd = (crop_y_max + crop_y_min) / 2, (crop_x_max + crop_x_min) / 2
        dy, dx = (crop_y_max - crop_y_min) / 2, (crop_x_max - crop_x_min) / 2

        gridy2x[:, 0] = (gridy2x[:, 0] - yd) / dy # Normalize to [-1, 1]
        gridy2x[:, 1] = (gridy2x[:, 1] - xd) / dx # Normalize to [-1, 1]

        cropy = torch.where(torch.abs(gridy2x) > 1., 0., 1.)
        masky = masky * cropy[:, 0] * cropy[:, 1]

        # sample query points
        sample_lst = np.random.choice(
            torch.nonzero(masky)[:, 0], self.sample_q, replace=False)
        
        gridy2z, _ = utils.gridy2x_erp2fish(gridy[sample_lst, :], 
            H, W, *self.hr_erp_shape, the_erp2fish, phi_erp2fish)
        gt = F.grid_sample(self.hr_erp_img.unsqueeze(0), 
            gridy2z.view(1, 1, self.sample_q, 2).flip(-1), mode='bicubic',
            align_corners=False).clamp_(0, 1)[0, :, 0, :].permute(1, 0)

        grid = gridy2x[sample_lst, :]

        cell = utils.celly2x_erp2fish(
            utils.make_cell(gridy[sample_lst, :], (H, W)),
            H, W, h, w, the_erp2fish, phi_erp2fish)
        cell[:, [0, 2, 4, 6, 8]] *= h # make cell independent to resolution
        cell[:, [1, 3, 5, 7, 9]] *= w # make cell independent to resolution
        
        return {'inp': inp, 'mask': mask, 'grid': grid, 'cell': cell, 'gt': gt}

    def fish2pers(self, h, w, H, W):
        the_erp2fish = random.uniform(-45, 45)
        phi_erp2fish = 0

        the_fish2pers = random.uniform(-45, 45)
        phi_fish2pers = random.uniform(-45, 45)

        the_erp2pers = the_erp2fish + the_fish2pers
        phi_erp2pers = phi_erp2fish + phi_fish2pers

        # get hr gridy projected to lr
        gridy = utils.make_coord((H, W))
        gridy2x, masky = utils.gridy2x_fish2pers(gridy, 
            H, W, h, w, the_fish2pers, phi_fish2pers)

        # get valid region on lr corresponding to hr
        gridx = utils.make_coord((h, w), flatten=False)
        gridx2y, maskx = utils.gridy2x_pers2fish(gridx,
            h, w, H, W, the_fish2pers, phi_fish2pers)
        maskx = maskx.view(1, h, w)

        # randomly crop on valid region of lr
        x0, y0 = self.crop_pos_fn(maskx, self.inp_size)
        x1, y1 = x0 + self.inp_size, y0 + self.inp_size
        mask = maskx[:, y0:y1, x0:x1]
        gridx2z, _ = utils.gridy2x_erp2fish(gridx[y0:y1, x0:x1, :], 
            h, w, *self.hr_erp_shape, the_erp2fish, phi_erp2fish)
        gridx2z = gridx2z.view(self.inp_size, self.inp_size, 2)
        inp = F.grid_sample(self.hr_erp_img.unsqueeze(0),
            gridx2z.unsqueeze(0).flip(-1), mode='bicubic',
            align_corners=False).clamp_(0, 1)[0] * mask
        
        # get valid region on hr corresponding to lr crop
        crop_y_min, crop_x_min = gridx[y0, x0, :]
        crop_y_max, crop_x_max = gridx[y1, x1, :]

        yd, xd = (crop_y_max + crop_y_min) / 2, (crop_x_max + crop_x_min) / 2
        dy, dx = (crop_y_max - crop_y_min) / 2, (crop_x_max - crop_x_min) / 2

        gridy2x[:, 0] = (gridy2x[:, 0] - yd) / dy # Normalize to [-1, 1]
        gridy2x[:, 1] = (gridy2x[:, 1] - xd) / dx # Normalize to [-1, 1]

        cropy = torch.where(torch.abs(gridy2x) > 1., 0., 1.)
        masky = masky * cropy[:, 0] * cropy[:, 1]

        # sample query points
        sample_lst = np.random.choice(
            torch.nonzero(masky)[:, 0], self.sample_q, replace=False)
        
        gridy2z, _ = utils.gridy2x_erp2pers(gridy[sample_lst, :], 
            H, W, *self.hr_erp_shape, the_erp2pers, phi_erp2pers)
        gt = F.grid_sample(self.hr_erp_img.unsqueeze(0),
            gridy2z.view(1, 1, self.sample_q, 2).flip(-1), mode='bicubic',
            align_corners=False).clamp_(0, 1)[0, :, 0, :].permute(1, 0)
        
        grid = gridy2x[sample_lst, :]

        cell = utils.celly2x_fish2pers(
            utils.make_cell(gridy[sample_lst, :], (H, W)),
            H, W, h, w, the_fish2pers, phi_fish2pers)
        cell[:, [0, 2, 4, 6, 8]] *= h # make cell independent to resolution
        cell[:, [1, 3, 5, 7, 9]] *= w # make cell independent to resolution

        return {'inp': inp, 'mask': mask, 'grid': grid, 'cell': cell, 'gt': gt}

    def fish2erp(self, h, w, H, W):
        the_fish2erp = random.uniform(-90, 90)
        phi_fish2erp = 0

        # get hr gridy projected to lr
        gridy = utils.make_coord((H, W))
        gridy2x, masky = utils.gridy2x_fish2erp(gridy, 
            H, W, h, w, the_fish2erp, phi_fish2erp)

        # get valid region on lr
        gridx = utils.make_coord((h, w), flatten=False)
        gridx2y, maskx = utils.gridy2x_erp2fish(gridx,
            h, w, H, W, the_fish2erp, phi_fish2erp)
        maskx = maskx.view(1, h, w)

        # randomly crop on valid region of lr
        x0, y0 = self.crop_pos_fn(maskx, self.inp_size)
        x1, y1 = x0 + self.inp_size, y0 + self.inp_size
        mask = maskx[:, y0:y1, x0:x1]
        gridx2z, _ = utils.gridy2x_erp2fish(gridx[y0:y1, x0:x1, :],
            h, w, *self.hr_erp_shape, the_fish2erp, phi_fish2erp)
        gridx2z = gridx2z.view(self.inp_size, self.inp_size, 2)
        inp = F.grid_sample(self.hr_erp_img.unsqueeze(0),
            gridx2z.unsqueeze(0).flip(-1), mode='bicubic',
            align_corners=False).clamp_(0, 1)[0] * mask
        
        # get valid region on hr corresponding to lr crop
        crop_y_min, crop_x_min = gridx[y0, x0, :]
        crop_y_max, crop_x_max = gridx[y1, x1, :]

        yd, xd = (crop_y_max + crop_y_min) / 2, (crop_x_max + crop_x_min) / 2
        dy, dx = (crop_y_max - crop_y_min) / 2, (crop_x_max - crop_x_min) / 2

        gridy2x[:, 0] = (gridy2x[:, 0] - yd) / dy # Normalize to [-1, 1]
        gridy2x[:, 1] = (gridy2x[:, 1] - xd) / dx # Normalize to [-1, 1]

        cropy = torch.where(torch.abs(gridy2x) > 1., 0., 1.)
        masky = masky * cropy[:, 0] * cropy[:, 1]

        # sample query points
        sample_lst = np.random.choice(
            torch.nonzero(masky)[:, 0], self.sample_q, replace=False)
        
        gridy2z, _ = gridy[sample_lst, :], None
        gt = F.grid_sample(self.hr_erp_img.unsqueeze(0),
            gridy2z.view(1, 1, self.sample_q, 2).flip(-1), mode='nearest',
            align_corners=False).clamp_(0, 1)[0, :, 0, :].permute(1, 0)
        
        grid = gridy2x[sample_lst, :]

        cell = utils.celly2x_fish2erp(
            utils.make_cell(gridy[sample_lst, :], (H, W)), 
            H, W, h, w, the_fish2erp, phi_fish2erp)
        cell[:, [0, 2, 4, 6, 8]] *= h # make cell independent to resolution
        cell[:, [1, 3, 5, 7, 9]] *= w # make cell independent to resolution

        return {'inp': inp, 'mask': mask, 'grid': grid, 'cell': cell, 'gt': gt}

    def pers2erp(self, h, w, H, W):
        the_pers2erp = random.uniform(-90, 90)
        phi_pers2erp = random.uniform(-45, 45)

        # get hr gridy projected to lr
        gridy = utils.make_coord((H, W))
        gridy2x, masky = utils.gridy2x_pers2erp(gridy,
            H, W, h, w, the_pers2erp, phi_pers2erp)

        # get valid region on lr
        gridx = utils.make_coord((h, w), flatten=False)
        gridx2y, maskx = utils.gridy2x_erp2pers(gridx,
            h, w, H, W, the_pers2erp, phi_pers2erp)
        maskx = maskx.view(1, h, w)

        # randomly crop on valid region of lr
        x0, y0 = self.crop_pos_fn(maskx, self.inp_size)
        x1, y1 = x0 + self.inp_size, y0 + self.inp_size
        mask = maskx[:, y0:y1, x0:x1]
        gridx2z, _ = utils.gridy2x_erp2pers(gridx[y0:y1, x0:x1, :],
            h, w, *self.hr_erp_shape, the_pers2erp, phi_pers2erp)
        gridx2z = gridx2z.view(self.inp_size, self.inp_size, 2)
        inp = F.grid_sample(self.hr_erp_img.unsqueeze(0),
            gridx2z.unsqueeze(0).flip(-1), mode='bicubic',
            align_corners=False).clamp_(0, 1)[0] * mask

        # get valid region on hr corresponding to lr crop
        crop_y_min, crop_x_min = gridx[y0, x0, :]
        crop_y_max, crop_x_max = gridx[y1, x1, :]

        yd, xd = (crop_y_max + crop_y_min) / 2, (crop_x_max + crop_x_min) / 2
        dy, dx = (crop_y_max - crop_y_min) / 2, (crop_x_max - crop_x_min) / 2

        gridy2x[:, 0] = (gridy2x[:, 0] - yd) / dy # Normalize to [-1, 1]
        gridy2x[:, 1] = (gridy2x[:, 1] - xd) / dx # Normalize to [-1, 1]

        cropy = torch.where(torch.abs(gridy2x) > 1., 0., 1.)
        masky = masky * cropy[:, 0] * cropy[:, 1]

        # sample query points
        sample_lst = np.random.choice(
            torch.nonzero(masky)[:, 0], self.sample_q, replace=False)
        
        gridy2z, _ = gridy[sample_lst, :], None
        gt = F.grid_sample(self.hr_erp_img.unsqueeze(0),
            gridy2z.view(1, 1, self.sample_q, 2).flip(-1), mode='nearest',
            align_corners=False).clamp_(0, 1)[0, :, 0, :].permute(1, 0)
        
        grid = gridy2x[sample_lst, :]

        cell = utils.celly2x_pers2erp(
            utils.make_cell(gridy[sample_lst, :], (H, W)),
            H, W, h, w, the_pers2erp, phi_pers2erp)
        cell[:, [0, 2, 4, 6, 8]] *= h # make cell independent to resolution
        cell[:, [1, 3, 5, 7, 9]] *= w # make cell independent to resolution

        return {'inp': inp, 'mask': mask, 'grid': grid, 'cell': cell, 'gt': gt}

    def pers2fish(self, h, w, H, W):
        the_erp2fish = random.uniform(-45, 45)
        phi_erp2fish = 0

        the_pers2fish = random.uniform(-45, 45)
        phi_pers2fish = random.uniform(-45, 45)

        the_erp2pers = the_erp2fish + the_pers2fish
        phi_erp2pers = phi_erp2fish + phi_pers2fish

        # get hr gridy projected to lr
        gridy = utils.make_coord((H, W))
        gridy2x, masky = utils.gridy2x_pers2fish(gridy, 
            H, W, h, w, the_pers2fish, phi_pers2fish)

        # get valid region on lr
        gridx = utils.make_coord((h, w), flatten=False)
        gridx2y, maskx = utils.gridy2x_fish2pers(gridx,
            h, w, H, W, the_pers2fish, phi_pers2fish)
        maskx = maskx.view(1, h, w)

        # randomly crop on valid region of lr
        x0, y0 = self.crop_pos_fn(maskx, self.inp_size)
        x1, y1 = x0 + self.inp_size, y0 + self.inp_size
        mask = maskx[:, y0:y1, x0:x1]
        gridx2z, _ = utils.gridy2x_erp2pers(gridx[y0:y1, x0:x1, :],
            h, w, *self.hr_erp_shape, the_erp2pers, phi_erp2pers)
        gridx2z = gridx2z.view(self.inp_size, self.inp_size, 2)
        inp = F.grid_sample(self.hr_erp_img.unsqueeze(0),
            gridx2z.unsqueeze(0).flip(-1), mode='bicubic', 
            align_corners=False).clamp_(0, 1)[0] * mask

        # get valid region on hr corresponding to lr crop
        crop_y_min, crop_x_min = gridx[y0, x0, :]
        crop_y_max, crop_x_max = gridx[y1, x1, :]

        yd, xd = (crop_y_max + crop_y_min) / 2, (crop_x_max + crop_x_min) / 2
        dy, dx = (crop_y_max - crop_y_min) / 2, (crop_x_max - crop_x_min) / 2

        gridy2x[:, 0] = (gridy2x[:, 0] - yd) / dy # Normalize to [-1, 1]
        gridy2x[:, 1] = (gridy2x[:, 1] - xd) / dx # Normalize to [-1, 1]

        cropy = torch.where(torch.abs(gridy2x) > 1., 0., 1.)
        masky = masky * cropy[:, 0] * cropy[:, 1]

        # sample query points
        sample_lst = np.random.choice(
            torch.nonzero(masky)[:, 0], self.sample_q, replace=False)
        
        gridy2z, _ = utils.gridy2x_erp2fish(gridy[sample_lst, :], 
            H, W, *self.hr_erp_shape, the_erp2fish, phi_erp2fish)
        gt = F.grid_sample(self.hr_erp_img.unsqueeze(0),
            gridy2z.view(1, 1, self.sample_q, 2).flip(-1), mode='bicubic',
            align_corners=False).clamp_(0, 1)[0, :, 0, :].permute(1, 0)
        
        grid = gridy2x[sample_lst, :]

        cell = utils.celly2x_pers2fish(
            utils.make_cell(gridy[sample_lst, :], (H, W)),
            H, W, h, w, the_pers2fish, phi_pers2fish)
        cell[:, [0, 2, 4, 6, 8]] *= h # make cell independent to resolution
        cell[:, [1, 3, 5, 7, 9]] *= w # make cell independent to resolution

        return {'inp': inp, 'mask': mask, 'grid': grid, 'cell': cell, 'gt': gt}