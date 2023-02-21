import os
import cv2
import time
import math
import shutil
import random

import torch
import numpy as np

from torchvision import transforms
from PIL import Image

from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter

def gridy2x_erp2pers(gridy, H, W, h, w, srcTHE, srcPHI):
    if gridy.dim() == 3:
        gridy = gridy.reshape(-1, 2)
        
    # scaling    
    wFOV = 90
    hFOV = float(H) / W * wFOV
    h_len = h*np.tan(np.radians(hFOV / 2.0))
    w_len = w*np.tan(np.radians(wFOV / 2.0))
    
    gridy = gridy.float()
    gridy[:, 0] *= h_len / h
    gridy[:, 1] *= w_len / w
    gridy = gridy.double()
    
    # H -> negative z-axis, W -> y-axis, place Warped_plane on x-axis
    gridy = gridy.flip(-1)
    gridy = torch.cat((torch.ones(gridy.shape[0], 1), gridy), dim=-1)
    
    # project warped planed onto sphere
    hr_norm = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    gridy /= hr_norm
    
    # set center position (theta, phi)
    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(srcTHE))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(srcPHI))
    
    gridy = torch.mm(torch.from_numpy(R1), gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(torch.from_numpy(R2), gridy.permute(1, 0)).permute(1, 0)

    # find corresponding sphere coordinate 
    lat = torch.arcsin(gridy[:, 2]) / np.pi * 2
    lon = torch.atan2(gridy[:, 1] , gridy[:, 0]) / np.pi
        
    gridx = torch.stack((lat, lon), dim=-1)
    gridx = gridx.float()
    
    # mask 
    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask = mask.float()

    return gridx, mask

def gridy2x_erp2fish(gridy, H, W, h, w, srcTHE, srcPHI):
    if gridy.dim() == 3:
        gridy = gridy.reshape(-1, 2)

    # scaling
    wFOV = 180
    hFOV = float(H) / W * wFOV
    h_len = h*np.sin(np.radians(hFOV / 2.0))
    w_len = w*np.sin(np.radians(wFOV / 2.0))
    
    gridy = gridy.float()
    gridy[:, 0] *= h_len / h
    gridy[:, 1] *= w_len / w
    gridy = gridy.double()
    
    # H -> negative z-axis, W -> y-axis, place Warped_plane on x-axis
    gridy = gridy.flip(-1)
    hr_norm = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    
    mask = torch.where(hr_norm > 1, 0.0, 1.0)
    hr_norm = torch.where(hr_norm > 1, 1.0, hr_norm)
    hr_xaxis = torch.sqrt(1 - hr_norm**2)
    gridy = torch.cat((hr_xaxis, gridy), dim=-1)
    
    # set center position (theta, phi)
    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(srcTHE))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(srcPHI))

    gridy = torch.mm(torch.from_numpy(R1), gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(torch.from_numpy(R2), gridy.permute(1, 0)).permute(1, 0)

    # find corresponding sphere coordinate 
    lat = torch.arcsin(gridy[:, 2].clamp_(-1+1e-6, 1-1e-6)) / np.pi * 2 # clamping to prevent arcsin explosion
    lon = torch.atan2(gridy[:, 1], gridy[:, 0]) / np.pi
    
    gridx = torch.stack((lat, lon), dim=-1)
    gridx = gridx.float()
    
    # mask
    mask = mask.squeeze(-1).float()
    
    return gridx, mask

def gridy2x_fish2erp(gridy, H, W, h, w, tgtTHE, tgtPHI):
    if gridy.dim() == 3:
        gridy = gridy.reshape(-1, 2)

    lat = gridy[:, 0] * np.pi / 2
    lon = gridy[:, 1] * np.pi

    z0 = torch.sin(lat)
    y0 = torch.cos(lat) * torch.sin(lon)
    x0 = torch.cos(lat) * torch.cos(lon)

    gridy = torch.stack((x0, y0, z0), dim=-1).double()

    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(tgtTHE))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(tgtPHI))

    R1_inv = torch.inverse(torch.from_numpy(R1))
    R2_inv = torch.inverse(torch.from_numpy(R2))

    gridy = torch.mm(R2_inv, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R1_inv, gridy.permute(1, 0)).permute(1, 0)

    z0 = gridy[:, 2]
    y0 = gridy[:, 1]
    x0 = gridy[:, 0]

    gridx = torch.stack((z0, y0), dim=-1)
    gridx = gridx.float()
    norm = torch.norm(gridx, p=2, dim=-1)

    mask = torch.where(x0 < 0, 0, 1)
    mask *= torch.where(norm > 1, 0, 1)
    mask = mask.float() 

    return gridx, mask

def gridy2x_fish2pers(gridy, H, W, h, w, srcTHE, srcPHI):
    if gridy.dim() == 3:
        gridy = gridy.reshape(-1, 2)

    # scaling    
    wFOV = 90
    hFOV = float(H) / W * wFOV
    h_len = h*np.tan(np.radians(hFOV / 2.0))
    w_len = w*np.tan(np.radians(wFOV / 2.0))
    
    gridy = gridy.float()
    gridy[:, 0] *= h_len / h
    gridy[:, 1] *= w_len / w
    gridy = gridy.double()
    
    # H -> negative z-axis, W -> y-axis, place Warped_plane on x-axis
    gridy = gridy.flip(-1)
    gridy = torch.cat((torch.ones(gridy.shape[0], 1), gridy), dim=-1)
    
    # project warped planed onto sphere
    hr_norm = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    gridy /= hr_norm
    
    # set center position (theta, phi)
    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(srcTHE))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(srcPHI))
    
    gridy = torch.mm(torch.from_numpy(R1), gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(torch.from_numpy(R2), gridy.permute(1, 0)).permute(1, 0)

    # find corresponding sphere coordinate     
    z0 = gridy[:, 2]
    y0 = gridy[:, 1]
    x0 = gridy[:, 0]

    gridx = torch.stack((z0, y0), dim=-1)
    gridx = gridx.float()
    
    # mask
    mask = torch.where(x0 < 0, 0, 1) # filtering in backplane
    mask = mask.float()
    
    return gridx, mask

def gridy2x_pers2erp(gridy, H, W, h, w, tgtTHE, tgtPHI):
    if gridy.dim() == 3:
        gridy = gridy.reshape(-1, 2)

    lat = gridy[:, 0] * np.pi / 2
    lon = gridy[:, 1] * np.pi

    z0 = torch.sin(lat)
    y0 = torch.cos(lat) * torch.sin(lon)
    x0 = torch.cos(lat) * torch.cos(lon)

    gridy = torch.stack((x0, y0, z0), dim=-1).double()

    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(tgtTHE))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(tgtPHI))

    R1_inv = torch.inverse(torch.from_numpy(R1))
    R2_inv = torch.inverse(torch.from_numpy(R2))

    gridy = torch.mm(R2_inv, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R1_inv, gridy.permute(1, 0)).permute(1, 0)

    z0 = gridy[:, 2]
    y0 = gridy[:, 1]
    x0 = gridy[:, 0]

    gridx = torch.stack((z0/x0, y0/x0), dim=-1)
    gridx = gridx.float()

    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = torch.where(x0 < 0 , 0, 1) * mask[:, 0] * mask[:, 1]
    mask = mask.float()

    return gridx, mask 

def gridy2x_pers2fish(gridy, H, W, h, w, tgtTHE, tgtPHI): 
    if gridy.dim() == 3:
        gridy = gridy.reshape(-1, 2)

    mask = torch.where(torch.norm(gridy, p=2, dim=-1) > 1., 0., 1.)

    # H -> negative z-axis, W -> y-axis, place Warped_plane on x-axis
    x_axis = torch.sqrt(1 - (gridy[:, 0]**2 + gridy[:, 1]**2).clamp_(0, 1))
    gridy = torch.cat((x_axis.unsqueeze(-1), gridy.flip(-1)), dim=-1).double()

    # set center position (theta, phi)
    y_axis = np.array([0.0, 1.0, 0.0], np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], np.float64)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(tgtTHE))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(tgtPHI))

    R1_inv = torch.inverse(torch.from_numpy(R1))
    R2_inv = torch.inverse(torch.from_numpy(R2))

    gridy = torch.mm(R2_inv, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R1_inv, gridy.permute(1, 0)).permute(1, 0)

    # find corresponding sphere coordinate
    z0 = gridy[:, 2]
    y0 = gridy[:, 1]
    x0 = gridy[:, 0]
    
    gridx = torch.stack((z0/x0, y0/x0), dim=-1).float()

    # mask
    mask *= torch.where(x0 < 0, 0, 1)
    mask_ = torch.where(torch.abs(gridx) > 1., 0., 1.)
    mask_ = mask_[:, 0] * mask_[:, 1]
    mask *= mask_
    mask = mask.float()

    return gridx, mask

def celly2x_erp2pers(celly, H, W, h, w, THETA, PHI):
    cellx, _ = gridy2x_erp2pers(celly, H, W, h, w, THETA, PHI)
    return shape_estimation(cellx)

def celly2x_erp2fish(celly, H, W, h, w, THETA, PHI):
    cellx, _ = gridy2x_erp2fish(celly, H, W, h, w, THETA, PHI)
    return shape_estimation(cellx)

def celly2x_fish2pers(celly, H, W, h, w, THETA, PHI):
    cellx, _ = gridy2x_fish2pers(celly, H, W, h, w, THETA, PHI)
    return shape_estimation(cellx)

def celly2x_fish2erp(celly, H, W, h, w, THETA, PHI):
    cellx, _ = gridy2x_fish2erp(celly, H, W, h, w, THETA, PHI)
    return shape_estimation(cellx)

def celly2x_pers2erp(celly, H, W, h, w, THETA, PHI):
    cellx, _ = gridy2x_pers2erp(celly, H, W, h, w, THETA, PHI)
    return shape_estimation(cellx)

def celly2x_pers2fish(celly, H, W, h, w, THETA, PHI):
    cellx, _ = gridy2x_pers2fish(celly, H, W, h, w, THETA, PHI)
    return shape_estimation(cellx)

def shape_estimation(cell):
    # Jacobian
    cell_1 = cell[7*cell.shape[0]//9:8*cell.shape[0]//9, :]\
           - cell[6*cell.shape[0]//9:7*cell.shape[0]//9, :]
    cell_2 = cell[5*cell.shape[0]//9:6*cell.shape[0]//9, :]\
           - cell[4*cell.shape[0]//9:5*cell.shape[0]//9, :]
    
    # Second-order derivatives in Hessian
    cell_3 = cell[7*cell.shape[0]//9:8*cell.shape[0]//9, :]\
           + cell[6*cell.shape[0]//9:7*cell.shape[0]//9, :]\
           - cell[8*cell.shape[0]//9:9*cell.shape[0]//9, :]*2
    cell_4 = cell[5*cell.shape[0]//9:6*cell.shape[0]//9, :]\
           + cell[4*cell.shape[0]//9:5*cell.shape[0]//9, :]\
           - cell[8*cell.shape[0]//9:9*cell.shape[0]//9, :]*2
           
    # Cross-term in Hessian
    cell_5 = cell[3*cell.shape[0]//9:4*cell.shape[0]//9, :]\
           - cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :]\
           - cell[1*cell.shape[0]//9:2*cell.shape[0]//9, :]\
           + cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :]\
           - cell[2*cell.shape[0]//9:3*cell.shape[0]//9, :]\
           + cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] 

    # cat[celly (q, 5), cellx (q, 5)]
    shape = torch.cat((cell_1, cell_2, 4*cell_3, 4*cell_4, cell_5), dim=-1)
    return shape