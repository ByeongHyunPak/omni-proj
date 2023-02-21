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

def square_crop_pos(mask, crop_size):
  mask = mask.squeeze(0)
  y, x = torch.nonzero(mask, as_tuple=True)
  idx = random.randint(0, len(x) - 1)

  y0, x0 = y[idx].item(), x[idx].item()
  y1, x1 = y0 + crop_size, x0 + crop_size

  if y1 >= mask.shape[-2] or x1 >= mask.shape[-1]:
    return square_crop_pos(mask, crop_size)

  crop_list = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
  for x_, y_ in crop_list:
    if mask[y_, x_] == 0:
      return square_crop_pos(mask, crop_size)
  return x0, y0

def to_pixel_samples(img):
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb

def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
  
def make_cell(coord, shape):
    coord_bot_left  = coord + torch.tensor([-1/shape[-2], -1/shape[-1]]).unsqueeze(0)
    coord_bot_right = coord + torch.tensor([-1/shape[-2],  1/shape[-1]]).unsqueeze(0)
    coord_top_left  = coord + torch.tensor([ 1/shape[-2], -1/shape[-1]]).unsqueeze(0)
    coord_top_right = coord + torch.tensor([ 1/shape[-2],  1/shape[-1]]).unsqueeze(0)
    coord_left  = coord + torch.tensor([-1/shape[-2], 0]).unsqueeze(0)
    coord_right = coord + torch.tensor([ 1/shape[-2], 0]).unsqueeze(0)
    coord_bot   = coord + torch.tensor([0, -1/shape[-1]]).unsqueeze(0)
    coord_top   = coord + torch.tensor([0,  1/shape[-1]]).unsqueeze(0)

    cell_side   = torch.cat((coord_left, coord_right, coord_bot, coord_top), dim=0)
    cell_corner = torch.cat((coord_bot_left, coord_bot_right, coord_top_left, coord_top_right), dim=0)
    cell = torch.cat((cell_corner, cell_side, coord), dim=0)
    return cell

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark=False

class Averager():
  def __init__(self):
    self.n = 0.0
    self.v = 0.0

  def add(self, v, n=1.0):
    self.v = (self.v * self.n + v * n) / (self.n + n)
    self.n += n

  def item(self):
    return self.v

class Timer():
  def __init__(self):
    self.v = time.time()

  def s(self):
    self.v = time.time()
    
  def t(self):
    return time.time() - self.v

def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)

_log_path = None

def log(obj, filename='log.txt'):
  print(obj)
  if _log_path is not None:
    with open(f"{_log_path}/{filename}", 'a') as f:
      print(obj, file=f)

def set_save_path(save_path):
  global _log_path
  _log_path = save_path
  if os.path.exists(save_path):
    if input(f"{save_path} exists, remove? (y/[n]): ") == 'y':
      shutil.rmtree(save_path)
      os.makedirs(save_path)
  else:
    os.makedirs(save_path)
  return log, SummaryWriter(os.path.join(save_path, 'tensorboard'))

def resize_fn(img, size):
  if isinstance(img, Image.Image): 
    return transforms.Resize(size, Image.BICUBIC)(img)
  
  elif torch.is_tensor(img):
    return transforms.ToTensor()(
      transforms.Resize(size, Image.BICUBIC)(
        transforms.ToPILImage()(img)))

def make_optimizer(param_list, optimizer_spec, load_sd=False):
  Optimizer = {
    'sgd': SGD,
    'adam': Adam,
  }[optimizer_spec['name']]
  optimizer = Optimizer(param_list, **optimizer_spec['args'])
  if load_sd:
    optimizer.load_state_dict(optimizer_spec['sd'])
  return optimizer

def compute_num_params(model, text=False):
  tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
  if text:
    if tot >= 1e6:
      return '{:.1f}M'.format(tot / 1e6)
    else:
      return '{:.1f}K'.format(tot / 1e3)
  else:
    return tot

def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        if isinstance(scale, list):
            valid = diff[..., shave[0]:-shave[0], shave[1]:-shave[1]]
        else:
            valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

def calc_mpsnr(sr, hr, mask, dataset=None, rgb_range=1):
    diff = mask * (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            pass
        else:
            raise NotImplementedError
        valid = diff
    else:
        valid = diff
    mask_factor = sr.shape[-2]*sr.shape[-1]/torch.sum(mask)
    mse = valid.pow(2).mean()*mask_factor
    return -10 * torch.log10(mse)