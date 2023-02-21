import os
import argparse

import yaml
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils

def make_data_loader(spec, tag=''):
  if spec is None:
    return None
  
  dataset = datasets.make(spec['dataset'])
  dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

  log(f"{tag} dataset: size={len(dataset)}")
  for k, v in dataset[0].items():
    log(f"  - {k}: shape={tuple(v.shape)}")

  return DataLoader(dataset, batch_size=spec['batch_size'],
    shuffle=(tag=='train'), num_workers=16, pin_memory=True)

def make_data_loaders():
  train_loader = make_data_loader(config.get('train_dataset'), tag='train')
  valid_loader = make_data_loader(config.get('valid_dataset'), tag='valid')
  return train_loader, valid_loader

def prepare_training():
  if os.path.exists(config.get('resume')):
    sv_file = torch.load(config['resume'])
    model = models.make(sv_file['model'], load_sd=True).cuda()
    optimizer = utils.make_optimizer(
      model.parameters(), sv_file['optimizer'], load_sd=True)
    epoch_start = sv_file['epoch'] + 1
    if config.get('multi_step_lr') is None:
      lr_scheduler = None
    else:
      lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
    for _ in range(epoch_start - 1):
      lr_scheduler.step()

  else:
    model = models.make(config['model']).cuda()
    optimizer = utils.make_optimizer(
      model.parameters(), config['optimizer'])
    epoch_start = 1
    if config.get('multi_step_lr') is None:
      lr_scheduler = None
    else:
      lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

  log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
  return model, optimizer, epoch_start, lr_scheduler

def train(train_loader, model, optimizer, epoch):
  model.train()
  loss_fn = nn.L1Loss()
  train_loss = utils.Averager()

  data_norm = config['data_norm']
  sub = torch.FloatTensor(data_norm['sub']).view(1, -1, 1, 1).cuda()
  div = torch.FloatTensor(data_norm['div']).view(1, -1, 1, 1).cuda()

  batch_size = config.get('train_dataset')['batch_size']
  repeat = config.get('train_dataset')['dataset']['args']['repeat']
  iter_per_epoch = len(train_loader) // batch_size * repeat

  iter_ = 0
  for batch in tqdm(train_loader, leave=False, desc='train'):
    for k, v in batch.items():
      batch[k] = v.cuda()
    
    inp = (batch['inp'] - sub) / div
    gt = (batch['gt'] - sub) / div
    grid = batch['grid']
    cell = batch['cell']
    
    inp = inp.view(inp.shape[0] * inp.shape[1], *inp.shape[2:])
    grid = grid.view(grid.shape[0] * grid.shape[1], *grid.shape[2:])
    cell = cell.view(cell.shape[0] * cell.shape[1], *cell.shape[2:])
    gt = gt.view(gt.shape[0] * gt.shape[1], *gt.shape[2:])

    hr = model(inp, grid, cell)
    loss = loss_fn(hr, gt)
    train_loss.add(loss.item(), gt.shape[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # tensorboard
    writer.add_scalars('loss', {'train': loss.item()}, (epoch-1)*iter_per_epoch+iter_)
    iter_ += 1

    hr, loss = None, None

  return train_loss.item()


def valid(valid_loader, model, epoch):
  model.eval()
  metric_fn = utils.calc_psnr
  valid_psnr = utils.Averager()

  data_norm = config['data_norm']
  sub = torch.FloatTensor(data_norm['sub']).view(1, -1, 1, 1).cuda()
  div = torch.FloatTensor(data_norm['div']).view(1, -1, 1, 1).cuda()

  batch_size = config.get('valid_dataset')['batch_size']
  repeat = config.get('valid_dataset')['dataset']['args']['repeat']
  iter_per_epoch = len(valid_loader) // batch_size * repeat

  iter_ = 0
  for batch in tqdm(valid_loader, leave=False, desc='valid'):
    for k, v in batch.items():
      batch[k] = v.cuda()
  
    inp = (batch['inp'] - sub) / div
    gt = batch['gt']
    grid = batch['grid']
    cell = batch['cell']
    
    inp = inp.view(inp.shape[0] * inp.shape[1], *inp.shape[2:])
    grid = grid.view(grid.shape[0] * grid.shape[1], *grid.shape[2:])
    cell = cell.view(cell.shape[0] * cell.shape[1], *cell.shape[2:])
    gt = gt.view(gt.shape[0] * gt.shape[1], *gt.shape[2:])

    hr = model(inp, grid, cell)
    hr = (hr * div + sub).clamp_(0, 1)
    psnr = metric_fn(hr, gt)
    valid_psnr.add(psnr.item(), gt.shape[0])

    # tensorboard
    writer.add_scalars('psnr', {'valid': psnr.item()}, (epoch-1)*iter_per_epoch + iter_)
    iter_ += 1

    hr, psnr = None, None

  return valid_psnr.item()    

def main(config_, save_path):
  global config, log, writer
  config = config_
  log, writer = utils.set_save_path(save_path)

  with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
    yaml.dump(config, f, sort_keys=False)
  
  if config.get('data_norm') is None:
    config['data_norm'] = {'sub': [0], 'div': [1]}
  
  train_loader, valid_loader = make_data_loaders()
  model, optimizer, epoch_start, lr_scheduler = prepare_training()

  n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
  if n_gpus > 1: model = nn.parallel.DataParallel(model)

  epoch_max = config['epoch_max']
  epoch_save = config['epoch_save']
  epoch_valid = config['epoch_valid']
  
  timer = utils.Timer()
  max_valid_psnr = 0

  for epoch in range(epoch_start, epoch_max + 1):
    t_epoch_start = timer.t()
    log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

    log_info.append('lr: {}'.format(optimizer.param_groups[0]['lr']))
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

    train_loss = train(train_loader, model, optimizer, epoch)

    if lr_scheduler is not None:
      lr_scheduler.step()

    log_info.append('train: loss={:.4f}'.format(train_loss))
    writer.add_scalars('loss', {'train': train_loss}, epoch)

    model_ = model.module if n_gpus > 1 else model
    model_spec = config['model']
    model_spec['sd'] = model_.state_dict()
    optimizer_spec = config['optimizer']
    optimizer_spec['sd'] = optimizer.state_dict()
    sv_file = {
      'model': model_spec,
      'optimizer': optimizer_spec,
      'epoch': epoch
    }
    torch.save(sv_file, os.path.join(save_path, 'epoch_last.pth'))

    if (epoch_save is not None) and (epoch % epoch_save == 0):
      torch.save(sv_file, f"{save_path}/epoch_{epoch}.pth")
        
    if valid_loader is not None:
      if (epoch_valid is not None) and (epoch % epoch_valid == 0):
        with torch.no_grad():
          valid_psnr = valid(valid_loader, model_, epoch)
          log_info.append('valid: psnr={:.4f}'.format(valid_psnr))
          writer.add_scalars('psnr', {'valid': valid_psnr}, epoch)

          if valid_psnr > max_valid_psnr:
            max_valid_psnr = valid_psnr
            torch.save(sv_file, f"{save_path}/epoch_best.pth")

    t = timer.t()
    prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
    t_epoch = utils.time_text(t - t_epoch_start)
    t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
    log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

    log(', '.join(log_info))
    writer.flush()
  
if __name__ == '__main__':
  import warnings
  warnings.filterwarnings('ignore')
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--config')
  parser.add_argument('--gpu', default='0')
  parser.add_argument('--resume', default=False)
  args = parser.parse_args()

  utils.set_seed(2023)
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    if args.resume is False:
      config['resume'] = '~'
    print('config loaded.')

  save_path = f"./save/{args.config.split('/')[-1][:-len('.yaml')]}"
  main(config, save_path)