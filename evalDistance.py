from __future__ import print_function

import sys
import os
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

from PIL import Image
import time
import os.path as osp
from os import listdir
from os.path import isfile, join
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import random

from aligned_reid.dataset import create_dataset
from aligned_reid.model.Model import Model
from aligned_reid.model.TripletLoss import TripletLoss
from aligned_reid.model.loss import global_loss
from aligned_reid.model.loss import local_loss

from aligned_reid.utils.utils import time_str
from aligned_reid.utils.utils import str2bool
from aligned_reid.utils.utils import tight_float_str as tfs
from aligned_reid.utils.utils import may_set_mode
from aligned_reid.utils.utils import load_state_dict
from aligned_reid.utils.utils import load_ckpt
from aligned_reid.utils.utils import save_ckpt
from aligned_reid.utils.utils import set_devices
from aligned_reid.utils.utils import AverageMeter
from aligned_reid.utils.utils import to_scalar
from aligned_reid.utils.utils import ReDirectSTD
from aligned_reid.utils.utils import set_seed
from aligned_reid.utils.utils import adjust_lr_exp
from aligned_reid.utils.utils import adjust_lr_staircase

from aligned_reid.dataset.PreProcessImage import PreProcessIm
from aligned_reid.utils.distance import compute_dist
from aligned_reid.utils.distance import normalize

class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
    parser.add_argument('-r', '--run', type=int, default=1)
    parser.add_argument('--set_seed', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke', 'combined','msmt17','douyin'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    # Only for training set.
    parser.add_argument('--resize_h_w', type=eval, default=(256, 128))
    parser.add_argument('--crop_prob', type=float, default=0)
    parser.add_argument('--crop_ratio', type=float, default=1)
    parser.add_argument('--ids_per_batch', type=int, default=32)
    parser.add_argument('--ims_per_id', type=int, default=4)

    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--normalize_feature', type=str2bool, default=True)
    parser.add_argument('--local_dist_own_hard_sample',
                        type=str2bool, default=False)
    parser.add_argument('-gm', '--global_margin', type=float, default=0.3)
    parser.add_argument('-lm', '--local_margin', type=float, default=0.3)
    parser.add_argument('-glw', '--g_loss_weight', type=float, default=1.)
    parser.add_argument('-llw', '--l_loss_weight', type=float, default=0.)
    parser.add_argument('-idlw', '--id_loss_weight', type=float, default=0.)

    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--model_weight_file', type=str, default='')

    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay_type', type=str, default='exp',
                        choices=['exp', 'staircase'])
    parser.add_argument('--exp_decay_at_epoch', type=int, default=76)
    parser.add_argument('--staircase_decay_at_epochs',
                        type=eval, default=(101, 201,))
    parser.add_argument('--staircase_decay_multiply_factor',
                        type=float, default=0.1)
    parser.add_argument('--total_epochs', type=int, default=150)

    args = parser.parse_known_args()[0]

    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None

    # The experiments can be run for several times and performances be averaged.
    # `run` starts from `1`, not `0`.
    self.run = args.run

    ###########
    # Dataset #
    ###########

    # If you want to exactly reproduce the result in training, you have to set
    # num of threads to 1.
    if self.seed is not None:
      self.prefetch_threads = 1
    else:
      self.prefetch_threads = 2

    self.dataset = args.dataset
    self.trainset_part = args.trainset_part

    # Image Processing

    # Just for training set
    self.crop_prob = args.crop_prob
    self.crop_ratio = args.crop_ratio
    self.resize_h_w = args.resize_h_w

    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
    self.im_std = [0.229, 0.224, 0.225]

    self.ids_per_batch = args.ids_per_batch
    self.ims_per_id = args.ims_per_id
    self.train_final_batch = False
    self.train_mirror_type = ['random', 'always', None][0]
    self.train_shuffle = True

    self.test_batch_size = 32
    self.test_final_batch = True
    self.test_mirror_type = ['random', 'always', None][2]
    self.test_shuffle = False

    dataset_kwargs = dict(
      name=self.dataset,
      resize_h_w=self.resize_h_w,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.train_set_kwargs = dict(
      part=self.trainset_part,
      ids_per_batch=self.ids_per_batch,
      ims_per_id=self.ims_per_id,
      final_batch=self.train_final_batch,
      shuffle=self.train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.train_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.test_set_kwargs = dict(
      part='test',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.test_set_kwargs.update(dataset_kwargs)

    ###############
    # ReID Model  #
    ###############

    self.local_dist_own_hard_sample = args.local_dist_own_hard_sample

    self.normalize_feature = args.normalize_feature

    self.local_conv_out_channels = 128
    self.global_margin = args.global_margin
    self.local_margin = args.local_margin

    # Identification Loss weight
    self.id_loss_weight = args.id_loss_weight

    # global loss weight
    self.g_loss_weight = args.g_loss_weight
    # local loss weight
    self.l_loss_weight = args.l_loss_weight

    #############
    # Training  #
    #############

    self.weight_decay = 0.0005

    # Initial learning rate
    self.base_lr = args.base_lr
    self.lr_decay_type = args.lr_decay_type
    self.exp_decay_at_epoch = args.exp_decay_at_epoch
    self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
    self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
    # Number of epochs to train
    self.total_epochs = args.total_epochs

    # How often (in batches) to log. If only need to log the average
    # information for each epoch, set this to a large value, e.g. 1e10.
    self.log_steps = 1e10

    # Only test and without training.
    self.only_test = args.only_test

    self.resume = args.resume

    #######
    # Log #
    #######

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    # 3) checkpoint will be saved
    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = osp.join(
        'exp/train',
        '{}'.format(self.dataset),
        #
        ('nf_' if self.normalize_feature else 'not_nf_') +
        ('ohs_' if self.local_dist_own_hard_sample else 'not_ohs_') +
        'gm_{}_'.format(tfs(self.global_margin)) +
        'lm_{}_'.format(tfs(self.local_margin)) +
        'glw_{}_'.format(tfs(self.g_loss_weight)) +
        'llw_{}_'.format(tfs(self.l_loss_weight)) +
        'idlw_{}_'.format(tfs(self.id_loss_weight)) +
        'lr_{}_'.format(tfs(self.base_lr)) +
        '{}_'.format(self.lr_decay_type) +
        ('decay_at_{}_'.format(self.exp_decay_at_epoch)
         if self.lr_decay_type == 'exp'
         else 'decay_at_{}_factor_{}_'.format(
          '_'.join([str(e) for e in args.staircase_decay_at_epochs]),
          tfs(self.staircase_decay_multiply_factor))) +
        'total_{}'.format(self.total_epochs),
        #
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Saving model weights and optimizer states, for resuming.
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
    # Just for loading a pretrained model; no optimizer states is needed.
    self.model_weight_file = args.model_weight_file
    
class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    #print('ims shape', ims.shape)
    global_feat, local_feat = self.model(ims)[:2]
    global_feat = global_feat.data.cpu().numpy()
    local_feat = local_feat.data.cpu().numpy()
    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return global_feat, local_feat

def get_id_from_name(name):
    return int(name[0:8])

def crop_img(img, pose):
  h = img.shape[0]
  w = img.shape[1]
  top = int(min(pose[:,0]) - h/16)
  bottom = int(max(pose[:,0]) + h/16)
  left = int(min(pose[:,1]) - w/16)
  right = int(max(pose[:,1]) + w/16)
  #print(h, w, top, bottom, left, right)
  croped_img = img[max([top,0]):min([bottom,h-1]), max([left,0]):min([right, w-1])]
  #print(croped_img.shape)
  return croped_img

def main():
  USE_POSE = False
    
  cfg = Config()
  TVT, TMO = set_devices(cfg.sys_device_ids)
  
  model = Model(local_conv_out_channels=cfg.local_conv_out_channels)
  # Model wrapper
  model_w = DataParallel(model)
    

  if cfg.model_weight_file != '':
    map_location = (lambda storage, loc: storage)
    sd = torch.load(cfg.model_weight_file, map_location=map_location)
    load_state_dict(model, sd)
    print('Loaded model weights from {}'.format(cfg.model_weight_file))
  else:
    return

  pre_process_im = PreProcessIm(prng=np.random, resize_h_w=(256, 128), im_mean=[0.486, 0.459, 0.408], im_std=[0.229, 0.224, 0.225], mirror_type=None, batch_dims='NCHW',scale=True)

  feat_extractor = ExtractFeature(model_w, TVT)

  test_set = create_dataset(**cfg.test_set_kwargs)
  
  #print(len(test_set.im_names))
  query_names = []
  query_ids = []
  gallery_names = []
  gallery_ids = []
  for i in range(len(test_set.marks)):
    if test_set.marks[i] == 0:
      query_names.append(test_set.im_names[i])
      query_ids.append(get_id_from_name(test_set.im_names[i]))
    elif test_set.marks[i] == 1:
      gallery_names.append(test_set.im_names[i])
      gallery_ids.append(get_id_from_name(test_set.im_names[i]))
  #print(len(query_names), len(gallery_names))
  
  iter_count = 0
  same_class_dist_sum = 0
  same_class_count = 0
  diff_class_dist_sum = 0
  diff_class_count = 0
  for q_id in range(len(query_names)):
    q_name = test_set.im_dir + '/' +query_names[q_id]
    #print('read', q_name)
    q_pose_name = q_name + '.txt'
    q_pose = np.array([])
    if os.path.isfile(q_pose_name):
      q_pose = np.loadtxt(q_pose_name)
    
    q_class_id = query_ids[q_id]
    for g_id in range(len(gallery_names)):
      g_name = test_set.im_dir + '/' + gallery_names[g_id]
      g_pose_name = g_name + '.txt'
      g_pose = np.array([])
      if os.path.isfile(g_pose_name):
        g_pose = np.loadtxt(g_pose_name)
    
      
      g_class_id = gallery_ids[g_id]
        
      if q_class_id != g_class_id:
        if random.random() > 0.01:
          continue
    
      q_im = np.asarray(Image.open(q_name))
      g_im = np.asarray(Image.open(g_name))
      #print(q_im.shape)
    
      if USE_POSE and q_pose.shape[0]==25 and g_pose.shape[0]==25:
        valid_pose_q = np.where(q_pose[:,2] > 0.1)
        valid_pose_g = np.where(g_pose[:,2] > 0.1)
        used_ids = np.intersect1d(valid_pose_q, valid_pose_g)
        h = q_im.shape[0]
        w = q_im.shape[1]
        valid_pose_q = np.where(q_pose[:,0] > 0)
        used_ids = np.intersect1d(used_ids, valid_pose_q)
        valid_pose_q = np.where(q_pose[:,0] < h-1)
        used_ids = np.intersect1d(used_ids, valid_pose_q)
        valid_pose_q = np.where(q_pose[:,1] > 0)
        used_ids = np.intersect1d(used_ids, valid_pose_q)
        valid_pose_q = np.where(q_pose[:,1] < w-1)
        used_ids = np.intersect1d(used_ids, valid_pose_q)
        
        h = g_im.shape[0]
        w = g_im.shape[1]
        valid_pose_g = np.where(g_pose[:,0] > 0)
        used_ids = np.intersect1d(used_ids, valid_pose_g)
        valid_pose_g = np.where(g_pose[:,0] < h-1)
        used_ids = np.intersect1d(used_ids, valid_pose_g)
        valid_pose_g = np.where(g_pose[:,1] > 0)
        used_ids = np.intersect1d(used_ids, valid_pose_g)
        valid_pose_g = np.where(g_pose[:,1] < w-1)
        used_ids = np.intersect1d(used_ids, valid_pose_g)
        
        
        if(len(used_ids) < 8):
          continue
        #print(used_ids)
        pose_q_used = q_pose[used_ids, 0:2]
        
        #print(pose_q_used)
        pose_g_used = g_pose[used_ids, 0:2]
        q_im = crop_img(q_im, pose_q_used)
        g_im = crop_img(g_im, pose_g_used)
        #print(used_ids)
        
      
      if random.random() > 0.5:
        h = q_im.shape[1]
        q_im = q_im[0:2*h/3, :]
      if random.random() > 0.5:
        h = g_im.shape[1]
        g_im = g_im[0:2*h/3, :]
      
      #print(q_im.shape, g_im.shape)
      q_im_preprocessed, _ = pre_process_im(q_im)
      g_im_preprocessed, _ = pre_process_im(g_im)
    
      img_set = []
      img_set.append(q_im_preprocessed)
      img_set.append(g_im_preprocessed)
      ims = np.stack(img_set, axis=0)
      global_feat, local_feat = feat_extractor(ims)
      if cfg.normalize_feature:
        global_feat = normalize(global_feat, axis=1)
      global_q_g_dist = compute_dist(global_feat, global_feat, type='euclidean')
      #print('dist',global_q_g_dist)
      if q_class_id == g_class_id:
        same_class_count += 1
        same_class_dist_sum += global_q_g_dist[0][1]
      else:
        diff_class_count += 1
        diff_class_dist_sum += global_q_g_dist[0][1]
      iter_count += 1
      if iter_count % 10 == 9:
        os.system('clear')
        print('same_class_count', same_class_count)
        print('same_class_dist_average', same_class_dist_sum / same_class_count)
        print('diff_class_count', diff_class_count)
        print('diff_class_dist_average', diff_class_dist_sum / diff_class_count)
           


if __name__ == '__main__':
  main()
    
