
from __future__ import print_function
import pickle
import sys
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse

from aligned_reid.dataset import create_dataset
from aligned_reid.model.Model import Model
from aligned_reid.model.myModel import myModel
from aligned_reid.model.resnet import resnet50
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

from PIL import Image



data = pickle.load(open('testData.dat','rb'))
ims = data[0]
im_namse = data[1]
#labels = data[2]
#mirrored = data[3]
# ims is a tuple
print('im number', len(ims))
for i in range(0, len(ims)):
    #im_data = ims[i].transpose(1, 2, 0)
    im_data = ims[i]
    print('im_data', type(im_data),im_data.dtype, im_data.shape)
    im_ori = Image.fromarray(im_data,'RGB')
    im_ori.save("test_images/%d.jpg"%i)
    
    # cut 
    im_shape = im_data.shape
    h = im_shape[0]
    w = im_shape[1]
    left_part = im_data[:, 0:2*w/3]
    right_part = im_data[:,w/3:-1]
    up_part = im_data[0:2*h/3,:]
    im = Image.fromarray(left_part,'RGB')
    im.save("test_images/%d_left.jpg"%i)
    im = Image.fromarray(right_part,'RGB')
    im.save("test_images/%d_right.jpg"%i)
    im = Image.fromarray(up_part,'RGB')
    im.save("test_images/%d_up.jpg"%i)
    
    # definition
    im = im_ori.resize((w/2, h/2))
    im = im.resize((w, h))
    im.save("test_images/%d_down.jpg"%i)
    
    # padding
    padded_im = 128 * np.ones((h, 2*w, 3), np.uint8)
    print('padded_im', type(padded_im),padded_im.dtype, padded_im.shape)
    padded_im[:, w/2:w+w/2] = im_ori
    im = Image.fromarray(padded_im,'RGB')
    im.save("test_images/%d_padded.jpg"%i)