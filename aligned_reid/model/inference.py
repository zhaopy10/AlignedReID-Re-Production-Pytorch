import sys
sys.path.insert(0, '.')

import numpy as np
import cv2
import os.path as osp
import torch
import torch.nn.functional as F
from PIL import Image
from Model import Model
from MobileNetV2_Relu_Scale_075 import MobileNetV2


#if __name__ == '__main__':
img_name = 'panda.jpg'
img = np.asarray(Image.open(img_name))

pre_process_im_func = PreProcessIm(prng=np.random, resize_h_w=(256, 128), im_mean=[0.486, 0.459, 0.408], im_std=[0.229, 0.224, 0.225], mirror_type=None, batch_dims='NCHW',scale=True)
img_pre = pre_process_im_func(img)

ims = np.expand_dims(img_pre, axis=0)
ims_var = torch.from_numpy(ims).float()

model = Model()
feat = model(ims_var)

print(feat.shape, feat)

