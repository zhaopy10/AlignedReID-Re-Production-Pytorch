from __future__ import print_function
import os
import os.path as osp
import cPickle as pickle
from scipy import io
import torch
import cv2
from PIL import Image
import math
import random
from skimage import exposure
from skimage.io import imsave, imread

def extend_ims(ims, crop=False, down_sample=False, padding=False, gamma=True):
    #print('initial ims', type(ims))
    ims_list = [] 
    im_num = 1
    if isinstance(ims, list):
        im_num = len(ims)
        ims_list.extend(ims)    
    else:
        ims_list.append(ims)
    extend = 1
    
    #print(len(ims_list))
    
    if gamma:
        extend += 1
        for i in range(0, im_num):
            im_data = ims_list[i]
            #print(type(im_data), im_data.shape)
            gamma = 0.85 + 0.3 * random.random()
            gain = 0.7 + 0.3 * random.random()
            gamma_corrected = exposure.adjust_gamma(im_data, gamma, gain)
            #imsave("original_img.jpg", im_data)
            #imsave("gamma_img.jpg", gamma_corrected)
            ims_list.append(gamma_corrected)
            #print(gamma, gain)
            #raw_input()
    
    if crop:
        extend += 1
        for i in range(0, im_num):
            im_data = ims_list[i]
            h = im_data.shape[0]
            w = im_data.shape[1]
            left_part = im_data[:, 0:2*w/3]
            #left_part = cv2.resize(left_part, (h, w), interpolation=cv2.INTER_LINEAR)
            right_part = im_data[:,w/3:-1]
            #right_part = cv2.resize(right_part, (h, w), interpolation=cv2.INTER_LINEAR)
            up_part = im_data[0:2*h/3,:]
            #up_part = cv2.resize(up_part, (h, w), interpolation=cv2.INTER_LINEAR)
            #ims_list.append(left_part)
            #ims_list.append(right_part)
            ims_list.append(up_part)
    
    if down_sample:
        extend += 1
        for i in range(0, im_num):
            im_data = ims_list[i]
            h = im_data.shape[0]
            w = im_data.shape[1]
            im_down = cv2.resize(im_data, (h/2, w/2), interpolation=cv2.INTER_LINEAR)
            im_down = cv2.resize(im_down, (h, w), interpolation=cv2.INTER_LINEAR)
            ims_list.append(im_down)
    
    if padding:
        extend += 1
        for i in range(0, im_num):
            im_data = ims_list[i]
            h = im_data.shape[0]
            w = im_data.shape[1]
            padded_im = 128 * np.ones((h, 2*w, 3), np.uint8)
            padded_im[:, w/2:w+w/2] = im_data
            #padded_im = cv2.resize(padded_im, (h, w), interpolation=cv2.INTER_LINEAR)
            ims_list.append(padded_im)
    if extend == 1:
        return ims_list, extend
    
    # random sample to 2^n
    
    #clip = math.floor(math.log(extend, 2))
    #new_size = int(math.pow(2, clip))
    new_size = extend
    random_ims_list = []
    #random_ims_list.extend(ims_list[0:im_num])
    random_ims_list.extend(random.sample(ims_list[:], new_size * im_num))
    #print('new_size', new_size, len(random_ims_list), type(random_ims_list))
    #print('original size', len(ims_list), type(ims_list))
    extend = new_size
    
    #print(len(random_ims_list))
    #ims_list = tuple(ims_list)
    #print('extended im list', extend, type(ims_list), len(ims_list))
    return random_ims_list, extend
    #return ims_list, extend
        
