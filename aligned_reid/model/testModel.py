from MobileNetV2_Relu_Scale import MobileNetV2
#from Model import Model
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from PIL import Image
import numpy as np
import tensorflow as tf
import scipy

'''
state_dict = torch.load('/home/corp.owlii.com/peiyao.zhao/reid/AlignedReID-Re-Production-Pytorch/train_test_mobilenetV2/ckpt.pth')
for key, value in state_dict.items():
    print('load key', key)
#print(state_dict['state_dicts'])
for item in state_dict['state_dicts']:
    for key, value in item.items():
        print('key', key)
'''


img = np.load('img.npy')
print(type(img), img.shape, img)
# NHWC -> NCHW
img = np.moveaxis(img, [0,1,2,3], [0,2,3,1])
img_var = torch.from_numpy(img).float()

net = MobileNetV2(pretrained=True)
out = net(img_var)
out = out.detach().numpy()
out = np.moveaxis(out, [0,1,2,3],[0,3,1,2])
print(type(out), out.shape)
print(out)
#print(out[:,100,100,:])


state_dict = net.state_dict()
print(len(state_dict.keys()))
for key, value in state_dict.items():
    print('key', key, 'type', type(value), 'size', value.shape)
    if not key.find('features.0.0')==-1:
        value = value.numpy()
        #value = np.moveaxis(value, [0,1,2,3], [3,2,0,1])
        #print(value)


scipy.io.savemat()
print('original')
#data = torch.load('/home/corp.owlii.com/peiyao.zhao/reid/AlignedReID-Re-Production-Pytorch/aligned_reid/model/tf_resnet_to_pth.pth')
#print(type(data), data)
#state_dict = torch.load('/home/corp.owlii.com/peiyao.zhao/reid/AlignedReID-Re-Production-Pytorch/aligned_reid/model/mobilenetv2_718.pth.tar')
#for key, value in state_dict.items():
#    print('key', key, 'type', type(value), 'size', value.shape)



