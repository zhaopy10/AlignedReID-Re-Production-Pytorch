import os
import os.path as osp
import cPickle as pickle
from scipy import io

import torch

state_dict = torch.load('../pretrained_model/model_weight.pth') 
for key, value in state_dict.items():
    print('key', key, 'value type', type(value), 'shape', value.shape)

'''
state_dict = torch.load('./train_with_msmt17/ckpt.pth') 
print('type', type(state_dict['state_dicts']))
for key, value in state_dict['state_dicts']:
    print('key', key, 'value type', type(value)) 
'''  
#model = torch.load('imagenet_resnet101.pth')  
'''
model = torch.load('train_with_msmt17/ckpt.pth')  
state_dict = model.state_dict()
print model
for key, value in state_dict.items():
    print('key', key, 'value shape', value.shape)
'''