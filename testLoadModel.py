import os
import os.path as osp
import cPickle as pickle
from scipy import io

import torch


#model = torch.load('imagenet_resnet101.pth')  
model = torch.load('train_test_cpu/graph.pth')  
state_dict = model.state_dict()
print model
for key, value in state_dict.items():
    print('key', key, 'value shape', value.shape)
