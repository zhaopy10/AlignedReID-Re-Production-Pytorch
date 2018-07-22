from MobileNetV2 import MobileNetV2

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


state_dict = torch.load('/home/corp.owlii.com/peiyao.zhao/reid/AlignedReID-Re-Production-Pytorch/train_test_mobilenetV2/ckpt.pth')
for key, value in state_dict.items():
    print('load key', key)
#print(state_dict['state_dicts'])
for item in state_dict['state_dicts']:
    for key, value in item.items():
        print('key', key)