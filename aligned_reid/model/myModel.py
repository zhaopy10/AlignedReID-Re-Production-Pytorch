import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .resnet import resnet50
from .MobileNetV2 import MobileNetV2

class myModel(nn.Module):
  
  def __init__(self):
    super(myModel, self).__init__()
    self.base = resnet50(pretrained=True)
  

  '''
  def __init__(self, local_conv_out_channels=128, num_classes=None):
    super(Model, self).__init__()
    self.base = MobileNetV2(pretrained=True)
    planes = 1280
    self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
    self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
    self.local_relu = nn.ReLU(inplace=True)

    if num_classes is not None:
      self.fc = nn.Linear(planes, num_classes)
      init.normal(self.fc.weight, std=0.001)
      init.constant(self.fc.bias, 0)
  '''

  def forward(self, x):
    """
    Returns:
      global_feat: shape [N, C]
      local_feat: shape [N, H, c]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    #print('feat shape', feat.shape)
    global_feat = F.avg_pool2d(feat, feat.size()[2:])
    # shape [N, C]
    global_feat = global_feat.view(global_feat.size(0), -1)
    
    return global_feat
