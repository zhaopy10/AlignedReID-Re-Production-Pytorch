import torch
import torch.nn as nn
import math

import torch.nn.functional as F

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(num_features=oup, track_running_stats=True),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def MobileNetV2(input_size=224, scale=1.0, pretrained=False):
    """Constructs a MobileNetV2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2_base()
    if pretrained:
        #filename = '/home/corp.owlii.com/peiyao.zhao/reid/AlignedReID-Re-Production-Pytorch/aligned_reid/model/mobilenet_v2_1.0_tf.pth'
        filename = '/home/corp.owlii.com/peiyao.zhao/reid/AlignedReID-Re-Production-Pytorch/aligned_reid/model/model_best_1.0_pyz.pth.tar'
        print('Load pretrained model from' + filename)
        #state_dict = torch.load(f=filename, map_location=lambda storage, loc: storage.cuda(0))
        state_dict = torch.load(f=filename)['state_dict']
        for key, value in state_dict.items():
            if 'classifier' in key:
                del state_dict[key]
                continue
            state_dict[key[7:]] = state_dict[key]
            state_dict.pop(key)
        #print(len(state_dict.keys()))
        #for key, value in state_dict.items():
        #    print('load key', key)
        model.load_state_dict(state_dict, False)
    return model

class MobileNetV2_base(nn.Module):
    def __init__(self, n_class=1000, input_size=224):
        super(MobileNetV2_base, self).__init__()
        # setting of inverted residual blocks
        width_mult = 1.0
        self.interverted_residual_setting = [
            # ex, oc, s
            [1, 16, 1],
            [6, 24, 2], [6, 24, 1],
            [6, 32, 2], [6, 32, 1], [6, 32, 1],
            [6, 64, 2], [6, 64, 1], [6, 64, 1], [6, 64, 1],
            [6, 96, 1], [6, 96, 1], [6, 96, 1],
            [6, 160, 2], [6, 160, 1], [6, 160, 1],
            [6, 320, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]


        # building inverted residual blocks
        for t, c, s in self.interverted_residual_setting:
            output_channel = int(c)
            self.features.append(InvertedResidual(input_channel, output_channel, s, t))
            input_channel = output_channel

        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        '''
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(self.last_channel, n_class),
        )
        '''
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = x.view(-1, self.last_channel)
        #x = x.mean(3).mean(2)
        #x = self.classifier(x)
        #return F.log_softmax(x, dim=1) #TODO not needed(?)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
