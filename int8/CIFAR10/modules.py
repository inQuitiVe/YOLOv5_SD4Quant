import torch
import math
from itertools import repeat
from torch.autograd.function import Function
from torch.autograd.variable import Variable
import os
import numpy as np
# import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import math
from torch.nn.modules.utils import _pair
from torch.nn.modules import conv as conv
import torch.nn.functional as F

from torch.nn import init
from torch.quantization import QuantStub, DeQuantStub

################################### mobile_net without quant #########################################
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding = 0 ,bias=False, groups=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.ReLU(inplace=False)
        )
        self.in_channels = in_planes
        self.groups      = groups

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=False, groups=1):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1)
        )
        self.in_channels = in_planes
        self.groups      = groups

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.convbnrelu1 = ConvBNReLU(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.convbnrelu2 = ConvBNReLU(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.convbn3 = ConvBN(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = ConvBN(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.convbnrelu1(x)
        out = self.convbnrelu2(out)
        out = self.convbn3(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2_without_quant(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2_without_quant, self).__init__()
        self.CONVBNRELU1 = ConvBNReLU(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers = self._make_layers(in_planes=32)
        self.CONVBNRELU2 = ConvBNReLU(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.CONVBNRELU1(x)
        out = self.layers(out)
        out = self.CONVBNRELU2(out)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

###################################### mobilenet with quant ############################################

class MobilenetV2_with_quant(nn.Module):
    def __init__(self):
        super(MobilenetV2_with_quant, self).__init__()
        self.quant      = QuantStub()
        self.body       = MobileNetV2_without_quant()
        self.dequant    = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.body(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True) #conv + bn + relu
            if type(m) == ConvBN:
                torch.quantization.fuse_modules(m, ['0', '1'], inplace=True) #conv + bn + relu


class Block_V1(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block_V1, self).__init__()
        self.convbnrelu1 = ConvBNReLU(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.convbnrelu2 = ConvBNReLU(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.convbnrelu1(x)
        out = self.convbnrelu2(out)
        return out


class MobileNetV1_without_quant(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNetV1_without_quant, self).__init__()
        self.convbnrelu1 = ConvBNReLU(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block_V1(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.convbnrelu1(x)
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class MobilenetV1_with_quant(nn.Module):
    def __init__(self):
        super(MobilenetV1_with_quant, self).__init__()
        self.quant      = QuantStub()
        self.body       = MobileNetV1_without_quant()
        self.dequant    = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.body(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True) #conv + bn + relu
            if type(m) == ConvBN:
                torch.quantization.fuse_modules(m, ['0', '1'], inplace=True) #conv + bn + relu
