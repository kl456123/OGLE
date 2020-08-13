# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

def loadtxt(filename, shape, load=True):
    if load:
        input = np.loadtxt(filename).reshape(shape).astype(np.float32)
        input = torch.from_numpy(input)
    else:
        input = torch.ones(*shape)
    # print(filename, input)
    return input

# input = torch.ones(1,3,5,5)

input = loadtxt('./assets/input.txt',(1,3,5,5))
weight = loadtxt('./assets/filter.txt', (3, 1, 3, 3))
bias = loadtxt('./assets/bias.txt', (3,))
stride = 2
pad = 2
dilation = 2
output = nn.functional.conv2d(
    input, weight, bias, stride=stride, padding=pad, dilation=dilation, groups=3)
print(output)
# print(input.shape)
