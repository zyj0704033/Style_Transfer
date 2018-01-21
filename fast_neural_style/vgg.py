from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, output_layers, requires_grad=False):
        super(Vgg16, self).__init__()
        self.output_layers = output_layers
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        slice_ = torch.nn.Sequential()
        slice_cnt = 0
        i = 1
        s = 1
        name = None
        for layer in list(vgg_pretrained_features):
            if isinstance(layer, nn.Conv2d):
                name = "conv{}_{}".format(s, i)
                slice_.add_module(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu{}_{}".format(s, i)
                slice_.add_module(name, layer)
                i += 1

            if isinstance(layer, nn.MaxPool2d):
                s += 1
                i = 1
                name = "pool{}".format(s)
                slice_.add_module(name, layer)

            if name in output_layers:
                setattr(self, 'slice' + str(slice_cnt), slice_)
                if slice_cnt == len(output_layers):
                    break
                slice_ = torch.nn.Sequential()
                slice_cnt += 1

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        for i in range(len(self.output_layers)):
            slice_ = getattr(self, 'slice' + str(i))
            print(slice_)

    def forward(self, X):
        h = X
        h_list = []
        for i in range(len(self.output_layers)):
            slice_ = getattr(self, 'slice'+str(i))
            h = slice_(h)
            h_list.append(h)
        vgg_outputs = namedtuple("VggOutputs", self.output_layers)
        out = vgg_outputs(*h_list)
        return out
