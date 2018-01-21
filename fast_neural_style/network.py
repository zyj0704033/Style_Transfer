from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
from totalvariationloss import TotalVariationLoss
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import copy
import functools

import init_weights
from fast_neural_style.perceptualcriterion import Perceptualcriterion
from fast_neural_style.vgg import Vgg16

class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a, b, c * d)  # resise F_XL into \hat F_XL

        G = torch.bmm(features, torch.transpose(features, 1, 2))  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(b * c * d)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias, use_dropout=False, padding_type='reflect'):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Generator(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(Generator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        prev_dim = 3
        next_dim = 32
        sequence = []
        sequence += [nn.Conv2d(prev_dim, next_dim, kernel_size=9, stride=1, padding=4),
                     norm_layer(next_dim),
                     nn.ReLU(True)
                     ]
        for i in range(2):
            prev_dim = next_dim
            next_dim *= 2
            sequence += [nn.Conv2d(prev_dim, next_dim, kernel_size=3, stride=2, padding=1),
                         norm_layer(next_dim),
                         nn.ReLU(True)
                         ]
        prev_dim = next_dim
        assert (prev_dim == next_dim == 128)
        for i in range(5):
            sequence += [ResnetBlock(next_dim, norm_layer, use_bias)]

        next_dim /= 2
        sequence += [nn.ConvTranspose2d(prev_dim, next_dim,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                     norm_layer(next_dim),
                     nn.ReLU(True)]
        prev_dim = next_dim
        next_dim /= 2
        sequence += [nn.ConvTranspose2d(prev_dim, next_dim,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                     norm_layer(next_dim),
                     nn.ReLU(True)]
        prev_dim = next_dim
        next_dim = 3
        sequence += [nn.Conv2d(prev_dim, next_dim, kernel_size=9, stride=1, padding=4),
                     norm_layer(next_dim),
                     nn.ReLU(True)
                     ]
        sequence += [nn.Tanh()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class PerceptualModel():
    def __init__(self, opt):
        # desired depth layers to compute style/content losses :
        print("build start")
        self.opt = opt
        self.Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor
        nb = opt.batch_size
        size = opt.image_size
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([transforms.Scale(opt.image_size),
                                              transforms.CenterCrop(opt.image_size),
                                              transforms.ToTensor()])

        self.vgg = Vgg16(opt.vgg_layers, requires_grad=False)
        self.image_tensor = self.Tensor(nb, 3, size, size)
        self.content_img = Variable(self.Tensor(nb, 3, size, size))
        self.style_img = self.image_loader(opt.style_image)
        self.generated_img = Variable(self.Tensor(nb, 3, size, size))
        norm_layer = get_norm_layer(norm_type=opt.norm)

        self.generator = Generator(norm_layer, opt.gpu_ids)
        self.tv = TotalVariationLoss(opt.reg_tv)
        init_weights.init_weights(self.generator)
        self.gram = GramMatrix()

        if opt.gpu_ids > 0:
            self.generator = self.generator.cuda()
            self.vgg.cuda()

        # copy style image batch wise
        a,b,c,d = self.style_img.size()
        torchvision.utils.save_image(self.style_img.data, opt.visualize_dir + '/style.jpg')
        # self.style_img = self.normalize(self.style_img)
        self.style_img=self.style_img.expand(nb,b,c,d)
        if self.opt.gpu_ids > 0:
            self.style_img = self.style_img.cuda()
        features_style = self.vgg(self.style_img)
        self.gram_style = [self.gram(y) for y in features_style]

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=opt.lr)
        print("build end")
        self.mse_loss = torch.nn.MSELoss()

    def image_loader(self, image_name):
        image = Image.open(image_name)
        image = Variable(self.preprocess(image))
        # fake batch dimension required to fit network's input dimensions
        image = image.unsqueeze(0)
        return image

    def forward(self, data):
        self.image_tensor.resize_(data.size()).copy_(data)
        self.content_img = Variable(self.image_tensor)
        if self.opt.gpu_ids > 0:
            self.content_img = self.content_img.cuda()
        self.generated_img = self.generator.forward(self.content_img)
        # self.content_img = self.normalize(self.content_img)
        # self.generated_img = self.normalize(self.generated_img)

        features_content = self.vgg(self.content_img)
        features_generated = self.vgg(self.generated_img)

        self.content_loss = self.opt.content_weight * self.mse_loss(features_generated.relu2_2, features_content.relu2_2.detach())

        self.style_loss = 0.
        for ft_g, gm_s in zip(features_generated, self.gram_style):
            gm_g = self.gram(ft_g)
            self.style_loss += self.mse_loss(gm_g, gm_s.detach())
        self.style_loss *= self.opt.style_weight
        
        self.tv_loss = self.tv(self.generated_img)

    def backward(self):
        self.generator_optimizer.zero_grad()
        self.content_loss.backward(retain_graph=True)
        self.style_loss.backward(retain_graph=True)
        self.tv_loss.backward(retain_graph=True)
        return self.content_loss, self.style_loss, self.tv_loss

    def normalize(self, batch):
        # normalize using imagenet mean and std
        batch = batch.clone()
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch -= Variable(mean)
        batch = batch / Variable(std)
        return batch

    def name(self):
        return 'SimpleModel'
