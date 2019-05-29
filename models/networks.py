import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F
import cv2
import numpy as np
from util.util import tensor2im
from torchvision.transforms import Grayscale
import time
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.2):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    # if which_model_netG == 'resnet_9blocks':
    netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    # elif which_model_netG == 'resnet_6blocks':
    # netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    # elif which_model_netG == 'unet_128':
    #     netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # elif which_model_netG == 'unet_256':
    #     netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # else:
    #     raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)

def define_G_Y(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)
    netG = yNetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    return init_net(netG, init_type, gpu_ids)

def define_D1(input_nc, ndf, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)
    netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    return init_net(netD, init_type, gpu_ids)

def define_D_Pixel(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)
    # if which_model_netD == 'resnet_9blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    # elif which_model_netD == 'resnet_6blocks':
    netD = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    # elif which_model_netD == 'unet_128':
    #     netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # elif which_model_netD == 'unet_256':
    #     netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # else:
    #     raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netD)
    return init_net(netD, init_type, gpu_ids)

def define_D_Dual(G_output_nc, ndf, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)
    netD = DualDiscriminator(G_output_nc, ndf, n_layers=3, norm_layer=norm_layer)
    return init_net(netD, init_type, gpu_ids)

##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class styleGANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(styleGANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class pixelL1Loss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor):
        super(pixelL1Loss, self).__init__()
        self.Tensor = tensor
        self.loss = nn.SmoothL1Loss()

    def __call__(self, input, real_C):
        channel = real_C.shape[1]
        assert(channel == 3)
        mask = ((real_C[:, 0, :, :] == 255) + (real_C[:, 1, :, :] == 255) + (real_C[:, 2, :, :] == 255) < 3).unsqueeze(1)
        maskidx = torch.cat((mask, mask, mask), dim=1)
        real_C_masked = torch.masked_select(real_C, maskidx)
        input_masked = torch.masked_select(input, maskidx)
        return self.loss(input_masked, real_C_masked)

def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)

def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)

def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr

class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        self.loss = torch.nn.SmoothL1Loss()

    def __call__(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma,
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        # return sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
        return sum(self.loss(a, b) for a, b in zip(pyr_input, pyr_target))

# def imgrad(img, opt):
#     img = torch.mean(img, dim=1, keepdim=True)
#
#     fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
#     kernel1 = Variable(torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
#     if len(opt.gpu_ids) > 0:
#         kernel1 = kernel1.cuda(opt.gpu_ids[0], async=True)
#     n_channels, _, kw, kh = kernel1.shape
#     # img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
#     grad_x = F.conv2d(img, kernel1, groups=1, padding=1)
#
#     fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#     kernel2 = Variable(torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
#     if len(opt.gpu_ids) > 0:
#         kernel2 = kernel2.cuda(opt.gpu_ids[0], async=True)
#     n_channels, _, kw, kh = kernel2.shape
#     # img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
#     grad_y = F.conv2d(img, kernel2, groups=1, padding=1)
#
#     return grad_y, grad_x

def imgrad(img, opt):
    img = torch.mean(img, dim=1, keepdim=True)
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
    if len(opt.gpu_ids) > 0:
        conv1.cuda(opt.gpu_ids[0])
    G_x = conv1(img).data
    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
    if len(opt.gpu_ids) > 0:
        conv2.cuda(opt.gpu_ids[0])
    G_y = conv2(img).data
    return G_x, G_y

# def imgrad(img, opt):
#     img = torch.mean(img, dim=1, keepdim=True)
#     fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
#     conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
#     weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
#     if len(opt.gpu_ids) > 0:
#         weight = weight.cuda(opt.gpu_ids[0], async=True)
#     conv1.weight = nn.Parameter(weight)
#     grad_x = conv1(img)
#
#     fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#     conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
#     weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
#     if len(opt.gpu_ids) > 0:
#         weight = weight.cuda(opt.gpu_ids[0], async=True)
#     conv2.weight = nn.Parameter(weight)
#     grad_y = conv2(img)
#
#     return grad_y, grad_x

def imgrad_yx(img, opt):
    # N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img, opt)
    return torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))
    # return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)


# class pixelBlurLoss(nn.Module):
#     def __init__(self, tensor=torch.FloatTensor):
#         super(pixelBlurLoss, self).__init__()
#         self.Tensor = tensor
#         self.loss = nn.MSELoss()
#
#     def __call__(self, input, real_B):
#         left = tensor2im(input)
#         right = tensor2im(real_B)
#         left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
#         right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
#         sobelLeft = cv2.Sobel(left, cv2.CV_64F, 1, 1, ksize=5)
#         sobelRight = cv2.Sobel(right, cv2.CV_64F, 1, 1, ksize=5)
#         return self.loss(sobelLeft, sobelRight)

class DoubleBlurLoss(nn.Module):
    def __init__(self, opt):
        super(DoubleBlurLoss, self).__init__()
        self.opt = opt

    def __call__(self, input, real):
        grad_real, grad_fake = imgrad_yx(real, self.opt), imgrad_yx(input, self.opt)
        return torch.mean(torch.abs(grad_real - grad_fake))

def reg_scalor(grad_yx):
    return torch.exp(-torch.abs(grad_yx))

class SingleBlurLoss(nn.Module):
    def __init__(self, opt):
        super(SingleBlurLoss, self).__init__()
        self.opt = opt

    def __call__(self, input):
        grad = imgrad_yx(input, self.opt)
        return torch.mean(reg_scalor(grad))

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class yNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', gpu_ids=[]):
        assert(n_blocks >= 0)
        super(yNetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downsampling_model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            downsampling_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(int(n_blocks / 2.0)):
            downsampling_model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        """
        up sampling content
        """

        content_residuals = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks - int(n_blocks / 2.0)):
            content_residuals += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        self.content_residuals = nn.Sequential(*content_residuals)

        self.content_upsampling = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.content_upsampling += nn.Sequential(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                     kernel_size=3, stride=2,
                                                     padding=1, output_padding=1,
                                                     bias=use_bias),
                                  norm_layer(int(ngf * mult / 2)),
                                  nn.ReLU(True))

        self.content_upsampling += nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),nn.Tanh())

        if len(gpu_ids) > 0:
            for net in self.content_upsampling:
                net.cuda(gpu_ids[0])
                init_weights(net)

        """
        upsampling painting
        """

        painting_residuals = []
        mult = 2 ** n_downsampling

        for i in range(n_blocks - int(n_blocks / 2.0)):
            painting_residuals += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        self.painting_residuals = nn.Sequential(*painting_residuals)

        self.painting_upsampling = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            self.painting_upsampling += nn.Sequential(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True))

        self.painting_upsampling += nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())

        if len(gpu_ids) > 0:
            for net in self.painting_upsampling:
                net.cuda(gpu_ids[0])
                init_weights(net)

        self.downmodel = nn.Sequential(*downsampling_model)
        # self.upmodel_b = nn.Sequential(*upsampling_model)
        # self.upmodel_c = nn.Sequential(*upsampling_model2)

    def forward(self, input):
        middle = self.downmodel(input)

        content_res = self.content_residuals(middle)
        cont = content_res
        content_uplist = [cont]
        for k in range(len(self.content_upsampling)):
            net = self.content_upsampling[k]
            cont = net(cont)
            if k == 2 or k == 5:
                content_uplist.append(cont)

        painting_res = self.painting_residuals(middle)
        paint = painting_res + content_uplist[0]
        for k in range(len(self.painting_upsampling)):
            net = self.painting_upsampling[k]
            paint = net(paint)
            if k == 2:
                paint = paint + content_uplist[1]
            elif k == 5:
                paint = paint + content_uplist[2]
        return paint, cont

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
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


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class DualDiscriminator(nn.Module):
    def __init__(self, G_output_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(DualDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(G_output_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                     norm_layer(1),
                     nn.LeakyReLU(0.2, True)
                     ]

        # if use_sigmoid:
        #     sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

        self.fc1 = nn.Linear(2700, 2048)
        self.fc2 = nn.Linear(2048, 1024)

    def forward(self, fake_B, real_C):
        # input fake_B and real_C, result is 30 * 30
        fake_B_map = self.model(fake_B)
        real_C_map = self.model(real_C)
        cross = fake_B_map + real_C_map

        combine = torch.cat(
            (fake_B_map.view(-1, 900), cross.view(-1, 900), real_C_map.view(-1, 900)), 0
        ).view(-1, 2700)

        out = F.relu(self.fc1(combine))
        out = F.sigmoid(self.fc2(out))

        return out


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
