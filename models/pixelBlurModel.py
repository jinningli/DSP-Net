import torch
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.util import tensor2im
from util.util import save_image
import os
import cv2
import numpy as np
import torch.nn.functional as F
import math
import time

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class PixelBlurModel(BaseModel):
    def name(self):
        return 'PixelBlurModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids

        self.loss_names = ['G_styleGAN', 'G_contentGAN', 'blur', 'D1_real', 'D1_fake']

        self.fake_B = None
        self.fake_C = None

        self.model_names = ['G', 'D1', 'D2']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        use_sigmoid = opt.no_lsgan
        self.netD1 = networks.define_D1(opt.input_nc + opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        self.netD2 = networks.define_D_Pixel(opt.output_nc, opt.input_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.fake_AB_pool = ImagePool(opt.pool_size)

        self.criterionStyleGAN = networks.styleGANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        # self.criterionBlur = networks.SingleBlurLoss(self.opt)
        # self.criterionBlur = networks.DoubleBlurLoss(self.opt)
        self.criterionBlur = networks.LapLoss()

        self.criterionContentGAN = torch.nn.L1Loss()

        self.criterionL1 = torch.nn.L1Loss()

        self.schedulers = []
        self.optimizers = []

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)

        self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_D1)

        self.optimizer_D2 = torch.optim.Adam([{'params':self.netD2.parameters()}, {'params':self.netG.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.999))

        for optimizer in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if opt.continue_train:
            print('Continue train. Loading the latest network.')
            self.load_networks(opt.which_epoch)

        self.print_networks(opt.verbose)

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        input_C = input['C']

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_C = input_C.cuda(self.gpu_ids[0], async=True)

        self.input_A = input_A
        self.input_B = input_B
        self.input_C = input_C

        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)

    def test(self):
        raise NotImplementedError("Test Function")

    def test_and_save(self, name):
        self.real_A = Variable(self.input_A, volatile=True)

        self.fake_B= self.netG(self.real_A)
        self.fake_C = self.netD2(self.fake_B)

        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.exists(os.path.join(self.save_dir, "val")):
            os.mkdir(os.path.join(self.save_dir, "val"))

        im = tensor2im(self.fake_B)
        save_image(im, os.path.join(self.save_dir, "val",  name + '_fake_B.png'))

        im = tensor2im(self.fake_C)
        save_image(im, os.path.join(self.save_dir, "val",  name + '_fake_C.png'))
        im = tensor2im(self.real_C)
        save_image(im, os.path.join(self.save_dir, "val",  name + '_real_C.png'))

    def backward_D1(self):
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD1(fake_AB.detach())
        self.loss_D1_fake = self.criterionStyleGAN(pred_fake, False) # target is real: False

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD1(real_AB)
        self.loss_D1_real = self.criterionStyleGAN(pred_real, True)

        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5
        self.loss_D1.backward(retain_graph=True)

    def backward_D2(self): # TODO Combine D1 and D2 loss??
        self.loss_D2_fake = 0.0
        self.loss_D2_real = 0.0

        assert(not self.opt.noPixelD2)
        pred_fake = self.netD2(self.fake_B)
        self.loss_D2_fake = self.criterionContentGAN(pred_fake, self.real_C)
        self.loss_D2_real = self.loss_D2_fake

        # Combined loss
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5

        self.loss_D2.backward(retain_graph=True)

    # def tenengrad(self, input):
    #     left = tensor2im(input)
    #     left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    #     sobelLeft = cv2.Sobel(left, cv2.CV_64F, 1, 1, ksize=5)
    #     self.loss_sobel = np.average(np.abs(sobelLeft))
    #     return 2 * sigmoid(- self.loss_sobel / 100.0)

    def backward_G(self): # TODO should backward twice??
        self.loss_G_styleGAN = 0.0
        self.loss_G_contentGAN = 0.0

        # painting style loss
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD1(fake_AB)
        self.loss_G_styleGAN = self.criterionStyleGAN(pred_fake, True)

        # semantic loss
        pred_real = self.netD2(self.fake_B)
        self.loss_G_contentGAN = self.criterionContentGAN(pred_real, self.real_C)

        # blur loss
        # self.loss_blur = self.criterionBlur(self.fake_B) * 10
        self.loss_blur = self.criterionBlur(self.fake_B, self.real_B) * 5

        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_styleGAN * self.loss_G_contentGAN + self.loss_blur

        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D1.zero_grad()
        self.backward_D1()
        self.optimizer_D1.step()

        self.optimizer_D2.zero_grad()
        self.backward_D2()
        self.optimizer_D2.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()