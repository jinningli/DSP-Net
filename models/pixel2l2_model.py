import torch
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.util import tensor2im
from util.util import save_image
import os
import time

class Pixel2l2Model(BaseModel):
    def name(self):
        return 'Pixel2l2Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        assert(opt.isTrain)
        self.gpu_ids = opt.gpu_ids
        self.loss_names = ['G_B', 'G_C', 'C_l1']
        self.fake_B = None
        self.fake_C = None
        self.model_names = ['G', 'D2']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netD2 = networks.define_D_Pixel(opt.output_nc, opt.input_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.criterionL1 = torch.nn.L1Loss()

        # initialize optimizers
        self.schedulers = []
        self.optimizers = []

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)

        self.optimizer_C = torch.optim.Adam([{'params':self.netD2.parameters()}, {'params':self.netG.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.999))

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

    def test_and_save(self, name):
        self.real_A = Variable(self.input_A, volatile=True)

        self.fake_B = self.netG(self.real_A)
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

    def backward_C(self):
        self.fake_C = self.netD2(self.fake_B)
        self.loss_C_l1 = self.criterionL1(self.fake_C, self.real_C)
        self.loss_C_l1.backward(retain_graph=True)

    def backward_G(self):
        self.loss_G_B = self.criterionL1(self.fake_B, self.real_B)

        self.fake_C = self.netD2(self.fake_B)
        self.loss_G_C = self.criterionL1(self.fake_C, self.real_C)

        self.loss_G = self.loss_G_B + self.loss_G_C
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()

        self.optimizer_C.zero_grad()
        self.backward_C()
        self.optimizer_C.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

