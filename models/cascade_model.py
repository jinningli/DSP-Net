import torch
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.util import tensor2im
from util.util import save_image
import os
import time

class CascadeModel(BaseModel):
    def name(self):
        return 'TriplePix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.fake_B = None
        self.fake_C = None

        self.loss_names = ['G_middle', 'G_right', 'D1_real', 'D1_fake', 'D2_real', 'D2_fake']

        if self.isTrain:
            self.model_names = ['G', 'D1', 'D2']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD1 = networks.define_D1(opt.input_nc + opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD2 = networks.define_D1(opt.input_nc + opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.fake_AC_pool = ImagePool(opt.pool_size)

            self.criterionMiddle = networks.styleGANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionRight = networks.styleGANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_G2 = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G2)

            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D1)

            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D2)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
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
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)

        self.fake_B = self.netG(self.real_A)
        self.fake_C = self.netG2(self.fake_B)

    # no backprop gradients
    def test(self):
        if self.opt.mode == 'Y':
            self.real_A = Variable(self.input_A, volatile=True)
            self.fake_B, self.fake_C = self.netG(self.real_A)
            self.real_B = Variable(self.input_B, volatile=True)
            self.real_C = Variable(self.input_C, volatile=True)
        else:
            self.real_A = Variable(self.input_A, volatile=True)
            self.fake_B = self.netG(self.real_A)
            self.real_B = Variable(self.input_B, volatile=True)

    def test_and_save(self, name):
        self.real_A = Variable(self.input_A, volatile=True)

        if self.opt.mode == 'Y':
            self.fake_B, self.fake_C = self.netG(self.real_A)
        else:
            self.fake_B= self.netG(self.real_A)

        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.exists(os.path.join(self.save_dir, "val")):
            os.mkdir(os.path.join(self.save_dir, "val"))

        im = tensor2im(self.fake_B)
        save_image(im, os.path.join(self.save_dir, "val",  name + '_fake_B.png'))
        if self.opt.mode == 'Y':
            im = tensor2im(self.fake_C)
            save_image(im, os.path.join(self.save_dir, "val",  name + '_fake_C.png'))

    def backward_D1(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD1(fake_AB.detach())
        self.loss_D1_fake = self.criterionStyleGAN(pred_fake, False) # target is real: False

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD1(real_AB)
        self.loss_D1_real = self.criterionStyleGAN(pred_real, True)

        # Combined loss
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5

        self.loss_D1.backward(retain_graph=True)

    def backward_D2(self): # TODO Combine D1 and D2 loss??
        self.loss_D2_fake = 0.0
        self.loss_D2_real = 0.0

        if self.opt.mode == 'PixelWise':
            pred_fake = self.netD2(self.fake_B)
            self.loss_D2_fake = self.criterionContentGAN(pred_fake, self.real_C, False)  # target is real: False

            pred_real = self.netD2(self.real_B)
            self.loss_D2_real = self.criterionContentGAN(pred_real, self.real_C, True)

        elif self.opt.mode == 'Dual':
            pred_fake = self.netD2(self.fake_B, self.real_C)
            self.loss_D2_fake = self.criterionContentGAN(pred_fake, False)  # target is real: False

            pred_real = self.netD2(self.real_B, self.real_C)
            self.loss_D2_real = self.criterionContentGAN(pred_real, True)

        elif self.opt.mode == 'Y':
            fake_AC = self.fake_AC_pool.query(torch.cat((self.real_A, self.fake_C), 1))
            pred_fake = self.netD2(fake_AC.detach())
            self.loss_D2_fake = self.criterionStyleGAN(pred_fake, False)  # target is real: False

            real_AC = torch.cat((self.real_A, self.real_C), 1)
            pred_real = self.netD2(real_AC)
            self.loss_D2_real = self.criterionContentGAN(pred_real, True)

        # Combined loss
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5

        self.loss_D2.backward(retain_graph=True)

    def backward_G(self): # TODO should backward twice??
        # First, G(A) should fake the discriminator
        self.loss_G_styleGAN = 0.0
        self.loss_G_contentGAN = 0.0

        if self.opt.mode != 'Y':
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD1(fake_AB)
            self.loss_G_styleGAN = self.criterionStyleGAN(pred_fake, True)

            if self.opt.mode == 'PixelWise':
                pred_real = self.netD2(self.fake_B)
                self.loss_G_contentGAN = self.criterionContentGAN(pred_real, self.real_C, True)
            elif self.opt.mode == 'Dual':
                pred_real = self.netD2(self.fake_B, self.real_C)
                self.loss_G_contentGAN = self.criterionContentGAN(pred_real, True)
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake_B = self.netD1(fake_AB)
            self.loss_G_styleGAN = self.criterionStyleGAN(pred_fake_B, True)

            fake_AC = torch.cat((self.real_A, self.fake_C), 1)
            pred_fake_C = self.netD2(fake_AC)
            self.loss_G_contentGAN = self.criterionContentGAN(pred_fake_C, True)

        # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        if self.opt.mode == 'Original':
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
            self.loss_G = self.loss_G_styleGAN + self.loss_G_contentGAN + self.loss_G_L1
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
            self.loss_G = self.loss_G_styleGAN + self.loss_G_contentGAN + self.loss_G_L1

        self.loss_G.backward(retain_graph=True)

    def backward_G_D1(self): # TODO
        # First, G(A) should fake the discriminator
        self.loss_G_styleGAN = 0.0

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake_B = self.netD1(fake_AB)
        self.loss_G_styleGAN = self.criterionStyleGAN(pred_fake_B, True)

        # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_styleGAN
                      # + self.loss_G_L1

        self.loss_G.backward(retain_graph=True)

    def backward_G_D2(self): # TODO
        # First, G(A) should fake the discriminator
        self.loss_G_contentGAN = 0.0

        fake_AC = torch.cat((self.real_A, self.fake_C), 1)
        pred_fake_C = self.netD2(fake_AC)
        self.loss_G_contentGAN = self.criterionContentGAN(pred_fake_C, True)

        # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G =  self.loss_G_contentGAN
                      # + self.loss_G_L1

        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D1.zero_grad()
        self.backward_D1()
        self.optimizer_D1.step()

        if not self.opt.mode == 'Original':
            self.optimizer_D2.zero_grad()
            self.backward_D2()
            self.optimizer_D2.step()

        if self.opt.mode != 'Original' and self.opt.back2:
            self.optimizer_G.zero_grad()
            self.backward_G_D1()
            self.optimizer_G.step()

            self.optimizer_G.zero_grad()
            self.backward_G_D2()
            self.optimizer_G.step()
        else:
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()