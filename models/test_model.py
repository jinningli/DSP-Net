from torch.autograd import Variable
from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        if opt.mode == 'PixelWise':
            self.model_names = ['G', 'D2']
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
            self.netD2 = networks.define_D_Pixel(opt.output_nc, opt.input_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        else:
            self.model_names = ['G']

            if self.opt.mode == 'Y':
                self.netG = networks.define_G_Y(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
            else:
                self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.load_networks(opt.which_epoch)
        self.fake_B = None
        self.real_B = None
        self.fake_C = None
        self.real_C = None

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.image_paths = input['A_paths']

        self.real_A = input_A
        self.real_B = input['B']
        self.real_C = input['C']

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        if self.opt.mode == 'Y':
            self.fake_B, self.fake_C = self.netG(self.real_A)
        else:
            self.fake_B = self.netG(self.real_A)
            if self.opt.mode == 'PixelWise':
                self.fake_C = self.netD2(self.fake_B)
