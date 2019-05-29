from .pix2pix_model import Pix2PixModel
from .doubleModel import DoubleModel
from .double2l2Model import Double2l2Model
from .pixel2l2_model import Pixel2l2Model
from .pixelBlurModel import PixelBlurModel

def create_model(opt):
    if opt.isTrain:
        if opt.mode == 'Double':
            model = DoubleModel()
        elif opt.mode == 'Double2l2':
            model = Double2l2Model()
        elif opt.mode == 'Pixel2l2':
            model = Pixel2l2Model()
        elif opt.mode == 'PixelBlur':
            model = PixelBlurModel()
        else:
            model = Pix2PixModel()
    else:
        from .test_model import TestModel
        model = TestModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
