import os
from options.test_options import TestOptions
from dataloader.creatDataLoader import CreateDataLoader
from models import create_model
from util.util import tensor2im, save_image
import ntpath

def save_img(tensor, save_name, save_path):
    im = tensor2im(tensor)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_image(im, os.path.join(save_path,save_name))

if __name__ == '__main__':
    opt = TestOptions().parse()
    setattr(opt, 'isTrain', False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.no_flip = True  # no flip
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        img_path = model.get_image_paths()
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]

        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        if not os.path.exists(model.save_dir):
            os.mkdir(model.save_dir)
        if not os.path.exists(os.path.join(model.save_dir, "test")):
            os.mkdir(os.path.join(model.save_dir, "test"))

        if model.real_A is not None:
            save_img(tensor=model.real_A, save_name=name+'_realA.png', save_path=os.path.join(model.save_dir, "test"))
        if model.fake_B is not None:
            save_img(tensor=model.fake_B, save_name=name+'_fakeB.png', save_path=os.path.join(model.save_dir, "test"))
        if model.real_B is not None:
            save_img(tensor=model.real_B, save_name=name+'_realB.png', save_path=os.path.join(model.save_dir, "test"))
        if model.fake_C is not None:
            save_img(tensor=model.fake_C, save_name=name + '_fakeC.png', save_path=os.path.join(model.save_dir, "test"))
        if model.real_C is not None:
            save_img(tensor=model.real_C, save_name=name + '_realC.png', save_path=os.path.join(model.save_dir, "test"))

        print("Processing " + img_path[0])
