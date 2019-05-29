import time
from options.train_options import TrainOptions
from dataloader.creatDataLoader import CreateDataLoader
from models import create_model

if __name__ == '__main__':
    opt = TrainOptions().parse()
    setattr(opt, 'isTrain', True)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    print('#Training pairs = %d' % dataset_size)

    model = create_model(opt)

    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        t_data = 0.0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            if i == 100:
                model.test_and_save(str(epoch) + '_' + str(i))
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                res = ''
                res += '[' + str(epoch) + "][" + str(epoch_iter) + '] Loss: '
                for k, v in losses.items():
                    res += '%s: %.5f ' % (k, v)
                res += "| AvgTime: %.3f" % t_data
                print(res)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        # print('End of epoch %d / %d \t Time Taken: %d sec' %
        #       (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
