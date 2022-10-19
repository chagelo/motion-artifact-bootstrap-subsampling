import time
from Dataset import create_dataset
from Models import create_model
from Options.train_options import TrainOptions

if __name__ == '__main__':
    opt = TrainOptions().parse()

    dataset = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)

    total_iters = 0

    start_time = time.time() # time for all train
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time() # time for entire epoch
        iter_data_time = time.time()   # time for data loading per iter
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            # every opt.print_freq size data, calculate t_data once.
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)   
            model.optimize_parameters()     # forward() and calculate loss, update weight.
            

            if total_iters % opt.save_loss_freq == 0:
                model.plot_current_losses(total_iters // opt.save_loss_freq)

            # every epoch save once
            if i == 0:
                model.save_images(epoch)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                print("epoch  {}, epoch_iter {} loss: {}".format(epoch, epoch_iter, losses))
            
            if total_iters % opt.save_latest_freq == 0:
                print("save the latest model (epoch %d, total_iters %d)" % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # cache model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # model.update_learning_rate()
        

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    print("Total time takn: %d sec" % (time.time() - start_time))
