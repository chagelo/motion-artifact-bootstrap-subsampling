import time
from Dataset import create_dataset
from Options.test_options import TestOptions
from Models import create_model


if __name__ == '__main__':
    opt = TestOptions().parse()

    dataset = create_dataset(opt)
    data_size = len(dataset)
    print('The number of test images = %d' % data_size)
    
    model = create_model(opt)
    model.setup(opt)

    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        img_path = model.get_image_paths()
        if i % 50 == 0:
            print('processing (%04d)-th image...%s' % (i, img_path))
        model.save_images()
        # exit(0)