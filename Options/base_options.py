import argparse
from Utils.util import *
import os
import Models, Dataset

class BaseOptions():
    """
    this class defines options used during both training and test time.

    """
    def __init__(self):
        """
        Reset the class; indicates the class hasn't been initailized
        """
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', default='/home1/ydliu/data/ixi-t2/mmbs', help='path to images (should have subfolders train, test, etc)')
        parser.add_argument('--save_path', type=str, default='/home1/ydliu/code/unet_v2/Images', help='intermediate result for train')
        parser.add_argiment('--log_dir', type=str, default='./Log')
        parser.add_argument('--exper_dir', type=str, default='mmbs_1', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='5', help='gpu ids: e.g. 0 0,2,3 0,2. use -1 for cpu')
        parser.add_argument('--checkpoints_dir',type=str, default='./checkpoints', help='saved models path')
        
        # model parameters
        # parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--net_name', type=str, default='unet', help='specify model architecture [unet | resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--netD', type=str, default='unet', help='specify model architecture [unet | resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--netG', type=str, default='unet', help='specify model architecture [unet | resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')    
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--custom_init', type=bool, default=False, help='whether use customize init or not, if True, use init_type specified method to init')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', default=True, type=bool, help='no dropout')
        parser.add_argument('--num_threads', default=2, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')


        # data augmentation
        parser.add_argument('--crop_size', type=int, default=256, help='croped size')
        parser.add_argument('--preprocess', type=str, default='', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        # parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--shuffle', type=bool, default='True', help='whether shuffle')

        # saved model suffix
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--verbose', default=True, type=bool, help='if specified, print more debugging information')
        
        parser.add_argument('--dataset_mode', type=str, default='mmbs', help='dataset for test')
        parser.add_argument('--model', type=str, default='mmbs', help='chooses which model to use. e.g. [cycle_gan | mmbs | test | unet]')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = Models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = Dataset.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


        args = vars(opt)

        # write options to file
        exper_dir = os.path.join(opt.checkpoints_dir, opt.exper_dir)
        mkdirs(exper_dir)
        file_name = os.path.join(exper_dir, '%s_opt.txt' % opt.phase)
        with open(file_name, 'wt') as opt_file:
            opt_file.write('---------- Options ----------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('------------ End ------------')
        opt_file.close()

        # mkdir for log saving
        mkdirs(os.path.join(opt.log_dir, opt.exper_dir, "trainloss"))

    def parse(self):
        """
        Parse our options, create checkpoints directory suffix, and set up gpu device.
        """
        opt = self.gather_options()
        
        # train or test
        opt.isTrain = self.isTrain

        # # process opt.suffix
        # # process opt.suffix
        # if opt.suffix:
        #     suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #     opt.name = opt.name + suffix

        # Print and save options
        self.print_options(opt)

        args = vars(opt)

        print('---------- Options ----------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('------------ End ------------')

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        

        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt
        return self.opt