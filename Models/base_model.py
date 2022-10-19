import torch
import os
from collections import OrderedDict
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
class BaseModel(ABC):
    """
    This class is a abstract base class for models.
    To create s subclass, you need to implement the following five functions:
    __init__:                       initialize the class; first call BaseModel.__init__(self, opt)
    set_input:                      unpack data from dataset and apply preprocessing.
    forward:                        produce intermediate results.
    optimize_parameters:            calculate losses, gradients and update network weights.
    modify_commandline_options:     (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """
        Initialize the BaseModel class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        
        
        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call BaseModel.__init__(self, opt)
        Then, you need to define four lists:
            self.loss_names (str list):          specify the training losses that you want to plot and save.
            self.model_names (str list):         define networks used in our training.
            self.visual_names (str list):        specify the images that you want to display and save.
            self.optimizers (optimizer list):    define and initialize optimizers. You can define one       
                                                 optimizer for each network. If two networks are updated at 
                                                 the same time, you can use itertools.chain to group them. 
                                                 See cycle_gan_model.py for an example.
        """

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.phase = opt.phase
        # get device name: CPU or GPU
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  
        # save all the checkpoints to save_dir
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.exper_dir)
        # save intermediate result to save_path
        self.save_path = os.path.join(opt.save_path, opt.exper_dir)
        # save log to log_path
        self.log_dir = os.path.join(opt.log_dir, opt.exper_dir, 'loss')
        # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        if opt.preprocess != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.writer = SummaryWriter(self.log_dir)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Add new model-specific options, and rewrite default values for existing options.
        
        Parameters:
            parser              original option parser
            is_train (bool)     whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser    
    
    @abstractmethod
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        """
        pass

    def setup(self, opt):
        """
        load and print networks; create schedulers

        Parameters:
            opt: Option class, stores all the experiment flags. need to be a subclass of BaseOptions class
        """
        # if self.isTrain:
        #     # self.optimizers is not none, and has been defined in <__init__> of BaseModel' subclass.
        #     self.schedulers = [network.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        
        self.print_networks(opt.verbose)

    def eval(self):
        """
        to eval mode during test time
        """
        for name in self.model_names:
            if isinstance(name, str):
                # HACK: origin: getattr(self, 'net_' + name)
                net = getattr(self, name)
                net.eval()
    
    def test(self):
        """
        forward function used in test time.
        """
        with torch.no_grad():
            self.forward()
            # TODO: visualization
            self.compute_visuals()

    # TODO: visualization
    def compute_visuals(self):
        """
        Origin version:
            Calculate additional output images for visdom and HTML visualization
        """
        pass

    def get_image_paths(self):
        """
        Return image paths that are used to load current data
        """
        return self.image_paths
    
    def update_learning_rate(self):
        """
        Update learning rates for all the networks; called at the end of every epoch
        """
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    # TODO: visualization
    def plot_current_losses(self, i):
        """
        Save current loss. train.py will call this, in tensorboad, the loss will be ploted.
        """
        loss = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                loss[name] = getattr(self, "loss_" + name)
        self.writer.add_scalars("Loss/train", loss, i)

    def get_current_losses(self):
        """
        Return traning losses / errors. train.py will print out these errors on console, and save them to a file.
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name)) # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def save_networks(self, epoch):
        """
        Save all the networks to the disk.

        Parameters:
            epoch (int): current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
    
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                # HACK: origin: getattr(self, 'net' + name)
                net = getattr(self, name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # HACK: origin: torch.save(net.module.cpu().state_dict(), save_path)
                    torch.save(net.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    # HACK: origin: torch.save(net.cpu().state_dict(), save_path), here, .cpu() might aim to save gpu memory
                    torch.save(net.state_dict(), save_path)

    # TODO      
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        pass
    
    def load_networks(self, epoch):
        """
        Load all the networks from the disk.
        
        Parameters:
            epoch: current epoch, saved model has format '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                # HACK: orgin: getattr(self, 'net' + name)
                net = getattr(self, name)
                # NOTE: usage of torch.nn.DataParallel
                if isinstance(net, torch.nn.DataParallel):
                    net = net.Module
                print(net)
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                
                net.load_state_dict(state_dict, strict=False)

    def print_networks(self, verbose):
        """
        Print the total number of parameters in the network and (if verbose) network architecture
        
        Parameters:
            verbose (bool): if verbose, print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                # NOTE: orgini version: getattr(self, 'net' + name)
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Such as gan, when g updates, we should freeze d.
        Parameters:
            nets (network list)   a list of networks
            requires_grad (bool)  whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad