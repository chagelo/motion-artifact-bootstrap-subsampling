from turtle import forward
import torch
from Models.base_model import BaseModel
from Models import network

class UnetModel(BaseModel):
    """
    This class implements the unet model, for learning a mapping from input images to output images given paired data.
    
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        add new model-specific options, and rewrite default values for existing options.


        Parameters:
            parser:             original option
            is_train (bool):    train or test
        
        Returns:
            the modified parser.
        
        Note:
            use parser.set_defaults to modify default value of parameter
            use parser.add_argument to add new model-specific options
        """
        return parser

    def __init__(self, opt):
        """
        Initialize the unet model class

        Paremeters:
            opt (Option class): store all experiment flag, subclass of BaseOptions
        """
        super().__init__(opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['L1']
        # HACK: specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # NOTE: See https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/eb6ae80412e23c09b4317b04d889f1af27526d2d/models/pix2pix_model.py#51 
        self.model_names = ['net']
        # define networks, names should match self.model_names
        self.net = network.define_net(opt.input_nc, opt.output_nc, opt.net_name, opt.norm, not opt.no_dropout, opt.custom_init, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=opt.lr)
            # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=1e-8)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.image = input['image'].to(self.device)
        self.groundtruth = input['gt'].to(self.device)
        # BUG
        # self.image_paths = input['image_paths']
        # self.gt_paths = input['image_paths']
    
    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters> and <test>.
        """

        self.denoised_img = self.net(self.image)  # G(A)

    def backward(self):
        """
        Calculate loss and backward
        """
        # NOTE: here name should match the self.loss_names
        self.loss_L1 = self.criterionL1(self.denoised_img, self.groundtruth)
        self.loss_L1.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()