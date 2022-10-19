import numpy as np
from Models.base_model import BaseModel
from . import network
import torch


class MmbstestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.
    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']  # only generator is needed.
        self.G = network.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.custom_init, opt.init_type, opt.init_gain, self.gpu_ids)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        #self.F = input['full'].to(self.device)
        #self.mask = input['mask'].to(self.device)
        self.D = torch.squeeze(input['down'], dim=0).to(self.device)
        self.scale = torch.squeeze(input['scale']).to(self.device)
        self.image_paths = input['save_path']

    def forward(self):
        """Run forward pass."""
        # self.mask = torch.squeeze(self.mask, dim=0)

        # k_origin = torch.fft.fftshift(torch.fft.fft2(self.F))
        # k_down = k_origin * self.mask
        # D = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(k_down)))

        self.recon_F = self.G(self.D) * self.scale  # G(real)
        self.recon_F = torch.mean(self.recon_F, dim=0)
        
    def save_images(self):
        c, w, h = self.recon_F.shape

        image = self.recon_F[0, :, :].cpu().detach().numpy()
        path = self.image_paths[0]
        
        # from matplotlib.pyplot import cm
        # import matplotlib.pyplot as plt
        # plt.imsave('/home1/ydliu/code/unet_v2/clean.jpg', image, cmap=cm.gray)
        # plt.imsave('/home1/ydliu/code/unet_v2/noisy.jpg', self.F[0, 0, :, :].cpu().detach().numpy(), cmap=cm.gray)
        # exit(0)
        np.save(path, image)
        


    def optimize_parameters(self):
        """No optimization for test model."""
        pass