import torch
import torch.nn as nn
import Models.network as network
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from Models.base_model import BaseModel
from Utils.util import ImagePool, mkdirs

class MmbsModel(BaseModel):
    """
    mr motion bootstrap subsampling, refer to "Unpaired MR Motion Artifact Deep Learning Using Outlier-Rejecting Bootstrap Aggregation", Gyutaek Oh, Jeong Eun Lee, and Jong Chul Ye, IEEE TMI
    tensorflow code: https://github.com/jongcye/MR_motion_bootstrap_subsampling
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
        parser.set_defaults(beta1=0.5)
        parser.set_defaults(beta2=0.999)

        parser.add_argument('--lambda_cycle', type=float, default=10)
        parser.add_argument('--pool_size', type=int, default=50, help='the size of the image pool')
        parser.set_defaults(norm='instance')
        # parser.add_argument('--augmentation', action='store_true', help='true if use data augmentation')
        return parser
        
    def __init__(self, opt):
        super().__init__(opt)

        self.lambda_cycle = opt.lambda_cycle
        self.loss_names = ['G', 'cycle', 'G_total', 'D']
        self.visual_names = ['G', 'D', 'G_total']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else: 
            pass

        self.G = network.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.custom_init, opt.init_type, opt.init_gain, self.gpu_ids)

        # TODO
        self.D = network.define_D(opt.input_nc, opt.ndf, opt.netD, opt.norm, opt.custom_init, opt.init_type, opt.init_gain, opt.gpu_ids)
        # Downsample net
        self.DS = network.DownSample

        if self.isTrain:
            assert (opt.input_nc == opt.output_nc)
            
            self.fake_pool = ImagePool(opt.pool_size)

            # define loss function
            self.criterionGAN = nn.MSELoss()
            self.criterionCycle = nn.L1Loss()

            # define optimizers, schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizersG = torch.optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizersD = torch.optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizersG)
            self.optimizers.append(self.optimizersD)

    def set_input(self, input):
        self.real_F = input['real_F'].to(self.device)
        self.real_D = input['real_D'].to(self.device)
        self.mask_F = input['mask_F'].to(self.device)
        self.mask_D = input['mask_D'].to(self.device)
    
    def save_images(self, epoch):
        """
        every epoch, save intermediate images once in ./Images/exper_dir/
        save one slice of self.recon_F, shape: [n, c, h, w] 
        """
        mkdirs(self.save_path)
        path = self.save_path + '/' + self.phase
        mkdirs(path)
        plt.imsave(path + '/epoch_{0}.jpg'.format(epoch), self.recon_F[0, 0, :, :].cpu().detach().numpy(), cmap=cm.gray)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_F = self.G(self.real_D)
        self.fake_D = self.DS(self.real_F, self.mask_F)

        self.recon_F = self.G(self.fake_D)
        self.recon_D = self.DS(self.fake_F, self.mask_D)

    def backwardG(self):
        """Calculate loss for generator"""
        temp = self.D(self.fake_F)
        self.loss_G = self.criterionGAN(temp, torch.ones_like(temp, requires_grad=False))
        self.loss_cycle = self.criterionCycle(self.real_F, self.recon_F) + self.criterionCycle(self.real_D, self.recon_D)
        self.loss_G_total = self.loss_G + self.lambda_cycle * self.loss_cycle
        self.loss_G_total.backward()

    def backwardD(self):
        """Calculate GAN loss for discriminator"""
        fake = self.fake_pool.query(self.fake_F)
        pred_real = self.D(self.real_F)
        pred_fake = self.D(fake.detach())

        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D = loss_D
        loss_D.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # downsampling and compute recon images from downsampled images 
        # G
        self.set_requires_grad([self.D], False)
        self.optimizersG.zero_grad()     # set G's gradients to zero
        self.backwardG()                # calcute G's gradients
        self.optimizersG.step()          # update G
        # D
        self.set_requires_grad([self.D], True)
        self.optimizersD.zero_grad()     # set D's gradients to zero
        self.backwardD()                # calcute D's gradients
        self.optimizersD.step()          # update D