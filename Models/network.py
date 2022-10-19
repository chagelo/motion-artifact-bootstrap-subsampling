import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torch.fft as fft
from torch.nn import init
from torch.optim import lr_scheduler

def init_net(net, custom_init=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)       the network to be initialized
        custom_init (bool)  whether use pytorch default initialization, default initialization might be better
        init_type (str)     the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)        scaling factor for normal, xavier and orthogonal
        gpu_ids (int list)  which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # NOTE: here we have only one gpu
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    if custom_init == True:
        init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)       network to be initialized
        init_type (str)     the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)   scaling factor for normal, xavier and orthogonal.
    
    Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class Identity(nn.Module):
    def forward(self, x):
        return 

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    """
    return a learning rate scheduler

    parameters:
        optimizer: the optimizer of the network
        opt (option class): stores all the experiment flags; needs to be a subclass of BaseOptions. opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    """

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def define_net(input_nc, output_nc, net_name, norm='batch', use_droup=False, custom_init=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Create a model, unet_128, unet_256, resnet_9blocks, resnet_6blocks...
    Parameters:
        input_nc (int)      the number of channels in input images
        output_nc (int)     the number of channels in output images
        net_name (str)      the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str)          the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool)  whether use dropout layers or not
        init_type (str)     the name of our initialization method
        init_gain (float)   scaling factor for normal, xavier and orthogonal
        gpu_ids (int list)  which GPUs the network runs on: e.g., 0,1,
    
    Return:
        an initialized model

    Implemented model:
        unet
        todo: resnet-based model from https://github.com/jcjohnson/fast-neural-style
    
    use <init_net> to initialize
    """

    if net_name == 'unet':
        net = Unet(input_nc, output_nc)
    else:
        raise NotImplementedError('architecture [%s] is not implemented' % net_name)
    return init_net(net, custom_init, init_type, init_gain, gpu_ids)

def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_droup=False, custom_init=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Create a model, unet_128, unet_256, resnet_9blocks, resnet_6blocks...
    Parameters:
        input_nc (int)      the number of channels in input images
        output_nc (int)     the number of channels in output images
        ngf (int)           the number of filters in the last conv layer
        netG (str)          the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str)          the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool)  whether use dropout layers or not
        custom_init (bool)  whether use customize init or pytorch default init
        init_type (str)     the name of our initialization method
        init_gain (float)   scaling factor for normal, xavier and orthogonal
        gpu_ids (int list)  which GPUs the network runs on: e.g., 0,1,
    
    Return:
        an initialized model

    Implemented model:
        unet
        todo: resnet-based model from https://github.com/jcjohnson/fast-neural-style
    
    use <init_net> to initialize
    """
    net = None
    norm_layer = get_norm_layer(norm)
    if netG == 'unet':
        net = UnetGenerator(input_nc, 5, ngf, norm_layer)
    else:
        raise NotImplementedError('architecture [%s] is not implemented' % netG)
    return init_net(net, custom_init, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, norm='batch', custom_init=False, init_type='instance', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator
    Parameters:
        input_nc (int)     the number of channels in input images
        ndf (int)          the number of filters in the first conv layer
        netD (str)         the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         the type of normalization layers used in the network.
        init_type (str)    the name of the initialization method.
        init_gain (float)  scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netD == 'unet':
        net = PatchGan(input_nc, ndf)
    else:
        raise NotImplementedError('architecture [%s] is not implemented' % netD)
    return init_net(net, custom_init, init_type, init_gain, gpu_ids)

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return 

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, c_mid = None) -> None:
        super().__init__()
        if not c_mid:
            c_mid = c_out
        self.doub_conv = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, padding = 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.doub_conv(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(c_in, c_out)
        )
    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, c_in, c_out, bilinear=True):
        super().__init__()

        if bilinear :
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(c_in, c_out, c_in // 2)
        else:
            self.up = nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride = 2)
            self.conv = DoubleConv(c_in, c_out)
    
    def forward(self, x1, x2):
        x2 = self.up(x2)

        diffX = x1.shape[2] - x2.shape[2]
        diffY = x1.shape[3] - x2.shape[3]

        x2 = F.pad(x2, (diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2))

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Unet(nn.Module):
    """
    Create a unet Module
    """
    def __init__(self, n_channels, n_classes=1, bilinear=False) -> None:
        """
        Parameters:
            input_nc (int)
            output_nc (int)
            bilinear (bool)     whether use bilinear to do upsampling or not
        """
        
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        

        # drop only begin a few layers or initial like origin papers?

        self.initconv = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear=bilinear)
        self.up2 = Up(512, 256 // factor, bilinear=bilinear)
        self.up3 = Up(256, 128 // factor, bilinear=bilinear)
        self.up4 = Up(128, 64, bilinear=bilinear)

        self.conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        ly1_out = self.initconv(x)
        ly2_out = self.down1(ly1_out)
        ly3_out = self.down2(ly2_out)
        ly4_out = self.down3(ly3_out)
        ly5_out = self.down4(ly4_out)

        x = self.up1(ly4_out, ly5_out)
        x = self.up2(ly3_out, x)
        x = self.up3(ly2_out, x)
        x = self.up4(ly1_out, x)

        x = self.conv(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c_in, c_out, 3),
            nn.InstanceNorm2d(c_out),
            nn.LeakyReLU())
    def forward(self, x):
        return self.convblock(x)

class UpBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.upblock = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c_in, c_out, 3, 1),
            nn.InstanceNorm2d(c_out),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.upblock(x)

class Ck(nn.Module):
    def __init__(self, c_in, c_out, s=2, norm='instance'):
        super().__init__()
        temp = [nn.ConstantPad2d(1, 1.0),
            nn.Conv2d(c_in, c_out, 4, s)]

        if norm == 'instance':
            temp.append(nn.InstanceNorm2d(c_out))
        temp.append(nn.LeakyReLU())

        self.ck = nn.Sequential(*temp)
    def forward(self, x):
        return self.ck(x)

def DownSample(x, mask):
    """
     x: [N, 2C, H, W] or [N, C, H, W]
    """
    shift = fft.fftshift(fft.fft2(x))
    downshift = shift * mask
    return torch.abs(fft.ifft2(fft.ifftshift(downshift)))


class PatchGan(nn.Module):
    def __init__(self, input_nc, ndf):
        super().__init__()
        self.C = nn.Sequential(
            Ck(input_nc, ndf, s=2, norm=None),
            Ck(ndf, ndf * 2, s=2, norm='instance'),
            Ck(ndf * 2, ndf * 4, s=2, norm='instance'),
            Ck(ndf * 4, ndf * 8, s=1, norm='instance'),
            Ck(ndf * 8, 1, s=1, norm=None)
        )
    def forward(self, x):
        return self.C(x)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, num_downs, ngf=64, norm_layer=nn.InstanceNorm2d):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        output_nc = input_nc
        self.output_nc = output_nc
        
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 16, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(output_nc * 2, output_nc, kernel_size=1, stride=1)

    def forward(self, input):
        """Standard forward"""
        x = self.model(input)
        x = self.conv(torch.cat([x, x + input], dim=1))
        
        if self.output_nc == 1:
            x = self.relu(x)
        return x

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downmaxpool = nn.MaxPool2d(2, 2)
        downconv1 = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1, padding_mode='reflect')
        downnorm1 = norm_layer(inner_nc)
        downrelu = nn.LeakyReLU(inplace=True)
        downconv2 = nn.Conv2d(inner_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1, padding_mode='reflect')
        downnorm2 = norm_layer(inner_nc)

        upsample = nn.Upsample(scale_factor=2)
        upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        down = [downconv1, downnorm1, downrelu, downconv2, downnorm2, downrelu]
        
        if outermost:
            up = [
                nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1, padding_mode='reflect'), 
                nn.InstanceNorm2d(inner_nc),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(inner_nc, inner_nc, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.InstanceNorm2d(inner_nc),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(inner_nc, outer_nc, kernel_size=1, stride=1)
            ]
            model = down + [submodule] + up
        elif innermost:
            up = [upsample, upconv]
            model = [downmaxpool] + down + up
        else:
            up = [
                nn.Conv2d(inner_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1, padding_mode='reflect'), 
                nn.InstanceNorm2d(inner_nc),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(inner_nc, inner_nc, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.InstanceNorm2d(inner_nc),
                nn.LeakyReLU(inplace=True),
                upsample,
                upconv
            ]
            model = [downmaxpool] + down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

if __name__ == '__main__':
    a = torch.rand([1, 2, 256, 256])
    net2 = UnetGenerator(2, 1, 5)
    print(net2)
    b = net2(a)
    
    print(b.shape)
    print(torch.__version__)
    print(torch.cuda.is_available())
    A = nn.Sequential(nn.Conv1d(2, 1, 1),
        nn.ConstantPad1d(1, 1.0)
    )
    print(A,*(A))
    