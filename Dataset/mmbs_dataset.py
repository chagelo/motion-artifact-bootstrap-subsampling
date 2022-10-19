import os
import numpy as np
import torch
from Dataset.base_dataset import BaseDataset
from Dataset.image_folder import make_dataset, make_savepath
from Utils.util import gen_mask
import torchvision.transforms as transforms

class MmbsDataset(BaseDataset):
    """
    As for MR_motion_bootstrap_subsampling task, This dataset class load a set of clean images to train.
    #TODO: complex image implementation
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # magnitude means \sqrt{real^2 + imaginary^2}, complex means [[real, imaginary]]
        parser.add_argument("--is_magnitude", type=bool, default=True, help="use magnitude or complex, the difference is the number of channels, magnitude image with channel 1, complex image with channel 2")
        parser.add_argument('--R', type=float, default=3, help='the downsampling rate')
        parser.set_defaults(norm='instance')
        parser.add_argument('--augment_data', type=bool, default=False)
        return parser

    def __init__(self, opt):
        """
        Initialization
        """
        super().__init__(opt)
        self.R = opt.R
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.image_path = make_dataset(os.path.join(self.dir, 'noartifact'))
        self.augmentation = opt.augment_data
        
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = np.load(image_path).astype(np.float32)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(self.opt.crop_size),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
        ])
        Input_F = transform(image)
        Input_D= Input_F.clone()

        mask_F = gen_mask(Input_F, self.R)
        mask_D = gen_mask(Input_D, self.R)

        k_orig_F = torch.fft.fftshift(torch.fft.fft2(Input_F))
        k_down_F = k_orig_F * mask_F
        tmp = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(k_down_F)))

        k_orig_D = torch.fft.fftshift(torch.fft.fft2(Input_D))
        k_down_D = k_orig_D * mask_D
        Input_D = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(k_down_D)))

        Scale_F = torch.std(tmp)
        Input_F = Input_F / Scale_F
        Scale_D = torch.std(Input_D)
        Input_D = Input_D / Scale_D

        if self.augmentation:
            Input_F, mask_F = self.augment_data(Input_F, mask_F)
            Input_D, mask_D = self.augment_data(Input_D, mask_D)
        return {"real_F": Input_F, "mask_F": mask_F, "real_D": Input_D, "mask_D": mask_D}
        
    def augment_data(self, image, mask):
        """
        augmentation:
        full: [c, h, w], origin image without downsampling
        mask: [c, h, w], used to downsampling
        """
        p_flip = np.random.rand(1)
        if p_flip > 0.85:
            image = torch.flip(image, 2)
            mask = np.flip(mask, 2)
        elif p_flip < 0.15:
            image = torch.flip(image, 1)
            mask = torch.flip(mask, 1)
        return image, mask
