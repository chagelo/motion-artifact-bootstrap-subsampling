import os
import numpy as np
import torch
from Dataset.base_dataset import BaseDataset
from Dataset.image_folder import make_dataset, make_savepath
from Utils.util import gen_mask
import torchvision.transforms as transforms

class MmbstestDataset(BaseDataset):
    """
    As for MR_motion_bootstrap_subsampling task, This dataset class load a set of clean images to train.
    #TODO: complex image implementation
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # magnitude means \sqrt{real^2 + imaginary^2}, complex means [[real, imaginary]]
        parser.add_argument("--is_magnitude", type=bool, default=True, help="use magnitude or complex, the difference is the number of channels, magnitude image with channel 1, complex image with channel 2")
        parser.add_argument('--R', type=float, default=3, help='the downsampling rate')
        parser.add_argument('--augment_data', type=bool, default=False)
        parser.add_argument('--N', type=int, default=15, help='the number of mask')
        return parser

    def __init__(self, opt):
        """
        Initialization
        """
        super().__init__(opt)
        self.N = opt.N
        self.R = opt.R
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.image_path = make_dataset(os.path.join(self.dir, 'artifact'))
        self.save_path = make_savepath(os.path.join(self.dir, 'result'), self.image_path)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = np.load(image_path).astype(np.float32)
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop(self.opt.crop_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
        ])
        F = transform(image)
        mask = []        
        for i in range(self.N):
            mask.append(gen_mask(F, self.R))
        # [N, 1, H, W]
        mask = torch.concat(mask, dim=0)
        k_origin = torch.fft.fftshift(torch.fft.fft2(F))
        k_down = k_origin * mask
        # [N, 1, H, W]
        D = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(k_down)))
        scale = torch.std(D)
        D = D / scale
        save_path = self.save_path[index]
        return {"down": D, "scale": scale, "save_path": save_path}
        # return {"full": F, "mask": mask, "save_path": save_path}
        
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
