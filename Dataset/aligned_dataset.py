import os
from Dataset.base_dataset import BaseDataset
from Dataset.image_folder import make_dataset
import numpy as np
from torchvision.transforms import transforms

class AlignedDataset(BaseDataset):
    """
    A dataset class for paired image dataset.
    """

    def __init__(self, opt):
        """
        Initialize this dataset class.
        """

        BaseDataset.__init__(self, opt)
        # train data dir or test data dir
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.image_path = make_dataset(os.path.join(self.dir, 'artifact'))
        if opt.phase == 'train':
            self.gt_path = make_dataset(os.path.join(self.dir, 'noartifact'))
        self.phase = opt.phase
        

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index): 
        """
        return data specified by index
        """

        image_path = self.image_path[index]
        
        img = np.load(image_path).astype(np.float32)
        img_scale = np.std(img)
        img_mean = np.mean(img)
        img = (img - img_mean) / img_scale
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(self.opt.crop_size),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
        ])
        img = transform(img)

        gt = None
        if self.phase == 'train':
            gt_path = self.gt_path[index]
            gt = np.load(gt_path)
            gt_scale = np.std(gt)
            gt_mean = np.mean(gt)
            gt = (gt - gt_mean) / gt_scale
            gt = transform(gt)
        
        # TODO: there might be a dic?
        return {"image": img, "gt": gt, "image_path": self.image_path, "gt_path": self.gt_path}