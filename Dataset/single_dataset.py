import os
import numpy as np
from Dataset.base_dataset import BaseDataset
from Dataset.image_folder import make_dataset, make_savepath
import torchvision.transforms as transforms

class SingleDataset(BaseDataset):
    """
    This dataset class load a set of images
    """

    def __init__(self, opt):
        """
        Initialization
        """
        super().__init__(opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.image_path = make_dataset(os.path.join(self.dir, 'artifact'))
        self.save_path = make_savepath(os.path.join(self.dir, 'result'), self.image_path)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = np.load(image_path).astype(np.float32)
        image = transforms.ToTensor()(image)
        
        save_path = self.save_path[index]

        return {"image": image, "save_path": save_path}
        