import os
import numpy as np
import torch
import random
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str or list of) -- a list of directory paths
    """
    if isinstance(paths, list):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def gen_mask(x, R):
    """
    generate downsampling mask
    x: 
    R: downsample rate (accelerate factor), 3 or 4
    """
    h, w = x.shape[-2:]
    mask = torch.zeros((h, w), dtype=torch.float32)
    
    nACS = round(h / (R ** 2))
    ACS_s = round((h - nACS) / 2)
    ACS_e = ACS_s + nACS
    mask[ACS_s:ACS_e, :] = 1
    torch.cuda.empty_cache()
    nSamples = int(h / R)
    r = np.floor(np.random.normal(h / 2, 70, nSamples))
    r = np.clip(r.astype(int), 0, h - 1)
    mask[r.tolist(), :] = 1
    # a = sorted(set(r.tolist()))
    # print(a, len(a))
    # if phase == 'train':
    # return torch.unsqueeze(mask, 0)
    # else:
    return torch.unsqueeze(torch.unsqueeze(mask, 0), 0)
    

def cal_all_metrics(pred_path, target_path):
    """
    input_path contains all input images, target_path contains all ground truth.
    """
    
    ssim_record = []
    psnr_record = []
    mse_record = []
    for root, dirs, files in os.walk(pred_path):
        for file in files:
            img1_path = os.path.join(pred_path, file)
            img2_path = os.path.join(target_path, file)
            img1 = np.squeeze(np.load(img1_path))
            img2 = np.load(img2_path)

            img1 = img1 / np.max(img1)
            img2 = img2 / np.max(img2)
            ssim_record.append(ssim(img1, img2))
            psnr_record.append(psnr(img1, img2))
            mse_record.append(mse(img1, img2))
    return np.mean(ssim_record), np.mean(psnr_record), np.mean(mse_record)

class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

# if __name__ == '__main__':
#     pass

if __name__ == '__main__':
    pred_path = '/home1/ydliu/data/ixi-t2/mmbs/test/result'
    target_path = '/home1/ydliu//data/ixi-t2/mmbs/test/noartifact'
    noisy_path = '/home1/ydliu//data/ixi-t2/mmbs/test/artifact'
    s, p, m = cal_all_metrics(noisy_path, target_path)
    print("simulated motion:", s, p, m)
    s, p, m = cal_all_metrics(pred_path, target_path)
    print("result:", s, p, m)