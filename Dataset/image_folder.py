import os
from Utils.util import mkdir

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.npy',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    
    return images[:min(max_dataset_size, len(images))]

def make_savepath(root, image_path):
    """
    Generate denoised images names for each image of output of model
    for test, root has format "test/simuartifact" or "test/noartifact"
    """
    mkdir(root)
    return [os.path.join(root, os.path.basename(image)) for image in image_path]