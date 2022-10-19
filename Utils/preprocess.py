import SimpleITK as sitk
import os
from PIL import Image
import numpy as np

def readPath(path):

    filepaths = []
    for root, dirs, filenames in os.walk(path):
        for name in filenames:
            filepath = os.path.join(root, name)
            filepaths.append(filepath)

    return filepaths

def resampleEach(volume_path, name, savepath, interpolator = sitk.sitkLinear, new_spacing = [0.4, 0.4, 5]):
    """
    resample a 3d vlolume with new space

    """
    volume = sitk.ReadImage(volume_path)

    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    resampled_image = sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())

    sitk.WriteImage(resampled_image, os.path.join(savepath, 'R_' + name))
    print(os.path.join(savepath, 'R_' + name) + ' finished!')

def resampleAll(files, savepath):
    """
    resample all volume in a specific directory

    """
    new_space = [1.0, 1.0, 1.0]

    if os.path.exists(savepath) == False:
        os.makedirs(savepath)

    for file in files:
        name = os.path.basename(file)
        resampleEach(file, name, savepath, new_spacing=new_space)

def check_physic_parame(path):
    
    par_set = set()

    x_max = 0
    y_min = 1000
    y_max = 0
    x_min = 1000
    for name in os.listdir(path):
        filename = os.path.join(path, name)
        img = sitk.ReadImage(filename)
        par_set.add(img.GetDirection())
        x, y, z = img.GetSize()
        x_max = max(x, x_max)
        y_max = max(y, y_max)
        x_min = min(x, x_min)
        y_min = min(y, y_min)

    if len(par_set) > 1:
        print("images have different physic paremeter!")

    print(y_max, x_max, x_min, y_min)


def sliceAllVolume(savepath, paths, ratio=0.5):
    """
    get middle half of volume
    """
    for file in paths:
        sitkimg = sitk.ReadImage(file)
        _, _, c = sitkimg.GetSize()
        num = int(c * ratio)
        l = (c - num) // 2
        r = l + num
        
        img = sitk.GetArrayFromImage(sitkimg[:, :, l:r])
        filename = os.path.splitext(os.path.basename(file))[0]

        for i in range(num):
            filepath = os.path.join(savepath, filename + str(i) + '.npy')
            np.save(filepath, img[i])
        print("{} finished ".format(filename))

if __name__ == "__main__":
    path = 'D:/data/tt2/resampled'
    savepath = 'D:/data/tt2/slice_2'
    dirs = ['noartifact', 'simuartifact']

    # test_path = 'D:\\code\\unet_mr\\Data\\train'
    # test_savepath = 'D:\\code\\unet_mr\\Data\\test'
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)


    # resampleAll(filepaths, savepath)
    # check_physic_parame(savepath)

    for dir in dirs:
        files = readPath(os.path.join(path, dir))
        p = os.path.join(savepath, dir)

        if not os.path.exists(p):
            os.makedirs(p)
        
        sliceAllVolume(savepath=p, paths=files)