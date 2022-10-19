"""
PatchGAN:
    PatchGAN output size [1, 30, 30] with input size [1, 256, 256]

"""
from matplotlib.pyplot import cm
import SimpleITK as sitk
import matplotlib.pyplot as plt
from Models import network
import torch
import numpy as np
from torchvision.transforms import ToTensor
from Utils.util import gen_mask

def run():
    path = "/home1/ydliu/data/t2/tt2/artifact/S96scan_1.nii"
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    net = network.define_G(1, 1, 64, 'unet')
    net.load_state_dict(torch.load('/home1/ydliu/code/unet_v2/checkpoints/test/latest_net_G.pth'))
    img_ = ToTensor()(img[8].astype(np.float32))

    mask = []
    for i in range(15):
        mask.append(gen_mask(img_, 3))
    mask = torch.concat(mask, dim=0)

    k_origin = torch.fft.fftshift(torch.fft.fft2(img_))
    k_down = k_origin * mask
    img = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(k_down)))
    print(img.shape)
    out = net(img)
    out = torch.mean(out, dim=0)
    img = img.detach().numpy()
    out = out.detach().numpy()
    print(img.shape, out.shape)
    plt.imsave('/home1/ydliu/test/real_input.jpg', img_[0], cmap=cm.gray)
    plt.imsave('/home1/ydliu/test/real_out.jpg', out[0], cmap=cm.gray)

def run2():
    #path1 = "/home1/ydliu/data/ixi-t2/mmbs/train/artifact/IXI662-Guys-1120-T2.nii_slice_95.npy"
    #path2 = "/home1/ydliu/data/ixi-t2/mmbs/train/noartifact/IXI662-Guys-1120-T2.nii_slice_95.npy"
    path1 = "/home1/ydliu/data/ixi-t2/mmbs/test/artifact/IXI362-HH-2051-T2.nii_slice_49.npy"
    path2 = "/home1/ydliu/data/ixi-t2/mmbs/test/noartifact/IXI362-HH-2051-T2.nii_slice_49.npy"
    path3 = "/home1/ydliu/data/ixi-t2/mmbs/test/result/IXI362-HH-2051-T2.nii_slice_49.npy"
    img1 = np.load(path1)
    img2 = np.load(path2)
    img3 = np.load(path3)
    plt.imsave('noisy.jpg', img1, cmap=cm.gray)
    plt.imsave('clean.jpg', img2, cmap=cm.gray)
    plt.imsave('result.jpg', img3, cmap=cm.gray)
run2()