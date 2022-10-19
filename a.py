from matplotlib.pyplot import cm
import SimpleITK as sitk
import matplotlib.pyplot as plt
from Models import network
import torch
import numpy as np
from torchvision.transforms import ToTensor
from Utils.util import gen_mask
import scipy.io as sio

noisy = np.load('/home1/ydliu/code/mmbs/Data/test/real/S86scan_2_slice9.npy')
# clean = np.load("/home1/ydliu/data/ixi-t2/mmbs/test/result/IXI379-Guys-0943-T2.nii_slice_66.npy")
official = sio.loadmat("/home1/ydliu/code/mmbs/Results/MR_motion_reduction/test_N=15/real/S86scan_2_slice9.npy")

plt.imsave('noisy.jpg', noisy, cmap=cm.gray)
# plt.imsave('clean.jpg', clean, cmap=cm.gray)
plt.imsave('official.jpg', official['data'], cmap=cm.gray)