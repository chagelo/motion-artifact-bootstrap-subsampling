"""
path/[artifact, noartifact, weak artifact]/*.nii
convert path/artifact/*.nii to dst/artifact/*.npy
"""

import os
import SimpleITK as sitk
import numpy as np
path = "/home1/ydliu/data/t2/tt2/"
dst = "/home1/ydliu/data/t2/test"

if not os.path.exists(dst):
    os.mkdir(dst)

for r, ds, fs in os.walk(path):
    if len(fs) != 0:
        for f in fs:
            basename = os.path.basename(r)
            f = os.path.splitext(f)[0]
            
            dir = os.path.join(dst, basename)
            if not os.path.exists(dir):
                os.mkdir(dir)
            file = os.path.join(dir, f)
            # print(file)
            img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(r, f)))
            h, w, c = img.shape
            for i in range(h // 4, h // 2 + h // 4):
                np.save(file + "_slice{0}.npy".format(i), img[i, :, :])