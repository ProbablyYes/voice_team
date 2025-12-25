import torch
import cv2
import numpy as np
import os


def load_dir(path, start, end):
    lmss = []
    imgs_paths = []
    for i in range(start, end):
        if os.path.isfile(os.path.join(path, str(i) + '.lms')):
            lms = np.loadtxt(os.path.join(
                path, str(i) + '.lms'), dtype=np.float32)
            lmss.append(lms)
            imgs_paths.append(os.path.join(path, str(i) + '.jpg'))
    lmss = np.stack(lmss)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lmss = torch.as_tensor(lmss).to(device)
    return lmss, imgs_paths
