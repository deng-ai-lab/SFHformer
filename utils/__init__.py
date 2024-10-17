from .common import AverageMeter, ListAverageMeter, read_img, write_img, hwc_to_chw, chw_to_hwc, read_img255
from .data_parallel import BalancedDataParallel

import numpy as np
import os
import cv2
import math

from skimage import metrics
from sklearn.metrics import mean_absolute_error

def MAE(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])

def PSNR(img1, img2):
    mse_ = np.mean((img1 - img2) ** 2 )
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)

def SSIM(img1, img2):
    return metrics.structural_similarity(img1, img2, data_range=1, channel_axis=2)

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def load_img16(filepath):
    return cv2.cvtColor(cv2.imread(filepath, -1), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))