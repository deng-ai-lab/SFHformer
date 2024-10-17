import os.path
import time
import cv2
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from math import ceil


def read_img255(filename):
    img0 = cv2.imread(filename)
    img1 = img0[:, :, ::-1].astype('float32') / 1.0
    return img1

# for lsui, augment and aling all need to resize to 256 if size is lower than this num

def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    if min(H, W) < size:
        if H <= W:
            for i in range(len(imgs)):
                imgs[i] = cv2.resize(imgs[i], (256, ceil(W * 256 / H)), interpolation=cv2.INTER_LINEAR)
        else:
            for i in range(len(imgs)):
                imgs[i] = cv2.resize(imgs[i], (ceil(H * 256 / W), 256), interpolation=cv2.INTER_LINEAR)

    H, W, _ = imgs[0].shape
    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    if not only_h_flip:
        # bad data augmentations for outdoor
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs


def align_for_valid(imgs=[], size_H=256, size_W=256):
    H, W, _ = imgs[0].shape
    Hc = size_H
    Wc = size_W

    if min(H, W) < size_H:
        if H <= W:
            for i in range(len(imgs)):
                imgs[i] = cv2.resize(imgs[i], (256, ceil(W * 256 / H)), interpolation=cv2.INTER_LINEAR)
        else:
            for i in range(len(imgs)):
                imgs[i] = cv2.resize(imgs[i], (ceil(H * 256 / W), 256), interpolation=cv2.INTER_LINEAR)

    H, W, _ = imgs[0].shape

    Hs = random.randint(0, H - Hc)
    Ws = random.randint(0, W - Wc)
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]
    return imgs


def align_for_test(imgs=[], local_size=32):
    H, W, _ = imgs[0].shape
    if min(H, W) < 256:
        if H <= W:
            for i in range(len(imgs)):
                imgs[i] = cv2.resize(imgs[i], (256, ceil(W * 256 / H)), interpolation=cv2.INTER_LINEAR)
        else:
            for i in range(len(imgs)):
                imgs[i] = cv2.resize(imgs[i], (ceil(H * 256 / W), 256), interpolation=cv2.INTER_LINEAR)
    H, W, _ = imgs[0].shape
    Hc = local_size * (H // local_size)
    Wc = local_size * (W // local_size)
    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]
    return imgs


# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, only_h_flip=False):
        super().__init__()
        train_list_haze = '/home/jxy/projects_dir/datasets/LSUI_new/train/train_hazy_list_3500.txt'

        with open(train_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = haze_names
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.train_data_dir = train_data_dir
        self.crop_size = crop_size
        self.only_h_flip = only_h_flip

    def get_images(self, index):

        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_path = os.path.join(self.train_data_dir, 'hazy', haze_name)
        gt_path = os.path.join(self.train_data_dir, 'GT', gt_name)
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        haze_img = read_img255(haze_path)
        gt_img = read_img255(gt_path)
        [haze_img, gt_img] = augment([haze_img, gt_img], size=self.crop_size, edge_decay=0., only_h_flip=self.only_h_flip)
        haze_img = np.ascontiguousarray(haze_img).astype('uint8')
        gt_img = np.ascontiguousarray(gt_img).astype('uint8')
        transform_haze = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)

        return {'source': haze, 'target': gt, 'filename': haze_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)

class TestData(data.Dataset):
    def __init__(self, local_size, val_data_dir, flag=True):
        super().__init__()

        val_list_haze = '/home/jxy/projects_dir/datasets/LSUI_new/test/test_hazy_list.TXT'

        with open(val_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = haze_names

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.flag = flag
        self.local_size = local_size

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        haze_path = os.path.join(self.val_data_dir, 'hazy', haze_name)
        gt_path = os.path.join(self.val_data_dir, 'GT', gt_name)
        haze_img = read_img255(haze_path)
        gt_img = read_img255(gt_path)
        [haze_img, gt_img] = align_for_test([haze_img, gt_img], local_size=self.local_size)
        haze_img = np.ascontiguousarray(haze_img).astype('uint8')
        gt_img = np.ascontiguousarray(gt_img).astype('uint8')
        transform_haze = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)

        return {'source': haze, 'target': gt, 'filename': haze_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
