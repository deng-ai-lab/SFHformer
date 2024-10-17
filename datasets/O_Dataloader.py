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
import math


def read_img255(filename):
    img0 = cv2.imread(filename)
    img1 = img0[:, :, ::-1].astype('float32') / 1.0
    return img1


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

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

def align(imgs=[], size_H=448, size_W=608):
    H, W, _ = imgs[0].shape
    Hc = size_H
    Wc = size_W
    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]
    return imgs


# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, only_h_flip=False):
        super().__init__()
        train_list_haze = '/home/jxy/projects_dir/datasets/O-HAZY/train.txt'

        with open(train_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('hazy')[0] + 'GT' + i.split('hazy')[1] for i in haze_names]
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.train_data_dir = train_data_dir
        self.crop_size = crop_size
        self.only_h_flip = only_h_flip

    def get_images(self, index):

        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_path = os.path.join(self.train_data_dir, 'train_hazy', haze_name)
        gt_path = os.path.join(self.train_data_dir, 'train_gt', gt_name)
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


class ValData(data.Dataset):
    def __init__(self, crop_size_H, crop_size_W, val_data_dir, flag=True):
        super().__init__()

        val_list_haze = '/home/jxy/projects_dir/O-HAZY/datasets/valid.txt'

        with open(val_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('hazy')[0] + 'GT' + i.split('hazy')[1] for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.flag = flag
        self.size_H = crop_size_H
        self.size_W = crop_size_W

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        haze_path = os.path.join(self.val_data_dir, 'val_hazy', haze_name)
        gt_path = os.path.join(self.val_data_dir, 'val_gt', gt_name)
        haze_img = read_img255(haze_path)
        gt_img = read_img255(gt_path)
        [haze_img, gt_img] = augment([haze_img, gt_img], size=self.size_H)
        haze_img = np.ascontiguousarray(haze_img).astype('uint8')
        gt_img = np.ascontiguousarray(gt_img).astype('uint8')
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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
    def __init__(self, crop_size_H, crop_size_W, val_data_dir, flag=True):
        super().__init__()

        val_list_haze = '/home/jxy/projects_dir/datasets/O-HAZY/test.txt'

        with open(val_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('hazy')[0] + 'GT' + i.split('hazy')[1] for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.flag = flag
        self.size_H = crop_size_H
        self.size_W = crop_size_W

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        haze_path = os.path.join(self.val_data_dir, 'test_hazy', haze_name)
        gt_path = os.path.join(self.val_data_dir, 'test_gt', gt_name)
        haze_img = read_img255(haze_path)
        gt_img = read_img255(gt_path)
        h, w, c = haze_img.shape
        # print(haze_img.shape)
        [haze_img, gt_img] = align([haze_img, gt_img], size_H=h-h%16, size_W=w-w%16)
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
