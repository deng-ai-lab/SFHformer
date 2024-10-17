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
    is_h_flip = 0
    if random.randint(0, 1) == 1:
        is_h_flip = 1
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    rot_deg = 0
    if not only_h_flip:
        # bad data augmentations for outdoor
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs, Hs, Ws, is_h_flip, rot_deg

def align(imgs=[], size_H=448, size_W=608):
    H, W, _ = imgs[0].shape
    Hc = size_H
    Wc = size_W
    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]
    return imgs


def align_for_test(imgs=[], local_size=32):
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
        train_list_haze = '/home/jxy/projects_dir/datasets/LOLdataset/our485/low/LIST.TXT'

        with open(train_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] + '.png' for i in haze_names]
        self.haze_names = haze_names
        self.gt_names = haze_names
        self.train_data_dir = train_data_dir
        self.crop_size = crop_size
        self.only_h_flip = only_h_flip

    def get_images(self, index):

        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_path = os.path.join(self.train_data_dir, 'low', haze_name)
        gt_path = os.path.join(self.train_data_dir, 'high', gt_name)
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        haze_img = read_img255(haze_path)
        gt_img = read_img255(gt_path)
        [haze_img, gt_img], Hs, Hw, is_h_flip, rot_deg = augment([haze_img, gt_img], size=self.crop_size, edge_decay=0., only_h_flip=self.only_h_flip)
        haze_img = np.ascontiguousarray(haze_img).astype('uint8')
        gt_img = np.ascontiguousarray(gt_img).astype('uint8')
        transform_haze = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)
        location = torch.tensor([Hs, Hw])
        aug = torch.tensor([is_h_flip, rot_deg])

        return {'source': haze, 'target': gt, 'filename': haze_name, 'location': location, 'aug':aug}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)


class TrainData_for_FiveK(data.Dataset):
    def __init__(self, crop_size, train_data_dir, only_h_flip=False):
        super().__init__()
        train_list_haze = '/home/jxy/projects_dir/datasets/FiveK/train.txt'

        with open(train_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = haze_names
        self.haze_names = haze_names
        self.gt_names = haze_names
        self.train_data_dir = train_data_dir
        self.crop_size = crop_size
        self.only_h_flip = only_h_flip

    def get_images(self, index):

        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_path = os.path.join(self.train_data_dir, 'input', haze_name)
        gt_path = os.path.join(self.train_data_dir, 'target', gt_name)
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        haze_img = read_img255(haze_path)
        gt_img = read_img255(gt_path)
        [haze_img, gt_img], Hs, Hw, is_h_flip, rot_deg = augment([haze_img, gt_img], size=self.crop_size, edge_decay=0., only_h_flip=self.only_h_flip)
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


class TrainData_for_LOLv2Real(data.Dataset):
    def __init__(self, crop_size, train_data_dir, only_h_flip=False):
        super().__init__()
        train_list_haze = '/home/jxy/projects_dir/datasets/LOLv2/Real_captured/Train.txt'

        with open(train_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = haze_names
        self.haze_names = haze_names
        self.gt_names = haze_names
        self.train_data_dir = train_data_dir
        self.crop_size = crop_size
        self.only_h_flip = only_h_flip

    def get_images(self, index):

        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_path = os.path.join(self.train_data_dir, 'Low', haze_name)
        gt_path = os.path.join(self.train_data_dir, 'Normal', gt_name)
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        haze_img = read_img255(haze_path)
        gt_img = read_img255(gt_path)
        [haze_img, gt_img], Hs, Hw, is_h_flip, rot_deg = augment([haze_img, gt_img], size=self.crop_size, edge_decay=0., only_h_flip=self.only_h_flip)
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


class TrainData_for_LOLv2Synthetic(data.Dataset):
    def __init__(self, crop_size, train_data_dir, only_h_flip=False):
        super().__init__()
        train_list_haze = '/home/jxy/projects_dir/datasets/LOLv2/Synthetic/Train.txt'

        with open(train_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = haze_names
        self.haze_names = haze_names
        self.gt_names = haze_names
        self.train_data_dir = train_data_dir
        self.crop_size = crop_size
        self.only_h_flip = only_h_flip

    def get_images(self, index):

        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_path = os.path.join(self.train_data_dir, 'Low', haze_name)
        gt_path = os.path.join(self.train_data_dir, 'Normal', gt_name)
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        haze_img = read_img255(haze_path)
        gt_img = read_img255(gt_path)
        [haze_img, gt_img], Hs, Hw, is_h_flip, rot_deg = augment([haze_img, gt_img], size=self.crop_size, edge_decay=0., only_h_flip=self.only_h_flip)
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

        val_list_haze = '/home/jxy/projects_dir/datasets/LOLdataset/eval15/low/LIST.TXT'

        with open(val_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] + '.png' for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = haze_names
        self.val_data_dir = val_data_dir
        self.flag = flag
        self.local_size = local_size

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        haze_path = os.path.join(self.val_data_dir, 'low', haze_name)
        gt_path = os.path.join(self.val_data_dir, 'high', gt_name)
        haze_img = read_img255(haze_path)
        gt_img = read_img255(gt_path)
        [haze_img, gt_img] = align_for_test([haze_img, gt_img], local_size=self.local_size)
        haze_img = np.ascontiguousarray(haze_img).astype('uint8')
        gt_img = np.ascontiguousarray(gt_img).astype('uint8')
        transform_haze = Compose([ToTensor()])
        #transform_haze = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)



        return {'source': haze, 'target': gt, 'filename': haze_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)


class TestData_for_FiveK(data.Dataset):
    def __init__(self, local_size, val_data_dir, flag=True):
        super().__init__()

        val_list_haze = '/home/jxy/projects_dir/datasets/FiveK/test.txt'

        with open(val_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]

        self.haze_names = haze_names
        self.gt_names = haze_names
        self.val_data_dir = val_data_dir
        self.flag = flag
        self.local_size = local_size

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        haze_path = os.path.join(self.val_data_dir, 'input', haze_name)
        gt_path = os.path.join(self.val_data_dir, 'target', gt_name)
        haze_img = read_img255(haze_path)
        gt_img = read_img255(gt_path)
        [haze_img, gt_img] = align_for_test([haze_img, gt_img], local_size=self.local_size)
        haze_img = np.ascontiguousarray(haze_img).astype('uint8')
        gt_img = np.ascontiguousarray(gt_img).astype('uint8')
        transform_haze = Compose([ToTensor()])
        #transform_haze = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)



        return {'source': haze, 'target': gt, 'filename': haze_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)


class TestData_for_LOLv2Real(data.Dataset):
    def __init__(self, local_size, val_data_dir, flag=True):
        super().__init__()

        val_list_haze = '/home/jxy/projects_dir/datasets/LOLv2/Real_captured/Test.txt'

        with open(val_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]

        self.haze_names = haze_names
        self.gt_names = haze_names
        self.val_data_dir = val_data_dir
        self.flag = flag
        self.local_size = local_size

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        haze_path = os.path.join(self.val_data_dir, 'Low', haze_name)
        gt_path = os.path.join(self.val_data_dir, 'Normal', gt_name)
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


class TestData_for_LOLv2Synthetic(data.Dataset):
    def __init__(self, local_size, val_data_dir, flag=True):
        super().__init__()

        val_list_haze = '/home/jxy/projects_dir/datasets/LOLv2/Synthetic/Test.txt'

        with open(val_list_haze) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]

        self.haze_names = haze_names
        self.gt_names = haze_names
        self.val_data_dir = val_data_dir
        self.flag = flag
        self.local_size = local_size

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        haze_path = os.path.join(self.val_data_dir, 'Low', haze_name)
        gt_path = os.path.join(self.val_data_dir, 'Normal', gt_name)
        haze_img = read_img255(haze_path)
        gt_img = read_img255(gt_path)
        [haze_img, gt_img] = align_for_test([haze_img, gt_img], local_size=self.local_size)
        haze_img = np.ascontiguousarray(haze_img).astype('uint8')
        gt_img = np.ascontiguousarray(gt_img).astype('uint8')
        transform_haze = Compose([ToTensor()])
        #transform_haze = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)



        return {'source': haze, 'target': gt, 'filename': haze_name}

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)