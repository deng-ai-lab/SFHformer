# -*- coding: utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
from datasets.Motion_Blur_Dataloader import TestHIDE, TestGoPro_whole
from numpy import *
from pytorch_msssim import ssim
from models import *
from line_profiler import LineProfiler
from utils.utils import *
from pytorch_ssim import SSIM
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='sfhformer_motion_blur', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--exp', default='motion_blur', type=str, help='experiment setting')
args = parser.parse_args()

torch.manual_seed(8001)

def valid(val_loader_full, network, index1, index2):
    PSNR_full = AverageMeter()
    SSIM_full = AverageMeter()
    # torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader_full:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        # names = batch['filename']

        output = torch.zeros_like(source_img).cuda()
        print(source_img.shape)
        B, C, H, W = source_img.shape
        count = torch.zeros([1, 1, H, W]).cuda()
        step = 32
        size = 128
        for i in range(H // step):
            for j in range(W // step):
                patch = source_img[:, :, i * step:i * step + size, j * step:j * step + size]
                with torch.no_grad():
                    output[:, :, i * step:i * step + size, j * step:j * step + size] += network(patch).clamp_(0, 1)
                count[:, :, i * step:i * step + size, j * step:j * step + size] += 1
        output = output / count

        # for k in range(B):
        #     name = names[k]
        #     top = transforms.ToPILImage()
        #     pic = top(target_img[k].cpu())
        #     pic.save(index1 + name)
        #
        #     pic = top(output[k].cpu())
        #     pic.save(index2 + name)

        mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
        psnr_full = 10 * torch.log10(1 / mse_loss).mean()
        PSNR_full.update(psnr_full.item(), source_img.size(0))

        ssim_full = ssim(output, target_img, data_range=1, size_average=False).mean()
        SSIM_full.update(ssim_full.item(), source_img.size(0))


    print(PSNR_full.avg, SSIM_full.avg)


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    print(setting_filename)
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    device_index = [0, 1, 2, 3]
    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network, device_ids=device_index).cuda()
    network.load_state_dict(torch.load('/home/jxy/projects_dir/motion_blur/sfhformer_motion_blur')['state_dict'])


    test_gopro_dir = '/home/jxy/projects_dir/datasets/Motion_blur/GoPro'
    test_hide_dir = '/home/jxy/projects_dir/datasets/Motion_blur/HIDE'
    test_realblurj_dir = '/home/jxy/projects_dir/datasets/Motion_blur/RealBlur_J'
    test_realblurr_dir = '/home/jxy/projects_dir/datasets/Motion_blur/RealBlur_R'


    test_gopro = TestGoPro_whole(8, test_gopro_dir)
    TestG = DataLoader(test_gopro,
                              batch_size=360,
                              num_workers=args.num_workers,
                              pin_memory=True)
    test_hide = TestHIDE(8, test_hide_dir)
    TestH = DataLoader(test_hide,
                              batch_size=360,
                              num_workers=args.num_workers,
                              pin_memory=True)

    valid(TestG, network, './target_for_gopro/', './result_for_gopro/')
    #valid(TestH, network, './target_for_hide/', './result_for_hide/')



