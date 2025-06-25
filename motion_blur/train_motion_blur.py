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
from datasets.Motion_Blur_Dataloader import TrainData, TestGoPro, TestHIDE, TestRealBlurR, TestRealBlurJ
from numpy import *
from pytorch_msssim import ssim
from models import *
from line_profiler import LineProfiler
from utils.utils import *
from pytorch_ssim import SSIM


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='sfhformer_motion_blur', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--exp', default='motion_blur', type=str, help='experiment setting')
args = parser.parse_args()

torch.manual_seed(8001)


def train(train_loader, network, criterion, optimizer):
    losses = AverageMeter()

    # torch.cuda.empty_cache()

    network.train()
    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        pred_img = network(source_img)
        label_img = target_img
        l3 = criterion(pred_img, label_img)
        loss_content = l3

        label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

        pred_fft3 = torch.fft.fft2(pred_img, dim=(-2, -1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

        f3 = criterion(pred_fft3, label_fft3)
        loss_fft = f3


        loss = loss_content + 0.5 * loss_fft
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(network.parameters(), 0.01)
        optimizer.step()

    return losses.avg



def valid(val_loader_full, network):
    PSNR_full = AverageMeter()
    SSIM_full = AverageMeter()

    # torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader_full:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():  # torch.no_grad() may cause warning
            output = network(source_img).clamp_(0, 1)  # we change this to [0,1]?

        mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
        psnr_full = 10 * torch.log10(1 / mse_loss).mean()
        PSNR_full.update(psnr_full.item(), source_img.size(0))

        ssim_full = ssim(output, target_img, data_range=1, size_average=False).mean()
        SSIM_full.update(ssim_full.item(), source_img.size(0))

    print(PSNR_full.avg, SSIM_full.avg)

    return PSNR_full.avg, SSIM_full.avg


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

    criterion = nn.L1Loss()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'], eps=1e-8)
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'], weight_decay=1e-3)
    else:
        raise Exception("ERROR: wrunsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=1e-7)

    train_dir = '/home/jxy/projects_dir/datasets/Motion_blur/train_GoPro_crops'
    test_gopro_dir = '/home/jxy/projects_dir/datasets/Motion_blur/test_GoPro_crops'
    test_hide_dir = '/home/jxy/projects_dir/datasets/Motion_blur/HIDE'
    test_realblurj_dir = '/home/jxy/projects_dir/datasets/Motion_blur/RealBlur_J'
    test_realblurr_dir = '/home/jxy/projects_dir/datasets/Motion_blur/RealBlur_R'
    train_dataset = TrainData(128, train_dir)
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              sampler=RandomSampler(train_dataset, num_samples=setting['batch_size'] * 800,
                                                    replacement=True),
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=True)
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=setting['batch_size'],
    #                           shuffle=True,
    #                           num_workers=args.num_workers,
    #                           pin_memory=True,
    #                           drop_last=True)
    test_gopro = TestGoPro(test_gopro_dir)
    TestG = DataLoader(test_gopro,
                              batch_size=240,
                              num_workers=args.num_workers,
                              pin_memory=True)
    test_hide = TestHIDE(8, test_hide_dir)
    TestH = DataLoader(test_hide,
                              batch_size=30,
                              num_workers=args.num_workers,
                              pin_memory=True)
    test_realblurj = TestRealBlurJ(test_realblurj_dir)
    TestJ = DataLoader(test_realblurj,
                              batch_size=45,
                              num_workers=args.num_workers,
                              pin_memory=True)
    test_realblurr = TestRealBlurR(test_realblurr_dir)
    TestR = DataLoader(test_realblurr,
                              batch_size=45,
                              num_workers=args.num_workers,
                              pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    # change test_str when you development new exp
    test_str = 'sfhformer_motion_blur'

    if not os.path.exists(os.path.join(save_dir, args.model + test_str + '.pth')):
        print('==> Start training, current model name: ' + args.model)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model, test_str))

        for testset in ["TestG", "TestH", "TestJ", "TestR"]:
            exec("best_{var_name}_psnr = 0".format(var_name=testset))
            exec("best_{var_name}_ssim = 0".format(var_name=testset))

        for epoch in tqdm(range(setting['epochs'] + 1)):
            train_loss = train(train_loader, network, criterion, optimizer)

            writer.add_scalar('train_loss', train_loss, epoch)

            scheduler.step()  # TODO

            if epoch % setting['eval_freq'] == 0:

                for testset in ["TestG"]:
                    exec("avg_{var_name}_psnr, avg_{var_name}_ssim = valid({var_name}, network)".format(
                        var_name=testset))
                    exec("writer.add_scalar('valid_{var_name}_psnr', avg_{var_name}_psnr, epoch)".format(
                        var_name=testset))
                    exec("writer.add_scalar('valid_{var_name}_ssim', avg_{var_name}_ssim, epoch)".format(
                        var_name=testset))

                torch.save({'state_dict': network.state_dict()},
                           os.path.join(save_dir, args.model + test_str + '_newest' + '.pth'))

                for testset in ["TestG"]:
                    writer.add_scalar('best_' + testset + '_psnr', eval("best_{}_psnr".format(testset)), epoch)
                    if eval("avg_{}_psnr".format(testset)) > eval("best_{}_psnr".format(testset)):
                        exec("best_{var_name}_psnr = avg_{var_name}_psnr".format(var_name=testset))
                        torch.save({'state_dict': network.state_dict()},
                                   os.path.join(save_dir, args.model + test_str + '_' + testset + '_best' + '.pth'))

                    writer.add_scalar('best_' + testset + '_ssim', eval("best_{}_ssim".format(testset)), epoch)
                    if eval("avg_{}_ssim".format(testset)) > eval("best_{}_ssim".format(testset)):
                        exec("best_{var_name}_ssim = avg_{var_name}_ssim".format(
                            var_name=testset))

            if epoch % 10 == -1:

                # avg_psnr, avg_ssim = valid(test_loader, network)
                for testset in ["TestH", "TestJ", "TestR"]:
                    exec("avg_{var_name}_psnr, avg_{var_name}_ssim = valid({var_name}, network)".format(
                        var_name=testset))
                    exec("writer.add_scalar('valid_{var_name}_psnr', avg_{var_name}_psnr, epoch)".format(
                        var_name=testset))
                    exec("writer.add_scalar('valid_{var_name}_ssim', avg_{var_name}_ssim, epoch)".format(
                        var_name=testset))

                for testset in ["TestH", "TestJ", "TestR"]:
                    writer.add_scalar('best_' + testset + '_psnr', eval("best_{}_psnr".format(testset)), epoch)
                    if eval("avg_{}_psnr".format(testset)) > eval("best_{}_psnr".format(testset)):
                        exec("best_{var_name}_psnr = avg_{var_name}_psnr".format(var_name=testset))
                        torch.save({'state_dict': network.state_dict()},
                                   os.path.join(save_dir, args.model + test_str + '_' + testset + '_best' + '.pth'))

                    writer.add_scalar('best_' + testset + '_ssim', eval("best_{}_ssim".format(testset)), epoch)
                    if eval("avg_{}_ssim".format(testset)) > eval("best_{}_ssim".format(testset)):
                        exec("best_{var_name}_ssim = avg_{var_name}_ssim".format(
                            var_name=testset))
    else:
        print('==> Existing trained model')
        exit(1)
