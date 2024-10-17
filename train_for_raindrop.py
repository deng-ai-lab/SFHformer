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
from datasets.Raindrop_Dataloader import TrainData, Test_a, Test_b
from numpy import *
from pytorch_msssim import ssim
from models import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='SFHformer_m', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--exp', default='raindrop', type=str, help='experiment setting')
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

        loss = loss_content + 0.1 * loss_fft
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

        psnr_full, sim = calculate_psnr_torch(target_img, output.clamp_(0, 1))
        PSNR_full.update(psnr_full.item(), source_img.size(0))

        ssim_full = sim
        SSIM_full.update(ssim_full.item(), source_img.size(0))

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
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: wrunsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-3)

    train_dir = '/home/jxy/projects_dir/datasets/Raindrop/train'
    test_a_dir = '/home/jxy/projects_dir/datasets/Raindrop/test_a'
    test_b_dir = '/home/jxy/projects_dir/datasets/Raindrop/test_b'
    train_dataset = TrainData(256, train_dir)
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    test_a = Test_a(test_a_dir)
    Testa = DataLoader(test_a,
                              batch_size=1,
                              num_workers=args.num_workers,
                              pin_memory=True)
    test_b = Test_b(test_b_dir)
    Testb = DataLoader(test_b,
                              batch_size=1,
                              num_workers=args.num_workers,
                              pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    # change test_str when you development new exp
    test_str = 'raindrop_sfhformer_m'

    if not os.path.exists(os.path.join(save_dir, args.model + test_str + '.pth')):
        print('==> Start training, current model name: ' + args.model)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model, test_str))

        for testset in ["Testa", "Testb"]:
            exec("best_{var_name}_psnr = 0".format(var_name=testset))
            exec("best_{var_name}_ssim = 0".format(var_name=testset))

        for epoch in tqdm(range(setting['epochs'] + 1)):
            train_loss = train(train_loader, network, criterion, optimizer)
            writer.add_scalar('train_loss', train_loss, epoch)
            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                for testset in ["Testa"]:
                    exec("avg_{var_name}_psnr, avg_{var_name}_ssim = valid({var_name}, network)".format(
                        var_name=testset))
                    exec("writer.add_scalar('valid_{var_name}_psnr', avg_{var_name}_psnr, epoch)".format(
                        var_name=testset))
                    exec("writer.add_scalar('valid_{var_name}_ssim', avg_{var_name}_ssim, epoch)".format(
                        var_name=testset))

                torch.save({'state_dict': network.state_dict()},
                           os.path.join(save_dir, args.model + test_str + '_newest' + '.pth'))

                for testset in ["Testa", "Testb"]:
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
