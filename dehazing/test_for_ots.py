import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
from datasets.ITS_Dataloader import TrainData, ValData, TestData_ots
from numpy import *
from pytorch_msssim import ssim
from models import *
from einops import rearrange
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='sfhformer_haze', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()

torch.manual_seed(8001)


def valid(val_loader448, network):
    PSNR448 = AverageMeter()
    SSIM_full = AverageMeter()
    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader448:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        _, _, H, W = target_img.shape

        with torch.no_grad():  # torch.no_grad() may cause warning
            output = network(source_img).clamp_(0, 1)  # we change this to [0,1]?

        mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
        psnr448 = 10 * torch.log10(1 / mse_loss).mean()
        PSNR448.update(psnr448.item(), source_img.size(0))

        down_ratio = max(1, round(min(H, W) / 256))
        ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                        F.adaptive_avg_pool2d(target_img, (int(H / down_ratio), int(W / down_ratio))),
                        data_range=1, size_average=False).mean()
        SSIM_full.update(ssim_val.item(), source_img.size(0))
        print(PSNR448.avg, SSIM_full.avg)

    return PSNR448.avg, SSIM_full.avg


if __name__ == '__main__':
    device_index = [0]
    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network, device_ids=device_index).cuda()
    network.load_state_dict(torch.load(
        '/home/jxy/projects_dir/dehazing/sfhformer_haze_ots.pth')[
                                'state_dict'])

    test_data_dir = '/home/jxy/projects_dir/datasets/SOTS/outdoor/'

    test_dataset = TestData_ots(8, test_data_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True)

    avg_psnr, avg_ssim = valid(test_loader, network)
    print(avg_psnr, avg_ssim)

