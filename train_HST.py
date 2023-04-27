import argparse
import os

import numpy as np
import cv2

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils import data
from torch import distributed as dist
import torch.optim as optim

import srdata

from model import HST

import utils_logger
import logging
import util_calculate_psnr_ssim as util

from my_loss import CharbonnierLoss


def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description='Train HST')

    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to the checkpoints for pretrained model",
    )
    parser.add_argument(
        '--distributed',
        action='store_true'
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument('--save_every', type=int, default=5000, help='save weights')
    parser.add_argument('--eval_every', type=int, default=5000, help='test network')

    parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint', help='path to save checkpoints')
    parser.add_argument('--valid_path', type=str, default='./valid', help='path to save intermediate validation results')

    args = parser.parse_args()

    return args

def data_sampler(dataset, shuffle=True, distributed=True):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def get_bare_model(network):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(network, DistributedDataParallel):
        network = network.module
    return network

def update_E(model, model_E, decay=0.999):
    netG = get_bare_model(model)
    netG_params = dict(netG.named_parameters())
    netE_params = dict(model_E.named_parameters())
    for k in netG_params.keys():
        netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1-decay)


def main():
    args = parse_args()

    checkpoint_save_path = args.ckpt_path
    test_image_save_path = args.valid_path
    if not os.path.exists(checkpoint_save_path):
        os.makedirs(checkpoint_save_path, exist_ok=True)
    if not os.path.exists(test_image_save_path):
        os.makedirs(test_image_save_path, exist_ok=True)

    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(test_image_save_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        # synchronize()
    
    # set random seeds and log
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    model = HST(img_size=64).to('cuda')
    if args.use_ema:
        model_E = HST(img_size=64).to('cuda').eval()

    if args.resume is not None:
        logger.info("load model: ", args.resume)
        if not os.path.isfile(args.resume):
            raise ValueError

        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt)
        if args.use_ema:
            model_E.load_state_dict(ckpt)

        logger.info("model checkpoint load!")

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1.e-4)

    loss_fn = torch.nn.L1Loss()
    # loss_fn = CharbonnierLoss(eps=1.e-6)
    loss_fn = loss_fn.to('cuda')

    trainset = srdata.Data_Train(patch_size=64)
    testset = srdata.Data_Test()

    data_loader = data.DataLoader(
        trainset, 
        batch_size=args.batch_size,
        sampler=data_sampler(trainset, shuffle=True, distributed=args.distributed),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    data_loader_test = data.DataLoader(
        testset, 
        batch_size=1,
        sampler=data_sampler(testset, shuffle=False, distributed=False),
        num_workers=1,
        pin_memory=True
    )


    if args.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )

    total_epochs = 1000000
    epoch_start = 1

    lr_decay = [100000, 250000]

    current_step = 0

    for epoch in range(epoch_start, total_epochs+1):

        m = model.module if args.distributed else model

        model.train()
        if args.use_ema:
            model_E.eval()

        for batch, pik in enumerate(data_loader):
            current_step += 1

            if current_step in lr_decay:
                for param in optimizer.param_groups:
                    param['lr'] = (param['lr'] * 0.5) if param['lr'] > 1.e-6 else 1.e-6

            learning_rate = optimizer.param_groups[0]['lr']

            lr = pik['L']
            hr = pik['H']
            filename = pik['N']

            filename = filename[0].split('/')[-1]

            optimizer.zero_grad()
            lr = lr.to('cuda')
            hr = hr.to('cuda')
            sr = model(lr)
            loss = loss_fn(sr, hr)
            loss_print = loss.item()
            loss.backward()
            optimizer.step()
            
            if args.use_ema:
                update_E(model, model_E)

            if current_step % 50 == 0 and torch.cuda.current_device() == 0:
                logger.info('Epoch: {}\tStep: {}\t[{}/{}]\t{}LR: {}'.format(
                    epoch,
                    current_step,
                    (batch + 1) * args.batch_size,
                    int(len(trainset)/args.gpus),
                    loss_print,
                    learning_rate))
            
            if not current_step % args.save_every and torch.cuda.current_device() == 0:
                model_dict = m.state_dict()
                torch.save(
                    model_dict,
                    os.path.join(checkpoint_save_path, 'model_{}.pt'.format(current_step))
                )
                if args.use_ema:
                    torch.save(
                        model_E.state_dict(),
                        os.path.join(checkpoint_save_path, 'model_E_{}.pt'.format(current_step))
                    )
            # valid
            if not current_step % args.eval_every and torch.cuda.current_device() == 0:

                model_E.eval()

                p = 0
                s = 0
                count = 0
                for batch in data_loader_test:
                    count += 1
                    lr = batch['L']
                    hr = batch['H']
                    filename = batch['N']
                    lr = lr.to('cuda')
                    filename = filename[0]
                    with torch.no_grad():
                        sr = model_E(lr)
                    sr = sr.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
                    sr = sr * 255.
                    sr = np.clip(sr.round(), 0, 255).astype(np.uint8)
                    hr = hr.squeeze(0).numpy().transpose(1, 2, 0)
                    hr = hr * 255.
                    hr = np.clip(hr.round(), 0, 255).astype(np.uint8)
                    
                    psnr = util.calculate_psnr(sr, hr, crop_border=4)
                    ssim = util.calculate_ssim(sr, hr, crop_border=4)
                    p += psnr
                    s += ssim
                    logger.info('{}: {}, {}'.format(filename, psnr, ssim))

                    cv2.imwrite(os.path.join(test_image_save_path, filename+'.png'), sr)

                p /= count
                s /= count

                logger.info("Epoch: {}, Step: {}, psnr: {}. ssim: {}.".format(epoch, current_step, p, s))
                model.train()

        if current_step > 400000:
            break
        
    logger.info('Done')

if __name__ == '__main__':
    main()