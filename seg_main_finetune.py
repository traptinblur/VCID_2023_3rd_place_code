# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------


import argparse
import datetime
import json
import numpy as np
import os
import time
import random
import importlib
from functools import partial

import cv2

import torch
from torch.nn import SyncBatchNorm
import torch.backends.cudnn as cudnn

import misc
from misc import NativeScalerWithGradNormCount as NativeScaler

import sys
sys.path.append("configs")

import hybrid_unet2

import albumentations as A

from seg_engine_finetune import train_fn, valid_fn


# ----------------------------- helper -----------------------------
def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def get_args_parser():
    parser = argparse.ArgumentParser('3d seg fine-tuning for ves', add_help=False)

    parser.add_argument("--config", help="train config file path")

    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


# ----------------------------- dataset -----------------------------
def read_image_mask(fragment_id, cfg):

    images = []
    idcs = cfg.slice_idcs
    for i in idcs:
        image = cv2.imread(cfg.comp_dataset_path + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)
        pad0 = (cfg.tile_size - image.shape[0] % cfg.tile_size)
        pad1 = (cfg.tile_size - image.shape[1] % cfg.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)

    mask = cv2.imread(cfg.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask = mask.astype(np.float32)
    mask /= 255.0
    
    return images, mask


def get_train_valid_dataset(cfg):
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    for fragment_id in cfg.frags:

        print(f"Reading fold {fragment_id} images...")
        start_time = time.time()
        image, mask = read_image_mask(fragment_id, cfg)
        print(f"Done loading in {(time.time()-start_time):.2f}s")

        x1_list = list(range(0, image.shape[1]-cfg.tile_size+1, cfg.stride))  # in vcid-tile-png, tmyok use mask.shape(inklabel)
        y1_list = list(range(0, image.shape[0]-cfg.tile_size+1, cfg.stride))

        print(f"Cropping fold {fragment_id} images...")
        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + cfg.tile_size
                x2 = x1 + cfg.tile_size
                if cfg.clean and image[y1:y2, x1:x2].sum()==0:
                    continue

                if fragment_id == cfg.valid_id:
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])

                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    if cfg.positive_only and mask[y1:y2, x1:x2].sum() == 0:
                        continue
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])
        print(f"Done processing in {(time.time()-start_time):.2f}s")

    return train_images, train_masks, valid_images, valid_masks, np.stack(valid_xyxys)


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)

    return aug


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, cfg, labels=None, transform=None, xyxys=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.xyxys = xyxys

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']

        image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32)/255.
        label = torch.from_numpy(label).permute(2, 0, 1).to(torch.float32)
        if self.xyxys is not None:
            return image.unsqueeze(0), label, self.xyxys[idx]
        else:
            return image.unsqueeze(0), label


# -----------------------------scheduler-----------------------------
def get_scheduler(cfg, optimizer):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.max_lr,
        epochs=cfg.epochs, steps_per_epoch=cfg.num_batches//cfg.accum_iter+1,
        pct_start=cfg.warmup_epoch/cfg.epochs, anneal_strategy=cfg.anneal_strategy
    )
    return scheduler


# -----------------------------loss-----------------------------
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def bce_dice(input, target, loss_weights=[1,0]):
    loss1 = loss_weights[0] * torch.nn.BCEWithLogitsLoss()(input, target)
    loss2 = loss_weights[1] * dice_loss(input, target)
    return (loss1 + loss2) / sum(loss_weights)


# -----------------------------main-----------------------------
def main(args):
    misc.init_distributed_mode(args)

    Logger = init_logger(log_file=args.log_path)
    if args.log_path and misc.is_main_process():
        Logger.info('\n\n-------- exp_info -----------------')
        Logger.info(datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'))
        if args.verbose:
            Logger.info(f"-------- {args.verbose} --------")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    for args.valid_id in args.folds:
        if args.log_path and misc.is_main_process():
            Logger.info(f'\n\n-------- exp on fold{args.valid_id} --------')

        # -- build dataset --
        train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset(args)

        dataset_train = CustomDataset(
            train_images, args, labels=train_masks, transform=get_transforms(data='train', cfg=args))
        dataset_val = CustomDataset(
            valid_images, args, labels=valid_masks, xyxys=valid_xyxys, transform=None)
        
        if args.distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        args.num_batches = len(data_loader_train)  # steps_per_epoch for 1cycle shceduler.

        fragment_id = args.valid_id

        valid_mask_gt = cv2.imread(args.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
        valid_mask_gt = valid_mask_gt / 255
        pad0 = (args.tile_size - valid_mask_gt.shape[0] % args.tile_size)
        pad1 = (args.tile_size - valid_mask_gt.shape[1] % args.tile_size)
        valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)

        # -- build model --
        if args.model_name == "hybrid_silu":
            model = hybrid_unet2.__dict__[args.model_name]()
        elif args.model_name == "hybrid_silu_r152":
            model = hybrid_unet2.__dict__[args.model_name]()
        elif args.model_name == "adaptive_silu":
            model = hybrid_unet2.__dict__[args.model_name](
                num_slices=args.num_slices, tr_slices=args.tr_slices, tr_idx=args.tr_idx
            )
        elif args.model_name == "adaptive_silu_r152":
            model = hybrid_unet2.__dict__[args.model_name](
                num_slices=args.num_slices, tr_slices=args.tr_slices, tr_idx=args.tr_idx
            )
        else:
            model = hybrid_unet2.__dict__[args.model_name]()


        if args.eval:
            segmentor_ckpt = torch.load(args.segmentor_ckpt[str(args.valid_id)], map_location="cpu")["model"]
            model.load_state_dict(segmentor_ckpt)
        else:
            if args.model_name[-3:] == "152":
                backbone_ckpt = torch.load(
                    "ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-7d1dacde.pth",
                    map_location="cpu"
                )["state_dict"]
            else:
                backbone_ckpt = torch.load(
                    "ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-44395bae.pth",
                    map_location="cpu"
                )["state_dict"]

            if args.model_name.split("_")[0] == "adaptive":
                msg = model.load_state_dict(backbone_ckpt, strict=False)
            else:
                msg = model.load_pretrained_weights(backbone_ckpt)
                
            print(msg)

        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.e6

        print('number of params (M): %.2f' % (n_parameters))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        print("min lr: %.2e" % args.init_lr)
        print("max lr: %.2e" % args.max_lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)

        # --EMA and SWA init--
        model_ema = None
        if args.ema:
            from timm.utils import ModelEmaV2
            model_ema = ModelEmaV2(model, decay=args.model_ema_decay, device=None)
            ema_best_score = 0.

        if args.distributed:
            model = SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        
        # -- criterion --
        criterion = partial(bce_dice, loss_weights=args.loss_weights)

        # validation only
        if args.eval:
            test_stats = valid_fn(data_loader_val, model, criterion, device, valid_mask_gt, args)
            if args.log_path and misc.is_main_process():
                Logger.info(f"Score of the network on fold {args.valid_id} : {test_stats['score']:.4f}")
            exit(0)

        # -- lr scheduler --
        scheduler = get_scheduler(args, optimizer)

        loss_scaler = NativeScaler()

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        best_score = 0.
        for epoch in range(1, args.epochs+1):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_fn(data_loader_train, model, criterion,
                                   optimizer, scheduler, loss_scaler, epoch,
                                   device, args, model_ema)
            train_msg = (f"Train on fold {args.valid_id} - "
                         f"Epoch: {epoch}, tr_Loss: {train_stats['loss']:.4f}, "
                         f"lr: {train_stats['lr']*1e4:.4f}e-4")

            test_stats = valid_fn(data_loader_val, model, criterion, device, valid_mask_gt, args)
            test_msg = (f"Validation on fold {args.valid_id} - "
                         f"Epoch: {epoch}, val_Loss: {test_stats['loss']:.4f}, "
                         f"Score: {test_stats['score']:.4f}")

            if args.log_path and misc.is_main_process():
                Logger.info(train_msg)
                Logger.info(test_msg)

            score = test_stats['score']
            update_best = score > best_score

            if update_best:
                best_loss = test_stats['loss']
                best_score = score

                best_verbose = (
                    f'********************Epoch {epoch} - Save Best Score: {best_score:.4f} Model '
                    f'with corresponding val_Loss: {best_loss:.4f}********************'
                )
                if args.log_path and misc.is_main_process():
                    Logger.info(best_verbose)

                to_save = {
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch,
                    'score': best_score,
                }
                checkpoint_path = (args.model_dir + \
                                   f'{args.model_name}_fold{args.valid_id}_slices{args.in_chans}_resolution{args.size}_stride{args.stride}_best.pth')
                misc.save_on_master(to_save, checkpoint_path)

            if model_ema is not None:

                ema_test_stats = valid_fn(data_loader_val, model_ema.module, criterion, device, valid_mask_gt, args)
                ema_test_msg = (f"EMA Validation on fold {args.valid_id} - "
                                f"Epoch: {epoch}, ema_val_Loss: {ema_test_stats['loss']:.4f}, "
                                f"ema_Score: {ema_test_stats['score']:.4f}")

                if args.log_path and misc.is_main_process():
                    Logger.info(ema_test_msg)

                ema_score = ema_test_stats['score']
                ema_update_best = ema_score > ema_best_score

                if ema_update_best:
                    ema_best_loss = ema_test_stats['loss']
                    ema_best_score = ema_score

                    ema_best_verbose = (
                        f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Epoch {epoch} - Save Best EMA Score: {ema_best_score:.4f} Model '
                        f'with corresponding ema_val_Loss: {ema_best_loss:.4f}<<<<<<<<<<<<<<<<<<<<<<<<<<'
                    )
                    if args.log_path and misc.is_main_process():
                        Logger.info(ema_best_verbose)

                    ema_to_save = {
                        'model': model_ema.module.state_dict(),
                        'epoch': epoch,
                        'score': ema_best_score,
                    }
                    ema_checkpoint_path = (args.model_dir + \
                                    f'ema_{args.model_name}_fold{args.valid_id}_slices{args.in_chans}_resolution{args.size}_stride{args.stride}_best.pth')
                    misc.save_on_master(ema_to_save, ema_checkpoint_path)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    CFG = importlib.import_module(args.config).CFG
    for attr in dir(CFG):
        if not attr.startswith("__"):
            setattr(args, attr, getattr(CFG, attr))

    for dir in [args.model_dir, args.log_dir]:
        os.makedirs(dir, exist_ok=True)

    main(args)
