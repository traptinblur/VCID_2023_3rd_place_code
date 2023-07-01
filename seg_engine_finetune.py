# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------


import math
import sys
import random
import numpy as np
import cv2
from typing import Iterable, Optional

import torch

import utils.misc as misc


# -----------------------------train, val-----------------------------
def normalization(x:torch.Tensor)->torch.Tensor:
    """input.shape=(batch,f1,f2,...)"""
    #[batch,f1,f2]->dim[1,2]
    dim=list(range(1,x.ndim))
    mean=x.mean(dim=dim,keepdim=True)
    std=x.std(dim=dim,keepdim=True)
    return (x-mean)/(std+1e-9)


def mixup(input, truth, clip=[0, 1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, truth, shuffled_labels, lam


def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def cutmix(input, target, alpha=1):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(input.size()[0]).cuda()

    target_a = target
    target_b = target[rand_index]
    yl, yh, xl, xh = rand_bbox(input.size(), lam)
    # 5 dim tensor
    input[:, :, :, yl:yh, xl:xh] = input[rand_index, :, :, yl:yh, xl:xh]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((yh - yl) * (xh - xl) / (input.size()[-1] * input.size()[-2]))
    return input, target_a, target_b, lam


# -----------------------------metrics-----------------------------
def dice_coef_torch(preds, targets, beta=0.5, smooth=1e-5):
    """https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """

    # flatten label and prediction tensors
    preds = preds.view(-1).float()
    targets = targets.view(-1).float()

    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice

def calc_fbeta(mask, mask_pred):
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0
    for th in [0.5]:#np.array(range(10, 50+1, 5)) / 100:
        
        dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)

        if dice > best_dice:
            best_dice = dice
            best_th = th
    
    return best_dice, best_th


def calc_cv(mask_gt, mask_pred):
    best_dice, best_th = calc_fbeta(mask_gt, mask_pred)

    return best_dice, best_th


def train_fn(train_loader, model, criterion, optimizer, scheduler, loss_scaler, epoch, device, cfg, model_ema=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = cfg.accum_iter

    optimizer.zero_grad()

    for step, (images, labels) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        images = images.to(device, non_blocking=True)

        if cfg.normalization:
            images = normalization(images)

        labels = labels.to(device, non_blocking=True)

        if step % accum_iter == 0:
            scheduler.step()

        do_mixup = False
        if random.random() < cfg.p_mixup:
            do_mixup = True
            images, labels, gt_masks_sfl, lam = (
                cutmix(images, labels) if random.random() < cfg.p_switch else \
                mixup(images, labels)
            )

        with torch.cuda.amp.autocast():
            y_preds = model(images)
            loss = criterion(y_preds, labels)
            if do_mixup:
                loss2 = criterion(y_preds, gt_masks_sfl)
                loss = loss * lam  + loss2 * (1 - lam)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=cfg.max_grad_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(step + 1) % accum_iter == 0)
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()

            if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def valid_fn(valid_loader, model, criterion, device, valid_mask_gt, cfg):
    """validation on GPU including calculating the metric
    """
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation:'

    # valid_mask_gt = torch.from_numpy(valid_mask_gt).to(device, non_blocking=True)
    mask_pred = torch.zeros(valid_mask_gt.shape).to(device, non_blocking=True)
    dummy_gt = torch.zeros(valid_mask_gt.shape).to(device, non_blocking=True)
    mask_count = torch.zeros(valid_mask_gt.shape).to(device, non_blocking=True)

    model.eval()

    for step, (images, labels, xyxys) in enumerate(metric_logger.log_every(valid_loader, 10, header)):
        images = images.to(device, non_blocking=True)

        if cfg.normalization:
            images = normalization(images)

        labels = labels.to(device, non_blocking=True)
        xyxys = xyxys.to(device, non_blocking=True)
        batch_size = labels.size(0)

        y_preds = model(images)
        loss = criterion(y_preds, labels)

        metric_logger.update(loss=loss.item())

        # make whole mask
        y_preds = torch.sigmoid(y_preds)
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            dummy_gt[y1:y2, x1:x2] += labels[i].squeeze(0)#valid_mask_gt[y1:y2, x1:x2]
            mask_count[y1:y2, x1:x2] += torch.ones((cfg.tile_size, cfg.tile_size), device=device)

    mask_pred /= (mask_count + 1e-7)
    dummy_gt /= (mask_count + 1e-7)
    dummy_gt = dummy_gt > 0

    # calc metric
    best_dice = dice_coef_torch(mask_pred>0.5, dummy_gt)
    metric_logger.update(score=best_dice.item())
    metric_logger.synchronize_between_processes()
    print("Averaged val stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
