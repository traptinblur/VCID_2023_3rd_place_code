import albumentations as A

# -----------------------------config-----------------------------
class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    comp_folder_name = 'vesuvius-challenge-ink-detection'
    comp_dataset_path = f'dataset/'

    device = "cuda"
    # ============== model cfg =============
    model_name = "adaptive_silu"

    ema = True
    model_ema_decay = 0.997
    positive_only = False
    clean = True
    normalization = True
    loss_weights = [1, 0]
    p_mixup = 0.6
    p_switch = 0.84  # switch mixup to cutmix
    rotate_limit = 180
    p_shiftscalerotate = 0.75

    in_chans = 28
    mid = 65 // 2
    start = mid - in_chans // 2
    end = mid + in_chans // 2
    slice_idcs = range(start, end)
    assert in_chans == len(slice_idcs)
    num_slices = in_chans
    tr_idx = [0, 2, 4]
    tr_slices = num_slices - tr_idx[-1]

    # ============== training cfg =============
    size = 384  # crop size
    tile_size = 384  # same as size
    stride = tile_size // 3

    batch_size = 4
    accum_iter = 4  # 1, maintain bs to 16
    use_amp = True
    epochs = 30

    max_lr = 1.5e-4
    init_lr = 1.5e-5
    warmup_epoch = 1  # can be float
    anneal_strategy = "cos"  # "linear"


    # ============== fold =============
    folds = [1, 2, 3, 4, 5]  # specify which fold needs experiment on
    frags = [1, 2, 3, 4, 5]  # specify which fold needs loaded

    # ============== fixed =============
    weight_decay = 1e-6
    max_grad_norm = 1000

    num_workers = 4
    pin_mem = True

    seed = 210

    # ============== set dataset path =============
    exp_name = f'{model_name}_size{size}'

    outputs_path = f'./outputs/{exp_name}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + 'models/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== augmentation =============
    train_aug_list = [
        A.HorizontalFlip(p=0.75),
        A.VerticalFlip(p=0.75),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=rotate_limit, p=p_shiftscalerotate),#p=0.75
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    ]

    verbose = (f"\n{len(slice_idcs)} slices {slice_idcs}, tr slices {tr_slices} with idices {tr_idx}\n"
               f"EMA state {ema}, with EMA decay {model_ema_decay}\n"
               f"mixup prob: {p_mixup}, switch to cutmix prob: {p_switch}\n"
               f"clean: {clean}, normalization: {normalization}\n"
               f"changing resolution {size} with spatio stride {stride}\n"
               f"model {model_name}\n"
               f"bs {batch_size}, acctual bs {batch_size*accum_iter}, epochs {epochs}\n"
               f"max_lr {max_lr}({init_lr}), decay strategy: {anneal_strategy}, warm up epoch: {warmup_epoch}\n"
               f"add {loss_weights[1]}dice loss and {loss_weights[0]}bce loss\n"
               f"rot limit {rotate_limit}, shiftscalerotate p = {p_shiftscalerotate}\n"
               f"no cutout"
               )