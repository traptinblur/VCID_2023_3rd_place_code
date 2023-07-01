# Kaggle Vesuvius Challenge - Ink Detection 3rd place solution

This repository contains Vesuvius Challenge - Ink Detection 3rd place solution training code.

More details about the solution please refer to [Kaggle forums of VCID](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417536)

## Hardware Requirements

One of:

* Nvidia RTX 3090 with 24G RAM
* Nvidia A100 with 40G RAM

When TRAIN on big resolution you may need one of:

* 4x Nvidia RTX 3090 with 24G RAM
* 4x Nvidia A100 with 40G RAM

## Base Libraries

- [pytorch image models](https://github.com/huggingface/pytorch-image-models)
- [segmentation_models pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [mmaction2](https://github.com/open-mmlab/mmaction2)

Environment configuration for reference can be found in [requirements.txt](https://github.com/traptinblur/VCID_2023_3rd_place_code/blob/main/requirements.txt) file.

## Data

The input data is required to have the following structure:

```
dataset
	└──train
        └── fragmentid
            ├── subvolume
            │   ├── 1.tif
            │   ├── 2.tif
            │   ├── 3.tif
            │   ├── ・・・
            │   └── 64.tif
            ├── mask.png
            ├── ir.png
            └── inklabels.png
```

The above 👆 structure is  directly decompressed from the kaggle official dataset.

Please run **fragment2_split.ipynb** before training. It will rename folder 2 to folder 6 then splitting fragment 2 to 3 parts.

After splitting, the file format is as follows：

```
dataset
	└──train
        └── 1
            ├── subvolume
            │   ├── 1.tif
            │   ├── 2.tif
            │   ├── 3.tif
            │   ├── ・・・
            │   └── 64.tif
            ├── mask.png
            ├── ir.png
            └── inklabels.png
        └── 2
        └── 3
        └── 4
        └── 5
        └── 6
```

## 3rd Place Settings

If you want reproduce our final solution, the settings need specified hyper-parameters provided in sheet below:

| model name         | misc config                       | slices-tr_slices | size-stride | bs   | epoch | lr/init_lr    | norm | mixup-switch2cutmix | ema    | fold1 score(ema_cv-cv/ema_lb-lb) | fold2 score(ema_cv-cv/ema_lb-lb) | fold3 score(ema_cv-cv/ema_lb-lb) | fold4 score(ema_cv-cv/ema_lb-lb) | fold5 score(ema_cv-cv/ema_lb-lb) |
| ------------------ | --------------------------------- | ---------------- | ----------- | ---- | ----- | ------------- | ---- | ------------------- | ------ | -------------------------------- | -------------------------------- | -------------------------------- | -------------------------------- | -------------------------------- |
| adaptive_silu      | 6 bce: 1 dice, best-lb, no cutout | 28-24            | 224-112     | 16   | 30    | 1.5e-4/7.5e-6 | TRUE | 0.6-0.84            | 0.997  | 0.6168-0.6404/?-0.71             |                                  |                                  |                                  |                                  |
| adaptive_silu      | bce only, no cutout               | 28-24            | 224-112     | 16   | 30    | 1.5e-4/1.5e-5 | TRUE | 0.6-0.84            | 0.9998 |                                  | 0.7018-0.6918/0.75(tta:0.76)     | 0.6979-0.6726/0.68               |                                  |                                  |
| adaptive_silu      | bce only                          | 28-24            | 224-112     | 16   | 30    | 1.5e-4/1.5e-5 | TRUE | 0.1-0.              | 0.997  |                                  |                                  |                                  | 0.7399-0.7418/0.72               |                                  |
| adaptive_silu      | 6 bce: 1 dice, no cutout          | 28/24            | 224-112     | 16   | 30    | 1.5e-4/1.5e-5 | TRUE | 0.6-0.84            | 0.9998 |                                  |                                  |                                  |                                  | 0.7404-0.7440/0.71(tta:0.74)     |
| adaptive_silu      | bce only, no cutout               | 28/24            | 384-128     | 16   | 30    | 1.5e-4/1.5e-5 | TRUE | 0.6-0.84            | 0.997  | 0.6106-0.6177/                   | 0.7110-0.7016/                   | 0.6852-0.7036/                   | 0.6833-0.7212/                   | 0.7303-0.7321/                   |
| adaptive_silu_r152 | bce only, no cutout               | 28-24            | 576-144     | 16   | 20    | 1.5e-4/1.5e-5 | TRUE | 0.6-0.84            | 0.999  |                                  | 0.7279-0.7159/                   | 0.6907-0.6605/                   |                                  | 0.7714-0.7476/                   |
| adaptive_silu_r152 | 1dice loss and 3bce loss          | 28-24            | 576-144     | 16   | 20    | 1.5e-4/1.5e-5 | TRUE | 0.4-0.5             | 0.9994 | 0.6392-0.6511/                   |                                  |                                  | 0.7670-0.7452/                   |                                  |

## Training

Download the pretrained weights:

```shell
sh download_pretrained_weights.sh
```

Change training hyper-parameters listed in **seg_main_finetune.py** .

Run training pipeline:

```shell
python seg_main_finetune.py
```

Run DDP training pipeline:

```shell
sh seg_main_finetune.sh
```

## Inference

Inference notebook can be found [here](https://www.kaggle.com/code/traptinblur/3rd-place-ensemble-576-8-384-6-224-8/notebook?scriptVersionId=135421024)

## License

This repository is released under the MIT license as found in the [LICENSE](https://github.com/traptinblur/VCID_2023_3rd_place_code/blob/main/LICENSE) file.
