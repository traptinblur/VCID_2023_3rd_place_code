# Kaggle Vesuvius Challenge - Ink Detection 3rd place solution

This repository contains Vesuvius Challenge - Ink Detection 3rd place solution training code.

More details about the solution please refer to [Kaggle forums of VCID](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417536).

## Hardware Requirements

One of:

* Nvidia RTX 3090 with 24GB VRAM
* Nvidia A100 with 40GB VRAM

When TRAIN on big resolution you may need one of:

* 4x Nvidia RTX 3090 with 24GB VRAM
* 4x Nvidia A100 with 40GB VRAM

## Base Libraries

- [pytorch image models](https://github.com/huggingface/pytorch-image-models)
- [segmentation_models pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [mmaction2](https://github.com/open-mmlab/mmaction2)

Environment configuration for reference can be found in the [requirements.txt](https://github.com/traptinblur/VCID_2023_3rd_place_code/blob/main/requirements.txt) file.

## Data

Dataset of VCID2023 can be found [here](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/data).

The input data is required to have the following structure:

```
dataset
  └──train
       └── fragment_id
             ├── subvolume
             │     ├── 00.tif
             │     ├── 01.tif
             │     ├── 02.tif
             │     ├── ・・・
             │     └── 64.tif
             ├── mask.png
             ├── ir.png
             └── inklabels.png
```

The above 👆 structure is  directly decompressed from the kaggle official dataset.

**Please run [fragment2_split.ipynb](https://github.com/traptinblur/VCID_2023_3rd_place_code/blob/main/fragment2_split.ipynb) before training** if you want to reproduce the solution result. It will rename folder 2 to folder 6 then splitting fragment 2 to 3 parts.

After splitting, the file format is as follows：

```
dataset
  └──train
       ├── 1
       │   ├── subvolume
       │   │     ├── 00.tif
       │   │     ├── 01.tif
       │   │     ├── 02.tif
       │   │     ├── ・・・
       │   │     └── 64.tif
       │   ├── mask.png
       │   ├── ir.png
       │   └── inklabels.png
       ├── 2
       ├── 3
       ├── 4
       ├── 5
       └── 6
```
## Training

Download the pretrained weights:

```shell
sh download_pretrained_weights.sh
```

The pretrained weights should be in the same directory as the dataset folder:

```
dataset
  └──train
weight1.pth
weight2.pth
```

Change training hyper-parameters listed in the [base_config.py]() .

Run the training pipeline:

```shell
CUDA_VISIBLE_DEVICES=0 python seg_main_finetune.py --config base_config
```

Run the DDP training pipeline:

```shell
sh seg_main_finetune.sh
```

## 3rd Place Settings

If you want reproduce our final solution, the settings need specified hyper-parameters provided in sheet below:

|    model  name     | bce/dice | misc_config | slices/tr_slices | size/stride |  bs  | epoch |  lr/init_lr   | norm | mixup/switch2cutmix |  ema   | fold1_score(cv/lb) | fold2_score(cv/lb) | fold3_score(cv/lb) | fold4_score(cv/lb) | fold5_score(cv/lb) |
| :----------------: | :------: | :---------: | :--------------: | ----------- | :--: | :---: | :-----------: | :--: | :-----------------: | :----: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|   adaptive_silu    |   6/1    | no  cutout  |      28/24       | 224/112     |  16  |  30   | 1.5e-4/7.5e-6 | TRUE |      0.6/0.84       | 0.997  |    0.6404/0.71     |                    |                    |                    |                    |
|   adaptive_silu    |   1/0    | no  cutout  |      28/24       | 224/112     |  16  |  30   | 1.5e-4/1.5e-5 | TRUE |      0.6/0.84       | 0.9998 |                    |    0.7018/0.75     |    0.6979/0.68     |                    |                    |
|   adaptive_silu    |   1/0    |             |      28/24       | 224/112     |  16  |  30   | 1.5e-4/1.5e-5 | TRUE |       0.1/0.        | 0.997  |                    |                    |                    |    0.7418/0.72     |                    |
|   adaptive_silu    |   6/1    | no  cutout  |      28/24       | 224/112     |  16  |  30   | 1.5e-4/1.5e-5 | TRUE |      0.6/0.84       | 0.9998 |                    |                    |                    |                    |    0.7440/0.71     |
|   adaptive_silu    |   1/0    | no  cutout  |      28/24       | 384/128     |  16  |  30   | 1.5e-4/1.5e-5 | TRUE |      0.6/0.84       | 0.997  |       0.6177       |       0.7110       |       0.7036       |       0.7212       |       0.7321       |
| adaptive_silu_r152 |   1/0    | no  cutout  |      28/24       | 576/144     |  16  |  20   | 1.5e-4/1.5e-5 | TRUE |      0.6/0.84       | 0.999  |                    |       0.7279       |       0.6907       |                    |       0.7714       |
| adaptive_silu_r152 |   3/1    |  no cutout  |      28/24       | 576/144     |  16  |  20   | 1.5e-4/1.5e-5 | TRUE |       0.4/0.5       | 0.9994 |       0.6511       |                    |                    |       0.7670       |                    |

These settings are listed in the (config)[] folder.

**For reproducing the solution, you only need to run the code below:**

```shell
sh reproduce.sh
```

## Inference

You can find the final solution checkpoints and inference pipeline [here](https://www.kaggle.com/code/traptinblur/3rd-place-ensemble-576-8-384-6-224-8#model).

If you want to test the weights you trained, you just need to:

* copy&edit the notebook
* upload your own weights for models with input sizes of 224, 384, 576 and 640
* change the path value in the `ckpts` dictionary defined in `CFG` class for each model
* save and submit

## License

This repository is released under the MIT license as found in the [LICENSE](https://github.com/traptinblur/VCID_2023_3rd_place_code/blob/main/LICENSE) file.
