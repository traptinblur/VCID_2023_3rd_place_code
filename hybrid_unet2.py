# reference: https://github.com/selimsef/xview3_solution/blob/main/zoo/unet.py
# --------------------------------------------------------


import os

import timm
import torch.hub
from torch.nn import Dropout2d
from torch.utils import model_zoo

from resnet3d_csn import ResNet3dCSN

denoiser = [
    "convnext_small",
    "resnet50",
    "tf_efficientnet_b3",
    "tf_efficientnet_b4",
    "tf_efficientnet_b7",
    "resnext50d_32x4d",
]

encoder_params = {
    "r50ir": {
        "filters": [64, 256, 512, 1024, 2048],
        "decoder_filters": [40, 64, 128, 256],
        "last_upsample": 32,
    },
    "r152ir": {
        "filters": [64, 256, 512, 1024, 2048],
        "decoder_filters": [40, 64, 128, 256],
        "last_upsample": 32,
    },
    "convnext_small": {
        "decoder_filters": [40, 64, 128],
        "last_upsample": 32,
    }
}

default_decoder_filters = [48, 96, 176, 256]
default_last = 48

import torch
from torch import nn
import torch.nn.functional as F


class BasicConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, activation=nn.SiLU, bias=True):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                            bias=bias)
        self.use_act = activation is not None
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class Conv1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=None, bias=bias)


class Conv3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=None)


class ConvSilu1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=nn.SiLU)


class ConvSilu3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=nn.SiLU)


class BasicUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.SiLU, mode='nearest'):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * 1
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1)
        self.use_act = activation is not None
        self.mode = mode
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = F.upsample(x, scale_factor=2, mode=self.mode)
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()[self.first_layer_params_names[0] + '.weight'][:, :3, ...] = pretrained_dict[
                self.first_layer_params_names[0] + '.weight'].data
            skip_layers = self.first_layer_params_names
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               not any(k.startswith(s) for s in skip_layers)}
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_names(self):
        return ['conv1.conv']


class TimmUnet_ves(AbstractModel):
    """
    2D conv unets
    available timm's backbones are resnet series, efficienet series and convnext series
    e.g.:
        models = [
        "convnext_small",
        "resnet50",
        "tf_efficientnet_b3",
        "tf_efficientnet_b4",
        "tf_efficientnet_b7",
        "resnext50d_32x4d",
        ]
    available model name refers to: {
        "convnext": "https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py",
        "resnet": "https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py",
        "efficientnet": "https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py",
    }
    wrappers name with "@register_model" decorator is available for encoder param.
    """
    def __init__(self, encoder="convnext_small", in_chans=1, num_class=1, pretrained=True, channels_last=False,
                 **kwargs):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        scale_factor = 2
        if encoder[:8] == "convnext":
            scale_factor = 4
        backbone_arch = encoder
        self.channels_last = channels_last
        backbone = timm.create_model(backbone_arch, features_only=True, in_chans=in_chans, pretrained=pretrained,
                                     **kwargs)
        
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.decoder_filters = default_decoder_filters
        self.last_upsample_filters = default_last
        if encoder in encoder_params:
            self.decoder_filters = encoder_params[encoder].get('decoder_filters', self.filters[:-1])
            self.last_upsample_filters = encoder_params[encoder].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])

        self.seg_head = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, num_class, scale_factor)

        self.name = "u-{}".format(encoder)

        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone

    def forward(self, x):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        enc_results = self.encoder(x)
        bottlenecks = self.bottlenecks
        x = enc_results[-1]

        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        seg_mask = self.seg_head(x).contiguous(memory_format=torch.contiguous_format)

        return seg_mask

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])


class HybridUnet(AbstractModel):
    """
    CSN 3D conv encoder with light-weight 2D conv decoders
    pretrained weight for encoder: https://mmaction2.readthedocs.io/en/latest/model_zoo/recognition_models.html#csn
    """
    def __init__(self, encoder='r50ir', in_chans=1, num_class=1, channels_last=False, scale_factor=2):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        self.channels_last = channels_last

        backbone = ResNet3dCSN(
            pretrained2d=False,
            pretrained=None,
            depth=int(encoder[1:-2]),
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=False,
            zero_init_residual=False,
            in_channels=in_chans
        )
        
        self.filters = encoder_params[encoder]["filters"]

        if encoder in encoder_params:
            self.decoder_filters = encoder_params[encoder].get('decoder_filters', self.filters[:-1])
            self.last_upsample_filters = encoder_params[encoder].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])

        self.seg_head = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, num_class, scale_factor)


        self.name = "u-{}".format(encoder)

        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        self.backbone = backbone

    def forward(self, x):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        enc_results = self.backbone(x)

        # temporal avgpool
        enc_results = [torch.mean(f, dim=2) for f in enc_results]

        x = enc_results[-1]

        bottlenecks = self.bottlenecks

        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        seg_mask = self.seg_head(x).contiguous(memory_format=torch.contiguous_format)

        return seg_mask

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])
    
    def load_pretrained_weights(self, state_dict):
        # Convert 3 channel weights to single channel
        # ref - https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1

        for i, key in enumerate(state_dict.keys()):
            if i == 1:
                break
            first_key = key
            print(f"sum first layer '{first_key}' weight to 1 channel")

        conv1_weight = state_dict[first_key]
        state_dict[first_key] = conv1_weight.sum(dim=1, keepdim=True)
        msg = self.load_state_dict(state_dict, strict=False)
        return msg
    

class HybridUnet_seg_cls(AbstractModel):
    """
    It's the same with HybridUnet class 
    but with an additional classification branch for stronger supervision

    ref1: https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/337468
    ref2: https://github.com/nvnnghia/nfl3_1st/blob/main/cnn/models/model_csn1.py#L24
    """
    def __init__(self, encoder='r50ir', in_chans=1, num_class=1,
                 channels_last=False, scale_factor=2, pool_type="avg"):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        self.channels_last = channels_last

        backbone = ResNet3dCSN(
            pretrained2d=False,
            pretrained=None,
            depth=int(encoder[1:-2]),
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=False,
            zero_init_residual=False,
            in_channels=in_chans#in_channels=1
        )
        
        self.filters = encoder_params[encoder]["filters"]

        if encoder in encoder_params:
            self.decoder_filters = encoder_params[encoder].get('decoder_filters', self.filters[:-1])
            self.last_upsample_filters = encoder_params[encoder].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])

        self.seg_head = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, num_class, scale_factor)


        self.name = "u-{}".format(encoder)

        # classification branch
        self.cls_head = nn.Linear(2048+1024, out_features=1)
        if pool_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)

        self._initialize_weights()
        self.backbone = backbone

    def forward(self, x):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        enc_results = self.backbone(x)

        # class branch
        x_fast = self.avg_pool(enc_results[-2])
        x_slow = self.avg_pool(enc_results[-1])

        x_cls = torch.cat((x_slow, x_fast), dim=1)

        x_cls = self.dropout(x_cls)
        x_cls = x_cls.flatten(1)
        cls_pred = self.cls_head(x_cls)

        # temporal avgpool
        enc_results = [torch.mean(f, dim=2) for f in enc_results]

        x = enc_results[-1]

        bottlenecks = self.bottlenecks

        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        seg_mask = self.seg_head(x).contiguous(memory_format=torch.contiguous_format)

        return seg_mask, cls_pred

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])
    
    def load_pretrained_weights(self, state_dict):
        # Convert 3 channel weights to single channel
        # ref - https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1

        for i, key in enumerate(state_dict.keys()):
            if i == 1:
                break
            first_key = key
            print(f"sum first layer '{first_key}' weight to 1 channel")

        conv1_weight = state_dict[first_key]
        state_dict[first_key] = conv1_weight.sum(dim=1, keepdim=True)
        msg = self.load_state_dict(state_dict, strict=False)
        return msg


class HybridUnet_adpt_seg_cls(AbstractModel):
    def __init__(self, encoder='r50ir', in_chans=3, num_class=1,
                 channels_last=False, scale_factor=2, pool_type="avg",
                 num_slices=28, tr_slices=24, tr_idx=[0,2,4]):
        
        self.tr_s = tr_slices
        self.num_slices = num_slices
        self.tr_idx = tr_idx
        assert self.num_slices == self.tr_s + self.tr_idx[-1]

        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        self.channels_last = channels_last

        backbone = ResNet3dCSN(
            pretrained2d=False,
            pretrained=None,
            depth=int(encoder[1:-2]),
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=False,
            zero_init_residual=False,
            in_channels=in_chans#in_channels=1
        )
        
        self.filters = encoder_params[encoder]["filters"]

        if encoder in encoder_params:
            self.decoder_filters = encoder_params[encoder].get('decoder_filters', self.filters[:-1])
            self.last_upsample_filters = encoder_params[encoder].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])

        self.seg_head = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, num_class, scale_factor)


        self.name = "u-{}".format(encoder)

        # class branch
        self.cls_head = nn.Linear(2048+1024, out_features=1)
        if pool_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)

        self._initialize_weights()
        self.backbone = backbone

    def forward(self, x):
        # adaptive select 3 parts
        tr_v = [
            x[:, :, idx:idx+self.tr_s,...] for idx in self.tr_idx
        ]

        x = torch.cat(tr_v, dim=1)

        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        enc_results = self.backbone(x)

        # class branch
        x_fast = self.avg_pool(enc_results[-2])
        x_slow = self.avg_pool(enc_results[-1])

        x_cls = torch.cat((x_slow, x_fast), dim=1)

        x_cls = self.dropout(x_cls)
        x_cls = x_cls.flatten(1)
        cls_pred = self.cls_head(x_cls)

        # temporal avgpool
        enc_results = [torch.mean(f, dim=2) for f in enc_results]

        x = enc_results[-1]

        bottlenecks = self.bottlenecks

        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        seg_mask = self.seg_head(x).contiguous(memory_format=torch.contiguous_format)

        return seg_mask, cls_pred

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])
    
    def load_pretrained_weights(self, state_dict):
        # Convert 3 channel weights to single channel
        # ref - https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1

        for i, key in enumerate(state_dict.keys()):
            if i == 1:
                break
            first_key = key
            print(f"sum first layer '{first_key}' weight to 1 channel")

        conv1_weight = state_dict[first_key]
        state_dict[first_key] = conv1_weight.sum(dim=1, keepdim=True)
        msg = self.load_state_dict(state_dict, strict=False)
        return msg


class HybridUnet_adaptive(HybridUnet):
    """
    modified model for ves challenge
    """
    def __init__(self, num_slices=28, tr_slices=24, tr_idx=[0,2,4], **kwargs):  # 28,16,[0,6,12]; 20,16,[0,2,4]
        super(HybridUnet_adaptive, self).__init__(in_chans=3, **kwargs)

        self.tr_s = tr_slices
        self.num_slices = num_slices
        self.tr_idx = tr_idx
        assert self.num_slices == self.tr_s + self.tr_idx[-1]

    def forward(self, x):
        # copy slice to channel
        # B, C, T, H, W = x.shape

        tr_v = [
            x[:, :, idx:idx+self.tr_s,...] for idx in self.tr_idx
        ]

        x = torch.cat(tr_v, dim=1)

        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        enc_results = self.backbone(x)

        # temporal avgpool
        enc_results = [torch.mean(f, dim=2) for f in enc_results]

        x = enc_results[-1]

        bottlenecks = self.bottlenecks

        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        seg_mask = self.seg_head(x).contiguous(memory_format=torch.contiguous_format)

        return seg_mask
    

class HybridUnet_adaptive_denoiser(HybridUnet):
    """
    modified model for ves challenge
    """
    def __init__(self, num_slices=28, tr_slices=15, tr_idx=[0,7,13], **kwargs):
        super(HybridUnet_adaptive_denoiser, self).__init__(in_chans=3, **kwargs)
        # 0-24, 2-26, 4-28
        # pred cat slices[0:15], pred cat slices[7:22], pred cat slices[13:28]
        self.tr_s = tr_slices
        self.num_slices = num_slices
        self.tr_idx = tr_idx
        assert self.num_slices == self.tr_s + self.tr_idx[-1]

    def forward(self, x, seg1_out):
        # copy slice to channel
        # B, C, T, H, W = x.shape

        if len(seg1_out.shape) == 4:
            seg1_out = seg1_out.unsqueeze(1)

        tr_v = [
            torch.cat([seg1_out, x[:, :, idx:idx+self.tr_s, ...]], dim=2) for idx in self.tr_idx
        ]

        x = torch.cat(tr_v, dim=1)

        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        enc_results = self.backbone(x)

        # temporal avgpool
        enc_results = [torch.mean(f, dim=2) for f in enc_results]

        x = enc_results[-1]

        bottlenecks = self.bottlenecks

        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        seg_mask = self.seg_head(x).contiguous(memory_format=torch.contiguous_format)

        return seg_mask


class StackedUnet(nn.Module):
    """
    modified model for ves challenge
    """
    def __init__(self, adaptive_silu_cfg, timmunet_cfg):
        super().__init__()
        self.unet1 = HybridUnet_adaptive(**adaptive_silu_cfg)
        self.unet2 = TimmUnet_ves(**timmunet_cfg)

    def forward(self, x):
        x1 = self.unet1(x)
        # x2 = self.unet2(x1.sigmoid())
        x2 = self.unet2(x1)
        return x1, x2


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class UnetDecoderLastConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, scale_factor=2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, 1)
        )

    def forward(self, x):
        return self.layer(x)
    

# parameter specified wrappers
def timm_unet(**kwargs):
    model = TimmUnet_ves(**kwargs)
    return model

def hybrid_silu(**kwargs):
    model = HybridUnet(encoder="r50ir", channels_last=False, scale_factor=2, **kwargs)
    return model


def hybrid_silu_r152(**kwargs):
    model = HybridUnet(encoder="r152ir", channels_last=False, scale_factor=2, **kwargs)
    return model


def hybrid_silu_seg_cls(**kwargs):
    model = HybridUnet_seg_cls(encoder="r50ir", in_chans=1, num_class=1, channels_last=False, scale_factor=2, **kwargs)
    return model


def hybrid_silu_seg_cls_r152(**kwargs):
    model = HybridUnet_seg_cls(encoder="r152ir", in_chans=1, num_class=1, channels_last=False, scale_factor=2, **kwargs)
    return model

def adaptive_silu(**kwargs):
    model = HybridUnet_adaptive(**kwargs)
    return model


def adaptive_silu_r152(**kwargs):
    model = HybridUnet_adaptive(encoder="r152ir", **kwargs)
    return model


def adaptive_denosier(**kwargs):
    model = HybridUnet_adaptive_denoiser(**kwargs)
    return model


def adaptive_denosier_r152(**kwargs):
    model = HybridUnet_adaptive_denoiser(encoder="r152ir", **kwargs)
    return model


def stacked_unet(cfg):
    adaptive_silu_default = dict(
        encoder="r50ir", num_slices=28, tr_slices=24, tr_idx=[0,2,4]
    )

    timmunet_default = dict(
        encoder="convnext_small", in_chans=1, num_class=1, pretrained=True
    )
    
    for k, v in cfg["adaptive_silu_cfg"].items():
        adaptive_silu_default[k] = v

    for k, v in cfg["timmunet_cfg"].items():
        timmunet_default[k] = v

    model = StackedUnet(adaptive_silu_cfg=adaptive_silu_default, timmunet_cfg=timmunet_default)
    return model


def adaptive_seg_cls(**kwargs):
    model = HybridUnet_adpt_seg_cls(**kwargs)
    return model


def adaptive_seg_cls_r152(**kwargs):
    model = HybridUnet_adpt_seg_cls(encoder="r152ir", **kwargs)
    return model


# model checking functions
def test_silu():
    model = hybrid_silu()
    print(model)
    print("================================")

    x = torch.rand(1, 1, 24, 224, 224)
    print(x.size())
    print("================================")

    y = model(x)
    print(y.size())
    print("================================")

    backbone_ckpt = torch.load(
        "/home/wuyx/ves/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-44395bae.pth",
        map_location="cpu"
    )["state_dict"]
    msg = model.load_pretrained_weights(backbone_ckpt)
    print(msg)
    print("================================")

    total_params = sum(
        param.numel() for param in model.parameters()
    )/1e6
    print(f"total_params: {total_params}")


def test_silu_seg_cls():
    model = hybrid_silu_seg_cls()
    print(model)
    print("================================")

    x = torch.rand(1, 1, 24, 224, 224)
    print(x.size())
    print("================================")

    seg_pred, cls_pred = model(x)
    print(seg_pred.size())
    print(cls_pred.size())
    print("================================")

    backbone_ckpt = torch.load(
        "/home/wuyx/ves/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-44395bae.pth",
        map_location="cpu"
    )["state_dict"]
    msg = model.load_pretrained_weights(backbone_ckpt)
    print(msg)
    print("================================")

    total_params = sum(
        param.numel() for param in model.parameters()
    )/1e6
    print(f"total_params: {total_params}")


def test_adpt_seg_cls():
    model = adaptive_seg_cls(num_slices=28, tr_slices=24, tr_idx=[0,2,4])
    print(model)
    print("================================")

    x = torch.rand(1, 1, 28, 224, 224)
    print(x.size())
    print("================================")

    seg_pred, cls_pred = model(x)
    print(seg_pred.size())
    print(cls_pred.size())
    print("================================")

    backbone_ckpt = torch.load(
        "/home/wuyx/ves/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-44395bae.pth",
        map_location="cpu"
    )["state_dict"]
    msg = model.load_state_dict(backbone_ckpt, strict=False)
    print(msg)
    print("================================")

    total_params = sum(
        param.numel() for param in model.parameters()
    )/1e6
    print(f"total_params: {total_params}")


def test_stacked_unet():
    adaptive_silu_cfg = dict(
        encoder="r50ir", num_slices=28, tr_slices=24, tr_idx=[0,2,4]
    )
    timmunet_cfg = dict(
        encoder="convnext_small", in_chans=1, num_class=1, pretrained=True
    )
    model_cfg = dict(adaptive_silu_cfg=adaptive_silu_cfg, timmunet_cfg=timmunet_cfg)

    model = stacked_unet(model_cfg)
    print(model)
    print("================================")

    x = torch.rand(1, 1, 28, 224, 224)
    print(x.size())
    print("================================")

    mask1, mask2 = model(x)
    print(mask1.size())
    print(mask2.size())
    print("================================")

    backbone_ckpt = torch.load(
        "/home/wuyx/ves/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-44395bae.pth",
        map_location="cpu"
    )["state_dict"]
    msg = model.unet1.load_state_dict(backbone_ckpt, strict=False)
    print(msg)
    print("================================")

    total_params = sum(
        param.numel() for param in model.parameters()
    )/1e6
    unet1_params = sum(
        param.numel() for param in model.unet1.parameters()
    )/1e6
    unet2_params = sum(
        param.numel() for param in model.unet2.parameters()
    )/1e6
    print(f"total_params: {total_params}")
    print(f"unet1_params: {unet1_params}")
    print(f"unet2_params: {unet2_params}")


def test_silu_adaptive():
    model = adaptive_silu(num_slices=28, tr_slices=24, tr_idx=[0,2,4])
    print(model)
    print("================================")

    x = torch.rand(1, 1, 28, 224, 224)
    print(x.size())
    print("================================")

    y = model(x)
    print(y.size())
    print("================================")

    backbone_ckpt = torch.load(
        "/home/wuyx/ves/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-44395bae.pth",
        map_location="cpu"
    )
    msg = model.load_state_dict(backbone_ckpt, strict=False)
    print(msg)
    print("================================")

    total_params = sum(
        param.numel() for param in model.parameters()
    )/1e6
    print(f"total_params: {total_params}")


def test_152_silu_adaptive():
    model = adaptive_silu(encoder="r152ir", num_slices=28, tr_slices=24, tr_idx=[0,2,4])
    print(model)
    print("================================")

    x = torch.rand(1, 1, 28, 224, 224)
    print(x.size())
    print("================================")

    y = model(x)
    print(y.size())
    print("================================")

    backbone_ckpt = torch.load(
        "/home/wuyx/ves/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-7d1dacde.pth",
        map_location="cpu"
    )["state_dict"]
    msg = model.load_state_dict(backbone_ckpt, strict=False)
    print(msg)
    print("================================")

    total_params = sum(
        param.numel() for param in model.parameters()
    )/1e6
    print(f"total_params: {total_params}")


def test_adaptive_denoiser():
    model = adaptive_denosier(num_slices=28, tr_slices=15, tr_idx=[0,7,13])
    print(model)
    print("================================")

    x = torch.rand(1, 1, 28, 224, 224)
    seg1_out = torch.rand(1, 1, 224, 224)
    print(x.size())
    print(seg1_out.size())
    print("================================")

    y = model(x, seg1_out)
    print(y.size())
    print("================================")

    backbone_ckpt = torch.load(
        "/home/wuyx/ves/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-44395bae.pth",
        map_location="cpu"
    )["state_dict"]
    msg = model.load_state_dict(backbone_ckpt, strict=False)
    print(msg)
    print("================================")

    total_params = sum(
        param.numel() for param in model.parameters()
    )/1e6
    print(f"total_params: {total_params}")


if __name__ == "__main__":
    test_silu_seg_cls()
    # test_silu()
    # test_adpt_seg_cls()
    # test_adaptive_denoiser()
    # test_silu_adaptive()
    # test_152_silu_adaptive()
    