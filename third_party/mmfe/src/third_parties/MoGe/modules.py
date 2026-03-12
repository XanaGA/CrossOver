from typing import *
from numbers import Number
import importlib
import itertools
import functools
import sys

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from third_parties.MoGe.utils import wrap_module_with_gradient_checkpointing


class ResidualConvBlock(nn.Module):  
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int = None, 
        hidden_channels: int = None, 
        kernel_size: int = 3, 
        padding_mode: str = 'replicate', 
        activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu', 
        in_norm: Literal['group_norm', 'layer_norm', 'instance_norm', 'none'] = 'layer_norm',
        hidden_norm: Literal['group_norm', 'layer_norm', 'instance_norm'] = 'group_norm',
    ):  
        super(ResidualConvBlock, self).__init__()  
        if out_channels is None:  
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation =='relu':
            activation_cls = nn.ReLU
        elif activation == 'leaky_relu':
            activation_cls = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        elif activation =='silu':
            activation_cls = nn.SiLU
        elif activation == 'elu':
            activation_cls = nn.ELU
        else:
            raise ValueError(f'Unsupported activation function: {activation}')

        self.layers = nn.Sequential(
            nn.GroupNorm(in_channels // 32, in_channels) if in_norm == 'group_norm' else \
                nn.GroupNorm(1, in_channels) if in_norm == 'layer_norm' else \
                nn.InstanceNorm2d(in_channels) if in_norm == 'instance_norm' else \
                nn.Identity(),
            activation_cls(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode),
            nn.GroupNorm(hidden_channels // 32, hidden_channels) if hidden_norm == 'group_norm' else \
                nn.GroupNorm(1, hidden_channels) if hidden_norm == 'layer_norm' else \
                nn.InstanceNorm2d(hidden_channels) if hidden_norm == 'instance_norm' else\
                nn.Identity(),
            activation_cls(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode)
        )
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()  
  
    def forward(self, x):  
        skip = self.skip_connection(x)  
        x = self.layers(x)
        x = x + skip
        return x  

class Resampler(nn.Sequential):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        type_: Literal['pixel_shuffle', 'nearest', 'bilinear', 'conv_transpose', 'pixel_unshuffle', 'avg_pool', 'max_pool'],
        scale_factor: int = 2, 
    ):
        if type_ == 'pixel_shuffle':
            nn.Sequential.__init__(self,
                nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            )
            for i in range(1, scale_factor ** 2):
                self[0].weight.data[i::scale_factor ** 2] = self[0].weight.data[0::scale_factor ** 2]
                self[0].bias.data[i::scale_factor ** 2] = self[0].bias.data[0::scale_factor ** 2]
        elif type_ in ['nearest', 'bilinear']:
            nn.Sequential.__init__(self,
                nn.Upsample(scale_factor=scale_factor, mode=type_, align_corners=False if type_ == 'bilinear' else None),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            )
        elif type_ == 'conv_transpose':
            nn.Sequential.__init__(self,
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            )
            self[0].weight.data[:] = self[0].weight.data[:, :, :1, :1]
        elif type_ == 'pixel_unshuffle':
            nn.Sequential.__init__(self,
                nn.PixelUnshuffle(scale_factor),
                nn.Conv2d(in_channels * (scale_factor ** 2), out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
            )
        elif type_ == 'avg_pool': 
            nn.Sequential.__init__(self,
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        elif type_ == 'max_pool':
            nn.Sequential.__init__(self,
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
                nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        else:
            raise ValueError(f'Unsupported resampler type: {type_}')

class MLP(nn.Sequential):
    def __init__(self, dims: Sequence[int]):
        nn.Sequential.__init__(self,
            *itertools.chain(*[
                (nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True))
                    for dim_in, dim_out in zip(dims[:-2], dims[1:-1])
            ]),
            nn.Linear(dims[-2], dims[-1]),
        )

class ConvStack(nn.Module):
    def __init__(self, 
        dim_in: List[Optional[int]],
        dim_res_blocks: List[int],
        dim_out: List[Optional[int]],
        resamplers: Union[Literal['pixel_shuffle', 'nearest', 'bilinear', 'conv_transpose', 'pixel_unshuffle', 'avg_pool', 'max_pool'], List],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_in_norm: Literal['layer_norm', 'group_norm' , 'instance_norm', 'none'] = 'layer_norm',
        res_block_hidden_norm: Literal['layer_norm', 'group_norm' , 'instance_norm', 'none'] = 'group_norm',
        activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu',
    ):
        super().__init__()
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(dim_in_, dim_res_block_, kernel_size=1, stride=1, padding=0) if dim_in_ is not None else nn.Identity() 
                for dim_in_, dim_res_block_ in zip(dim_in if isinstance(dim_in, Sequence) else itertools.repeat(dim_in), dim_res_blocks)
        ])
        self.resamplers = nn.ModuleList([
            Resampler(dim_prev, dim_succ, scale_factor=2, type_=resampler) 
                for i, (dim_prev, dim_succ, resampler) in enumerate(zip(
                    dim_res_blocks[:-1], 
                    dim_res_blocks[1:], 
                    resamplers if isinstance(resamplers, Sequence) else itertools.repeat(resamplers)
                ))
        ])
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                *(
                    ResidualConvBlock(
                        dim_res_block_, dim_res_block_, dim_times_res_block_hidden * dim_res_block_, 
                        activation=activation, in_norm=res_block_in_norm, hidden_norm=res_block_hidden_norm
                    ) for _ in range(num_res_blocks[i] if isinstance(num_res_blocks, list) else num_res_blocks)
                )
            ) for i, dim_res_block_ in enumerate(dim_res_blocks)
        ])
        self.output_blocks = nn.ModuleList([
            nn.Conv2d(dim_res_block_, dim_out_, kernel_size=1, stride=1, padding=0) if dim_out_ is not None else nn.Identity() 
                for dim_out_, dim_res_block_ in zip(dim_out if isinstance(dim_out, Sequence) else itertools.repeat(dim_out), dim_res_blocks)
        ])

    def enable_gradient_checkpointing(self):
        for i in range(len(self.resamplers)):
            self.resamplers[i] = wrap_module_with_gradient_checkpointing(self.resamplers[i])
        for i in range(len(self.res_blocks)):
            for j in range(len(self.res_blocks[i])):
                self.res_blocks[i][j] = wrap_module_with_gradient_checkpointing(self.res_blocks[i][j])

    def forward(self, in_features: List[torch.Tensor]):
        out_features = []
        for i in range(len(self.res_blocks)):
            feature = self.input_blocks[i](in_features[i])
            if i == 0:
                x = feature
            elif feature is not None:
                x = x + feature
            x = self.res_blocks[i](x)
            out_features.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return out_features


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv

class MoGe_1_Head(nn.Module):
    def __init__(
        self, 
        num_features: int,
        dim_in: int, 
        dim_out: List[int], 
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        projection_spatial: List[int] = None
    ):
        super().__init__()
        self.projection_spatial = projection_spatial
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_proj, kernel_size=1, stride=1, padding=0,) for _ in range(num_features)
        ])

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch + 2, out_ch),
                *(ResidualConvBlock(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", in_norm=res_block_norm) for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        ])

        self.output_block = nn.ModuleList([
            self._make_output_block(
                dim_upsample[-1] + 2, dim_out_, dim_times_res_block_hidden, last_res_blocks, last_conv_channels, last_conv_size, res_block_norm,
            ) for dim_out_ in dim_out
        ])
    
    def _make_upsampler(self, in_channels: int, out_channels: int):
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(self, dim_in: int, dim_out: int, dim_times_res_block_hidden: int, last_res_blocks: int, last_conv_channels: int, last_conv_size: int, res_block_norm: Literal['group_norm', 'layer_norm']):
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            *(ResidualConvBlock(last_conv_channels, last_conv_channels, dim_times_res_block_hidden * last_conv_channels, activation='relu', norm=res_block_norm) for _ in range(last_res_blocks)),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_conv_channels, dim_out, kernel_size=last_conv_size, stride=1, padding=last_conv_size // 2, padding_mode='replicate'),
        )
            
    def forward(self, hidden_states: torch.Tensor, image: torch.Tensor):
        img_h, img_w = self.projection_spatial if self.projection_spatial is not None else (32, 32)

        x = torch.stack([
            proj(
                feat.permute(0, 2, 1)
                    .unflatten(2, (int(feat.shape[1] ** 0.5), int(feat.shape[1] ** 0.5)))
                    .contiguous()
            )
            for proj, (feat, clstoken) in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)
        
        # Upsample stage
        # (patch_h, patch_w) -> (patch_h * 2, patch_w * 2) -> (patch_h * 4, patch_w * 4) -> (patch_h * 8, patch_w * 8)
        for i, block in enumerate(self.upsample_blocks):
            # UV coordinates is for awareness of image aspect ratio
            uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        
        # (patch_h * 8, patch_w * 8) -> (img_h, img_w)
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)

        if isinstance(self.output_block, nn.ModuleList):
            output = [torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) for block in self.output_block]
        else:
            output = torch.utils.checkpoint.checkpoint(self.output_block, x, use_reentrant=False)
        
        return output

class MoGe_2_Head(nn.Module):
    def __init__(self, backbone_dim: int, projection_dim: int):
        super().__init__()
        self.projection_head = ConvStack(
            dim_in=[backbone_dim],
            dim_res_blocks=[backbone_dim],
            dim_out=[projection_dim],
            resamplers=[None],
        )
        raise NotImplementedError("MoGe_2_Head is not implemented")