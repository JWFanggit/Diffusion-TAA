from einops import rearrange, repeat, reduce
import numpy as np
import torch.nn as nn
from torch.nn.modules.utils import _pair

from weight_init import trunc_normal_, constant_init_, kaiming_init_
class PatchEmbed(nn.Module):
    """Images to Patch Embedding.

    Args:
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        tube_size (int): Size of temporal field of one 3D patch.
        in_channels (int): Channel num of input features. Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        conv_type (str): Type for convolution layer. Defaults to 'Conv2d'.
    """

    def __init__(self,
                 img_size,
                 patch_size,
                 tube_size=1,
                 in_channels=3,
                 embed_dims=192,
                 conv_type='Conv3d'):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)

        num_patches = \
            (self.img_size[1] // self.patch_size[1]) * \
            (self.img_size[0] // self.patch_size[0])
        assert (num_patches * self.patch_size[0] * self.patch_size[1] ==
                self.img_size[0] * self.img_size[1],
                'The image size H*W must be divisible by patch size')
        self.num_patches = num_patches

        # Use conv layer to embed
        if conv_type == 'Conv2d':
            self.projection = nn.Conv2d(
                in_channels,
                embed_dims,
                kernel_size=patch_size,
                stride=patch_size)
        elif conv_type == 'Conv3d':
            self.projection = nn.Conv3d(
                in_channels,
                embed_dims,
                kernel_size=(tube_size, patch_size, patch_size),
                stride=(tube_size, patch_size, patch_size))
        else:
            raise TypeError(f'Unsupported conv layer type {conv_type}')

        self.init_weights(self.projection)

    def init_weights(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            kaiming_init_(module.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(module, 'bias') and module.bias is not None:
            constant_init_(module.bias, constant_value=0)

    def forward(self, x):
        layer_type = type(self.projection)
        if layer_type == nn.Conv3d:
            x = rearrange(x, 'b t c h w -> b c t h w')
            x = self.projection(x)
            x = rearrange(x, 'b c t h w -> b t (h w) c')
        elif layer_type == nn.Conv2d:
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            x = self.projection(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
        else:
            raise TypeError(f'Unsupported conv layer type {layer_type}')

        return x