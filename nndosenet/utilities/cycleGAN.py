import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Type, List, Tuple
from abc import abstractmethod
import math
import numpy as np
import torch as th
from collections import OrderedDict
# Assuming these imports exist in your project structure
# from .fp16_util import convert_module_to_f16, convert_module_to_f32
# from .utils import softmax_helper, sigmoid_helper
# from .utils import InitWeights_He
# from batchgenerators.augmentations.utils import pad_nd_image
# from .utils import no_op, to_cuda, maybe_to_torch
# from scipy.ndimage.filters import gaussian_filter
# from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder # Keep for typing hint in UNetDecoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks # Not used directly now
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op, get_matching_convtransp
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks # Used in Decoder


class CycleGANGenerator(nn.Module):
    """
    Generator network based on ResidualEncoderUNet, adapted for CycleGAN.
    - Takes an image from the source domain.
    - Outputs an image for the target domain.
    - Disables deep supervision.
    - Uses Tanh activation on the final output.
    """
    def __init__(self,
                 input_channels: int,
                 output_channels: int, # Changed from num_classes
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None, # Default to InstanceNorm later if None
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()

        # Set default norm_op to InstanceNorm if None (common for GANs)
        if norm_op is None:
            # Assuming conv_op gives dimension (e.g., nn.Conv3d -> use InstanceNorm3d)
            dim = conv_op.__name__[-2:] # '2d' or '3d'
            norm_op = getattr(nn, f"InstanceNorm{dim}")
            if norm_op_kwargs is None:
                 norm_op_kwargs = {'affine': False} # CycleGAN often uses affine=False

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # --- Encoder ---
        # NOTE: return_skips MUST be True for the U-Net structure
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, # Ensure skips are returned
                                       disable_default_stem=False, stem_channels=stem_channels)

        # --- Decoder ---
        # NOTE: deep_supervision is explicitly set to False
        # NOTE: Using output_channels instead of num_classes
        self.decoder = UNetDecoder(self.encoder,
                                   output_channels, # Pass output_channels here
                                   n_conv_per_stage_decoder,
                                   deep_supervision=False, # Disable deep supervision
                                   nonlin_first=False) # Assuming default

        # --- Final Activation ---
        # Common practice in GANs to scale output, typically to [-1, 1]
        self.final_activation = nn.Tanh()

    def forward(self, x):
        skips = self.encoder(x)
        # Decoder returns only the final output when deep_supervision=False
        decoder_output = self.decoder(skips)
        # Apply final activation
        output_image = self.final_activation(decoder_output)
        return output_image

    # compute_conv_feature_map_size might not be directly relevant for CycleGAN training loop
    # but keep it if you need it for analysis.
    # def compute_conv_feature_map_size(self, input_size):
    #     # ... (implementation from ResidualEncoderUNet) ...

class PatchGANDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator for 3D (adapt for 2D if needed)."""
    def __init__(self, input_channels, base_features=64, n_layers=3, norm_op=nn.InstanceNorm3d, use_sigmoid=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_channels (int)  -- the number of channels in input images
            base_features (int)   -- the number of filters in the first conv layer
            n_layers (int)      -- the number of conv layers in the discriminator
            norm_op             -- normalization layer type
            use_sigmoid (bool)  -- if True, use sigmoid activation (LSGAN uses no activation)
        """
        super().__init__()
        if norm_op is not None and norm_op == nn.BatchNorm3d : # Or BatchNorm2d
             # Use BatchNorm with affine=True, track_running_stats=True
             norm_op_kwargs = {'affine': True, 'track_running_stats': True}
             use_bias = False # BatchNorm has bias
        elif norm_op is not None and norm_op == nn.InstanceNorm3d: # Or InstanceNorm2d
             norm_op_kwargs = {'affine': False, 'track_running_stats': False}
             use_bias = True # InstanceNorm doesn't have learnable bias/scale
        else:
             norm_op_kwargs = {}
             use_bias = True

        kernel_size = 4
        padding = 1 # Adjust padding based on kernel_size and desired output size

        # Determine Conv operation based on norm_op dimension
        conv_op = nn.Conv3d if norm_op == nn.InstanceNorm3d or norm_op == nn.BatchNorm3d else nn.Conv2d

        sequence = [conv_op(input_channels, base_features, kernel_size=kernel_size, stride=2, padding=padding),
                    nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                conv_op(base_features * nf_mult_prev, base_features * nf_mult, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_op(base_features * nf_mult, **norm_op_kwargs),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            conv_op(base_features * nf_mult_prev, base_features * nf_mult, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias),
            norm_op(base_features * nf_mult, **norm_op_kwargs),
            nn.LeakyReLU(0.2, True)
        ]

        # Output 1 channel prediction map
        sequence += [conv_op(base_features * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding)]

        if use_sigmoid: # Not recommended for LSGAN loss
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
