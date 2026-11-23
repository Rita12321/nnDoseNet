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
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder as OrigUNetDecoder # Keep original name if needed elsewhere
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
# from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks # Not used directly now
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op, get_matching_convtransp
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks # Used in Decoder

# --- TimestepEmbedding --- (Unchanged, looks okay)
class TimestepEmbedding(nn.Module):
    def __init__(self, dim): # dim is the target embedding dimension (e.g., 128)
        super().__init__()

        # Calculate the feature dimension created by the sin/cos embedding part
        # This is often based on a fraction of the target dimension, e.g., dim / 4
        half_base_dim = dim // 4 # e.g., 128 // 4 = 32
        embedding_input_dim = half_base_dim * 2 # = 64 (since we concat sin and cos)

        # Linear layers for the MLP
        self.linear1 = nn.Linear(embedding_input_dim, dim) # Accepts 64 features, outputs target dim (128)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim) # Accepts target dim, outputs target dim

    def forward(self, t):
        # Use the same base dimension for calculating frequencies
        half_base_dim = self.linear1.in_features // 2 # Recalculate from linear1's in_features (64 // 2 = 32)

        # Calculate sinusoidal position embeddings
        embeddings = math.log(10000) / (half_base_dim - 1)
        embeddings = torch.exp(torch.arange(half_base_dim, device=t.device) * -embeddings)
        # Ensure t is [batch_size, 1] for broadcasting
        if t.dim() == 1:
           t = t.unsqueeze(-1)
        embeddings = t.float() * embeddings.unsqueeze(0) # [batch, 1] * [1, half_base_dim] -> [batch, half_base_dim]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # Shape: [batch, half_base_dim * 2] = [batch, 64]

        # Project through MLP
        emb = self.linear1(embeddings) # linear1 now correctly accepts 64 features
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb

# --- SelfAttention --- (Corrected)
class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, head_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        # If head_dim is not specified, try to divide channels by num_heads
        if head_dim is None:
            assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
            self.head_dim = in_channels // num_heads
        else:
            self.head_dim = head_dim
            assert num_heads * head_dim == in_channels, "num_heads * head_dim must equal in_channels"

        self.scale = self.head_dim ** -0.5

        # Use GroupNorm with a reasonable number of groups, e.g., 32 or ensure it divides in_channels
        num_groups = 32
        if in_channels < num_groups or in_channels % num_groups != 0:
             # Find the largest divisor of in_channels <= 32
             num_groups = max(g for g in range(1, min(in_channels, 32) + 1) if in_channels % g == 0)

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        # Use a single Conv3D for QKV projection
        self.qkv = nn.Conv3d(in_channels, in_channels * 3, kernel_size=1, bias=False) # Bias often omitted in attention qkv
        # Projection layer
        self.proj = nn.Conv3d(in_channels, in_channels, kernel_size=1) # Keep bias here maybe

    def forward(self, x):
        # Input x: [batch, channels, depth, height, width]
        b, c, d, h, w = x.shape
        assert c == self.in_channels, f"Input channels {c} != initialized channels {self.in_channels}"

        x_norm = self.norm(x)
        qkv_out = self.qkv(x_norm) # [b, c*3, d, h, w]

        # Reshape and permute for multi-head attention
        # Reshape: [b, 3, num_heads, head_dim, d*h*w]
        # Permute: [3, b, num_heads, d*h*w, head_dim] (for q) or [3, b, num_heads, head_dim, d*h*w] (for k, v)
        qkv_reshaped = qkv_out.reshape(b, 3, self.num_heads, self.head_dim, d * h * w).permute(1, 0, 2, 4, 3)
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2] # q,k,v: [b, num_heads, seq_len, head_dim]

        # Scaled Dot-Product Attention
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale # [b, num_heads, seq_len, seq_len]
        attn = attn_logits.softmax(dim=-1)

        # Weighted sum of values
        out = attn @ v # [b, num_heads, seq_len, head_dim]

        # Reshape and project back
        out = out.permute(0, 1, 3, 2).reshape(b, c, d, h, w) # Back to [b, channels, d, h, w]
        out = self.proj(out)

        # Residual connection
        return x + out


# --- StackedResidualBlocks_diffuse --- (Corrected)
class StackedResidualBlocks_diffuse(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 time_embed_dim: int, # Added
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):
        super().__init__()
        assert n_blocks > 0, 'n_blocks must be > 0'
        assert block in [BasicBlockD, BottleneckD], 'block must be BasicBlockD or BottleneckD'
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks
        if not isinstance(bottleneck_channels, (tuple, list)):
            bottleneck_channels = [bottleneck_channels] * n_blocks

        # Store for time projection
        self.final_output_channels = output_channels[-1]
        self.time_embed_dim = time_embed_dim

        if block == BasicBlockD:
            blocks = nn.Sequential(
                block(conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias,
                      norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                      squeeze_excitation, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], output_channels[n], kernel_size, 1, conv_bias, norm_op,
                        norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                        squeeze_excitation, squeeze_excitation_reduction_ratio) for n in range(1, n_blocks)]
            )
        else: # BottleneckD
            blocks = nn.Sequential(
                block(conv_op, input_channels, bottleneck_channels[0], output_channels[0], kernel_size,
                      initial_stride, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                      nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], bottleneck_channels[n], output_channels[n], kernel_size,
                        1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation,
                        squeeze_excitation_reduction_ratio) for n in range(1, n_blocks)]
            )
        self.blocks = blocks
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

        # Time embedding projection: Map time embedding dim to the output channels of this block sequence
        self.time_mlp = nn.Sequential(
            nn.SiLU(), # Apply activation first
            nn.Linear(self.time_embed_dim, self.final_output_channels)
        )

    def forward(self, x, t_emb):
        # x: input feature map [b, c_in, d, h, w]
        # t_emb: time embedding [b, time_embed_dim]

        # Process input through residual blocks
        x = self.blocks(x) # Output shape: [b, c_out, d', h', w']

        # Project time embedding and reshape for broadcasting
        # t_emb shape: [b, time_embed_dim] -> [b, c_out] -> [b, c_out, 1, 1, 1]
        t_emb_proj = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Add time embedding (element-wise)
        # Broadcasting will automatically expand the spatial dimensions (1, 1, 1)
        x = x + t_emb_proj
        return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "Invalid input_size format"
        output = np.int64(0) # Use numpy int64 for potential large numbers
        current_size = list(input_size)
        for block_idx, block_module in enumerate(self.blocks):
             # Need compute_conv_feature_map_size method on BasicBlockD/BottleneckD
             # Assuming it exists and returns the contribution of that block
             if hasattr(block_module, 'compute_conv_feature_map_size'):
                 output += block_module.compute_conv_feature_map_size(current_size)

             # Update size based on stride only for the first block if initial_stride > 1
             if block_idx == 0:
                 current_size = [i // s for i, s in zip(current_size, self.initial_stride)]
             # Subsequent blocks have stride 1, so size doesn't change within the sequence
        return output

# --- ResidualEncoder --- (Corrected)
class ResidualEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 time_embed_dim: int, # Required for StackedResidualBlocks_diffuse
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv', # 'conv' means strided conv, 'max' or 'avg' means explicit pooling
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):
        super().__init__()
        # Input validation
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int): features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int): n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int): strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        assert len(bottleneck_channels) == n_stages
        assert len(kernel_sizes) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(strides) == n_stages

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]
        self.return_skips = return_skips
        self.time_embed_dim = time_embed_dim # Store for passing down

        # Stem
        if not disable_default_stem:
            if stem_channels is None: stem_channels = features_per_stage[0] // strides[0][0] # Guess reasonable stem channels
            # Stem uses StackedConvBlocks (no time embedding needed here usually)
            self.stem = StackedConvBlocks(
                num_convs=1, # Simple stem
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                kernel_size=kernel_sizes[0], # Use first stage kernel size
                initial_stride=1, # Stem usually doesn't downsample, stages handle it
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs
            )
            current_input_channels = stem_channels
        else:
            self.stem = None
            current_input_channels = input_channels

        # Stages
        self.stages = nn.ModuleList()
        for s in range(n_stages):
            stage_stride = self.strides[s]
            stage_features = features_per_stage[s]
            stage_kernel_size = kernel_sizes[s]
            stage_n_blocks = n_blocks_per_stage[s]
            stage_bottleneck_channels = bottleneck_channels[s]

            # Determine pooling and stride for the residual blocks
            pooling_layer = None
            stride_for_residual_block = stage_stride # Default: use stride in the first conv of the block

            if pool_type != 'conv' and any(st > 1 for st in stage_stride):
                 # If explicit pooling is used, residual block stride should be 1
                 pool_op = get_matching_pool_op(conv_op, pool_type=pool_type)
                 pooling_layer = pool_op(kernel_size=stage_stride, stride=stage_stride) # Pool kernel usually matches stride
                 stride_for_residual_block = 1 # Stride is handled by pooling

            res_blocks = StackedResidualBlocks_diffuse(
                n_blocks=stage_n_blocks,
                conv_op=conv_op,
                input_channels=current_input_channels,
                output_channels=stage_features,
                kernel_size=stage_kernel_size,
                initial_stride=stride_for_residual_block,
                time_embed_dim=self.time_embed_dim, # Pass time dim here
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                block=block,
                bottleneck_channels=stage_bottleneck_channels,
                stochastic_depth_p=stochastic_depth_p,
                squeeze_excitation=squeeze_excitation,
                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
            )

            # Combine pooling (if any) and residual blocks
            if pooling_layer is not None:
                stage_module = nn.Sequential(pooling_layer, res_blocks)
            else:
                stage_module = res_blocks

            self.stages.append(stage_module)
            current_input_channels = stage_features # Output of current stage is input for next

        self.output_channels = features_per_stage # Store final output channels per stage


    def forward(self, x, t_emb):
        skips = []
        if self.stem is not None:
            x = self.stem(x)
        # Skip connection from the stem (or input if no stem)
        if self.return_skips:
            skips.append(x)

        for s_idx, stage_module in enumerate(self.stages):
            # Check if stage module is Sequential (pool + blocks) or just blocks
            if isinstance(stage_module, nn.Sequential):
                x = stage_module[0](x) # Apply pooling if it exists
                x = stage_module[1](x, t_emb) # Apply blocks with time embedding
            else: # It's just the StackedResidualBlocks_diffuse
                x = stage_module(x, t_emb) # Apply blocks with time embedding

            if self.return_skips:
                # Don't add the absolute final output as a skip for the decoder
                if s_idx < len(self.stages) - 1:
                    skips.append(x)

        if self.return_skips:
             # The final output 'x' is the bottleneck, add it too
             skips.append(x)
             return skips # Return list [stem/input_skip, stage1_skip, ..., bottleneck]
        else:
             return x # Return only the final output (bottleneck)

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        current_size = list(input_size)

        if self.stem is not None:
             if hasattr(self.stem, 'compute_conv_feature_map_size'):
                  output += self.stem.compute_conv_feature_map_size(current_size)
             # Assume stem stride is 1, so current_size doesn't change here

        for s_idx, stage_module in enumerate(self.stages):
            stage_stride = self.strides[s_idx]
            res_blocks_module = stage_module[1] if isinstance(stage_module, nn.Sequential) else stage_module
            pooling_module = stage_module[0] if isinstance(stage_module, nn.Sequential) else None

            # Account for pooling layer size change
            if pooling_module is not None:
                 current_size = [i // st for i, st in zip(current_size, stage_stride)]

            # Account for residual blocks (which might also have stride in the first block if no pooling)
            if hasattr(res_blocks_module, 'compute_conv_feature_map_size'):
                 output += res_blocks_module.compute_conv_feature_map_size(current_size)

            # Update size based on stride ONLY if stride was handled by the residual block (i.e., no pooling)
            if pooling_module is None:
                current_size = [i // st for i, st in zip(current_size, stage_stride)]

        return output

class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

class DiffusionUNet_3(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int, # Number of downsampling stages in encoder (e.g., 4 means stem + 4 stages)
                 features_per_stage: Union[int, List[int], Tuple[int, ...]], # e.g., [64, 128, 256, 512]
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]], # Strides for each encoder stage
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]], # ResBlocks per encoder stage
                 num_classes: int, # Output classes for segmentation
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]], # Convs per decoder stage
                 time_embed_dim: int = 128, # Dimension of the time embedding
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = nn.InstanceNorm3d, # Example default
                 norm_op_kwargs: dict = {'eps': 1e-5, 'affine': True}, # Example default
                 dropout_op: Union[None, Type[_DropoutNd]] = None, # Example default
                 dropout_op_kwargs: dict = None, # Example default
                 nonlin: Union[None, Type[torch.nn.Module]] = nn.LeakyReLU, # Example default
                 nonlin_kwargs: dict = {'inplace': True}, # Example default
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD, # Residual block type
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None, # Bottleneck channels for BottleneckD
                 stem_channels: int = None, # Channels after initial stem convolution
                 use_attention: bool = True # Flag to enable/disable bottleneck attention
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int): # This one is for the encoder, seems okay
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        # Ensure norm_op is correctly matched if passed as string etc.
        # norm_op = get_matching_norm_op(norm_op) if isinstance(norm_op, str) else norm_op
        if isinstance(n_conv_per_stage_decoder, int):
            # Should have n_stages elements, not n_stages - 1
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * n_stages
        # The assertion should check against n_stages
        # assert len(n_conv_per_stage_decoder) == n_stages, \
        #     f"n_conv_per_stage_decoder must have as many entries as n_stages ({n_stages}). " \
        #     f"Got: {len(n_conv_per_stage_decoder)}"

        self.encoder = ResidualEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            time_embed_dim=time_embed_dim, # Pass time dim
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            block=block,
            bottleneck_channels=bottleneck_channels,
            return_skips=True, # MUST be true for U-Net structure
            disable_default_stem=False, # Usually want a stem
            stem_channels=stem_channels,
            pool_type='conv' # Use strided convs for downsampling by default
        )

        self.use_attention = use_attention
        if self.use_attention:
             bottleneck_feature_count = self.encoder.output_channels[-1]
             self.attention = SelfAttention(bottleneck_feature_count)
        else:
             self.attention = nn.Identity() # If attention is disabled

        self.decoder = UNetDecoder(
             encoder=self.encoder,
             num_classes=num_classes,
             n_conv_per_stage=n_conv_per_stage_decoder,
             deep_supervision=deep_supervision,
             # nonlin_first=False # Optional parameter for StackedConvBlocks
         )

        # Time embedding layer
        self.time_embedding = TimestepEmbedding(time_embed_dim)


    def forward(self, x, t):
        # x: Input image/volume [b, c, d, h, w]
        # t: Timestep tensor [b,] or [b, 1]
        # print(f"DEBUG: Input t shape to DiffusionUNet_3: {t.shape}")
        assert t.dim() == 1 or (t.dim() == 2 and t.shape[1] == 1), \
               f"Input t to DiffusionUNet_3 should be [batch] or [batch, 1], but got {t.shape}"
        # 1. Compute time embedding
        t_emb = self.time_embedding(t) # Shape: [b, time_embed_dim]

        # 2. Run Encoder
        # Skips will be a list: [stem_out, stage1_out, ..., bottleneck_out]
        skips = self.encoder(x, t_emb)

        # 3. Apply Attention (optional) to the bottleneck
        # Bottleneck is the last element in the skips list
        bottleneck = skips[-1]
        attended_bottleneck = self.attention(bottleneck) # Applies Identity if use_attention=False
        skips[-1] = attended_bottleneck # Replace original bottleneck with attended one

        # 4. Run Decoder
        # Decoder takes the list of skips (now including the attended bottleneck)
        output = self.decoder(skips) # Pass skips list, no t_emb needed here directly

        return output

    def compute_conv_feature_map_size(self, input_size):
        # Crude estimation: sum of encoder and decoder contributions
        # Note: Decoder calculation is complex and currently returns 0.
        encoder_size = self.encoder.compute_conv_feature_map_size(input_size)
        # decoder_size = self.decoder.compute_conv_feature_map_size(input_size) # Currently inaccurate
        # For now, primarily rely on encoder size or manually sum parameters after init.
        # print("Warning: Total compute_conv_feature_map_size includes inaccurate decoder size.")
        # return encoder_size + decoder_size
        return encoder_size # Return only encoder size for now