import jax
import random as r
import numpy as np
import jax.numpy as jnp
from flax import nnx
from typing import Callable
import math
from abc import abstractmethod

class PositionalEncoding(nnx.Module):
    """
        Sinusoidal positional encoding module.

        This module encodes scalar input positions (e.g., sequence indices or 
        continuous values such as time steps) into a higher-dimensional space 
        using sinusoidal functions of different frequencies. The encoding 
        provides a way to inject positional information into models that 
        otherwise lack an inherent notion of order, such as Transformers.

        The encoding is defined as:
            PE(x) = [sin(x * θ_1), ..., sin(x * θ_d), cos(x * θ_1), ..., cos(x * θ_d)]

        where θ_i are frequencies spaced geometrically between 1 and 1/10000.

        Parameters
        ----------
        num_features : int, default=32
            Dimensionality of the encoding vector. Must be an even number, 
            since half of the features are sine terms and the other half are cosine terms.
    """    
    def __init__(self,num_features:int=32):
        self.num_output_features = num_features
    def __call__(self, x: jax.Array):
        half_dim = self.num_output_features // 2
        emb = math.log(10000) / (half_dim)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        return emb

class TimeEmbedding(nnx.Module):

    def __init__(self,dim:int=32,*,rngs: nnx.Rngs):
        self.dim = dim
        self.pos_encoder = PositionalEncoding(self.dim)
        self.output_features_dim =  self.dim * 4 # Projecting the embedding into a 128 dimensional space
        self.time_mlp_1 = nnx.Linear(in_features=self.dim, out_features=self.output_features_dim, rngs=rngs)
        self.time_mlp_2 = nnx.Linear(in_features=self.output_features_dim, out_features=self.output_features_dim, rngs=rngs)

    def __call__(self, inputs: jax.Array):
        x = self.pos_encoder(inputs)
        x = self.time_mlp_1(x)
        x = nnx.silu(x)
        x = self.time_mlp_2(x)        
        return x 

class MultiHeadAttention(nnx.Module):
    def __init__(self, num_input_features: int, num_heads: int = 8, use_bias: bool = False,
                kernel_init: Callable = nnx.initializers.xavier_uniform(),*,rngs: nnx.Rngs):
        self.num_input_features = num_input_features
        self.num_heads = num_heads
        self.mlp_1 = nnx.Linear(in_features=num_input_features, out_features=num_input_features*3, kernel_init=kernel_init, use_bias = use_bias, rngs=rngs)
        self.mlp_2 = nnx.Linear(in_features=num_input_features, out_features=num_input_features, kernel_init=kernel_init, rngs=rngs)

    def __call__(self, inputs: jax.Array):
        batch, h, w, channels = inputs.shape
        inputs = inputs.reshape(batch, h*w, channels)
        batch, n, channels = inputs.shape
        scale = (self.num_input_features // self.num_heads) ** -0.5
        qkv = self.mlp_1(inputs)
        qkv = jnp.reshape(
            qkv, (batch, n, 3, self.num_heads, channels // self.num_heads)
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attention = nnx.softmax(attention, axis=-1)

        x = (attention @ v).swapaxes(1, 2).reshape(batch, n, channels)
        x = self.mlp_2(x)
        x = jnp.reshape(x, (batch, int(x.shape[1]** 0.5), int(x.shape[1]** 0.5), -1))
        return x

class ConvNormAct(nnx.Module):
    def __init__(self, in_features: int, out_features: int = 32, num_groups: int = 8,*, rngs: nnx.Rngs): 
        self.conv = nnx.Conv(in_features=in_features,
                             out_features=out_features,
                             kernel_size=(3, 3),
                             rngs=rngs)
        self.gnorm = nnx.GroupNorm(num_features=out_features, num_groups=num_groups, rngs=rngs)  

    def __call__(self, inputs: jax.Array):
        conv = self.conv(inputs)
        norm = self.gnorm(conv)
        activation = nnx.silu(norm)
        return activation        

class ResnetBlock(nnx.Module):
    def __init__(self, in_features: int, embedded_time_features: int = None, out_features: int = 32, num_groups: int = 8,*, rngs: nnx.Rngs): 

        self.block_1 = ConvNormAct(in_features=in_features,
                                  out_features=out_features,
                                  num_groups=num_groups,
                                  rngs=rngs)
        self.block_2 = ConvNormAct(in_features=out_features,
                                  out_features=out_features,
                                  num_groups=num_groups,
                                  rngs=rngs)      
        self.res_conv = nnx.Conv(in_features=in_features,
                                 out_features=out_features,
                                 kernel_size=(1, 1),
                                 padding="SAME",
                                 rngs=rngs)   
        
        self.embedded_time_features = embedded_time_features
        if embedded_time_features is not None:
            self.time_mlp = nnx.Linear(in_features=embedded_time_features,
                                       out_features=out_features,
                                       rngs=rngs)                                                                                            

    def __call__(self, inputs: jax.Array, embedded_time: jax.Array | None = None):
        x = self.block_1(inputs)
        if self.embedded_time_features is not None: 
            embedded_time = nnx.silu(embedded_time)
            embedded_time = self.time_mlp(embedded_time)
            x = jnp.expand_dims(jnp.expand_dims(embedded_time, 1), 1) + x
        x = self.block_2(x)
        res_conv = self.res_conv(inputs)
        return x + res_conv

class TimestepBlock(nnx.Module):
    """
    Any module where it takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def __call__(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nnx.Module):
    """
    A sequential module that passes timestep embeddings to children that
    support it as an extra input.
    """

    def __init__(self, *layers: nnx.Module):
        self.layers = layers

    def __call__(self, x, emb):
        for layer in self.layers:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

def conv_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size,
    rngs: nnx.Rngs,
    stride=None,
    padding=None,
    **kwargs,
):
    """
    Create a 1D, 2D, or 3D convolution module in Flax nnx.
    
    Args:
        dims: number of spatial dims (1, 2, or 3).
        in_channels: input channels.
        out_channels: output channels.
        kernel_size: int or tuple.
        rngs: nnx.Rngs object.
        stride: int or tuple, optional.
        padding: "SAME", "VALID", int, or tuple, optional.
    """
    if dims not in (1, 2, 3):
        raise ValueError(f"unsupported dimensions: {dims}")

    # expand scalar -> tuple
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * dims

    # expand stride if provided
    if stride is not None and isinstance(stride, int):
        stride = (stride,) * dims

    # expand padding if provided
    if padding is not None:
        if isinstance(padding, int):
            padding = tuple((padding, padding) for _ in range(dims))
        elif isinstance(padding, tuple) and isinstance(padding[0], int):
            padding = tuple((p, p) for p in padding)

    # collect arguments
    conv_args = dict(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=kernel_size,
        rngs=rngs,
    )

    if stride is not None:
        conv_args["strides"] = stride
    if padding is not None:
        conv_args["padding"] = padding

    return nnx.Conv(**conv_args, **kwargs)

class Upsample(nnx.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    :param rngs: nnx.Rngs object containing random keys for initializing parameters
                 and any stochastic operations (e.g., dropout). Required in Flax nnx.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,*,rngs: nnx.Rngs):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims=dims, in_channels=channels, out_channels=self.out_channels, kernel_size=3, padding=1,rngs=rngs)

    def __call__(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = jax.image.resize(x, (x.shape[0],
                                    x.shape[1]*2,
                                    x.shape[2]*2,
                                    x.shape[3]*2,
                                    x.shape[4]), method="nearest")
        else:
            x = jax.image.resize(x,(x.shape[0],
                                    x.shape[1]*2,
                                    x.shape[2]*2,
                                    x.shape[3]), method="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    :param rngs: nnx.Rngs object containing random keys for initializing parameters
                 and any stochastic operations (e.g., dropout). Required in Flax nnx.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        *,
        rngs: nnx.Rngs
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = [nnx.GroupNorm(num_features=channels, num_groups=32, rngs=rngs),
                          nnx.silu,
                          conv_nd(dims=dims, in_channels=channels, out_channels=self.out_channels, kernel_size=3, padding=1,rngs=rngs)]

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims,rngs=rngs)
            self.x_upd = Upsample(channels, False, dims,rngs=rngs)
        elif down:
            self.h_upd = Downsample(channels, False, dims,rngs=rngs)
            self.x_upd = Downsample(channels, False, dims,rngs=rngs)
        else:
            identity = lambda x: x
            self.h_upd = self.x_upd = identity


        self.emb_layers = nnx.Sequential(
            nnx.silu,
            nnx.Linear(in_features=emb_channels, out_features=2 * self.out_channels if use_scale_shift_norm else self.out_channels, rngs=rngs)
        )

        self.out_layers = [ nnx.GroupNorm(num_features=self.out_channels, num_groups=32, rngs=rngs),
                            nnx.silu,
                            nnx.Dropout(rate=dropout),
                            conv_nd(dims=dims, in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1,rngs=rngs,
                                    kernel_init=nnx.initializers.constant(0),
                                    bias_init=nnx.initializers.constant(0))]

        if self.out_channels == channels:
            identity = lambda x: x
            self.skip_connection = identity
        elif use_conv:
            self.skip_connection = conv_nd(dims=dims, in_channels=channels, out_channels=self.out_channels, kernel_size=3, padding=1,rngs=rngs)
        else:
            self.skip_connection = conv_nd(dims=dims, in_channels=channels, out_channels=self.out_channels, kernel_size=1, rngs=rngs)

    def __call__(self, x, emb):
        if self.updown:
            in_rest, in_conv = nnx.Sequential(*self.in_layers[:-1]), self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = nnx.Sequential(*self.in_layers)(x)
        emb_out = self.emb_layers(emb)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], nnx.Sequential(*self.out_layers[1:])
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = nnx.Sequential(*self.out_layers)(h)
        return self.skip_connection(x) + h

class OpenAI_UNetModel(nnx.Module):
    """

    Originally ported from here and adapted to jax flax nnx .
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/openaimodel.py#L413.
    
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :param rngs: nnx.Rngs object containing random keys for initializing parameters
                 and any stochastic operations (e.g., dropout). Required in Flax nnx.
    """    
    def __init__(self,
                image_size,
                in_channels,
                model_channels,
                out_channels,
                num_res_blocks,
                attention_resolutions,
                dropout=0,
                channel_mult=(1, 2, 4, 8),
                conv_resample=True,
                dims=2,
                num_heads=1,
                num_head_channels=-1,
                num_heads_upsample=-1,
                use_scale_shift_norm=False,
                resblock_updown=False,
                use_new_attention_order=False,
                *,
                rngs: nnx.Rngs): 


        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.time_embed = TimeEmbedding(model_channels,rngs=rngs)

        ch = input_ch = int(channel_mult[0] * model_channels)
        time_embed_dim = model_channels * 4

        self.input_blocks = [TimestepEmbedSequential(conv_nd(dims=dims,
                                                             in_channels=in_channels,
                                                              out_channels=ch, 
                                                              kernel_size=3, 
                                                              padding=1,
                                                              rngs=rngs) )]


        temp_input = jnp.ones([10, 128, 128, 128])
        embedded_time = self.time_embed(jnp.ones([10, ]))
        # temp_output = self.input_blocks[0](temp_input,embedded_time)
        # print(f"temp_input:{temp_input.shape}")
        # print(f"temp_output:{temp_output.shape}")

        # exit()

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        tmp_res = ResBlock(ch,
                           time_embed_dim,
                           dropout,
                           out_channels=int(channel_mult[0] * model_channels),
                           dims=dims,
                           use_scale_shift_norm=use_scale_shift_norm,
                           rngs=rngs)
        
        tmp_res_output = tmp_res(temp_input,embedded_time)

        print(tmp_res_output.shape)
        
        
        exit()
        # ResBlock(ch,
        #                    time_embed_dim,
        #                    dropout,
        #                    out_channels=int(channel_mult[0] * model_channels),
        #                    use_scale_shift_norm=use_scale_shift_norm)

        # for level, mult in enumerate(channel_mult):
        #     for _ in range(num_res_blocks):
        #         layers = [
        #             ResBlock(
        #                 ch,
        #                 time_embed_dim,
        #                 dropout,
        #                 out_channels=int(mult * model_channels),
        #                 dims=dims,
        #                 use_checkpoint=use_checkpoint,
        #                 use_scale_shift_norm=use_scale_shift_norm,
        #             )
        #         ]
        #         ch = int(mult * model_channels)
        #         if ds in attention_resolutions:
        #             layers.append(
        #                 AttentionBlock(
        #                     ch,
        #                     use_checkpoint=use_checkpoint,
        #                     num_heads=num_heads,
        #                     num_head_channels=num_head_channels,
        #                     use_new_attention_order=use_new_attention_order,
        #                 )
        #             )
        #         self.input_blocks.append(TimestepEmbedSequential(*layers))
        #         self._feature_size += ch
        #         input_block_chans.append(ch)
        #     if level != len(channel_mult) - 1:
        #         out_ch = ch
        #         self.input_blocks.append(
        #             TimestepEmbedSequential(
        #                 ResBlock(
        #                     ch,
        #                     time_embed_dim,
        #                     dropout,
        #                     out_channels=out_ch,
        #                     dims=dims,
        #                     use_checkpoint=use_checkpoint,
        #                     use_scale_shift_norm=use_scale_shift_norm,
        #                     down=True,
        #                 )
        #                 if resblock_updown
        #                 else Downsample(
        #                     ch, conv_resample, dims=dims, out_channels=out_ch
        #                 )
        #             )
        #         )
        #         ch = out_ch
        #         input_block_chans.append(ch)
        #         ds *= 2
        #         self._feature_size += ch

        # temp_input = jnp.ones([10, 128, 128, 1])
        # embedded_time = self.time_embed(jnp.ones([10, ]))
        # temp_output = self.input_blocks[0](temp_input,embedded_time)
        # print(f"temp_input:{temp_input.shape}")
        # print(f"temp_output:{temp_output.shape}")

        # # print(len(self.input_blocks))
        # # print(f"ch:{ch}")
        # exit()


    def __call__(self, inputs: jax.Array, time: jax.Array | None = None):

        embedded_time = self.time_embed(time)

        return embedded_time        
