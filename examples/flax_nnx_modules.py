import jax
import random as r
import numpy as np
import jax.numpy as jnp
from flax import nnx
from typing import Callable
import math
from abc import abstractmethod
from functools import partial

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    )
    args = timesteps[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

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

def avg_pool_nd(dims: int,
    window_shape,
    stride=None,
    padding=None,
    **kwargs,
):
    """
    Create a 1D, 2D, or 3D convolution module in Flax nnx.
    
    Args:
        dims: number of spatial dims (1, 2, or 3).
        window_shape: int or tuple.
        stride: int or tuple, optional.
        padding: "SAME", "VALID", int, or tuple, optional.
    """
    if dims not in (1, 2, 3):
        raise ValueError(f"unsupported dimensions: {dims}")

    # expand scalar -> tuple
    if isinstance(window_shape, int):
        window_shape = (window_shape,) * dims

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
    avg_pool_args = dict(
        window_shape=window_shape,
    )

    if stride is not None:
        avg_pool_args["strides"] = stride
    if padding is not None:
        avg_pool_args["padding"] = padding

    return partial(nnx.avg_pool, **avg_pool_args, **kwargs)

class Downsample(nnx.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    :param rngs: nnx.Rngs object containing random keys for initializing parameters
                 and any stochastic operations (e.g., dropout). Required in Flax nnx.                 
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,*,rngs: nnx.Rngs):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims=dims, in_channels=self.channels, out_channels=self.out_channels, kernel_size=3, stride=stride, padding=1,rngs=rngs)
            
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, window_shape=stride, stride=stride)
        
    def __call__(self, x):
        assert x.shape[-1] == self.channels
        return self.op(x)    

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
        assert x.shape[-1] == self.channels
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

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], nnx.Sequential(*self.out_layers[1:])
            scale, shift = jnp.split(emb_out, 2, axis=1)
            n_middle = h.ndim - shift.ndim
            shift = shift.reshape(shift.shape[0], *([1]*n_middle), shift.shape[-1])
            scale = scale.reshape(scale.shape[0], *([1]*n_middle), scale.shape[-1])
            h = out_norm(h) * (1 + scale) + shift           
            h = out_rest(h)
        else:
            n_middle = h.ndim - emb_out.ndim
            emb_out = emb_out.reshape(emb_out.shape[0], *([1]*n_middle), emb_out.shape[-1])
            h = h + emb_out
            h = nnx.Sequential(*self.out_layers)(h)
        return self.skip_connection(x) + h

class QKVAttention(nnx.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def __call__(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x T x (3 * H * C)] tensor of Qs, Ks, and Vs.
        :return: an [N x T x (H * C)] tensor after attention.
        """
        bs, length, width,  = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jnp.einsum(
            "bic,bjc->bij",
            (q * scale).reshape(bs * self.n_heads, length, ch),
            (k * scale).reshape(bs * self.n_heads, length, ch),
        )  # More stable with f16 than dividing afterwards
        weight = nnx.softmax(weight, axis=-1)
        a = jnp.einsum('bij,bjc->bic', weight, v.reshape(bs * self.n_heads, length, ch))
        return a.reshape(bs, length, -1)

class QKVAttentionLegacy(nnx.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def __call__(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x T x (H * 3 * C) ] tensor of Qs, Ks, and Vs.
        :return: an [N x T x (H * C) ] tensor after attention.
        """    
        bs, length, width,  = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = jnp.split(qkv.reshape(bs * self.n_heads, length, ch * 3), 3, axis=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jnp.einsum(
            "bic,bjc->bij", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = nnx.softmax(weight, axis=-1)
        a = jnp.einsum('bij,bjc->bic', weight, v)
        return a.reshape(bs, length, -1)

class AttentionBlock(nnx.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
        *,
        rngs: nnx.Rngs        
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = nnx.GroupNorm(num_features=channels, num_groups=32, rngs=rngs)
        self.qkv = conv_nd(dims=1, in_channels=channels, out_channels=channels * 3, kernel_size=1, rngs=rngs)

        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = conv_nd(dims=1, in_channels=channels, out_channels=channels, kernel_size=1,rngs=rngs,
                                    kernel_init=nnx.initializers.constant(0),
                                    bias_init=nnx.initializers.constant(0))

    def __call__(self, x):
        b, *spatial, c = x.shape
        x = x.reshape(b, -1, c)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, *spatial, c)


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

        time_embed_dim = model_channels * 4
        self.time_embed = nnx.Sequential(
            nnx.Linear(in_features=model_channels, out_features=time_embed_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(in_features=time_embed_dim, out_features=time_embed_dim, rngs=rngs)
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = [TimestepEmbedSequential(conv_nd(dims=dims,
                                                             in_channels=in_channels,
                                                              out_channels=ch, 
                                                              kernel_size=3, 
                                                              padding=1,
                                                              rngs=rngs) )]

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        rngs=rngs
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            rngs=rngs
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            rngs=rngs
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch,rngs=rngs
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                rngs=rngs
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                rngs=rngs
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                rngs=rngs
            ),
        )
        self._feature_size += ch

        self.output_blocks = []
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        rngs=rngs
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            rngs=rngs
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            rngs=rngs
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch,rngs=rngs)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch


        self.out = nnx.Sequential(
                    nnx.GroupNorm(num_features=ch, num_groups=32, rngs=rngs), 
                    nnx.silu,
                    conv_nd(dims=dims, in_channels=input_ch, out_channels=out_channels, kernel_size=3, padding=1,rngs=rngs,
                                                        kernel_init=nnx.initializers.constant(0),
                                                        bias_init=nnx.initializers.constant(0)))

    def __call__(self, x: jax.Array, timesteps: jax.Array):

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h = x
        hs = []
        for index,module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for index,module in enumerate(self.output_blocks):
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = module(h, emb)
        return self.out(h)
