import jax
import random as r
import numpy as np
import jax.numpy as jnp
from flax import nnx
from typing import Callable
import math

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
        emb = math.log(10000) / (half_dim - 1)
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
        x = nnx.gelu(x)
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


class UNet(nnx.Module):
    def __init__(self, 
                 input_features: int,
                 input_feature_maps: int,
                 output_features: int,
                 base_feature_channels: int = 8, 
                 features_scale_factor: tuple = (1, 2, 4, 8), 
                 num_groups: int = 8,
                 embed_time: bool = True,
                 *,
                 rngs: nnx.Rngs): 

        self.feature_map_conv = nnx.Conv(in_features=input_features,
                                         out_features=input_feature_maps, 
                                         kernel_size=(7, 7), 
                                         padding=((3,3), (3,3)),
                                         rngs=rngs) 

        self.embed_time = embed_time
        if embed_time is True:
            self.time_emb = TimeEmbedding(base_feature_channels,rngs=rngs)

        self.sampling_dims = [base_feature_channels * i for i in features_scale_factor]

        # Downsampling blocks
        self.down_blocks = []
        for i, ds_dim in enumerate(self.sampling_dims):
            if i==0:
                in_features = input_feature_maps
            else:
                in_features = self.sampling_dims[i-1]
            res_1 = ResnetBlock(in_features=in_features,
                               embedded_time_features=4*base_feature_channels,
                               out_features=ds_dim,
                               num_groups=num_groups,
                               rngs=rngs)

            res_2 = ResnetBlock(in_features=ds_dim,
                               embedded_time_features=4*base_feature_channels,
                               out_features=ds_dim,
                               num_groups=num_groups,
                               rngs=rngs)

            att = MultiHeadAttention(ds_dim,rngs=rngs)

            gnorm = nnx.GroupNorm(num_features=ds_dim, num_groups=num_groups, rngs=rngs)  

            # downsample conv (except last)
            down_conv = None
            if i != len(self.sampling_dims) - 1:
                down_conv = nnx.Conv(in_features=ds_dim,
                                     out_features=ds_dim,
                                     kernel_size=(4, 4), 
                                     strides=(2, 2), 
                                     rngs=rngs)

            self.down_blocks.append((res_1, res_2, att, gnorm, down_conv))

        
        # Latent blocks
        self.latent_res_1 = ResnetBlock(in_features=self.sampling_dims[-1],
                                        embedded_time_features=4*base_feature_channels,
                                        out_features=self.sampling_dims[-1],
                                        num_groups=num_groups,
                                        rngs=rngs)

        self.latent_att = MultiHeadAttention(self.sampling_dims[-1],rngs=rngs)
        self.latent_gnorm = nnx.GroupNorm(num_features=self.sampling_dims[-1], num_groups=num_groups, rngs=rngs)  
        self.latent_res_2 = ResnetBlock(in_features=self.sampling_dims[-1],
                                        embedded_time_features=4*base_feature_channels,
                                        out_features=self.sampling_dims[-1],
                                        num_groups=num_groups,
                                        rngs=rngs)


        reversed_sampling_dims = list(reversed(self.sampling_dims))
        # Downsampling blocks
        self.up_blocks = []
        for i, ds_dim in enumerate(reversed_sampling_dims):
            if i==0:
                in_features = ds_dim * 2
            else:
                in_features = ds_dim + reversed_sampling_dims[i-1]

            res_1 = ResnetBlock(in_features=in_features,
                                embedded_time_features=4*base_feature_channels,
                                out_features=ds_dim,
                                num_groups=num_groups,
                                rngs=rngs)

            res_2 = ResnetBlock(in_features=ds_dim,
                                embedded_time_features=4*base_feature_channels,
                                out_features=ds_dim,
                                num_groups=num_groups,
                                rngs=rngs)

            att = MultiHeadAttention(ds_dim,rngs=rngs)

            gnorm = nnx.GroupNorm(num_features=ds_dim, num_groups=num_groups, rngs=rngs)  

            # downsample conv (except last)
            up_conv = None
            if i != len(self.sampling_dims) - 1:
                up_conv = nnx.ConvTranspose(in_features=ds_dim,
                                            out_features=ds_dim,
                                            kernel_size=(4, 4), 
                                            strides=(2, 2), 
                                            rngs=rngs)

            self.up_blocks.append((res_1, res_2, att, gnorm, up_conv))

        # Final ResNet block and output convolutional layer
        self.final_res = ResnetBlock(in_features=self.sampling_dims[0],
                                     embedded_time_features=4*base_feature_channels,
                                     out_features=base_feature_channels,
                                     num_groups=num_groups,
                                     rngs=rngs)
        self.final_conv = nnx.Conv(in_features=base_feature_channels,
                                   out_features=output_features,
                                   kernel_size=(1, 1), 
                                   padding="SAME",
                                   rngs=rngs)       


    def __call__(self, inputs: jax.Array, time: jax.Array | None = None):

        x = self.feature_map_conv(inputs)
        embedded_time = self.time_emb(time)

        pre_downsampling = []
        for (res_1, res_2, att, gnorm, down_conv) in self.down_blocks:
            x = res_1(x,embedded_time)
            x = res_2(x,embedded_time)
            attn_out = att(x)
            gnorm_out = gnorm(attn_out)
            x = gnorm_out + x
            # Saving this output for residual connection with the upsampling layer
            pre_downsampling.append(x)
            if down_conv is not None:
                x = down_conv(x)

        # latent
        x = self.latent_res_1(x,embedded_time)
        attn_out = self.latent_att(x)
        gnorm_out = self.latent_gnorm(attn_out)
        x = gnorm_out + x
        x = self.latent_res_2(x,embedded_time)

        # Upsampling 
        for up_index,(res_1, res_2, att, gnorm, up_conv) in enumerate(self.up_blocks):
            x = jnp.concatenate([pre_downsampling.pop(), x], -1)
            x = res_1(x,embedded_time)
            x = res_2(x,embedded_time)
            attn_out = att(x)
            gnorm_out = gnorm(attn_out)
            x = gnorm_out + x
            if up_conv is not None:
                x = up_conv(x)

        print(f"nnx,ups_final_x.shape:{x.shape}")

        x = self.final_res(x,embedded_time)
        x = self.final_conv(x)

        print(f"nnx,final_x.shape:{x.shape}")

        return x        
