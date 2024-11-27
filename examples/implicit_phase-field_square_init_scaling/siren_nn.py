from flax import nnx
import jax
import jax.numpy as jnp
from jax import random

class Siren(nnx.Module):
  def __init__(self, din: int, dout: int, hidden_layers:list, weight_scale:float=1.0):

    self.in_features, self.out_features = din, dout
    self.omega=30
    self.weight_scale = weight_scale
    siren_layers = [hidden_layers[0]] + hidden_layers

    layer_sizes = [self.in_features] +  siren_layers + [self.out_features]
    key = random.PRNGKey(0)
    keys = random.split(key, len(layer_sizes) - 1)
    self.NN_params = []

    for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        weight_key, bias_key = random.split(keys[i])
        if i==0:
            weight_variance = self.weight_scale / in_dim
        # if i==len(layer_sizes)-2:
        #     weight_variance = jnp.sqrt(6 / in_dim) / self.omega
        else:
            weight_variance = self.weight_scale*jnp.sqrt(6 / in_dim) / self.omega
        
        weights = nnx.Param(random.uniform(weight_key, (in_dim, out_dim), jnp.float32, minval=-weight_variance, maxval=weight_variance))
        bias_variance = jnp.sqrt(1 / in_dim)
        biases = nnx.Param(random.uniform(bias_key, (int(out_dim),), jnp.float32, minval=-bias_variance, maxval=bias_variance))
        self.NN_params.append((weights, biases))

  def __call__(self, x: jax.Array):
        for (w, b) in self.NN_params[:-1]:
            x = x @ w + b
            x = jnp.sin(self.omega*x)
        final_w, final_b = self.NN_params[-1]
        x = x @ final_w + final_b
        return x