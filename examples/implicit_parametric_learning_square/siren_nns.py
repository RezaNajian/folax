from flax import nnx
import jax
import jax.numpy as jnp
from jax import random

class Siren(nnx.Module):
  def __init__(self,input_size: int,output_size: int, 
                hidden_layers:list,omega:float=30,
                weight_scale:float=3.0):

    self.in_features=input_size
    self.out_features=output_size
    self.omega=omega
    self.weight_scale=weight_scale
    siren_layers = hidden_layers

    layer_sizes = [self.in_features] +  siren_layers + [self.out_features]
    key = random.PRNGKey(0)
    keys = random.split(key, len(layer_sizes) - 1)
    self.NN_params = []

    for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        weight_key, bias_key = random.split(keys[i])
        if i==0:
            weight_variance = self.weight_scale / in_dim
        elif i==len(layer_sizes)-2:
            weight_variance = jnp.sqrt(6 / in_dim) / self.omega
        else:
            weight_variance = self.weight_scale * jnp.sqrt(6 / in_dim) / self.omega
        
        weights = nnx.Param(random.uniform(weight_key, (in_dim, out_dim), jnp.float32, minval=-weight_variance, maxval=weight_variance))
        bias_variance = jnp.sqrt(1 / in_dim)
        biases = nnx.Param(random.uniform(bias_key, (int(out_dim),), jnp.float32, minval=-bias_variance, maxval=bias_variance))
        self.NN_params.append((weights, biases))

  def __call__(self, x: jax.Array):
        for (w, b) in self.NN_params[:-1]:
            x = x @ w + b
            x = jnp.sin(self.omega*x)
        final_w, final_b = self.NN_params[-1]
        return x @ final_w + final_b
  
class ModulatedSiren(nnx.Module):
  def __init__(self, synthesis_input_dim:int, synthesis_output_dim:int,
                     modulator_input_dim:int, hidden_layers:list,
                     omega:float=30,weight_scale:float=3.0,
                     modulator_skip_connections:bool=True):

    self.synthesis_input_dim=synthesis_input_dim
    self.modulator_input_dim=modulator_input_dim
    self.in_features=synthesis_input_dim + modulator_input_dim
    self.out_features=synthesis_output_dim
    self.omega=omega
    self.weight_scale=weight_scale
    self.skip_connect=modulator_skip_connections

    synthesis_layers = [synthesis_input_dim] + hidden_layers + [synthesis_output_dim]
    modulator_layers = [modulator_input_dim] + hidden_layers

    def synthesis_initialize_params(layers:list):
        key = random.PRNGKey(0)
        keys = random.split(key, len(layers) - 1)
        layers_params = []
        for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            weight_key, bias_key = random.split(keys[i])
            if i==0:
                weight_variance = self.weight_scale / in_dim
            elif i==len(layers)-2:
                weight_variance = jnp.sqrt(6 / in_dim) / self.omega
            else:
                weight_variance = self.weight_scale * jnp.sqrt(6 / in_dim) / self.omega
            weights = nnx.Param(random.uniform(weight_key, (in_dim, out_dim), jnp.float32, minval=-weight_variance, maxval=weight_variance))
            bias_variance = jnp.sqrt(1 / in_dim)
            biases = nnx.Param(random.uniform(bias_key, (int(out_dim),), jnp.float32, minval=-bias_variance, maxval=bias_variance))
            layers_params.append((weights, biases))
        return layers_params
    
    def modulator_initialize_params(layers:list):
        key = random.PRNGKey(0)
        keys = random.split(key, len(layers) - 1)
        layers_params = []
        for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            weight_key, bias_key = random.split(keys[i])
            if self.skip_connect and i>0:
                weights = nnx.Param(jax.nn.initializers.lecun_normal()(weight_key,(in_dim+self.modulator_input_dim, out_dim)))
            else:
                weights = nnx.Param(jax.nn.initializers.lecun_normal()(weight_key,(in_dim, out_dim)))
            biases = nnx.Param(jnp.zeros(out_dim))
            layers_params.append((weights, biases))
        return layers_params
        
    self.synthesis_params = synthesis_initialize_params(synthesis_layers)
    self.modulator_params = modulator_initialize_params(modulator_layers)

  def __call__(self, x: jax.Array):
        
        x_modul = x[:,0:self.modulator_input_dim]
        x_synth = x[:,self.modulator_input_dim:]
        if self.skip_connect:
            x_modul_init = x_modul.copy()

        for i in range(len(self.modulator_params)):
            (w_modul, b_modul) = self.modulator_params[i]
            if self.skip_connect and i>0:
                x_modul_skipped = jnp.hstack((x_modul,x_modul_init.copy()))
                x_modul = jax.nn.relu(x_modul_skipped @ w_modul + b_modul)
            else:
                x_modul = jax.nn.relu(x_modul @ w_modul + b_modul)

            (w_synth, b_synth) = self.synthesis_params[i]
            x_synth = jnp.sin(self.omega * (x_synth @ w_synth + b_synth + x_modul))

        final_w_synth, final_b_synth = self.synthesis_params[-1]
        return x_synth @ final_w_synth + final_b_synth