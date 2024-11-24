
from typing import Any,Literal
from collections.abc import Sequence
from flax import nnx
# JAX imports
from jax.nn import relu,sigmoid,swish,tanh,leaky_relu,elu
from jax.numpy import sin
import jax
import jax.numpy as jnp
from jax import random
from jax._src import core
from jax._src import dtypes
from jax._src.typing import Array, ArrayLike
from jax._src.util import set_module
# FOL imports
from fol.tools.usefull_functions import *
from fol.tools.decoration_functions import *


export = set_module('jax.nn.initializers')
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex
RealNumeric = Any 
KeyArray = Array

@export
def siren_initializer(
  layer_type: Literal["input"] | Literal["hidden"] | Literal["output"],  
  omega: RealNumeric=30.0,
  weight_scale: RealNumeric=1.0,
  dtype: DTypeLikeInexact = jnp.float_
) -> nnx.Initializer:

  def init(key: KeyArray,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
    dtype = dtypes.canonicalize_dtype(dtype)
    named_shape = core.as_named_shape(shape)
    in_dim, out_dim = named_shape[0],named_shape[1]

    if layer_type == "input": variance = weight_scale / in_dim
    elif layer_type == "hidden": variance = weight_scale * jnp.sqrt(6 / in_dim) / omega
    elif layer_type == "output": variance = jnp.sqrt(6 / in_dim) / omega
    else:
      raise ValueError(
        f"invalid layer type for siren initializer: {layer_type}")
    
    if jnp.issubdtype(dtype, jnp.floating):
        return random.uniform(key, (in_dim, out_dim), dtype, minval=-variance, maxval=variance)
    else:
        return ValueError(f"invalid dtype for siren initializer: {dtype}")

  return init


class MLP(nnx.Module):
  def __init__(self,input_size: int,
                    output_size: int,
                    hidden_layers:list,
                    activation_function_settings:dict,
                    kernel_init:nnx.Initializer):

    self.in_features=input_size
    self.out_features=output_size
    self.hidden_layers = hidden_layers

    self.activation_function_settings={"name":"sin",
                                       "argument_scale":30,
                                       "init_scale":1}
    self.activation_function_settings = UpdateDefaultDict(self.activation_function_settings,
                                                            activation_function_settings)
    
    rngs = nnx.Rngs(42)
    layers = []
    last_layer = self.in_features # out_featuers of the last layer
    act_name = self.activation_function_settings["name"]
    for i,hidden_layer in enumerate(self.hidden_layers):

        if kernel_init==siren_initializer:
            if i==0:layer_type="input"
            else: layer_type="hidden"
            layers.append(nnx.Linear(in_features=last_layer, 
                                    out_features=hidden_layer, 
                                    rngs=rngs,kernel_init=kernel_init(layer_type=layer_type)))
        else:
            layers.append(nnx.Linear(in_features=last_layer, 
                                    out_features=hidden_layer, 
                                    rngs=rngs,kernel_init=kernel_init))
        
        if act_name=="sin":
            @jax.jit
            def arg_scale(x):
                return self.activation_function_settings["argument_scale"] * x

            layers.append(arg_scale)

        layers.append(globals()[act_name])
        last_layer = hidden_layer

    # now add the last layer
    if kernel_init==siren_initializer:    
        layers.append(nnx.Linear(in_features=last_layer, 
                                 out_features=hidden_layer, 
                                 rngs=rngs,kernel_init=kernel_init(layer_type="output")))
    else:
        layers.append(nnx.Linear(in_features=last_layer, 
                                 out_features=self.out_features,
                                 rngs=rngs,kernel_init=kernel_init))

    self.nn = nnx.Sequential(*layers)
    del layers

  def __call__(self, x: jax.Array):
        x = self.nn(x)
        return x
  
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
  
class HyperNetworks(nnx.Module):
    def __init__(self,synthesizer_NN_settings:dict,
                      modulator_NN_settings:dict,
                      coupling_settings:dict={}):
                
        self.synthesizer_NN_settings = {"input_layer_dim":None,
                                        "hidden_layers":[50,50],
                                        "output_layer_dim":None,
                                        "activation_function":"sin",
                                        "omega":30,"weight_scale":3.0}  

        self.modulator_NN_settings = {"input_layer_dim":None,
                                      "hidden_layers":[50,50],
                                      "activation_function":"relu",
                                      "fully_connected_layers":True,
                                      "skip_connections":True}  

        self.coupling_settings = {"shift_coupling":True,
                                  "scale_coupling":False,
                                  "modulator_to_synthesizer_coupling_mode":"all_to_all"} # other coupling options: last_to_all,last_to_last                  

        self.synthesizer_NN_settings = UpdateDefaultDict(self.synthesizer_NN_settings,
                                                       synthesizer_NN_settings)

        self.modulator_NN_settings = UpdateDefaultDict(self.modulator_NN_settings,
                                                       modulator_NN_settings)
        
        self.coupling_settings = UpdateDefaultDict(self.coupling_settings,
                                                   coupling_settings)
        
        if self.synthesizer_NN_settings["input_layer_dim"]==None:
            fol_error(f"input_layer_dim of the synthesizer network should be specified in the synthesizer_NN_settings !")

        if self.synthesizer_NN_settings["output_layer_dim"]==None:
            fol_error(f"output_layer_dim of the synthesizer network should be specified in the synthesizer_NN_settings !")

        if self.modulator_NN_settings["input_layer_dim"]==None:
            fol_error(f"input_layer_dim of the modulator network should be specified in the modulator_NN_settings !")

        self.in_features = self.modulator_NN_settings["input_layer_dim"]

        self.out_features = self.synthesizer_NN_settings["output_layer_dim"]

        if self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "all_to_all":
            if self.synthesizer_NN_settings["hidden_layers"] != self.modulator_NN_settings["hidden_layers"]:
                fol_error(f"for all_to_all modulator to synthesizer coupling, hidden layers of synthesizer and modulator NNs should be identical !")

        elif self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "last_to_all":
            if not all(x==self.modulator_NN_settings["hidden_layers"][-1] for x in self.synthesizer_NN_settings["hidden_layers"]):
                fol_error(f"for last_to_all modulator to synthesizer coupling, the last hidden layer of modulator NN should be equall to all synthesizer NN layers !")

        elif self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "last_to_last":
            if self.modulator_NN_settings["hidden_layers"][-1]!=self.synthesizer_NN_settings["hidden_layers"][-1]:
                fol_error(f"for last_to_last modulator to synthesizer coupling, the last layer of synthesizer and modulator NNs should be identical !")

        else:
            valid_options=["all_to_all","last_to_all","last_to_last"]
            fol_error(f"valid options for modulator_to_synthesizer_coupling_mode are {valid_options} !")

        self.initialize_synthesizer()
        self.initialize_modulator()
        
    def initialize_synthesizer(self):
        layers = [self.synthesizer_NN_settings["input_layer_dim"]]
        layers += self.synthesizer_NN_settings["hidden_layers"]
        layers += [self.synthesizer_NN_settings["output_layer_dim"]]
        weight_scale = self.synthesizer_NN_settings["weight_scale"]
        key = random.PRNGKey(0)
        keys = random.split(key, len(layers) - 1)
        self.synthesizer_params = []
        for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            weight_key, bias_key = random.split(keys[i])
            if i==0:
                weight_variance = weight_scale / in_dim
            elif i==len(layers)-2:
                weight_variance = jnp.sqrt(6 / in_dim) / self.synthesizer_NN_settings["omega"]
            else:
                weight_variance = weight_scale * jnp.sqrt(6 / in_dim) / self.synthesizer_NN_settings["omega"]
            weights = nnx.Param(random.uniform(weight_key, (in_dim, out_dim), jnp.float32, minval=-weight_variance, maxval=weight_variance))
            bias_variance = jnp.sqrt(1 / in_dim)
            biases = nnx.Param(random.uniform(bias_key, (int(out_dim),), jnp.float32, minval=-bias_variance, maxval=bias_variance))
            self.synthesizer_params.append((weights, biases))

        self.synthesizer_act_func = globals()[self.synthesizer_NN_settings["activation_function"]]
        self.synthesizer_act_func_multiplier = 1.0
        if self.synthesizer_NN_settings["activation_function"]=="sin":
            self.synthesizer_act_func_multiplier = self.synthesizer_NN_settings["omega"]

    def initialize_modulator(self): 
        layers = [self.modulator_NN_settings["input_layer_dim"]]
        layers += self.modulator_NN_settings["hidden_layers"] 
        self.modulator_skip_connections = self.modulator_NN_settings["skip_connections"]
        fully_connected_layers = self.modulator_NN_settings["fully_connected_layers"]
        self.modulator_input_dim = self.modulator_NN_settings["input_layer_dim"]
        key = random.PRNGKey(0)
        keys = random.split(key, len(layers) - 1)
        self.modulator_params = []
        for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            weight_key, bias_key = random.split(keys[i])
            if self.modulator_skip_connections and fully_connected_layers: 
                if i>0:
                    weights = nnx.Param(jax.nn.initializers.lecun_normal()(weight_key,(in_dim+self.modulator_input_dim, out_dim)))
                else:
                    weights = nnx.Param(jax.nn.initializers.lecun_normal()(weight_key,(in_dim, out_dim)))
            elif not fully_connected_layers:
                weights = nnx.Param(jax.nn.initializers.lecun_normal()(weight_key,(self.modulator_input_dim, out_dim)))
            elif not self.modulator_skip_connections and fully_connected_layers:
                weights = nnx.Param(jax.nn.initializers.lecun_normal()(weight_key,(in_dim, out_dim)))            

            biases = nnx.Param(jnp.zeros(out_dim))
            self.modulator_params.append((weights, biases))

        self.modulator_act_func = globals()[self.modulator_NN_settings["activation_function"]]

    def __call__(self, x: jax.Array):

        x_modul = x[:,0:self.modulator_input_dim]
        x_synth = x[:,self.modulator_input_dim:]

        if self.modulator_skip_connections:
            x_modul_init = x_modul.copy()

        # case 1: when all layers and neurons of modulator and synthesizer networks are coupled
        if self.coupling_settings["modulator_to_synthesizer_coupling_mode"]=="all_to_all":
            for i in range(len(self.modulator_params)):
                (w_modul, b_modul) = self.modulator_params[i]
                (w_synth, b_synth) = self.synthesizer_params[i]
                # first compute modul NN 
                if self.modulator_NN_settings["fully_connected_layers"]:
                    if self.modulator_skip_connections and i>0:
                        x_modul_skipped = jnp.hstack((x_modul,x_modul_init.copy()))
                        x_modul_not_act = x_modul_skipped @ w_modul + b_modul
                    else:
                        x_modul_not_act = (x_modul @ w_modul + b_modul)
                    
                    x_modul = self.modulator_act_func(x_modul_not_act)
                else:
                    x_modul_not_act = x_modul_init @ w_modul + b_modul
                    x_modul = x_modul_not_act

                # now compute synth NN 
                x_synth = self.synthesizer_act_func(self.synthesizer_act_func_multiplier * (x_synth @ w_synth + b_synth + x_modul))

        # case 2: when last layer's neurons of modulator are coupled to all synthesizer networks layers and neurons 
        elif self.coupling_settings["modulator_to_synthesizer_coupling_mode"]=="last_to_all":
            # first fw propagate the modulator
            for i in range(len(self.modulator_params)):
                (w_modul, b_modul) = self.modulator_params[i]
                if self.modulator_skip_connections and i>0:
                    x_modul_skipped = jnp.hstack((x_modul,x_modul_init.copy()))
                    x_modul = self.modulator_act_func(x_modul_skipped @ w_modul + b_modul)
                else:
                    x_modul = self.modulator_act_func(x_modul @ w_modul + b_modul)

            # then fw propagate the synthesizer
            for (w_synth, b_synth) in self.synthesizer_params[:-1]:
                x_synth = self.synthesizer_act_func(self.synthesizer_act_func_multiplier * (x_synth @ w_synth + b_synth + x_modul))

        # final fw propagate of the synthesizer
        final_w_synth, final_b_synth = self.synthesizer_params[-1]
        return x_synth @ final_w_synth + final_b_synth