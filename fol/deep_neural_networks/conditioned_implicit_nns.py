from flax import nnx
from jax.nn import relu,sigmoid,swish,tanh,leaky_relu,elu
from jax.numpy import sin
import jax
import jax.numpy as jnp
from jax import random
from fol.tools.usefull_functions import *
from fol.tools.decoration_functions import *

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
  
class HyperNetworks(nnx.Module):
    def __init__(self,synthesis_NN_settings:dict,modulator_NN_settings:dict,
                coupling_settings:dict={}):
                
        self.synthesis_NN_settings = {"input_layer_dim":None,
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
                                "modulation_to_synthesis_coupling_mode":"all_to_all"} # other coupling options: last_to_all,last_to_last                  

        self.synthesis_NN_settings = UpdateDefaultDict(self.synthesis_NN_settings,
                                                    synthesis_NN_settings)

        self.modulator_NN_settings = UpdateDefaultDict(self.modulator_NN_settings,
                                                    modulator_NN_settings)
        
        self.coupling_settings = UpdateDefaultDict(self.coupling_settings,
                                                    coupling_settings)
        
        if self.synthesis_NN_settings["input_layer_dim"]==None:
            fol_error(f"input_layer_dim of the synthesis network should be specified in the synthesis_NN_settings !")

        if self.synthesis_NN_settings["output_layer_dim"]==None:
            fol_error(f"output_layer_dim of the synthesis network should be specified in the synthesis_NN_settings !")

        if self.modulator_NN_settings["input_layer_dim"]==None:
            fol_error(f"input_layer_dim of the modulator network should be specified in the modulator_NN_settings !")

        # if self.modulator_NN_settings["fully_connected_layers"]==False and self.modulator_NN_settings["skip_connections"]==True:
        #     fol_error(f"in the modulator network, fully_connected_layers can not be False when skip_connections is True !")

        self.in_features = self.modulator_NN_settings["input_layer_dim"]

        self.out_features = self.synthesis_NN_settings["output_layer_dim"]

        if self.coupling_settings["modulation_to_synthesis_coupling_mode"] == "all_to_all":
            if self.synthesis_NN_settings["hidden_layers"] != self.modulator_NN_settings["hidden_layers"]:
                fol_error(f"for all_to_all modulation to synthesis coupling, hidden layers of synthesis and modulator NNs should be identical !")

        elif self.coupling_settings["modulation_to_synthesis_coupling_mode"] == "last_to_all":
            if not all(x==self.modulator_NN_settings["hidden_layers"][-1] for x in self.synthesis_NN_settings["hidden_layers"]):
                fol_error(f"for last_to_all modulation to synthesis coupling, the last hidden layer of modulator NN should be equall to all synthesis NN layers !")

        elif self.coupling_settings["modulation_to_synthesis_coupling_mode"] == "last_to_last":
            if self.modulator_NN_settings["hidden_layers"][-1]!=self.synthesis_NN_settings["hidden_layers"][-1]:
                fol_error(f"for last_to_last modulation to synthesis coupling, the last layer of synthesis and modulator NNs should be identical !")

        else:
            valid_options=["all_to_all","last_to_all","last_to_last"]
            fol_error(f"valid options for modulation_to_synthesis_coupling_mode are {valid_options} !")

        self.initialize_synthesis()
        self.initialize_modulator()
        
    def initialize_synthesis(self):
        layers = [self.synthesis_NN_settings["input_layer_dim"]]
        layers += self.synthesis_NN_settings["hidden_layers"]
        layers += [self.synthesis_NN_settings["output_layer_dim"]]
        weight_scale = self.synthesis_NN_settings["weight_scale"]
        key = random.PRNGKey(0)
        keys = random.split(key, len(layers) - 1)
        self.synthesis_params = []
        for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            weight_key, bias_key = random.split(keys[i])
            if i==0:
                weight_variance = weight_scale / in_dim
            elif i==len(layers)-2:
                weight_variance = jnp.sqrt(6 / in_dim) / self.synthesis_NN_settings["omega"]
            else:
                weight_variance = weight_scale * jnp.sqrt(6 / in_dim) / self.synthesis_NN_settings["omega"]
            weights = nnx.Param(random.uniform(weight_key, (in_dim, out_dim), jnp.float32, minval=-weight_variance, maxval=weight_variance))
            bias_variance = jnp.sqrt(1 / in_dim)
            biases = nnx.Param(random.uniform(bias_key, (int(out_dim),), jnp.float32, minval=-bias_variance, maxval=bias_variance))
            self.synthesis_params.append((weights, biases))

        self.synthesis_act_func = globals()[self.synthesis_NN_settings["activation_function"]]
        self.synthesis_act_func_multiplier = 1.0
        if self.synthesis_NN_settings["activation_function"]=="sin":
            self.synthesis_act_func_multiplier = self.synthesis_NN_settings["omega"]

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

        # case 1: when all layers and neurons of modulator and synthesiser networks are coupled
        if self.coupling_settings["modulation_to_synthesis_coupling_mode"]=="all_to_all":
            for i in range(len(self.modulator_params)):
                (w_modul, b_modul) = self.modulator_params[i]
                (w_synth, b_synth) = self.synthesis_params[i]
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
                x_synth = self.synthesis_act_func(self.synthesis_act_func_multiplier * (x_synth @ w_synth + b_synth + x_modul))

        # case 2: when last layer's neurons of modulator are coupled to all synthesiser networks layers and neurons 
        elif self.coupling_settings["modulation_to_synthesis_coupling_mode"]=="last_to_all":
            # first fw propagate the modulation
            for i in range(len(self.modulator_params)):
                (w_modul, b_modul) = self.modulator_params[i]
                if self.modulator_skip_connections and i>0:
                    x_modul_skipped = jnp.hstack((x_modul,x_modul_init.copy()))
                    x_modul = self.modulator_act_func(x_modul_skipped @ w_modul + b_modul)
                else:
                    x_modul = self.modulator_act_func(x_modul @ w_modul + b_modul)

            # then fw propagate the synthesis
            for (w_synth, b_synth) in self.synthesis_params[:-1]:
                x_synth = self.synthesis_act_func(self.synthesis_act_func_multiplier * (x_synth @ w_synth + b_synth + x_modul))

        # final fw propagate of the synthesis
        final_w_synth, final_b_synth = self.synthesis_params[-1]
        return x_synth @ final_w_synth + final_b_synth