from flax import nnx
from jax.nn import relu,sigmoid,swish,tanh,leaky_relu,elu
from jax.numpy import sin
import jax
import jax.numpy as jnp
from jax._src.typing import Array, ArrayLike
from jax import random
from fol.tools.usefull_functions import *
from fol.tools.decoration_functions import *

def layer_init_factopry(key:Array,
                        in_dim:int,
                        out_dim:int,
                        activation_settings:dict):
    
    """
    Initializes weights and biases for a layer based on activation settings.

    Args:
        key (Array): PRNG key for random initialization.
        in_dim (int): Number of input features.
        out_dim (int): Number of output features.
        activation_settings (dict): Dictionary containing activation configuration, including type.

    Returns:
        Tuple[Array, Array]: Initialized weights and biases.
    """

    if activation_settings["type"]=="sin":
        return siren_init(key,in_dim,out_dim,activation_settings)
    else:
        if activation_settings["type"] in ["relu","leaky_relu","elu"]:
            init_weights = nnx.initializers.he_uniform()(key,(in_dim,out_dim))
        elif activation_settings["type"] == "tanh":
            init_weights = nnx.initializers.glorot_uniform()(key,(in_dim,out_dim))
        else:
            init_weights = nnx.initializers.lecun_uniform()(key,(in_dim,out_dim))
        init_biases = nnx.initializers.zeros(key,(out_dim,))

        return init_weights,init_biases

def siren_init(key:Array,in_dim:int,out_dim:int,activation_settings:dict):

    """
    Custom initialization method for SIREN layers.

    Args:
        key (Array): PRNG key for random initialization.
        in_dim (int): Number of input features.
        out_dim (int): Number of output features.
        activation_settings (dict): Dictionary containing SIREN-specific initialization parameters:
            - "current_layer_idx": Index of the current layer.
            - "total_num_layers": Total number of layers.
            - "initialization_gain": Weight scale for initialization.
            - "prediction_gain": Omega factor for SIREN layers.

    Returns:
        Tuple[Array, Array]: Initialized weights and biases.
    """

    weight_key, bias_key = random.split(key)
    current_layer_idx = activation_settings["current_layer_idx"]
    total_num_layers = activation_settings["total_num_layers"]
    weight_scale = activation_settings["initialization_gain"]
    omega = activation_settings["prediction_gain"]

    if current_layer_idx == 0: weight_variance = weight_scale / in_dim
    elif current_layer_idx == total_num_layers-2: weight_variance = jnp.sqrt(6 / in_dim) / omega
    else: weight_variance = weight_scale * jnp.sqrt(6 / in_dim) / omega
    
    init_weights = random.uniform(weight_key, (in_dim, out_dim), jnp.float32, minval=-weight_variance, maxval=weight_variance)
    bias_variance = jnp.sqrt(1 / in_dim)
    init_biases = random.uniform(bias_key, (int(out_dim),), jnp.float32, minval=-bias_variance, maxval=bias_variance)
    return init_weights,init_biases

class MLP(nnx.Module):
    """
    A multi-layer perceptron (MLP) with customizable activation functions, skip connections, 
    and initialization strategies.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        hidden_layers (list): List of integers specifying the number of units in each hidden layer.
        activation_settings (dict, optional): Settings for activation functions. Defaults to:
            - "type": Activation type (e.g., "sin", "relu", etc.).
            - "prediction_gain": Gain for scaling activations (default 30 for "sin").
            - "initialization_gain": Gain for weight initialization (default 1).
        use_bias (bool, optional): Whether to include biases in the layers. Defaults to True.
        skip_connections_settings (dict, optional): Settings for skip connections. Defaults to:
            - "active": Whether to enable skip connections.
            - "frequency": Frequency of skip connections (in layers).

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        NN_params (list): List of layer parameters (weights and biases).
        act_func (Callable): Activation function.
        act_func_gain (float): Gain applied to activations.
        fw_func (Callable): Forward pass method (with or without skip connections).
    """
    def __init__(self,input_size:int,
                    output_size: int, 
                    hidden_layers:list,
                    activation_settings:dict={},
                    use_bias:bool=True,
                    skip_connections_settings:dict={}):

        self.in_features=input_size
        self.out_features=output_size
        self.skip_connections_settings = skip_connections_settings

        default_activation_settings={"type":"sin",
                                    "prediction_gain":30,
                                    "initialization_gain":1}
        activation_settings = UpdateDefaultDict(default_activation_settings,
                                                activation_settings)
        
        default_skip_connections_settings = {"active":False,"frequency":None}
        self.skip_connections_settings = UpdateDefaultDict(default_skip_connections_settings,
                                                            self.skip_connections_settings)   
        
        self.NN_params = []
        layer_sizes = [self.in_features] +  hidden_layers + [self.out_features]
        activation_settings["total_num_layers"] = len(layer_sizes)
        key = random.PRNGKey(0)
        keys = random.split(key, len(layer_sizes) - 1)
        for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            activation_settings["current_layer_idx"] = i
            if self.skip_connections_settings["active"] and i>0 and \
                (i%self.skip_connections_settings["frequency"]==0):
                init_weights,init_biases = layer_init_factopry(keys[i],in_dim+self.in_features,out_dim,activation_settings)
            else:
                init_weights,init_biases = layer_init_factopry(keys[i],in_dim,out_dim,activation_settings)
            if use_bias:
                self.NN_params.append((nnx.Param(init_weights),nnx.Param(init_biases)))
            else:
                self.NN_params.append((nnx.Param(init_weights),jnp.zeros(init_biases.shape)))

        act_name = activation_settings["type"]
        self.act_func = globals()[act_name]
        if act_name=="sin":
            self.act_func_gain = activation_settings["prediction_gain"]
        else:
            self.act_func_gain = 1 
        
        if self.skip_connections_settings["active"]:
            self.fw_func = self.forward_skip
        else:
            self.fw_func = self.forward

    def compute_x(self,w:nnx.Param,prev_x:jax.Array,b:nnx.Param):
        """
        Computes the output of a layer without skip connections.

        Args:
            w (nnx.Param): Weight matrix.
            prev_x (jax.Array): Input to the layer.
            b (nnx.Param): Bias vector.

        Returns:
            jax.Array: Output of the layer.
        """
        return prev_x @ w + b
    
    def forward(self,x: jax.Array,nn_params:list[tuple[nnx.Param, nnx.Param]]):
        """
        Forward pass through the MLP without skip connections.

        Args:
            x (jax.Array): Input to the network.
            nn_params (list[tuple[nnx.Param, nnx.Param]]): List of layer parameters (weights and biases).

        Returns:
            jax.Array: Output of the network.
        """
        for (w, b) in nn_params[:-1]:
            x = self.compute_x(w,x,b)
            x = self.act_func(self.act_func_gain*x)
        final_w, final_b = nn_params[-1]
        return self.compute_x(final_w,x,final_b)
    
    def compute_x_skip(self,w:nnx.Param,prev_x:jax.Array,in_x:jax.Array,b:nnx.Param):
        """
        Computes the output of a layer with skip connections.

        Args:
            w (nnx.Param): Weight matrix.
            prev_x (jax.Array): Input to the current layer.
            in_x (jax.Array): Original input to the network for skip connection.
            b (nnx.Param): Bias vector.

        Returns:
            jax.Array: Output of the layer.
        """
        return jnp.hstack((prev_x,in_x.copy())) @ w + b
    
    def forward_skip(self,x:jax.Array,nn_params:list[tuple[nnx.Param, nnx.Param]]):
        """
        Forward pass through the MLP with skip connections.

        Args:
            x (jax.Array): Input to the network.
            nn_params (list[tuple[nnx.Param, nnx.Param]]): List of layer parameters (weights and biases).

        Returns:
            jax.Array: Output of the network.
        """
        in_x = x.copy()
        layer_num = 0
        for (w, b) in nn_params[0:-1]:
            if layer_num>0 and layer_num%self.skip_connections_settings["frequency"]==0:
                x = self.compute_x_skip(w,x,in_x,b)
            else:
                x = self.compute_x(w,x,b)
            x = self.act_func(self.act_func_gain*x)
            layer_num += 1

        final_w, final_b = nn_params[-1]

        if layer_num%self.skip_connections_settings["frequency"]==0:
            return self.compute_x_skip(final_w,x,in_x,final_b)
        else:
            return self.compute_x(final_w,x,final_b)

    def __call__(self, x: jax.Array):
        """
        Perform a forward pass through the network using the configured forward method.

        Args:
            x (jax.Array): Input to the network.

        Returns:
            jax.Array: Output of the network.
        """
        return self.fw_func(x,self.NN_params)

  
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
        omega = self.synthesizer_NN_settings["omega"]
        key = random.PRNGKey(0)
        keys = random.split(key, len(layers) - 1)
        self.synthesizer_params = []
        for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            if i==0:layer_type="input"
            elif i==len(layers)-2:layer_type="output"
            else:layer_type="hidden"
            self.synthesizer_params.append(siren_init(keys[i],layer_type,in_dim,out_dim,omega,weight_scale))

        self.synthesizer_act_func = globals()[self.synthesizer_NN_settings["activation_function"]]
        self.synthesizer_act_func_multiplier = 1.0
        if self.synthesizer_NN_settings["activation_function"]=="sin":
            self.synthesizer_act_func_multiplier = omega

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

        if self.modulator_skip_connections or \
            not self.modulator_NN_settings["fully_connected_layers"]:
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