"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: November, 2024
 License: FOL/LICENSE
"""
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
    def __init__(self,input_size:int=0,
                      output_size: int=0, 
                      hidden_layers:list=[],
                      activation_settings:dict={},
                      use_bias:bool=True,
                      fully_connected_layers:bool=True,
                      skip_connections_settings:dict={}):

        self.in_features=input_size
        self.out_features=output_size
        self.hidden_layers = hidden_layers
        self.fully_connected_layers = fully_connected_layers
        self.skip_connections_settings = skip_connections_settings

        default_activation_settings={"type":"sin",
                                    "prediction_gain":30,
                                    "initialization_gain":1}
        activation_settings = UpdateDefaultDict(default_activation_settings,
                                                activation_settings)
        
        default_skip_connections_settings = {"active":False,"frequency":None}
        self.skip_connections_settings = UpdateDefaultDict(default_skip_connections_settings,
                                                            self.skip_connections_settings) 

        if not self.fully_connected_layers and skip_connections_settings["active"]:
            fol_error(f"fully_connected_layers:{self.fully_connected_layers} and active skip_connections are not allowed !")

        self.NN_params = []
        layer_sizes = [self.in_features] +  hidden_layers
        if self.out_features != 0:
            layer_sizes += [self.out_features]

        activation_settings["total_num_layers"] = len(layer_sizes)
        key = random.PRNGKey(0)
        keys = random.split(key, len(layer_sizes) - 1)
        for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            activation_settings["current_layer_idx"] = i
            if self.skip_connections_settings["active"] and i>0 and \
                (i%self.skip_connections_settings["frequency"]==0):
                init_weights,init_biases = layer_init_factopry(keys[i],in_dim+self.in_features,out_dim,activation_settings)
            elif not self.fully_connected_layers:
                activation_settings["current_layer_idx"] = 0
                init_weights,init_biases = layer_init_factopry(keys[i],self.in_features,out_dim,activation_settings)
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
            self.compute_x_func = self.compute_x_skip
            self.fw_func = self.forward_skip
        else:
            self.compute_x_func = self.compute_x
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

class HyperNetwork(nnx.Module):
    """
    A neural network-based hypernetwork module that integrates a modulator network
    and a synthesizer network with configurable coupling mechanisms for forward propagation.

    Attributes:
        modulator_nn (MLP): The modulator neural network.
        synthesizer_nn (MLP): The synthesizer neural network.
        in_features (int): Number of input features for the modulator network.
        out_features (int): Number of output features for the synthesizer network.
        coupling_settings (dict): Configuration dictionary specifying the coupling mode
            and additional settings. Default settings include:
            - "shift_coupling" (bool): Whether to include shift coupling.
            - "scale_coupling" (bool): Whether to include scale coupling.
            - "modulator_to_synthesizer_coupling_mode" (str): Mode of coupling. Options:
              "all_to_all", "last_to_all", "last_to_last".
    """
    def __init__(self,modulator_nn:MLP,synthesizer_nn:MLP,coupling_settings:dict={}):

        self.modulator_nn = modulator_nn
        self.synthesizer_nn = synthesizer_nn

        self.in_features = self.modulator_nn.in_features
        self.out_features = self.synthesizer_nn.out_features
        
        self.coupling_settings = {"shift_coupling":True,
                                  "scale_coupling":False,
                                  "modulator_to_synthesizer_coupling_mode":"all_to_all"} # other coupling options: last_to_all,last_to_last                  

        self.coupling_settings = UpdateDefaultDict(self.coupling_settings,coupling_settings)

        if self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "all_to_all":
            if self.modulator_nn.hidden_layers != self.synthesizer_nn.hidden_layers:
                fol_error(f"for all_to_all modulator to synthesizer coupling, hidden layers of synthesizer and modulator NNs should be identical !")
            self.fw_func = self.all_to_all_fw
        elif self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "last_to_all":
            if not all(x==self.modulator_nn.hidden_layers[-1] for x in self.synthesizer_nn.hidden_layers):
                fol_error(f"for last_to_all modulator to synthesizer coupling, the last hidden layer of modulator NN should be equall to all synthesizer NN layers !")
            self.fw_func = self.last_to_all_fw
        elif self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "last_to_last":
            if self.modulator_nn.hidden_layers[-1]!=self.synthesizer_nn.hidden_layers[-1]:
                fol_error(f"for last_to_last modulator to synthesizer coupling, the last layer of synthesizer and modulator NNs should be identical !")
            if not self.modulator_nn.fully_connected_layers:
                fol_error(f"for last_to_last modulator to synthesizer coupling, the modulator NN should have fully connected layers !")
            self.fw_func = self.last_to_last_fw
        else:
            valid_options=["all_to_all","last_to_all","last_to_last"]
            fol_error(f"valid options for modulator_to_synthesizer_coupling_mode are {valid_options} !")

    def all_to_all_fw(self,x:jax.Array,modulator_nn:MLP,synthesizer_nn:MLP):
        """
        Implements the "all-to-all" forward propagation coupling mechanism.

        In this mode, each layer of the modulator network influences the corresponding 
        layer of the synthesizer network during forward propagation. This requires 
        that both networks have identical architectures (same number and sizes of hidden layers).

        Process:
        1. The input `x` is split into two parts:
           - `x_modul`: Input to the modulator network.
           - `x_synth`: Input to the synthesizer network.
        2. For each layer:
           - Compute the output of the modulator network (`x_modul`).
           - Compute the output of the synthesizer network (`x_synth`).
           - Add the modulator output (`x_modul`) to the synthesizer output (`x_synth`).
           - Apply the activation functions to both modulator and synthesizer outputs.
        3. At the final layer, only the synthesizer output is computed.

        Parameters:
            x (jax.Array): Input data, where the first `in_features` columns correspond 
                to the modulator network and the remaining columns to the synthesizer network.
            modulator_nn (MLP): The modulator neural network.
            synthesizer_nn (MLP): The synthesizer neural network.

        Returns:
            jax.Array: The output of the synthesizer network after applying the "all-to-all" coupling.
        """
        x_modul = x[:,0:modulator_nn.in_features]
        x_synth = x[:,modulator_nn.in_features:]

        if not modulator_nn.fully_connected_layers:
            x_modul_init = x_modul.copy()

        for i in range(len(modulator_nn.NN_params)):
            (w_modul, b_modul) = modulator_nn.NN_params[i]
            (w_synth, b_synth) = synthesizer_nn.NN_params[i]
            # compute x_modul
            if modulator_nn.fully_connected_layers:
                x_modul = modulator_nn.compute_x_func(w_modul,x_modul,b_modul)
            else:
                x_modul = modulator_nn.compute_x_func(w_modul,x_modul_init,b_modul)
            # now compute x_synth
            x_synth = synthesizer_nn.compute_x_func(w_synth,x_synth,b_synth)
            # add x_modul
            x_synth += x_modul
            # now apply modul activation
            x_modul = modulator_nn.act_func(modulator_nn.act_func_gain*x_modul)
            # now apply synth activation
            x_synth = synthesizer_nn.act_func(synthesizer_nn.act_func_gain*x_synth)

        final_w_synth, final_b_synth = synthesizer_nn.NN_params[-1]
        return synthesizer_nn.compute_x_func(final_w_synth,x_synth,final_b_synth)     

    def last_to_all_fw(self,x:jax.Array,modulator_nn:MLP,synthesizer_nn:MLP):
        """
        Implements the "last-to-all" forward propagation coupling mechanism.

        In this mode, the output of the last layer of the modulator network 
        is added to all layers of the synthesizer network during forward propagation.

        Process:
        1. The input `x` is split into two parts:
           - `x_modul`: Input to the modulator network.
           - `x_synth`: Input to the synthesizer network.
        2. The modulator network is fully propagated, producing its final output (`x_modul`).
        3. For each layer of the synthesizer network:
           - Compute the output of the synthesizer network (`x_synth`).
           - Add the final output of the modulator network (`x_modul`) to the synthesizer output.
           - Apply the activation function to the synthesizer output.
        4. At the final layer, only the synthesizer output is computed.

        Parameters:
            x (jax.Array): Input data, where the first `in_features` columns correspond 
                to the modulator network and the remaining columns to the synthesizer network.
            modulator_nn (MLP): The modulator neural network.
            synthesizer_nn (MLP): The synthesizer neural network.

        Returns:
            jax.Array: The output of the synthesizer network after applying the "last-to-all" coupling.
        """
        x_modul = x[:,0:modulator_nn.in_features]
        x_synth = x[:,modulator_nn.in_features:]

        # first modulator fw
        x_modul = modulator_nn(x_modul)

        for i in range(len(synthesizer_nn.NN_params)-1):
            (w_synth, b_synth) = synthesizer_nn.NN_params[i]
            # now compute x_synth
            x_synth = synthesizer_nn.compute_x_func(w_synth,x_synth,b_synth)
            # add x_modul
            x_synth += x_modul
            # now apply synth activation
            x_synth = synthesizer_nn.act_func(synthesizer_nn.act_func_gain*x_synth)

        final_w_synth, final_b_synth = synthesizer_nn.NN_params[-1]
        return synthesizer_nn.compute_x_func(final_w_synth,x_synth,final_b_synth)    
    
    def last_to_last_fw(self,x:jax.Array,modulator_nn:MLP,synthesizer_nn:MLP):
        """
        Implements the "last-to-last" forward propagation coupling mechanism.

        In this mode, only the final outputs of the modulator and synthesizer networks are coupled.
        This configuration is useful when the modulator's output acts as a direct control 
        or influence on the synthesizer's final output.

        Process:
        1. The input `x` is split into two parts:
           - `x_modul`: Input to the modulator network.
           - `x_synth`: Input to the synthesizer network.
        2. The modulator network is fully propagated, producing its final output (`x_modul`).
        3. The synthesizer network is propagated layer by layer without modification until the last layer.
        4. At the final layer:
           - Add the final output of the modulator network (`x_modul`) to the synthesizer output.
           - Compute the final synthesizer output.

        Parameters:
            x (jax.Array): Input data, where the first `in_features` columns correspond 
                to the modulator network and the remaining columns to the synthesizer network.
            modulator_nn (MLP): The modulator neural network.
            synthesizer_nn (MLP): The synthesizer neural network.

        Returns:
            jax.Array: The output of the synthesizer network after applying the "last-to-last" coupling.
        """
        x_modul = x[:,0:modulator_nn.in_features]
        x_synth = x[:,modulator_nn.in_features:]

        # first modulator fw
        x_modul = modulator_nn(x_modul)

        for i in range(len(synthesizer_nn.NN_params)-1):
            (w_synth, b_synth) = synthesizer_nn.NN_params[i]
            # now compute x_synth
            x_synth = synthesizer_nn.compute_x_func(w_synth,x_synth,b_synth)
            # now apply synth activation
            x_synth = synthesizer_nn.act_func(synthesizer_nn.act_func_gain*x_synth)

        # add x_modul
        x_synth += x_modul

        # final layer
        final_w_synth, final_b_synth = synthesizer_nn.NN_params[-1]
        return synthesizer_nn.compute_x_func(final_w_synth,x_synth,final_b_synth) 

    def __call__(self, x: jax.Array):
        return self.fw_func(x,self.modulator_nn,self.synthesizer_nn)