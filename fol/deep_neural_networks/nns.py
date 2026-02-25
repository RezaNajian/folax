"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: November, 2024
 License: FOL/LICENSE
"""
import copy
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
    Initialize a layer weight matrix and bias vector.

    This helper selects an initialization scheme based on the requested
    activation type. For sinusoidal activations it delegates to :func:`siren_init`.
    For other activations it chooses a standard initializer commonly used for
    stable training.

    Args:
        key (Array):
            PRNG key used to sample random initial parameters.
        in_dim (int):
            Input feature dimension of the layer.
        out_dim (int):
            Output feature dimension of the layer.
        activation_settings (dict):
            Activation configuration dictionary. Must include the key ``"type"``.
            For ``"type" == "sin"``, the settings must also include the SIREN
            parameters required by :func:`siren_init`.

    Returns:
        Tuple[Array, Array]:
            A tuple ``(weights, biases)`` where ``weights`` has shape
            ``(in_dim, out_dim)`` and ``biases`` has shape ``(out_dim,)``.

    Raises:
        KeyError:
            If ``"type"`` is missing from ``activation_settings`` or if required
            SIREN keys are missing when ``type == "sin"``.
        ValueError:
            If ``in_dim`` or ``out_dim`` is not positive.
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
    Initialize weights and biases for a SIREN (sinusoidal) layer.

    This initializer is designed for sinusoidal representation networks and
    chooses a uniform distribution bound that depends on the layer index and the
    SIREN frequency parameter (omega). The behavior matches common SIREN
    initialization practice and supports gain scaling strategies.

    References:
        Sitzmann et al. (2020), "Implicit neural representations with periodic
        activation functions", NeurIPS.
        Yeom et al. (2024), "Fast Training of Sinusoidal Neural Fields via
        Scaling Initialization", arXiv:2410.04779.

    Args:
        key (Array):
            PRNG key used to sample random initial parameters.
        in_dim (int):
            Input feature dimension of the layer.
        out_dim (int):
            Output feature dimension of the layer.
        activation_settings (dict):
            Dictionary containing SIREN initialization parameters. Required keys
            are ``"current_layer_idx"``, ``"total_num_layers"``,
            ``"initialization_gain"``, and ``"prediction_gain"``.

    Returns:
        Tuple[Array, Array]:
            A tuple ``(weights, biases)`` where ``weights`` has shape
            ``(in_dim, out_dim)`` and ``biases`` has shape ``(out_dim,)``.

    Raises:
        KeyError:
            If any required key is missing from ``activation_settings``.
        ValueError:
            If ``in_dim`` or ``out_dim`` is not positive, or if
            ``total_num_layers`` is inconsistent with ``current_layer_idx``.
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
    init_biases = jnp.zeros(out_dim)
    return init_weights,init_biases

class MLP(nnx.Module):
    """
    General-purpose multi-layer perceptron (MLP) building block.

    This module can be used to construct a wide range of MLP-based architectures,
    including neural fields, DeepONets, and standard feed-forward networks. It
    supports a variety of activation functions such as ``"relu"``, ``"leaky_relu"``,
    ``"elu"``, and ``"tanh"``, with a sinusoidal activation (SIREN-style ``"sin"``)
    as the default option. The sinusoidal activation is particularly suited for
    neural fields and implicit representations that must capture high-frequency
    behavior.

    In addition to standard dense layers, the MLP can:

    1. Use optional skip connections with a configurable frequency to improve
       gradient flow and expressivity.
    2. Apply Fourier feature mappings at the input layer, optionally with
        learnable Fourier frequencies, to better represent high-frequency
        signals in coordinate-based models.

    These capabilities make the class suitable as a reusable backbone for
    conditional neural fields, operator-learning networks such as DeepONets,
    and other coordinate-based models.

    Args:
        name (str):
            Name identifier for the network instance.
        input_size (int, optional):
            Number of input features. Default is ``0``.
        output_size (int, optional):
            Number of output features. If ``0``, the network can be configured as
            a feature extractor without a final output layer. Default is ``0``.
        hidden_layers (list, optional):
            Hidden layer widths. Default is ``[]``.
        activation_settings (dict, optional):
            Activation configuration dictionary. Common keys include ``"type"``,
            ``"prediction_gain"``, and ``"initialization_gain"``. Missing entries
            are filled using internal defaults. Default is ``{}``.
        use_bias (bool, optional):
            If ``True``, create trainable biases. If ``False``, biases are fixed
            zeros. Default is ``True``.
        skip_connections_settings (dict, optional):
            Skip connection configuration dictionary. Expected keys are
            ``"active"`` and ``"frequency"``. Missing entries are filled using
            internal defaults. Default is ``{}``.
        fourier_feature_settings (dict, optional):
            Fourier feature mapping configuration dictionary. Expected keys are
            ``"active"``, ``"type"``, ``"size"``, ``"frequency_scale"``, and
            ``"learn_frequency"``. Missing entries are filled using internal
            defaults. Default is ``{}``.

    Raises:
        KeyError:
            If required activation configuration keys are missing after defaults
            are applied.
        ValueError:
            If any provided dimension (input, output, hidden) is negative or if
            Fourier feature settings are inconsistent.
    """
    @print_with_timestamp_and_execution_time
    def __init__(self,name:str,
                      input_size:int=0,
                      output_size: int=0,
                      hidden_layers:list=[],
                      activation_settings:dict={},
                      use_bias:bool=True,
                      skip_connections_settings:dict={},
                      fourier_feature_settings:dict={}):
        self.name = name
        self.in_features=input_size
        self.out_features=output_size
        self.hidden_layers = hidden_layers
        self.activation_settings = activation_settings
        self.use_bias = use_bias
        self.skip_connections_settings = skip_connections_settings
        self.fourier_feature_settings = fourier_feature_settings

        default_activation_settings={"type":"sin",
                                    "prediction_gain":30,
                                    "initialization_gain":1}
        self.activation_settings = UpdateDefaultDict(default_activation_settings,
                                                     self.activation_settings)

        default_skip_connections_settings = {"active":False,"frequency":1}
        self.skip_connections_settings = UpdateDefaultDict(default_skip_connections_settings,
                                                            self.skip_connections_settings)

        default_fourier_feature_settings = {"active":False,
                                            "type":"Gaussian",
                                            "size":int(self.in_features),
                                            "frequency_scale":1.0,
                                            "learn_frequency":False}
        self.fourier_feature_settings = UpdateDefaultDict(default_fourier_feature_settings,
                                                            self.fourier_feature_settings)

        self.InitialNetworkParameters()

        if self.fourier_feature_settings["active"] and self.fourier_feature_settings["learn_frequency"]:
            fol_info(f"MLP network is initialized by {self.total_num_weights} weights and {self.total_num_biases} biases, {self.B.size} learnable fourier feature frequencies !")
        else:
            fol_info(f"MLP network is initialized by {self.total_num_weights} weights and {self.total_num_biases} biases !")

        act_name = self.activation_settings["type"]
        self.act_func = globals()[act_name]
        if act_name=="sin":
            self.act_func_gain = self.activation_settings["prediction_gain"]
        else:
            self.act_func_gain = 1

        if self.skip_connections_settings["active"]:
            self.fw_func = self.ForwardSkip
        else:
            self.fw_func = self.Forward

    def InitialNetworkParameters(self):
        """
        Initialize network parameters and optional Fourier feature mapping.

        This method constructs the layer size list, initializes each layer weight
        matrix and bias vector using :func:`layer_init_factopry`, and stores the
        parameters in ``self.nn_params``. When Fourier features are enabled, it also
        creates the frequency matrix ``B`` and defines ``self.input_mapping`` to map
        the input coordinates to sinusoidal features.

        The initialization accounts for skip connections by increasing the input
        dimension of layers that receive concatenated inputs.

        Args:
            None

        Returns:
            None

        Raises:
            KeyError:
                If required entries are missing from ``activation_settings`` or
                ``fourier_feature_settings``.
            ValueError:
                If hidden layer sizes are invalid or if Fourier feature dimensions
                are inconsistent with ``input_size``.
        """
        layer_sizes = [2*self.fourier_feature_settings["size"] if self.fourier_feature_settings["active"] else self.in_features] + self.hidden_layers

        if self.out_features != 0:
            layer_sizes += [self.out_features]

        activation_settings = copy.deepcopy(self.activation_settings)
        activation_settings["total_num_layers"] = len(layer_sizes)

        key = random.PRNGKey(0)
        keys = random.split(key, len(layer_sizes) - 1)

        self.nn_params = nnx.List([])
        self.total_num_weights = 0
        self.total_num_biases = 0
        for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            activation_settings["current_layer_idx"] = i
            if self.skip_connections_settings["active"] and i>0 and \
                (i%self.skip_connections_settings["frequency"]==0):
                init_weights,init_biases = layer_init_factopry(keys[i],in_dim+self.in_features,out_dim,activation_settings)
            else:
                init_weights,init_biases = layer_init_factopry(keys[i],in_dim,out_dim,activation_settings)

            self.total_num_weights += init_weights.size
            if self.use_bias:
                self.nn_params.append((nnx.Param(init_weights),nnx.Param(init_biases)))
                self.total_num_biases += init_biases.size
            else:
                self.nn_params.append((nnx.Param(init_weights),nnx.Variable(jnp.zeros_like(init_biases))))

        # # now compute B matrix for fourier_feature_mapping
        if self.fourier_feature_settings["active"]:
            rand_key = random.PRNGKey(43)
            B_initial_values = float(self.fourier_feature_settings["frequency_scale"]) * random.normal(rand_key, (self.fourier_feature_settings["size"], self.in_features))
            if self.fourier_feature_settings["learn_frequency"]:
                self.B = nnx.Param(B_initial_values)
            else:
                self.B = nnx.Variable(B_initial_values)
            self.input_mapping = lambda x: jnp.concatenate([jnp.sin((2.*np.pi*x) @ self.B.T), jnp.cos((2.*np.pi*x) @ self.B.T)],axis=-1)
        else:
            self.input_mapping = lambda x: x

    def GetName(self):
        """
        Return the name identifier of this MLP instance.

        Args:
            None

        Returns:
            str:
                Name of the network instance.

        Raises:
            None
        """
        return self.name

    def CountTrainableParams(self):
        """
        Count trainable parameters registered as ``nnx.Param`` in this module.

        This method extracts the module parameter state and sums the number of
        scalar entries across all leaves to compute the total number of trainable
        parameters.

        Args:
            None

        Returns:
            int:
                Total number of trainable scalar parameters.

        Raises:
            None
        """
        params = nnx.state(self, nnx.Param)
        return  sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))

    def ComputeX(self,w:nnx.Param,prev_x:jax.Array,b:nnx.Param):
        """
        Compute an affine layer output without skip connections.

        Args:
            w (nnx.Param):
                Weight matrix for the current layer.
            prev_x (jax.Array):
                Input activations to the current layer.
            b (nnx.Param):
                Bias vector for the current layer.

        Returns:
            jax.Array:
                Layer pre-activation output computed as ``prev_x @ w + b``.

        Raises:
            ValueError:
                If the input dimensions are incompatible for matrix multiplication.
        """
        return prev_x @ w + b

    def Forward(self,x: jax.Array,nn_params:list[tuple[nnx.Param, nnx.Param]]):
        """
        Forward pass through the MLP without skip connections.

        For each hidden layer, this method applies an affine transform followed by
        the configured activation function (and gain if applicable). The final layer
        is applied as a linear transform without an activation.

        Args:
            x (jax.Array):
                Input array to the network.
            nn_params (list[tuple[nnx.Param, nnx.Param]]):
                Sequence of ``(weights, biases)`` for each layer.

        Returns:
            jax.Array:
                Network output after applying all layers.

        Raises:
            ValueError:
                If parameter shapes do not match the input activation shapes.
        """
        for (w, b) in nn_params[:-1]:
            x = self.ComputeX(w,x,b)
            x = self.act_func(self.act_func_gain*x)
        final_w, final_b = nn_params[-1]
        return self.ComputeX(final_w,x,final_b)

    def ComputeXSkip(self,w:nnx.Param,prev_x:jax.Array,in_x:jax.Array,b:nnx.Param):
        """
        Compute an affine layer output with a skip connection.

        This method concatenates the current activation ``prev_x`` with the original
        network input ``in_x`` along the feature axis and applies an affine transform.

        Args:
            w (nnx.Param):
                Weight matrix for the current layer.
            prev_x (jax.Array):
                Current layer activations.
            in_x (jax.Array):
                Original input to the network that is injected via a skip path.
            b (nnx.Param):
                Bias vector for the current layer.

        Returns:
            jax.Array:
                Layer pre-activation output computed from the concatenated input.

        Raises:
            ValueError:
                If concatenation or matrix multiplication dimensions are incompatible.
        """
        return jnp.hstack((prev_x,in_x.copy())) @ w + b

    def ForwardSkip(self,x:jax.Array,nn_params:list[tuple[nnx.Param, nnx.Param]]):
        """
        Forward pass through the MLP with periodic skip connections.

        Skip connections are applied by concatenating the original input to the
        network with the current activation every ``skip_connections_settings["frequency"]``
        layers (after the first layer). Hidden layers apply the configured activation,
        while the last layer remains linear.

        Args:
            x (jax.Array):
                Input array to the network (before Fourier mapping if enabled).
            nn_params (list[tuple[nnx.Param, nnx.Param]]):
                Sequence of ``(weights, biases)`` for each layer.

        Returns:
            jax.Array:
                Network output after applying all layers with skip connections.

        Raises:
            KeyError:
                If ``"frequency"`` is missing from ``skip_connections_settings``.
            ValueError:
                If parameter shapes do not match the concatenated activation shapes.
        """
        in_x = x.copy()
        layer_num = 0
        for (w, b) in nn_params[0:-1]:
            if layer_num>0 and layer_num%self.skip_connections_settings["frequency"]==0:
                x = self.ComputeXSkip(w,x,in_x,b)
            else:
                x = self.ComputeX(w,x,b)
            x = self.act_func(self.act_func_gain*x)
            layer_num += 1

        final_w, final_b = nn_params[-1]

        if layer_num%self.skip_connections_settings["frequency"]==0:
            return self.ComputeXSkip(final_w,x,in_x,final_b)
        else:
            return self.ComputeX(final_w,x,final_b)

    def __call__(self, x: jax.Array):
        """
        Evaluate the MLP on an input array.

        This method first applies the configured input mapping. If Fourier features
        are enabled, the input is mapped to sinusoidal features; otherwise it is
        passed through unchanged. The mapped input is then propagated using the
        selected forward function (with or without skip connections).

        Args:
            x (jax.Array):
                Input array to evaluate.

        Returns:
            jax.Array:
                Network output.

        Raises:
            ValueError:
                If the input mapping or forward pass encounters incompatible shapes.
        """
        return self.fw_func(self.input_mapping(x),self.nn_params)

class HyperNetwork(nnx.Module):
    """
    Hypernetwork that modulates a synthesizer MLP using a modulator MLP.

    The hypernetwork couples a modulator network to a synthesizer network and
    applies modulation during the synthesizer forward pass. In this
    implementation, the supported coupled variable is a bias-like shift that is
    added to synthesizer layer activations.

    Coupling modes supported by ``coupling_settings["modulator_to_synthesizer_coupling_mode"]`` are:

    1. ``"all_to_all"``: layer-wise coupling between modulator and synthesizer.
    2. ``"last_to_all"``: modulator final output is injected into every synthesizer layer.
    3. ``"one_modulator_per_synthesizer_layer"``: a separate modulator is created per synthesizer layer.

    Args:
        name (str):
            Name identifier for the hypernetwork instance.
        modulator_nn (MLP):
            Modulator network producing conditioning signals.
        synthesizer_nn (MLP):
            Synthesizer network producing task outputs.
        coupling_settings (dict, optional):
            Coupling configuration dictionary. Common keys are
            ``"coupled_variable"`` and ``"modulator_to_synthesizer_coupling_mode"``.
            Default is ``{}``.

    Raises:
        KeyError:
            If required entries are missing from ``coupling_settings``.
        ValueError:
            If an unsupported coupling mode is requested or if network shapes
            are inconsistent with the chosen coupling mode.
    """
    @print_with_timestamp_and_execution_time
    def __init__(self,name:str,
                      modulator_nn:MLP,
                      synthesizer_nn:MLP,
                      coupling_settings:dict={}):
        self.name = name
        self.modulator_nn = modulator_nn
        self.synthesizer_nn = synthesizer_nn

        self.in_features = self.modulator_nn.in_features
        self.out_features = self.synthesizer_nn.out_features

        self.coupling_settings = {"coupled_variable":"shift",
                                  "modulator_to_synthesizer_coupling_mode":"all_to_all"} # other coupling options: last_to_all,last_to_last

        self.coupling_settings = UpdateDefaultDict(self.coupling_settings,coupling_settings)

        if self.coupling_settings["coupled_variable"] != "shift":
            coupled_variable = self.coupling_settings["coupled_variable"]
            fol_error(f"coupled_variable {coupled_variable} is not supported, options are shift")

        if self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "all_to_all":
            if self.modulator_nn.hidden_layers != self.synthesizer_nn.hidden_layers:
                fol_error(f"for all_to_all modulator to synthesizer coupling, hidden layers of synthesizer and modulator NNs should be identical !")
            self.fw_func = self.all_to_all_fw
            self.total_num_weights = self.modulator_nn.total_num_weights + \
                                    self.synthesizer_nn.total_num_weights
            self.total_num_biases = self.modulator_nn.total_num_biases +\
                                    self.synthesizer_nn.total_num_biases
        elif self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "last_to_all":
            synthesizer_modulated_biases = self.synthesizer_nn.total_num_biases
            synthesizer_modulated_biases -= self.synthesizer_nn.out_features # subtract the last linear layer
            modulator_original_out_features = self.modulator_nn.out_features

            self.modulator_nn.out_features = synthesizer_modulated_biases
            fol_info(f" the out_features of modulator network is changed from {modulator_original_out_features} to \
                        the total number of the modulated biases of the synthesizer network {synthesizer_modulated_biases}")

            self.modulator_nn.InitialNetworkParameters()
            fol_info(f"the modulator network is re-initialized by {self.modulator_nn.total_num_weights} weights and {self.modulator_nn.total_num_biases} biases !")
            self.fw_func = self.last_to_all_fw
            self.total_num_weights = self.modulator_nn.total_num_weights + \
                                    self.synthesizer_nn.total_num_weights
            self.total_num_biases = self.modulator_nn.total_num_biases + \
                                    self.synthesizer_nn.total_num_biases
        elif self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "one_modulator_per_synthesizer_layer":
            self.total_num_biases = self.synthesizer_nn.total_num_biases
            self.total_num_weights = self.synthesizer_nn.total_num_weights
            self.modulator_nns = nnx.List([])
            for i in range(len(self.synthesizer_nn.hidden_layers)):
                synthesizer_layer_biases = self.synthesizer_nn.hidden_layers[i]
                synthesizer_layer_modulator = MLP(name=f"synthesizer_layer_{i}_modulator",
                                                  input_size=self.modulator_nn.in_features,
                                                  output_size=synthesizer_layer_biases,
                                                  hidden_layers=self.modulator_nn.hidden_layers,
                                                  activation_settings=self.modulator_nn.activation_settings,
                                                  use_bias=self.modulator_nn.use_bias,
                                                  skip_connections_settings=self.modulator_nn.skip_connections_settings,
                                                  fourier_feature_settings=self.modulator_nn.fourier_feature_settings)
                self.modulator_nns.append(synthesizer_layer_modulator)
                self.total_num_biases += synthesizer_layer_modulator.total_num_biases
                self.total_num_weights += synthesizer_layer_modulator.total_num_weights

            fol_info(f" created {len(self.synthesizer_nn.hidden_layers)} modulators, i.e., one modulator per synthesizer layer !")
            # delete modulator_nn
            del self.modulator_nn
            # set fw function
            self.fw_func = self.one_modulator_per_synthesizer_layer_fw
        else:
            valid_options=["all_to_all","last_to_all","one_modulator_per_synthesizer_layer"]
            fol_error(f"valid options for modulator_to_synthesizer_coupling_mode are {valid_options} !")

        fol_info(f"hyper network has {self.total_num_weights} weights and {self.total_num_biases} biases !")

    def GetName(self):
        """
        Return the name identifier of this hypernetwork instance.

        Args:
            None

        Returns:
            str:
                Name of the hypernetwork instance.

        Raises:
            None
        """
        return self.name

    def CountTrainableParams(self):
        """
        Count trainable parameters registered as ``nnx.Param`` in this module.

        Args:
            None

        Returns:
            int:
                Total number of trainable scalar parameters.

        Raises:
            None
        """
        params = nnx.state(self, nnx.Param)
        return  sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))

    def all_to_all_fw(self,latent_array:jax.Array,coord_matrix:jax.Array,
                            modulator_nn:MLP,synthesizer_nn:MLP):
        """
        Forward pass for the ``"all_to_all"`` coupling mode.

        In this mode, the modulator and synthesizer networks are assumed to have
        identical layer structures. At each layer, the modulator activation is
        added to the synthesizer activation (shift coupling), and both networks
        apply their respective activation functions.

        The computation proceeds as follows.

        Step 1:
            Reshape ``latent_array`` to a row-vector input for the modulator
            network and initialize synthesizer activations using
            ``coord_matrix``.

        Step 2:
            For each coupled layer, compute the modulator pre-activation (with
            optional skip connections), compute the synthesizer pre-activation
            (with optional skip connections), add the modulator output to the
            synthesizer activations, and apply the activation functions for
            both networks.

        Step 3:
            Apply the final synthesizer linear layer to the last synthesizer
            activations and return the resulting output.

        Args:
            latent_array (jax.Array):
                Conditioning input for the modulator network.
            coord_matrix (jax.Array):
                Coordinate or feature input for the synthesizer network.
            modulator_nn (MLP):
                Modulator network providing per-layer conditioning.
            synthesizer_nn (MLP):
                Synthesizer network producing the final output.

        Returns:
            jax.Array:
                Output of the synthesizer network after applying all-to-all
                coupling.

        Raises:
            ValueError:
                If the modulator and synthesizer architectures are incompatible
                for layer-wise coupling.
        """

        x_modul = latent_array.reshape(-1,1).T
        x_synth = coord_matrix

        if modulator_nn.skip_connections_settings["active"]:
            x_modul_init = x_modul.copy()

        if synthesizer_nn.skip_connections_settings["active"]:
            x_synth_init = x_synth.copy()

        layer_num = 0
        for i in range(len(modulator_nn.nn_params)):
            (w_modul, b_modul) = modulator_nn.nn_params[i]
            (w_synth, b_synth) = synthesizer_nn.nn_params[i]

            # first compute x_modul
            if layer_num>0 and modulator_nn.skip_connections_settings["active"] and layer_num%modulator_nn.skip_connections_settings["frequency"]==0:
                x_modul = modulator_nn.ComputeXSkip(w_modul,x_modul,x_modul_init,b_modul)
            else:
                x_modul = modulator_nn.ComputeX(w_modul,x_modul,b_modul)

            # now compute x_synth
            if layer_num>0 and synthesizer_nn.skip_connections_settings["active"] and layer_num%synthesizer_nn.skip_connections_settings["frequency"]==0:
                x_synth = synthesizer_nn.ComputeXSkip(w_synth,x_synth,x_synth_init,b_synth)
            else:
                x_synth = synthesizer_nn.ComputeX(w_synth,x_synth,b_synth)

            # add x_modul
            x_synth = jax.vmap(lambda row: row + x_modul.reshape(-1))(x_synth)

            # now apply modul activation
            x_modul = modulator_nn.act_func(modulator_nn.act_func_gain*x_modul)
            # now apply synth activation
            x_synth = synthesizer_nn.act_func(synthesizer_nn.act_func_gain*x_synth)
            # update layer index
            layer_num += 1

        final_w_synth, final_b_synth = synthesizer_nn.nn_params[-1]

        if layer_num>0 and synthesizer_nn.skip_connections_settings["active"] and layer_num%synthesizer_nn.skip_connections_settings["frequency"]==0:
            return synthesizer_nn.ComputeXSkip(final_w_synth,x_synth,x_synth_init,final_b_synth)
        else:
            return synthesizer_nn.ComputeX(final_w_synth,x_synth,final_b_synth)

    def last_to_all_fw(self,latent_array:jax.Array,coord_matrix:jax.Array,
                            modulator_nn:MLP,synthesizer_nn:MLP):
        """
        Forward pass for the ``"last_to_all"`` coupling mode.

        In this mode, the modulator is first evaluated once to produce a single
        modulation vector. That vector is then split into chunks matching the bias
        sizes of the synthesizer hidden layers, and each chunk is added to the
        corresponding synthesizer layer activations (shift coupling).

        The computation proceeds as follows.

        Step 1:
            Forward propagate ``latent_array`` through the modulator network to
            produce a global modulation vector ``x_modul``.

        Step 2:
            Initialize synthesizer activations using ``coord_matrix``.

        Step 3:
            For each synthesizer hidden layer, compute the layer pre-activation
            (with optional skip connections), take the slice of ``x_modul``
            matching that layer width, add this slice to the synthesizer
            activations, and apply the synthesizer activation function.

        Step 4:
            Apply the final synthesizer linear layer and return the output.

        Args:
            latent_array (jax.Array):
                Conditioning input for the modulator network.
            coord_matrix (jax.Array):
                Coordinate or feature input for the synthesizer network.
            modulator_nn (MLP):
                Modulator network producing a global modulation vector.
            synthesizer_nn (MLP):
                Synthesizer network producing the final output.

        Returns:
            jax.Array:
                Output of the synthesizer network after applying last-to-all
                coupling.

        Raises:
            ValueError:
                If the modulator output size does not match the total number of
                modulated synthesizer hidden biases.
        """

        # first modulator fw
        x_modul = modulator_nn(latent_array.reshape(-1,1).T).flatten()
        x_synth = coord_matrix

        if synthesizer_nn.skip_connections_settings["active"]:
            x_synth_init = x_synth.copy()

        # now synthesizer fw with shift modulation coupling
        x_modul_itr = 0
        layer_num = 0
        for i in range(len(synthesizer_nn.nn_params)-1):
            (w_synth, b_synth) = synthesizer_nn.nn_params[i]
            # get num of the hidden layer biases
            num_hidden_biases = b_synth.shape[0]
            # now compute x_synth
            if layer_num>0 and synthesizer_nn.skip_connections_settings["active"] and layer_num%synthesizer_nn.skip_connections_settings["frequency"]==0:
                x_synth = synthesizer_nn.ComputeXSkip(w_synth,x_synth,x_synth_init,b_synth)
            else:
                x_synth = synthesizer_nn.ComputeX(w_synth,x_synth,b_synth)
            # add x_modul
            this_layer_x_modul = x_modul[x_modul_itr:x_modul_itr+num_hidden_biases]
            x_synth = jax.vmap(lambda row: row + this_layer_x_modul)(x_synth)
            # now apply synth activation
            x_synth = synthesizer_nn.act_func(synthesizer_nn.act_func_gain*x_synth)
            # update x_modul_itr
            x_modul_itr += num_hidden_biases
            # update layer index
            layer_num += 1

        # now apply last linear layer
        final_w_synth, final_b_synth = synthesizer_nn.nn_params[-1]
        if layer_num>0 and synthesizer_nn.skip_connections_settings["active"] and layer_num%synthesizer_nn.skip_connections_settings["frequency"]==0:
            return synthesizer_nn.ComputeXSkip(final_w_synth,x_synth,x_synth_init,final_b_synth)
        else:
            return synthesizer_nn.ComputeX(final_w_synth,x_synth,final_b_synth)

    def one_modulator_per_synthesizer_layer_fw(self,latent_array:jax.Array,coord_matrix:jax.Array,
                                               modulator_nns:list[MLP],synthesizer_nn:MLP):
        """
        Forward pass for ``"one_modulator_per_synthesizer_layer"`` coupling.

        In this mode, each synthesizer hidden layer has a dedicated modulator network
        that is evaluated on the same ``latent_array``. The resulting modulation vector
        is added to the corresponding synthesizer layer activations (shift coupling).

        The computation proceeds as follows.

        Step 1:
            Initialize synthesizer activations using ``coord_matrix``.

        Step 2:
            For each synthesizer hidden layer, compute the synthesizer layer
            pre-activation (with optional skip connections), evaluate the corresponding
            modulator to obtain a modulation vector of matching size, add this modulation
            to the synthesizer activations, and apply the synthesizer activation
            function.

        Step 3:
            Apply the final synthesizer linear layer and return the output.

        Args:
            latent_array (jax.Array):
                Conditioning input shared across all modulators.
            coord_matrix (jax.Array):
                Coordinate or feature input for the synthesizer network.
            modulator_nns (list[MLP]):
                List of modulator networks, one per synthesizer hidden layer.
            synthesizer_nn (MLP):
                Synthesizer network producing the final output.

        Returns:
            jax.Array:
                Output of the synthesizer network after applying per-layer
                modulator coupling.

        Raises:
            ValueError:
                If the number of modulators does not match the number of synthesizer
                hidden layers.
        """
        x_synth = coord_matrix
        if synthesizer_nn.skip_connections_settings["active"]:
            x_synth_init = x_synth.copy()

        layer_num = 0
        for i in range(len(synthesizer_nn.nn_params)-1):
            (w_synth, b_synth) = synthesizer_nn.nn_params[i]
            # now compute x_synth
            if layer_num>0 and synthesizer_nn.skip_connections_settings["active"] and layer_num%synthesizer_nn.skip_connections_settings["frequency"]==0:
                x_synth = synthesizer_nn.ComputeXSkip(w_synth,x_synth,x_synth_init,b_synth)
            else:
                x_synth = synthesizer_nn.ComputeX(w_synth,x_synth,b_synth)
            # compute x_modul
            this_layer_x_modul = modulator_nns[i](latent_array.reshape(-1,1).T).flatten()
            # now add to x_synth
            x_synth = jax.vmap(lambda row: row + this_layer_x_modul)(x_synth)
            # now apply synth activation
            x_synth = synthesizer_nn.act_func(synthesizer_nn.act_func_gain*x_synth)
            # update layer index
            layer_num += 1

        # now apply last linear layer
        final_w_synth, final_b_synth = synthesizer_nn.nn_params[-1]
        if layer_num>0 and synthesizer_nn.skip_connections_settings["active"] and layer_num%synthesizer_nn.skip_connections_settings["frequency"]==0:
            return synthesizer_nn.ComputeXSkip(final_w_synth,x_synth,x_synth_init,final_b_synth)
        else:
            return synthesizer_nn.ComputeX(final_w_synth,x_synth,final_b_synth)

    def __call__(self, latent_array: jax.Array,coord_matrix: jax.Array):
        """
        Evaluate the hypernetwork for a batch of latent inputs and shared coordinates.

        This method vectorizes the selected coupling forward function over the first
        axis of ``latent_array`` using ``jax.vmap``. The coordinate input is mapped
        using the synthesizer's input mapping (e.g., Fourier features) before being
        passed into the coupling function.

        Args:
            latent_array (jax.Array):
                Batch of latent conditioning vectors. The first axis is treated as
                the batch dimension.
            coord_matrix (jax.Array):
                Coordinate/features array shared across the batch of latents.

        Returns:
            jax.Array:
                Batched synthesizer outputs, one per latent vector.

        Raises:
            KeyError:
                If the configured coupling mode is missing from ``coupling_settings``.
            ValueError:
                If inputs have incompatible shapes for the selected coupling mode.
        """
        if self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "one_modulator_per_synthesizer_layer":
            return jax.vmap(self.fw_func,in_axes=(0, None, None, None))(latent_array,self.synthesizer_nn.input_mapping(coord_matrix),self.modulator_nns,self.synthesizer_nn)
        else:
            return jax.vmap(self.fw_func,in_axes=(0, None, None, None))(latent_array,self.synthesizer_nn.input_mapping(coord_matrix),self.modulator_nn,self.synthesizer_nn)
