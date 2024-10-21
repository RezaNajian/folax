"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""

from typing import Tuple 
import jax
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial
from optax import GradientTransformation
from flax import nnx
from .deep_network import DeepNetwork
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *

class ExplicitParametricOperatorLearning(DeepNetwork):
    """
    A class for explicit parametric operator learning in deep neural networks.

    This class extends the `DeepNetwork` base class and is designed specifically 
    for learning parametric operators where spatial fields like predicted displacement
    are explicitly modeled. It inherits all the attributes and methods from `DeepNetwork` and introduces 
    additional components to handle control parameters.

    Attributes:
        name (str): The name assigned to the neural network model for identification purposes.
        control (Control): An instance of the Control class used for the parametric learning.
        loss_function (Loss): An instance of the Loss class representing the objective function to be minimized during training.
        flax_neural_network (Module): The Flax neural network model (inherited from flax.nnx.Module) that defines the architecture and forward pass of the network.
        optax_optimizer (GradientTransformation): The Optax optimizer used to compute and apply gradients during the training process.
        checkpoint_settings (dict): A dictionary of configurations used to manage checkpoints, saving model states and parameters during or after training. Defaults to an empty dictionary.
     
    """

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation,
                 checkpoint_settings:dict={},
                 working_directory='.'
                 ):
        super().__init__(name,loss_function,flax_neural_network,
                         optax_optimizer,checkpoint_settings,
                         working_directory)
        self.control = control
        
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize the explicit parametric operator learning model, its components, and control parameters.

        This method extends the initialization process defined in the `DeepNetwork` base class by
        ensuring that the control parameters used for parametric learning are also initialized.
        It handles both the initialization of core deep learning components (loss function, 
        checkpoint settings, neural network state restoration) and the initialization of 
        the control parameters essential for explicit parametric learning tasks.

        Parameters:
        ----------
        reinitialize : bool, optional
            If True, forces reinitialization of the model and its components, including control parameters,
            even if they have been initialized previously. Default is False.

        """

        if self.initialized and not reinitialize:
            return

        super().Initialize(reinitialize)

        if not self.control.initialized:
            self.control.Initialize(reinitialize)

        self.initialized = True

        # now check if the input output layers size match with 
        # loss and control sizes, this is explicit parametric learning
        if not hasattr(self.flax_neural_network, 'in_features'):
            fol_error(f"the provided flax neural netwrok does not have in_features "\
                      "which specifies the size of the input layer ") 

        if not hasattr(self.flax_neural_network, 'out_features'):
            fol_error(f"the provided flax neural netwrok does not have out_features "\
                      "which specifies the size of the output layer") 

        if self.flax_neural_network.in_features != self.control.GetNumberOfVariables():
            fol_error(f"the size of the input layer is {self.flax_neural_network.in_features} "\
                      f"does not match the size of control variables {self.control.GetNumberOfVariables()}")

        if self.flax_neural_network.out_features != self.loss_function.GetNumberOfUnknowns():
            fol_error(f"the size of the output layer is {self.flax_neural_network.out_features} " \
                      f" does not match the size of unknowns of the loss function {self.loss_function.GetNumberOfUnknowns()}")

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Computes the loss value for a single data point.

        This method computes the network's output for a single input data point, 
        applies the control parameters, and evaluates the loss function.

        Parameters
        ----------
        x_set : Tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing the input data and corresponding target labels.
        nn_model : nnx.Module
            The Flax neural network model.

        Returns
        -------
        jnp.ndarray
            The loss value for the single data point.
        """
        nn_output = nn_model(x_set[0])
        control_output = self.control.ComputeControlledVariables(x_set[0])
        return self.loss_function.ComputeSingleLoss(control_output,nn_output)

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLossValue(self,batch_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Computes the loss values for a batch of data.

        This method computes the network's output for a batch of input data, applies the control parameters,
        and evaluates the loss function for the entire batch. It aggregates the results and returns
        summary statistics (min, max, avg) for the batch losses.

        Parameters
        ----------
        batch_set : Tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing a batch of input data and corresponding target labels.
        nn_model : nnx.Module
            The Flax neural network model.

        Returns
        -------
        Tuple[jnp.ndarray, dict]
            The mean loss for the batch and a dictionary of loss statistics (min, max, avg, total).
        """

        batch_losses,(batch_mins,batch_maxs,batch_avgs) = jax.vmap(self.ComputeSingleLossValue,(0,None))(batch_set,nn_model)
        loss_name = self.loss_function.GetName()
        total_mean_loss = jnp.mean(batch_losses)
        return total_mean_loss, ({loss_name+"_min":jnp.min(batch_mins),
                                         loss_name+"_max":jnp.max(batch_maxs),
                                         loss_name+"_avg":jnp.mean(batch_avgs),
                                         "total_loss":total_mean_loss})

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def Predict(self,batch_X):
        """
        Generates predictions for a batch of input data.

        This method computes the network's predictions for a batch of input data. 
        It maps the network outputs to the full degree of freedom (DoF) vector using the loss function.

        Parameters
        ----------
        batch_X : jnp.ndarray
            A batch of input data.

        Returns
        -------
        jnp.ndarray
            The predicted outputs, mapped to the full DoF vector.
        """
        batch_Y = self.flax_neural_network(batch_X)
        return vmap(self.loss_function.GetFullDofVector)(batch_X,batch_Y)

    def Finalize(self):
        pass