"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: October, 2024
 License: FOL/LICENSE
"""

from typing import Iterator,Tuple 
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

class ImplicitParametricOperatorLearning(DeepNetwork):
    """
    A class for implicit parametric operator learning in deep neural networks.

    This class extends the `DeepNetwork` base class and is designed specifically 
    for learning implicit parametric operators. These operators account for control 
    parameters that influence the spatial fields, such as predicted displacements or 
    other modeled phenomena. The class inherits all attributes and methods from 
    `DeepNetwork` and introduces additional components for handling parametric inputs.

    Attributes:
        name (str): Identifier for the neural network model.
        control (Control): An instance of the `Control` class for managing parametric inputs 
            and influencing model predictions.
        loss_function (Loss): An instance of the `Loss` class defining the objective 
            function for training.
        flax_neural_network (nnx.Module): The Flax-based neural network module that specifies 
            the architecture and forward computation.
        optax_optimizer (GradientTransformation): The Optax optimizer used to manage 
            gradient updates during training.
        checkpoint_settings (dict): Configuration dictionary for checkpointing, specifying 
            how and when to save model states and parameters during training. Defaults to an empty dictionary.
        working_directory (str): Directory path where model files, checkpoints, and logs 
            will be stored. Defaults to the current directory ('.').
    """

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation):
        """
        Initializes an instance of the `ImplicitParametricOperatorLearning` class.

        Parameters:
            name (str): The name assigned to the neural network model.
            control (Control): The control mechanism or parameters guiding 
                parametric operator learning.
            loss_function (Loss): The objective function to be minimized during training.
            flax_neural_network (nnx.Module): The Flax-based neural network module defining 
                the model's architecture and behavior.
            optax_optimizer (GradientTransformation): The optimizer for updating model weights 
                based on gradient information.
            checkpoint_settings (dict, optional): Configuration for managing model checkpoints. 
                Defaults to an empty dictionary.
            working_directory (str, optional): Path to the directory where files will be stored. 
                Defaults to the current directory ('.').
        """
        super().__init__(name,loss_function,flax_neural_network,
                         optax_optimizer)
        self.control = control
        
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize the implicit parametric operator learning model and its components.

        This method extends the initialization process defined in the `DeepNetwork` base class 
        by including the initialization of control parameters critical for parametric learning. 
        It ensures that all components, such as the loss function, checkpoint settings, 
        neural network state, and control parameters, are correctly set up. The method also validates 
        the consistency between the neural network architecture and the dimensions of the loss 
        function and control parameters.

        Parameters:
        ----------
        reinitialize : bool, optional
            If True, forces the reinitialization of all components, including the control parameters, 
            even if they have already been initialized. Default is False.

        Raises:
        -------
        ValueError:
            If the neural network's input (`in_features`) or output (`out_features`) dimensions 
            do not match the expected sizes based on the control parameters and loss function.
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

        if self.flax_neural_network.out_features != len(self.loss_function.dofs):
            fol_error(f"the size of the output layer is {self.flax_neural_network.out_features} " \
                      f" does not match the number of the loss function {self.loss_function.dofs}")

        # if self.flax_neural_network.in_features != self.control.GetNumberOfVariables():
        #     fol_error(f"the size of the input layer is {self.flax_neural_network.in_features} "\
        #               f"does not match the input size implicit/neural field which is {self.control.GetNumberOfVariables() + 3}")

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Compute the loss value for a single data point.

        This method evaluates the neural network's output for a single input data point, 
        applies control parameter transformations, and calculates the loss using the 
        specified loss function. It ensures the computation focuses on relevant indices 
        (e.g., non-Dirichlet boundary conditions) and integrates control effects.

        Parameters
        ----------
        x_set : Tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing:
            - The input data (jnp.ndarray): Represents the independent variables used as inputs to the model.
            - The target labels (jnp.ndarray): Corresponding ground-truth values for the input data.
        nn_model : nnx.Module
            The Flax-based neural network model to evaluate.

        Returns
        -------
        jnp.ndarray
            The computed loss value for the given data point, encapsulating the 
            discrepancy between model predictions and target labels, adjusted for 
            control variables.
        
        Notes
        -----
        - The loss computation considers only the non-Dirichlet indices to exclude 
          fixed boundary conditions from influencing the loss calculation.
        - Control variables are computed and applied to ensure parameterized influence 
          on the model output during the loss evaluation.
        """
        nn_output = nn_model(x_set[0],self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        control_output = self.control.ComputeControlledVariables(x_set[0])
        return self.loss_function.ComputeSingleLoss(control_output,nn_output)

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def Predict(self,batch_X):
        """
        Generate predictions for a batch of input data.

        This method computes the model's predictions for each sample in a batch of input data. 
        The predictions are mapped to the full degree of freedom (DoF) vector by incorporating 
        the loss function's non-Dirichlet indices and additional transformations.

        Parameters
        ----------
        batch_X : jnp.ndarray
            A 2D array representing the batch of input data, where each row corresponds 
            to an individual input sample.

        Returns
        -------
        jnp.ndarray
            A 2D array containing the predicted outputs for the batch, where each row represents 
            the full DoF vector corresponding to an input sample.

        Notes
        -----
        - For each input sample, the method computes the neural network's output, extracts the 
          non-Dirichlet indices defined by the loss function, and reconstructs the full DoF vector.
        - This approach ensures that predictions respect the boundary conditions or fixed values 
          associated with Dirichlet boundaries.
        - The method processes the batch iteratively, where each input sample is handled individually.

        Raises
        ------
        ValueError
            If the input batch does not have a valid shape (e.g., empty or incorrectly formatted).
        """
        prediction = []
        for i in range(batch_X.shape[0]):
            nn_output = self.flax_neural_network(batch_X[i],self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
            full_dof = self.loss_function.GetFullDofVector(batch_X[i],nn_output)
            prediction.append(full_dof)
        return jnp.array(prediction)

    def Finalize(self):
        pass