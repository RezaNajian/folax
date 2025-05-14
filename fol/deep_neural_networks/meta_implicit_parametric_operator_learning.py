"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: December, 2024
 License: FOL/LICENSE
"""

from typing import Tuple 
import jax
import jax.numpy as jnp
from functools import partial
from optax import GradientTransformation
from flax import nnx
from .implicit_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *
from .nns import HyperNetwork

class MetaImplicitParametricOperatorLearning(ImplicitParametricOperatorLearning):
    """
    A meta-learning framework for implicit parametric operator learning.

    This class extends `ImplicitParametricOperatorLearning` by introducing meta-learning capabilities, 
    such as latent loop optimization. It enables efficient learning of parametric operators using deep 
    neural networks and advanced optimization techniques.

    Attributes
    ----------
    name : str
        The name of the neural network model for identification purposes.
    control : Control
        An instance of the `Control` class that manages the parametric learning process.
    loss_function : Loss
        The objective function to minimize during training.
    flax_neural_network : HyperNetwork
        The Flax-based hypernetwork model defining the architecture and forward pass.
    main_loop_optax_optimizer : GradientTransformation
        The Optax optimizer used for the primary optimization loop.
    latent_step : float
        Step size for latent loop optimization.
    num_latent_iterations : int
        Number of iterations for latent loop optimization.
    checkpoint_settings : dict
        Configuration dictionary for managing checkpoints. Defaults to an empty dictionary.
    working_directory : str
        Path to the working directory where model outputs and checkpoints are saved. Defaults to the current directory.
    """

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:HyperNetwork,
                 main_loop_optax_optimizer:GradientTransformation,
                 latent_step_size:float=1e-2,
                 num_latent_iterations:int=3):
        """
        Initializes the `MetaImplicitParametricOperatorLearning` class.

        This constructor sets up the meta-learning framework by initializing attributes and configurations 
        needed for training and optimization, including latent loop parameters.

        Parameters
        ----------
        name : str
            The name assigned to the neural network model for identification purposes.
        control : Control
            An instance of the `Control` class that manages the parametric learning process.
        loss_function : Loss
            An instance of the `Loss` class representing the objective function to minimize.
        flax_neural_network : HyperNetwork
            The Flax-based hypernetwork model defining the architecture and forward pass.
        main_loop_optax_optimizer : GradientTransformation
            The Optax optimizer for the primary optimization loop.
        latent_step_size : float, optional
            The step size for latent loop optimization. Default is 1e-2.
        num_latent_iterations : int, optional
            The number of iterations for latent loop optimization. Default is 3.

        Returns
        -------
        None
        """
        super().__init__(name,control,loss_function,flax_neural_network,
                         main_loop_optax_optimizer)
        
        self.latent_step = latent_step_size
        self.num_latent_iterations = num_latent_iterations
        
    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,orig_features:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Computes the loss value for a single input using latent loop optimization.

        This method calculates the loss by optimizing a latent code for the given input features. The latent code is 
        iteratively updated using gradient descent, and the final loss is computed by comparing the neural network's 
        output with the control output based on the loss function's non-Dirichlet indices.

        Parameters
        ----------
        orig_features : Tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing the input features, where:
            - The first element is used to compute control variables.
            - The second element (if applicable) may contain additional feature data.
        nn_model : nnx.Module
            The neural network model used for predictions.

        Returns
        -------
        jnp.ndarray
            The computed loss value as a scalar.

        Notes
        -----
        - The latent code is initialized to zeros and updated iteratively using gradient descent.
        - The number of iterations and step size for updating the latent code are determined by the 
        `num_latent_iterations` and `latent_step` attributes, respectively.
        - This method uses JAX's `jit` for just-in-time compilation and `grad` for automatic differentiation 
        to compute the gradients of the loss function with respect to the latent code.
        """       
        latent_code = jnp.zeros(nn_model.in_features)
        control_output = self.control.ComputeControlledVariables(orig_features[0])

        @jax.jit
        def loss(input_latent_code):
            nn_output = nn_model(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
            return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]

        loss_latent_grad_fn = jax.grad(loss)
        for _ in range(self.num_latent_iterations):
            grads = loss_latent_grad_fn(latent_code)
            latent_code -= self.latent_step * grads / jnp.linalg.norm(grads)
        
        nn_output = nn_model(latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        return self.loss_function.ComputeSingleLoss(control_output,nn_output)

    @print_with_timestamp_and_execution_time
    def Predict(self,batch_X:jnp.ndarray):
        """
        Generates predictions for a batch of input data using latent loop optimization.

        This method processes a batch of input features and computes predictions for each sample by:
        1. Initializing a latent code for the input sample.
        2. Iteratively updating the latent code using gradient descent to minimize the loss.
        3. Using the optimized latent code to compute the neural network output.
        4. Mapping the network's output to the full degree of freedom (DoF) vector based on the loss function.

        Parameters
        ----------
        batch_X : jnp.ndarray
            A batch of input features for which predictions are required.

        Returns
        -------
        jnp.ndarray
            A batch of predicted outputs, where each prediction corresponds to a full DoF vector.

        Notes
        -----
        - The latent code is initialized to zeros and optimized iteratively for each input sample.
        - The number of iterations and step size for the latent loop optimization are determined by the 
        `num_latent_iterations` and `latent_step` attributes, respectively.
        - JAX's `jit` and `grad` are used for just-in-time compilation and automatic differentiation 
        to compute gradients for latent code optimization.
        - The predictions are processed in parallel using `jax.vmap` for efficiency.
        - This method maps the neural network's output to the full DoF vector, including both Dirichlet 
        and non-Dirichlet indices.
        """
        def predict_single_sample(sample_x:jnp.ndarray):

            latent_code = jnp.zeros(self.flax_neural_network.in_features)
            control_output = self.control.ComputeControlledVariables(sample_x)

            @jax.jit
            def loss(input_latent_code):
                nn_output = self.flax_neural_network(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
                return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]

            loss_latent_grad_fn = jax.grad(loss)
            for _ in range(self.num_latent_iterations):
                grads = loss_latent_grad_fn(latent_code)
                latent_code -= self.latent_step * grads / jnp.linalg.norm(grads)

            nn_output = self.flax_neural_network(latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]

            return self.loss_function.GetFullDofVector(sample_x,nn_output)

        return jnp.array(jax.vmap(predict_single_sample)(batch_X))
    
    @print_with_timestamp_and_execution_time
    def PredictDynamics(self,initial_u:jnp.ndarray,num_steps:int):
        """
        Simulates the temporal evolution of the system over multiple time steps using latent loop optimization.

        This method performs sequential predictions starting from an initial state, computing the state at each 
        subsequent time step based on the output of the previous step. At each time step:
        1. A latent code is initialized and optimized to minimize the loss function for the current input state.
        2. The optimized latent code is used to generate the neural network output.
        3. The output is mapped to the full degree of freedom (DoF) vector.
        4. The updated DoF vector becomes the input for the next time step.

        Parameters
        ----------
        initial_u : jnp.ndarray
            The initial condition or state of the system at time step zero. Should be a batch of input vectors.
        
        num_steps : int
            The number of time steps to simulate the system forward in time.

        Returns
        -------
        jnp.ndarray
            An array containing the predicted system states over time.
            The first row corresponds to the initial condition, and each subsequent row corresponds to the predicted 
            state at the next time step. Shape is (num_steps + 1, DoF).

        Notes
        -----
        - Latent code optimization is performed at each time step to generate an accurate prediction of the system's 
        next state.
        - The optimization loop uses `jax.lax.scan` to efficiently perform a fixed number of latent updates.
        - The state update across time steps is performed using `jax.lax.scan` to enable efficient sequential processing.
        - The function uses `jax.vmap` to enable parallel processing of batch inputs at each time step.
        - The model implicitly learns the system dynamics by optimizing the latent representation without requiring 
        explicit temporal modeling.
        - The final output includes both the initial state and the predicted states across all time steps, stacked vertically.
        """ 
        def predict_single_step(sample_u: jnp.ndarray):
            latent_code = jnp.zeros(self.flax_neural_network.in_features)
            control_output = self.control.ComputeControlledVariables(sample_u)
            @jax.jit
            def loss(input_latent_code):
                nn_output = self.flax_neural_network(
                    input_latent_code, self.loss_function.fe_mesh.GetNodesCoordinates()
                ).flatten()[self.loss_function.non_dirichlet_indices]
                return self.loss_function.ComputeSingleLoss(control_output, nn_output)[0]
            loss_latent_grad_fn = jax.grad(loss)
            
            @jax.jit
            def update_latent(latent_code):
                def single_update_latent_fn(state, _):
                    grads = loss_latent_grad_fn(state)
                    update = self.latent_step * grads / jnp.linalg.norm(grads)  
                    return state - update, None  
                latent_code, _ = jax.lax.scan(single_update_latent_fn, latent_code, xs=None, length=self.num_latent_iterations)
                return latent_code

            latent_code = update_latent(latent_code)           
            @jax.jit
            def compute_output(latent_code):
                nn_output = self.flax_neural_network(
                    latent_code, self.loss_function.fe_mesh.GetNodesCoordinates()
                ).flatten()[self.loss_function.non_dirichlet_indices]
                return self.loss_function.GetFullDofVector(sample_u, nn_output)      
            return compute_output(latent_code)
        
        parallel_predict_fn = jax.vmap(predict_single_step)

        def scan_fn(u, _):
            u_next = parallel_predict_fn(u)
            return u_next, u_next

        _, dynamic_u = jax.lax.scan(scan_fn, initial_u.reshape(-1,1).T, None, length=num_steps)

        return jnp.vstack((initial_u.reshape(-1,1).T,jnp.squeeze(dynamic_u)))

    def Finalize(self):
        pass