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
from implicit_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *
from fol.deep_neural_networks.nns import HyperNetwork
from fol.loss_functions.thermo_mechanics import ThermoMechanicsLoss3DTetra as _ThermoMechanicsLoss3DTetra


class CastingThermoMechanicsLoss3DTetra(_ThermoMechanicsLoss3DTetra):
    """Casting-local bridge between legacy FE assembly and named element fields."""

    def ComputeElementResidualAndJacobian(
        self,
        elem_xyz,
        elem_controls,
        elem_dofs,
        elem_t0,
        elem_BC,
        elem_mask_BC,
        transpose_jac,
    ):
        elem_dofs_by_node = jnp.reshape(
            elem_dofs, (-1, self.number_dofs_per_node)
        )
        elem_var_dict = {
            "XYZ": elem_xyz,
            "K": jnp.reshape(elem_controls, (-1,)),
            "T0": jnp.reshape(elem_t0, (-1,)),
        }
        for dof_index, dof_name in enumerate(self.dofs):
            dof_values = elem_dofs_by_node[:, dof_index]
            elem_var_dict[dof_name] = dof_values
            elem_var_dict[dof_name + "_mask"] = jnp.zeros_like(dof_values)
            elem_var_dict[dof_name + "_mask_value"] = jnp.zeros_like(dof_values)

        _, residual, jacobian = self.ComputeElement(elem_var_dict)
        jacobian = jax.lax.cond(
            jnp.asarray(transpose_jac),
            lambda matrix: matrix.T,
            lambda matrix: matrix,
            jacobian,
        )
        return self.ApplyDirichletBCOnElementResidualAndJacobian(
            residual, jacobian, elem_BC, elem_mask_BC
        )

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

    def _BuildThermoMechanicalBatchVariables(self, control_output, nn_output):
        """Build the named, batched fields expected by ``FiniteElementLoss``."""
        num_nodes = self.loss_function.fe_mesh.GetNumberOfNodes()
        num_dofs = self.loss_function.number_dofs_per_node

        predicted_fields = jnp.reshape(nn_output, (num_nodes, num_dofs))
        boundary_mask = jnp.zeros(num_nodes * num_dofs)
        boundary_values = jnp.zeros(num_nodes * num_dofs)
        boundary_mask = boundary_mask.at[self.loss_function.dirichlet_indices].set(1.0)
        boundary_values = boundary_values.at[self.loss_function.dirichlet_indices].set(control_output)

        boundary_mask = jnp.reshape(boundary_mask, (num_nodes, num_dofs))
        boundary_values = jnp.reshape(boundary_values, (num_nodes, num_dofs))

        batch_vars = {
            "K": jnp.ones((1, num_nodes)),
            "T0": jnp.reshape(self.loss_function.thermal_loss_settings["T0"], (1, num_nodes)),
        }
        for dof_index, dof_name in enumerate(self.loss_function.GetDOFs()):
            batch_vars[dof_name] = predicted_fields[None, :, dof_index]
            batch_vars[dof_name + "_mask"] = boundary_mask[None, :, dof_index]
            batch_vars[dof_name + "_mask_value"] = boundary_values[None, :, dof_index]

        return batch_vars

    def _ComputeThermoMechanicalLoss(self, control_output, nn_output):
        """Evaluate total and per-physics losses using the named-field FE API."""
        batch_vars = self._BuildThermoMechanicalBatchVariables(control_output, nn_output)
        connectivity = self.loss_function.fe_mesh.GetElementsNodes(
            self.loss_function.element_type
        )
        element_vars = {
            "XYZ": self.loss_function.fe_mesh.GetNodesCoordinates()[connectivity, :]
        }
        for key, value in batch_vars.items():
            element_vars[key] = value[0, connectivity]

        element_thermal_losses, element_mechanical_losses = jax.vmap(
            lambda variables: (
                self.loss_function.ComputeElementThermal(variables)[0],
                self.loss_function.ComputeElementMechanical(variables)[0],
            )
        )(element_vars)
        thermal_loss = jnp.sum(element_thermal_losses)
        mechanical_loss = jnp.sum(element_mechanical_losses)
        batch_loss = (thermal_loss + mechanical_loss) \
            ** self.loss_function.loss_function_exponent

        return batch_loss, (
            batch_loss,
            batch_loss,
            batch_loss,
            thermal_loss,
            mechanical_loss,
        )
        
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
        # latent_code = jnp.zeros(nn_model.in_features)
        control_output = self.control.ComputeControlledVariables(orig_features[0])

        # @jax.jit
        # def loss(input_latent_code):
        #     nn_output = nn_model(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        #     return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]

        # loss_latent_grad_fn = jax.grad(loss)
        # for _ in range(self.num_latent_iterations):
        #     grads = loss_latent_grad_fn(latent_code)
        #     latent_code -= self.latent_step * grads / jnp.linalg.norm(grads)
        
        nn_output = nn_model(
            orig_features[0], self.loss_function.fe_mesh.GetNodesCoordinates()
        ).flatten()
        return self._ComputeThermoMechanicalLoss(control_output, nn_output)
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

            # latent_code = jnp.zeros(self.flax_neural_network.in_features)
            control_output = self.control.ComputeControlledVariables(sample_x)

            # @jax.jit
            # def loss(input_latent_code):
            #     nn_output = self.flax_neural_network(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
            #     return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]

            # loss_latent_grad_fn = jax.grad(loss)
            # for _ in range(self.num_latent_iterations):
            #     grads = loss_latent_grad_fn(latent_code)
            #     latent_code -= self.latent_step * grads / jnp.linalg.norm(grads)

            nn_output = self.flax_neural_network(
                sample_x, self.loss_function.fe_mesh.GetNodesCoordinates()
            ).flatten()
            return nn_output.at[self.loss_function.dirichlet_indices].set(control_output)

        return jnp.array(jax.vmap(predict_single_sample)(batch_X))

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLatent(self,batch_X:jnp.ndarray):
        @nnx.jit
        def compute_single_latent(sample_x:jnp.ndarray):

            latent_code = jnp.zeros(self.flax_neural_network.in_features)
            control_output = self.control.ComputeControlledVariables(sample_x)

            @nnx.jit
            def loss(input_latent_code):
                nn_output =  self.flax_neural_network(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
                return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]

            loss_latent_grad_fn = jax.grad(loss)
            for _ in range(self.num_latent_iterations):
                latent_code -= self.latent_step * loss_latent_grad_fn(latent_code)

            return latent_code

        return jnp.array(jax.vmap(compute_single_latent)(batch_X))
    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValueStaggered1(self,orig_features:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
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
            return self.loss_function.ComputeSingleLossStaggered(control_output,nn_output)[0]

        loss_latent_grad_fn = jax.grad(loss)
        for _ in range(self.num_latent_iterations):
            grads = loss_latent_grad_fn(latent_code)
            latent_code -= self.latent_step * grads / jnp.linalg.norm(grads)
        
        nn_output = nn_model(latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        return (self.loss_function.ComputeSingleLossStaggered(control_output,nn_output)[1],self.loss_function.ComputeSingleLossStaggered(control_output,nn_output)[3])

    def ComputeSingleLossValueStaggered2(self,orig_features:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
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
        return (self.loss_function.ComputeSingleLossStaggered(control_output,nn_output)[2],self.loss_function.ComputeSingleLossStaggered(control_output,nn_output)[3])



    @print_with_timestamp_and_execution_time
    def Predict_wo_Autodec(self,batch_X:jnp.ndarray):
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
        def predict_single_sample(latent_code:jnp.ndarray):

            nn_output = self.flax_neural_network(
                latent_code, self.loss_function.fe_mesh.GetNodesCoordinates()
            ).flatten()
            return nn_output.at[self.loss_function.dirichlet_indices].set(
                self.loss_function.dirichlet_values
            )

        return jnp.array(jax.vmap(predict_single_sample)(batch_X))

    def Finalize(self):
        pass
