"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2025
 License: FOL/LICENSE
"""

from typing import Iterator,Tuple 
import jax
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial
from optax import GradientTransformation
from flax import nnx
from deep_network_multiphysics import DeepNetwork
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *

class DeepONetParametricOperatorLearning(DeepNetwork):

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation):

        super().__init__(name,loss_function,flax_neural_network,
                         optax_optimizer)
        self.control = control
        
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
 
        if self.initialized and not reinitialize:
            return

        super().Initialize(reinitialize)

        if not self.control.initialized:
            self.control.Initialize(reinitialize)

        self.initialized = True

    @print_with_timestamp_and_execution_time
    def Predict(self,batch_control:jnp.ndarray):
        batch_X = jax.vmap(self.control.ComputeControlledVariables)(batch_control)
        batch_Y =jax.vmap(self.flax_neural_network,(0,None))(batch_X,self.loss_function.fe_mesh.GetNodesCoordinates())
        batch_Y = batch_Y.reshape(batch_X.shape[0], -1)[:,self.loss_function.non_dirichlet_indices]
        return jax.vmap(self.loss_function.GetFullDofVector)(batch_X,batch_Y)

    def Finalize(self):
        pass

class DataDrivenDeepONetParametricOperatorLearning(DeepONetParametricOperatorLearning):

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_output = self.control.ComputeControlledVariables(x_set[0])
        nn_output = nn_model(control_output,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        return self.loss_function.ComputeSingleLoss(x_set[1],nn_output)
    
class PhysicsInformedDeepONetParametricOperatorLearning(DeepONetParametricOperatorLearning):

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_output = self.control.ComputeControlledVariables(x_set[0])
        nn_output = nn_model(control_output,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        return self.loss_function.ComputeSingleLoss(control_output.flatten(),nn_output)

class PhysicsInformedDeepONetParametricOperatorLearningDBC(DeepONetParametricOperatorLearning):

    def _ComputeThermoMechanicalLoss(self, control_output, nn_output):
        num_nodes = self.loss_function.fe_mesh.GetNumberOfNodes()
        num_dofs = self.loss_function.number_dofs_per_node
        predicted_fields = nn_output.reshape(num_nodes, num_dofs)

        boundary_mask = jnp.zeros(num_nodes * num_dofs)
        boundary_values = jnp.zeros(num_nodes * num_dofs)
        boundary_mask = boundary_mask.at[self.loss_function.dirichlet_indices].set(1.0)
        boundary_values = boundary_values.at[self.loss_function.dirichlet_indices].set(control_output)
        boundary_mask = boundary_mask.reshape(num_nodes, num_dofs)
        boundary_values = boundary_values.reshape(num_nodes, num_dofs)

        element_nodes = self.loss_function.fe_mesh.GetElementsNodes(
            self.loss_function.element_type
        )
        element_vars = {
            "XYZ": self.loss_function.fe_mesh.GetNodesCoordinates()[element_nodes, :],
            "K": jnp.ones((num_nodes,))[element_nodes],
            "T0": self.loss_function.thermal_loss_settings["T0"][element_nodes],
        }
        for dof_index, dof_name in enumerate(self.loss_function.GetDOFs()):
            element_vars[dof_name] = predicted_fields[:, dof_index][element_nodes]
            element_vars[dof_name + "_mask"] = boundary_mask[:, dof_index][element_nodes]
            element_vars[dof_name + "_mask_value"] = boundary_values[:, dof_index][element_nodes]

        element_thermal_losses, element_mechanical_losses = jax.vmap(
            lambda variables: (
                self.loss_function.ComputeElementThermal(variables)[0],
                self.loss_function.ComputeElementMechanical(variables)[0],
            )
        )(element_vars)
        thermal_loss = jnp.sum(element_thermal_losses)
        mechanical_loss = jnp.sum(element_mechanical_losses)
        total_loss = (thermal_loss + mechanical_loss) \
            ** self.loss_function.loss_function_exponent
        return total_loss, (
            total_loss, total_loss, total_loss, thermal_loss, mechanical_loss
        )

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_output = self.control.ComputeControlledVariables(x_set[0])
        nn_output = nn_model(
            x_set[0], self.loss_function.fe_mesh.GetNodesCoordinates()
        ).flatten()
        return self._ComputeThermoMechanicalLoss(control_output, nn_output)

    @print_with_timestamp_and_execution_time
    def Predict(self,batch_control:jnp.ndarray):
        batch_X = jax.vmap(self.control.ComputeControlledVariables)(batch_control)
        batch_Y =jax.vmap(self.flax_neural_network,(0,None))(batch_control,self.loss_function.fe_mesh.GetNodesCoordinates())
        batch_Y = batch_Y.reshape(batch_X.shape[0], -1)
        return batch_Y.at[:, self.loss_function.dirichlet_indices].set(batch_X)
