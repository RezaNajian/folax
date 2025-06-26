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
from .deep_network import DeepNetwork
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *

class FourierParametricOperatorLearning(DeepNetwork):

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
        mesh_size = int(self.loss_function.fe_mesh.GetNumberOfNodes()**0.5)
        batch_size = batch_control.shape[0]
        batch_X = jax.vmap(self.control.ComputeControlledVariables)(batch_control)
        batch_X = batch_X.reshape(batch_size,mesh_size,mesh_size,1)
        batch_Y =self.flax_neural_network(batch_X).reshape(batch_size,-1)[:,self.loss_function.non_dirichlet_indices]
        batch_X = batch_X.reshape(batch_size,-1)
        return jax.vmap(self.loss_function.GetFullDofVector)(batch_X,batch_Y)

    def Finalize(self):
        pass

class DataDrivenFourierParametricOperatorLearning(FourierParametricOperatorLearning):

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_output = self.control.ComputeControlledVariables(x_set[0])
        mesh_size = int(self.loss_function.fe_mesh.GetNumberOfNodes()**0.5)
        control_output = control_output.reshape(1,mesh_size,mesh_size,1)
        nn_output = nn_model(control_output).flatten()[self.loss_function.non_dirichlet_indices]
        return self.loss_function.ComputeSingleLoss(x_set[1],nn_output)
    
class PhysicsInformedFourierParametricOperatorLearning(FourierParametricOperatorLearning):

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_output = self.control.ComputeControlledVariables(x_set[0])
        mesh_size = int(self.loss_function.fe_mesh.GetNumberOfNodes()**0.5)
        nn_output = nn_model(control_output.reshape(1,mesh_size,mesh_size,1)).flatten()[self.loss_function.non_dirichlet_indices]
        return self.loss_function.ComputeSingleLoss(control_output.flatten(),nn_output)