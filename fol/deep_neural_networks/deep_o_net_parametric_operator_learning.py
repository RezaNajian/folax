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
