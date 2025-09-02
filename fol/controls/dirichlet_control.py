"""
 Authors: Kianoosh Taghikhani, https://github.com/Kianoosh1989
 Date: August, 2025
 License: FOL/LICENSE
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial
from jax.nn import sigmoid
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.loss import Loss
from fol.tools.decoration_functions import *
import jax

class DirichletControl(Control):
    
    def __init__(self,control_name: str,control_settings: dict, fe_mesh: Mesh,fe_loss:Loss):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh
        self.loss_function = fe_loss
        

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        if self.initialized and not reinitialize:
            return

        self.dirichlet_indices = self.loss_function.dirichlet_indices
        self.dirichlet_values = self.loss_function.dirichlet_values
        self.dofs = self.loss_function.loss_settings.get("ordered_dofs")
        self.dirichlet_bc_dict = self.loss_function.loss_settings.get("dirichlet_bc_dict")
        self.dim = self.loss_function.loss_settings.get("compute_dims")
    
        self.num_control_vars = 3   #len(self.dirichlet_indices) / 2
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
        self.initialized = True

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self,variable_vector:jnp.array):
        dirichlet_values = jnp.zeros_like(self.dirichlet_values)
        slice_len = int(len(dirichlet_values)/6)
        dirichlet_values = dirichlet_values.at[slice_len:2*slice_len].set(variable_vector[0])
        dirichlet_values = dirichlet_values.at[3*slice_len:4*slice_len].set(variable_vector[1])
        dirichlet_values = dirichlet_values.at[5*slice_len:6*slice_len].set(variable_vector[2])
        
        return dirichlet_values

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass