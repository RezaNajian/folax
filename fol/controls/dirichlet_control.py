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
import numpy as np

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

        self.boundary_node_ids = jnp.array(self.fe_mesh.GetNodeSet("right"))
        self.dirichlet_indices = self.loss_function.dirichlet_indices
       
        self.dofs = self.loss_function.loss_settings.get("ordered_dofs")
        self.dirichlet_bc_dict = self.loss_function.loss_settings.get("dirichlet_bc_dict")
        self.dim = self.loss_function.loss_settings.get("compute_dims")
    
        self.num_control_vars = 3   #len(self.dirichlet_indices) / 2
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
        self.initialized = True

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self,variable_vector:jnp.array):
        all_dofs = jnp.arange(3*self.fe_mesh.GetNumberOfNodes())
        dof_values = jnp.zeros_like(all_dofs, dtype=jnp.float32)
        dof_values = dof_values.at[3*self.boundary_node_ids].set(jnp.full(self.boundary_node_ids.shape, variable_vector[0], dtype=jnp.float32))
        dof_values = dof_values.at[3*self.boundary_node_ids+1].set(jnp.full(self.boundary_node_ids.shape, variable_vector[1], dtype=jnp.float32))
        dof_values = dof_values.at[3*self.boundary_node_ids+2].set(jnp.full(self.boundary_node_ids.shape, variable_vector[2], dtype=jnp.float32))

        dirichlet_values = dof_values[self.dirichlet_indices]
        return dirichlet_values

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass