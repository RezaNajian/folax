"""
 Authors: Kianoosh Taghikhani, https://github.com/Kianoosh1989
 Date: February, 2025
 License: FOL/License.txt
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh

class DirichletControl(Control):
    def __init__(self,control_name: str,control_settings: dict, fe_mesh: Mesh):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        if self.initialized and not reinitialize:
            return
        
        self.dirichlet_dofs_boundary_dict = self.settings["dirichlet_bc_dict"]
        self.dirichlet_dofs_learning_dict = self.settings["parametric_boundary_learning"]
        self.control_dirichlet_dict = {}
        self.dirichlet_nodes_len = 0
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
        if not self.dirichlet_dofs_learning_dict.keys() <= self.dirichlet_dofs_boundary_dict.keys():
            error_msg = f"provided DoFs:{list(self.dirichlet_dofs_learning_dict.keys())} do not match the DoFs of provided loss function {list(self.dirichlet_dofs_boundary_dict.keys())}"
            fol_error(error_msg)
        
        self.num_control_vars = 0
        for dof,boundary_list in self.dirichlet_dofs_learning_dict.items():
            for boundary in boundary_list:
                if not boundary in self.dirichlet_dofs_boundary_dict[dof].keys():
                    error_msg = f"boundary {boundary} does not exist in dof {dof} settings of the loss's bc"
                    fol_error(error_msg)
            self.num_control_vars += len(boundary_list)
        
        for dof, boundary_dict in self.dirichlet_dofs_boundary_dict.items():
            self.control_dirichlet_dict[dof] = {}
            for boundary_name in boundary_dict.keys():
                if boundary_name in self.dirichlet_dofs_learning_dict[dof]:
                    self.control_dirichlet_dict[dof][boundary_name] = jnp.ones(len(self.fe_mesh.GetNodeSet(boundary_name)))
                    self.dirichlet_nodes_len += len(self.fe_mesh.GetNodeSet(boundary_name))
                else:
                    self.control_dirichlet_dict[dof][boundary_name] = jnp.zeros(len(self.fe_mesh.GetNodeSet(boundary_name)))
                    self.dirichlet_nodes_len += len(self.fe_mesh.GetNodeSet(boundary_name))
        self.initialized = True
        
    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self,variable_vector:jnp.array):
        dirichlet_values = jnp.zeros(self.dirichlet_nodes_len)
        control_var_index = 0
        start_index = 0
        end_index = 0
        for dof,boundary_list in self.control_dirichlet_dict.items():
            for boundary in boundary_list:
                end_index += int(self.control_dirichlet_dict[dof][boundary].shape[0])
                dirichlet_values = dirichlet_values.at[start_index:end_index].set(variable_vector[control_var_index] * self.control_dirichlet_dict[dof][boundary])
                start_index += int(self.control_dirichlet_dict[dof][boundary].shape[0])
            control_var_index += 1
        return dirichlet_values
    
    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass