"""
 Authors: Kianoosh Taghikhani, https://github.com/Kianoosh1989
 Date: August, 2025
 License: FOL/LICENSE
"""
from  fol.controls.control import Control
import jax.numpy as jnp
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.fe_loss_multiphysics import FiniteElementLoss
from fol.tools.decoration_functions import *
import jax 

class DirichletControl(Control):

    """
    Implements a Dirichlet boundary condition control mechanism for Finite Element
    simulations. This class identifies a set of Dirichlet DOFs (degrees of freedom)
    specified by the user, maps them to global indices, and provides functionality to
    update these controlled Dirichlet values based on a given vector of control
    variables.

    Parameters
    ----------
    control_name : str
        Name of the control object.
    control_settings : dict
        Dictionary defining control configuration. Must contain the key
        "learning_boundary", which maps DOF names to the boundary names over which
        they are controlled.
    fe_mesh : Mesh
        Finite element mesh object from which boundary node sets are extracted.
    fe_loss : FiniteElementLoss
        Loss function object providing Dirichlet DOF information and initial values.

    Attributes
    ----------
    settings : dict
        Configuration dictionary passed from the constructor.
    fe_mesh : Mesh
        Finite element mesh reference.
    loss_function : FiniteElementLoss
        Reference to the FE loss function.
    dirichlet_values : jnp.ndarray
        Initial Dirichlet values for all DOFs in the system.
    dirichlet_indices : jnp.ndarray
        Global indices of DOFs that are subject to Dirichlet boundary conditions.
    learning_dirichlet_starts : jnp.ndarray
        Starting offsets for each controlled variable’s segment in the flattened
        `learning_dirichlet_indices` array.
    learning_dirichlet_sizes : jnp.ndarray
        Number of Dirichlet DOFs controlled by each control variable.
    learning_dirichlet_indices : jnp.ndarray
        Flattened list of all Dirichlet DOF indices that are controlled by the
        current control settings.
    num_control_vars : int
        Number of control variables (e.g., DOFs × boundaries that are learnable).
    num_controlled_vars : int
        Total number of individual Dirichlet DOFs affected by the controller.
    initialized : bool
        Whether initialization has been performed.
    """
    
    def __init__(self,control_name: str,control_settings: dict, fe_mesh: Mesh,fe_loss:FiniteElementLoss):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh
        self.loss_function = fe_loss
        
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:

        """
        Initializes the Dirichlet control structure by identifying all Dirichlet DOFs
        that are designated as learnable according to `control_settings`. This method
        constructs flattened index mappings that allow fast and JAX-friendly control
        updates during optimization.

        The method builds:
        - `learning_dirichlet_starts` : segment start indices
        - `learning_dirichlet_sizes`  : segment sizes
        - `learning_dirichlet_indices`: flattened list of all controlled DOF indices
        
        """

        if self.initialized and not reinitialize:
            return
        
        self.loss_function.Initialize()

        self.dirichlet_values = self.loss_function.dirichlet_values
        self.dirichlet_indices = self.loss_function.dirichlet_indices

        self.learning_dirichlet_starts = []
        self.learning_dirichlet_sizes = []
        self.learning_dirichlet_indices = jnp.array([], dtype=jnp.int32)
        for learning_dof,learning_boundaryies in self.settings["learning_boundary"].items():
            dof_index = self.loss_function.GetDOFs().index(learning_dof)
            for learning_boundary_name in learning_boundaryies:
                learning_boundary_node_ids = jnp.asarray(self.fe_mesh.GetNodeSet(learning_boundary_name))
                learning_dirichlet_bc_indices = self.loss_function.number_dofs_per_node*learning_boundary_node_ids + dof_index
                learning_dirichlet_bc_idx = jnp.where(self.dirichlet_indices[:, None] == learning_dirichlet_bc_indices)[0]
                self.learning_dirichlet_starts.append(self.learning_dirichlet_indices.size)
                self.learning_dirichlet_sizes.append(learning_dirichlet_bc_indices.size)
                self.learning_dirichlet_indices = jnp.hstack([self.learning_dirichlet_indices,learning_dirichlet_bc_idx])

        self.learning_dirichlet_starts = jnp.array(self.learning_dirichlet_starts)
        self.learning_dirichlet_sizes = jnp.array(self.learning_dirichlet_sizes)

        self.num_control_vars = len(self.settings["learning_boundary"].keys())
        self.num_controlled_vars = self.learning_dirichlet_indices.size
        self.initialized = True

    def ComputeControlledVariables(self,variable_vector:jnp.array):
        """
        Computes the updated Dirichlet boundary values given a vector of control
        variables. Each control variable applies to a specific segment of
        Dirichlet DOFs defined during initialization.
        """
        dirichlet_values = jnp.copy(self.dirichlet_values)
        values_per_index = jnp.repeat(variable_vector, self.learning_dirichlet_sizes)
        return dirichlet_values.at[self.learning_dirichlet_indices].set(values_per_index)

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass

# """
#  Authors: Kianoosh Taghikhani, https://github.com/Kianoosh1989
#  Date: February, 2025
#  License: FOL/License.txt
# """
# from  .control import Control
# import jax.numpy as jnp
# from jax import jit
# from functools import partial
# from fol.tools.decoration_functions import *
# from fol.mesh_input_output.mesh import Mesh

# class DirichletControl(Control):
#     def __init__(self,control_name: str,control_settings: dict, fe_mesh: Mesh):
#         super().__init__(control_name)
#         self.settings = control_settings
#         self.fe_mesh = fe_mesh

#     @print_with_timestamp_and_execution_time
#     def Initialize(self,reinitialize=False) -> None:
#         if self.initialized and not reinitialize:
#             return
        
#         self.dirichlet_dofs_boundary_dict = self.settings["dirichlet_bc_dict"]
#         self.dirichlet_dofs_learning_dict = self.settings["parametric_boundary_learning"]
#         self.control_dirichlet_dict = {}
#         self.dirichlet_nodes_len = 0
#         self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
#         if not self.dirichlet_dofs_learning_dict.keys() <= self.dirichlet_dofs_boundary_dict.keys():
#             error_msg = f"provided DoFs:{list(self.dirichlet_dofs_learning_dict.keys())} do not match the DoFs of provided loss function {list(self.dirichlet_dofs_boundary_dict.keys())}"
#             fol_error(error_msg)
        
#         self.num_control_vars = 0
#         for dof,boundary_list in self.dirichlet_dofs_learning_dict.items():
#             for boundary in boundary_list:
#                 if not boundary in self.dirichlet_dofs_boundary_dict[dof].keys():
#                     error_msg = f"boundary {boundary} does not exist in dof {dof} settings of the loss's bc"
#                     fol_error(error_msg)
#             self.num_control_vars += len(boundary_list)
        
#         for dof, boundary_dict in self.dirichlet_dofs_boundary_dict.items():
#             self.control_dirichlet_dict[dof] = {}
#             for boundary_name in boundary_dict.keys():
#                 if boundary_name in self.dirichlet_dofs_learning_dict[dof]:
#                     self.control_dirichlet_dict[dof][boundary_name] = jnp.ones(len(self.fe_mesh.GetNodeSet(boundary_name)))
#                     self.dirichlet_nodes_len += len(self.fe_mesh.GetNodeSet(boundary_name))
#                 else:
#                     self.control_dirichlet_dict[dof][boundary_name] = jnp.zeros(len(self.fe_mesh.GetNodeSet(boundary_name)))
#                     self.dirichlet_nodes_len += len(self.fe_mesh.GetNodeSet(boundary_name))
#         self.initialized = True
        
#     @partial(jit, static_argnums=(0,))
#     def ComputeControlledVariables(self,variable_vector:jnp.array):
#         dirichlet_values = jnp.zeros(self.dirichlet_nodes_len)
#         control_var_index = 0
#         start_index = 0
#         end_index = 0
#         for dof,boundary_list in self.control_dirichlet_dict.items():
#             for boundary in boundary_list:
#                 end_index += int(self.control_dirichlet_dict[dof][boundary].shape[0])
#                 dirichlet_values = dirichlet_values.at[start_index:end_index].set(variable_vector[control_var_index] * self.control_dirichlet_dict[dof][boundary])
#                 start_index += int(self.control_dirichlet_dict[dof][boundary].shape[0])
#             control_var_index += 1
#         return dirichlet_values
    
#     @print_with_timestamp_and_execution_time
#     def Finalize(self) -> None:
#         pass

