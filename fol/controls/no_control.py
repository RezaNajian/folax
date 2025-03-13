"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial
from jax.nn import sigmoid
from fol.mesh_input_output.mesh import Mesh
from fol.tools.decoration_functions import *

class NoControl(Control):
    def __init__(self,control_name: str, fe_mesh: Mesh):
        super().__init__(control_name)
        self.fe_mesh = fe_mesh

    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:
        self.num_control_vars = self.fe_mesh.GetNumberOfNodes()
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
        self.initialized = True

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self,variable_vector:jnp.array):
        return variable_vector
    