"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: October, 2024
 License: FOL/LICENSE
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh
import jax

class MultiControl(Control):

    def __init__(self,name: str, controls: list[Control]):
        super().__init__(name)
        self.controls = controls
        self.controls_dict : dict[str, Control] = {}

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:

        if not self.initialized or reinitialize:
            self.initialized = True
            for control in self.controls:
                control.Initialize()
                self.controls_dict[control.GetName()] = control

    def ComputeControlledVariables(self,variable_vector:jnp.array):
        return variable_vector
    
    @partial(jit, static_argnums=(0,))
    def ComputeBatchControlledVariables(self,batch_variable_dict:dict[str, jnp.array]) -> None:
        batch_dict = {}
        for control_name,control in self.controls_dict.items():
            batch_dict[control_name] = control.ComputeBatchControlledVariables(batch_variable_dict[control_name])
        return batch_dict

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass