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

class IdentityControl(Control):
    """
    Identity control mapping control variables directly to controlled variables.

    This control implements a trivial identity mapping in which the input
    control vector is returned unchanged as the controlled variable vector.
    It is primarily intended for cases where no spatial or functional
    parameterization is required, such as debugging, testing, or optimization
    workflows where the control variables already represent the desired
    physical quantities.

    The number of control variables and the number of controlled variables
    are both equal to ``num_vars`` and are fixed at construction time.

    Args:
        control_name (str):
            Name identifier for the control instance.
        control_settings (dict):
            Configuration dictionary. This argument is accepted for interface
            consistency but is not used by the identity mapping.
        num_vars (int):
            Number of control variables and controlled variables.
    """
    def __init__(self,control_name: str,control_settings: dict, num_vars):
        super().__init__(control_name)
        self.num_vars = num_vars
        self.settings = control_settings

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize the identity control.

        This method sets the number of control variables and controlled
        variables to ``num_vars``. Initialization is idempotent and may be
        forced to re-run by setting ``reinitialize=True``.

        Args:
            reinitialize (bool, optional):
                If ``True``, forces reinitialization even if already initialized.
                Default is ``False``.

        Returns:
            None
        """
        if self.initialized and not reinitialize:
            self.initialized = True
        self.num_control_vars = self.num_vars
        self.num_controlled_vars = self.num_vars

    def ComputeControlledVariables(self,variable_vector:jnp.array):
        """
        Return the controlled variables corresponding to the input vector.

        This identity mapping returns the input control vector unchanged.

        Args:
            variable_vector (jax.numpy.ndarray):
                Control variable vector of length ``num_control_vars``.

        Returns:
            jax.numpy.ndarray:
                Controlled variable vector equal to ``variable_vector``.
        """
        return variable_vector

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        """
        Finalize the identity control.

        This method is provided for API consistency with the base
        :class:`fol.controls.control.Control`. The default implementation
        performs no action.

        Returns:
            None
        """
        pass