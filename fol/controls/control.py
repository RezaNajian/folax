"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from abc import ABC, abstractmethod
import jax.numpy as jnp
from functools import partial
from jax import jit
import jax
from fol.tools.decoration_functions import *

class Control(ABC):
    """Base abstract control class.

    The base abstract control class has the following responsibilities.
        1. Initalizes and finalizes the model.

    """
    def __init__(self, control_name: str) -> None:
        self.__name = control_name
        self.__initialized = False

    def GetName(self) -> str:
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        """Initializes the control.

        This method initializes the control. This is only called once in the whole training process.

        """
        pass

    @abstractmethod
    def GetNumberOfVariables(self):
        """Returns number of variables of the control.

        """
        pass
    
    @abstractmethod
    def GetNumberOfControlledVariables(self):
        """Returns number of controlled variables

        """
        pass

    @abstractmethod
    def ComputeControlledVariables(self,variable_vector:jnp.array) -> None:
        """Computes the controlled variables for the given variables.

        """
        pass

    @print_with_timestamp_and_execution_time     
    @partial(jit, static_argnums=(0,))
    def ComputeBatchControlledVariables(self,batch_variable_vector:jnp.array) -> None:
        """Computes the controlled variables for the given batch variables.

        """
        return jnp.squeeze(jax.vmap(self.ComputeControlledVariables,(0))(batch_variable_vector))

    @abstractmethod
    def ComputeJacobian(self,variable_vector:jnp.array) -> None:
        """Computes jacobian of the control w.r.t input variable vector.

        """
        pass

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the control.

        This method finalizes the control. This is only called once in the whole training process.

        """
        pass



