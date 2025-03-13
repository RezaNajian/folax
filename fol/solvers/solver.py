"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/LICENSE
"""
from abc import ABC, abstractmethod
from fol.tools.decoration_functions import *
import jax.numpy as jnp
import jax

class Solver(ABC):
    """Base abstract solver class.

    The base abstract control class has the following responsibilities.
        1. Initalizes and finalizes the solver.

    """
    def __init__(self, solver_name: str) -> None:
        self.__name = solver_name

    def GetName(self) -> str:
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        """Initializes the loss.

        This method initializes the loss. This is only called once in the whole training process.

        """
        pass

    @abstractmethod
    def Solve(self,control_vars:jnp.array,dofs:jnp.array):
        pass

    @print_with_timestamp_and_execution_time
    def BatchSolve(self,batch_control_vars:jnp.array,batch_dofs:jnp.array):
        return jnp.squeeze(jax.vmap(self.Solve, (0,0))(batch_control_vars,batch_dofs))

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the solver.

        This method finalizes the solver.
        """
        pass



