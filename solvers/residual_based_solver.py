"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/License.txt
"""
from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import jit,grad,jit
from functools import partial
from  .solver import Solver

class ResidualBasedSolver(Solver):
    """Residual base solver class.

    """
    def __init__(self, solver_name: str) -> None:
        super().__init__(solver_name)

    def Initialize(self) -> None:
        pass

    @partial(jit, static_argnums=(0,))
    def Solve(self,tangent_matrix:jnp.array,residual_vector:jnp.array,dofs_vector:jnp.array):
        delta_dofs = jnp.linalg.solve(tangent_matrix, -residual_vector)
        return dofs_vector + delta_dofs

    def Finalize(self) -> None:
        pass



