"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/License.txt
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from  .residual_based_solver import ResidualBasedSolver
from tools import *

class FiniteElementSolver(ResidualBasedSolver):
    """FE-based solver class.

    """
    @print_with_timestamp_and_execution_time
    def __init__(self, solver_name: str, fe_loss_function) -> None:
        super().__init__(solver_name)
        self.fe_loss_function = fe_loss_function

    # @partial(jit, static_argnums=(0,))
    @print_with_timestamp_and_execution_time
    def SingleSolve(self,current_control_vars,current_dofs):
        applied_BC_dofs = self.fe_loss_function.ApplyBCOnDOFs(current_dofs)
        R,K_mat = self.fe_loss_function.ComputeResidualsAndStiffness(current_control_vars,applied_BC_dofs)
        applied_BC_R = self.fe_loss_function.ApplyBCOnR(R)
        applied_BC_K_mat = self.fe_loss_function.ApplyBCOnMatrix(K_mat)
        return self.Solve(applied_BC_K_mat,applied_BC_R,applied_BC_dofs)

    # @partial(jit, static_argnums=(0,))
    @print_with_timestamp_and_execution_time
    def BatchSolve(self,batch_control_vars,batch_dofs):
        return jnp.squeeze(jax.vmap(self.SingleSolve, (0,0))(batch_control_vars,batch_dofs))





