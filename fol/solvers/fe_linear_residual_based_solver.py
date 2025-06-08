"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/LICENSE
"""
import jax.numpy as jnp
from  .fe_solver import FiniteElementSolver
from fol.tools.decoration_functions import *

class FiniteElementLinearResidualBasedSolver(FiniteElementSolver):
    """Residual base linear solver class.

    """
    @print_with_timestamp_and_execution_time
    def Solve(self,current_control_vars:jnp.array,current_dofs:jnp.array):
        BC_applied_dofs = self.fe_loss_function.ApplyDirichletBCOnDofVector(current_dofs)
        BC_applied_jac,BC_applied_r = self.fe_loss_function.ComputeJacobianMatrixAndResidualVector(
                                            current_control_vars,BC_applied_dofs)
        
        norm_max = jnp.max(jnp.abs(BC_applied_r))
        fol_info(f"max residual component:{norm_max}")
        norm_min = jnp.min(jnp.abs(BC_applied_r))
        fol_info(f"min residual component:{norm_min}")

        threshold = 1e-5
        num_large_res = jnp.sum(jnp.abs(BC_applied_r)>threshold)
        fol_info(f"num large residual component:{num_large_res}")

        delta_dofs = self.LinearSolve(BC_applied_jac,BC_applied_r,BC_applied_dofs)

        return BC_applied_dofs + delta_dofs

    @print_with_timestamp_and_execution_time
    def SolveMulti(self,current_control_vars:jnp.array,current_dofs:jnp.array):
        BC_applied_dofs = self.fe_loss_function.ApplyDirichletBCOnDofVector(current_dofs)
        BC_applied_jac,BC_applied_r = self.fe_loss_function.ComputeJacobianMatrixAndResidualVector(
                                            current_control_vars,BC_applied_dofs)
        
        delta_dofs = self.LinearSolve(BC_applied_jac,BC_applied_r,BC_applied_dofs)
        return BC_applied_dofs + delta_dofs




