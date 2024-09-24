"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/License.txt
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
        
        delta_dofs = self.LinearSolve(BC_applied_jac,BC_applied_r,BC_applied_dofs)

        return BC_applied_dofs + delta_dofs

        





