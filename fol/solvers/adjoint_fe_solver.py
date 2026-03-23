"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: January, 2025
 License: FOL/LICENSE
"""
import jax.numpy as jnp
from  .fe_solver import FiniteElementSolver
from fol.tools.decoration_functions import *
from fol.responses.fe_response import FiniteElementResponse

class AdjointFiniteElementSolver(FiniteElementSolver):

    @print_with_timestamp_and_execution_time
    def __init__(self, adj_fe_solver_name: str, fe_response: FiniteElementResponse, adj_fe_solver_settings:dict={}) -> None:
        super().__init__(adj_fe_solver_name,fe_response.fe_loss,adj_fe_solver_settings)
        self.fe_response = fe_response
  
    @print_with_timestamp_and_execution_time
    def Solve(self,current_control_vars:jnp.array,current_dofs:jnp.array,current_adjoint_dofs:jnp.array):
        BC_applied_jac,BC_applied_rhs = self.fe_response.ComputeAdjointJacobianMatrixAndRHSVector(current_control_vars,current_dofs)
        # here we need to multiply by -1 since the solver later mutiplies by -1
        BC_applied_rhs *= -1
        return self.LinearSolve(BC_applied_jac,BC_applied_rhs,current_adjoint_dofs)

        





