"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: July, 2024
 License: FOL/License.txt
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from  .residual_based_solver import ResidualBasedSolver
from fol.tools.decoration_functions import *

class NonLinearSolver(ResidualBasedSolver):
    """Nonlinear solver class.

    """
    @print_with_timestamp_and_execution_time
    def __init__(self, solver_name: str, fe_loss_function, 
                 max_num_itr = 10, relative_error=1e-8, absolute_error=1e-8) -> None:
        super().__init__(solver_name)
        self.fe_loss_function = fe_loss_function
        self.max_num_itr = max_num_itr
        self.relative_error = relative_error
        self.absolute_error = absolute_error

    # @partial(jit, static_argnums=(0,))
    @print_with_timestamp_and_execution_time
    def SingleSolve(self,current_control_vars,current_dofs):
        applied_BC_dofs = self.fe_loss_function.ApplyBCOnDOFs(current_dofs)
        for i in range(self.max_num_itr):
            R,K_mat = self.fe_loss_function.ComputeResidualsAndStiffness(current_control_vars,applied_BC_dofs)
            applied_BC_R = self.fe_loss_function.ApplyBCOnR(R)
            res_norm = jnp.linalg.norm(applied_BC_R,ord=2)
            if res_norm<self.absolute_error:
                print(f"NonLinearSolver::SingleSolve: converged; iterations:{i+1},residuals_norm:{res_norm}")
                return applied_BC_dofs
                
            applied_BC_K_mat = self.fe_loss_function.ApplyBCOnMatrix(K_mat)
            delta_dofs = jnp.linalg.solve(applied_BC_K_mat, -applied_BC_R)
            delta_norm = jnp.linalg.norm(delta_dofs,ord=2)
            applied_BC_dofs += delta_dofs

            if delta_norm<self.relative_error:
                print(f"NonLinearSolver::SingleSolve: converged; iterations:{i+1},delta_norm:{delta_norm},residuals_norm:{res_norm}")
                return applied_BC_dofs
            elif i+1==self.max_num_itr:
                print(f"NonLinearSolver::SingleSolve: maximum num iterations:{i+1} acheived,delta_norm:{delta_norm},residuals_norm:{res_norm}")
                return applied_BC_dofs
            else:
                print(f"NonLinearSolver::SingleSolve: iteration:{i+1},delta_norm:{delta_norm},residuals_norm:{res_norm}")

    # @partial(jit, static_argnums=(0,))
    @print_with_timestamp_and_execution_time
    def BatchSolve(self,batch_control_vars,batch_dofs):
        return jnp.squeeze(jax.vmap(self.SingleSolve, (0,0))(batch_control_vars,batch_dofs))





