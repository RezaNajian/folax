"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: July, 2024
 License: FOL/LICENSE
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *
from fol.loss_functions.fe_loss import FiniteElementLoss

class FiniteElementNonLinearResidualBasedSolver(FiniteElementLinearResidualBasedSolver):
    """Nonlinear solver class.

    """

    @print_with_timestamp_and_execution_time
    def __init__(self, fe_solver_name: str, fe_loss_function: FiniteElementLoss, fe_solver_settings:dict={}) -> None:
        super().__init__(fe_solver_name,fe_loss_function,fe_solver_settings)
        self.nonlinear_solver_settings = {"rel_tol":1e-8,
                                           "abs_tol":1e-8,
                                           "maxiter":20,
                                           "load_incr":5,
                                           "line_search":False,
                                           "line_search_step_size":1.0,
                                           "line_search_reduction_factor":0.5,
                                           "line_search_min_step_size":1e-3,
                                           "line_search_maxiter":10}

    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:
        super().Initialize() 
        if "nonlinear_solver_settings" in self.fe_solver_settings.keys():
            self.nonlinear_solver_settings = UpdateDefaultDict(self.nonlinear_solver_settings,
                                                                self.fe_solver_settings["nonlinear_solver_settings"])

    def _ComputeResidualNorm(self,current_control_vars,trial_dofs):
        _, trial_r = self.fe_loss_function.ComputeJacobianMatrixAndResidualVector(
            current_control_vars, trial_dofs)
        trial_res_norm = jnp.linalg.norm(trial_r,ord=2)
        return trial_res_norm

    def _PerformLineSearch(self,current_control_vars,applied_BC_dofs,delta_dofs,current_res_norm):
        step_size = self.nonlinear_solver_settings["line_search_step_size"]
        reduction_factor = self.nonlinear_solver_settings["line_search_reduction_factor"]
        min_step_size = self.nonlinear_solver_settings["line_search_min_step_size"]
        max_line_search_iters = self.nonlinear_solver_settings["line_search_maxiter"]

        accepted_dofs = applied_BC_dofs.at[:].add(step_size * delta_dofs)
        accepted_res_norm = self._ComputeResidualNorm(current_control_vars, accepted_dofs)
        accepted_step_size = step_size

        if (not jnp.isnan(accepted_res_norm)) and accepted_res_norm < current_res_norm:
            return accepted_dofs, accepted_res_norm, accepted_step_size, True

        for _ in range(1, max_line_search_iters):
            step_size *= reduction_factor
            if step_size < min_step_size:
                break

            trial_dofs = applied_BC_dofs.at[:].add(step_size * delta_dofs)
            trial_res_norm = self._ComputeResidualNorm(current_control_vars, trial_dofs)
            if (not jnp.isnan(trial_res_norm)) and trial_res_norm < current_res_norm:
                return trial_dofs, trial_res_norm, step_size, True

            accepted_dofs = trial_dofs
            accepted_res_norm = trial_res_norm
            accepted_step_size = step_size

        return accepted_dofs, accepted_res_norm, accepted_step_size, False

    @print_with_timestamp_and_execution_time
    def Solve(self,current_control_vars,current_dofs_np:np.array):
        current_dofs = jnp.array(current_dofs_np)
        load_increament = self.nonlinear_solver_settings["load_incr"]
        for load_fac in range(load_increament):
            fol_info(f"loadStep; increment:{load_fac+1}")
            applied_BC_dofs = self.fe_loss_function.ApplyDirichletBCOnDofVector(current_dofs,(load_fac+1)/load_increament)
            for i in range(self.nonlinear_solver_settings["maxiter"]):
                BC_applied_jac,BC_applied_r = self.fe_loss_function.ComputeJacobianMatrixAndResidualVector(
                                                                    current_control_vars,applied_BC_dofs)
                res_norm = jnp.linalg.norm(BC_applied_r,ord=2)
                if jnp.isnan(res_norm):
                    fol_info("Residual norm is NaN, check inputs!")
                    raise(ValueError("res_norm contains nan values!"))
                if res_norm<self.nonlinear_solver_settings["abs_tol"]:
                    fol_info(f"converged; iterations:{i+1},residuals_norm:{res_norm}")
                    break
                    
                delta_dofs = self.LinearSolve(BC_applied_jac,BC_applied_r,applied_BC_dofs)
                if self.nonlinear_solver_settings["line_search"]:
                    applied_BC_dofs, trial_res_norm, step_size, line_search_accepted = self._PerformLineSearch(
                        current_control_vars, applied_BC_dofs, delta_dofs, res_norm)
                    effective_delta_dofs = step_size * delta_dofs
                    delta_norm = jnp.linalg.norm(effective_delta_dofs,ord=2)
                    if line_search_accepted:
                        fol_info(f"line search accepted; step_size:{step_size},trial_residuals_norm:{trial_res_norm}")
                    else:
                        fol_info(f"line search reached minimum step; step_size:{step_size},trial_residuals_norm:{trial_res_norm}")
                else:
                    delta_norm = jnp.linalg.norm(delta_dofs,ord=2)
                    applied_BC_dofs = applied_BC_dofs.at[:].add(delta_dofs)

                if delta_norm<self.nonlinear_solver_settings["rel_tol"]:
                    fol_info(f"converged; iterations:{i+1},delta_norm:{delta_norm},residuals_norm:{res_norm}")
                    break
                elif i+1==self.nonlinear_solver_settings["maxiter"]:
                    fol_info(f"maximum num iterations:{i+1} acheived,delta_norm:{delta_norm},residuals_norm:{res_norm}")
                    break
                else:
                    fol_info(f"iteration:{i+1},delta_norm:{delta_norm},residuals_norm:{res_norm}")
            current_dofs = current_dofs.at[self.fe_loss_function.non_dirichlet_indices].set(applied_BC_dofs[self.fe_loss_function.non_dirichlet_indices])
        return applied_BC_dofs




