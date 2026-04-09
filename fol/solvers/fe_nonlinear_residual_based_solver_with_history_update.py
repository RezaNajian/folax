"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: July, 2024
 License: FOL/LICENSE
"""
import jax.numpy as jnp
from  .fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *


class FiniteElementNonLinearResidualBasedSolverWithStateUpdate(FiniteElementNonLinearResidualBasedSolver):
    """
    Nonlinear residual-based finite element solver with Gauss-point state updates.

    This solver extends
    :class:`~fol.solvers.fe_nonlinear_residual_based_solver.FiniteElementNonLinearResidualBasedSolver`
    by coupling global Newton–Raphson iterations for the degrees of freedom
    with local updates of Gauss-point (integration-point) state variables
    such as internal or history-dependent quantities.

    The solution procedure is performed using incremental load steps defined
    by ``nonlinear_solver_settings["load_incr"]``. For each load step, the
    solver applies boundary conditions, performs Newton iterations, updates
    local state variables, and records convergence information.

    This class is intended for nonlinear, history-dependent problems such as
    elastoplasticity, damage, or viscoelasticity, where consistent updates of
    integration-point state variables are required.
    """
    @print_with_timestamp_and_execution_time
    def Solve(self,current_control_vars:jnp.array,current_dofs:jnp.array,current_state:jnp.array=None,return_all_steps:bool=False):
        """
        Solve the nonlinear finite element system using incremental loading.

        The solver advances the solution through a sequence of load steps.
        Within each load step, Newton–Raphson iterations are performed to
        enforce global equilibrium. At every Newton iteration, the Jacobian
        matrix, residual vector, and updated Gauss-point state variables are
        assembled by the loss function.

        Convergence is declared when at least one of the following conditions
        is satisfied: the residual norm falls below the absolute tolerance,
        the update norm falls below the relative tolerance, or the maximum
        number of Newton iterations is reached.

        Args:
            current_control_vars (jax.numpy.ndarray):
                Control variables used to assemble the system (e.g., material
                parameters).
            current_dofs (jax.numpy.ndarray):
                Initial degrees of freedom representing the current global
                state.
            current_state (jax.numpy.ndarray, optional):
                Initial Gauss-point state variables. If not provided, a zero
                state field is initialized based on the element type and
                integration scheme.
            return_all_steps (bool, optional):
                If ``False`` (default), only the final load step solution and
                state are returned. If ``True``, the solution and state for
                all load steps are returned.

        Returns:
            Tuple[jax.numpy.ndarray, jax.numpy.ndarray, dict]:
                The updated degrees of freedom, the corresponding Gauss-point
                state variables, and a dictionary containing convergence
                history for each load step.

        Raises:
            ValueError:
                If the residual norm becomes NaN during the Newton iterations,
                indicating a breakdown of the nonlinear solve.
        """
        current_dofs = jnp.asarray(current_dofs)
        current_control_vars = jnp.asarray(current_control_vars)
        num_load_steps = self.nonlinear_solver_settings["load_incr"]
        # --- init per-GP state (committed) once ---
        nelem = self.fe_loss_function.fe_mesh.GetNumberOfElements(self.fe_loss_function.element_type)
        ngp   = self.fe_loss_function.fe_element.GetIntegrationData()[0].shape[0]
        if current_state is None:
            if self.fe_loss_function.element_type=="quad":
                current_state = jnp.zeros((nelem, ngp, 4))
            else:
                current_state = jnp.zeros((nelem, ngp, 7))
        else:
            current_state = jnp.asarray(current_state)

        solution_history_dict = {}
        for load_step in range(1,num_load_steps+1):
            load_step_value = (load_step)/num_load_steps
            # increment load
            current_dofs = self.fe_loss_function.ApplyDirichletBCOnDofVector(current_dofs,load_step_value)
            newton_converged = False
            solution_history_dict[load_step] = {"res_norm":[],"delta_dofs_norm":[]}
            for i in range(1,self.nonlinear_solver_settings["maxiter"]+1):
                new_state,BC_applied_jac,BC_applied_r = self.fe_loss_function.ComputeJacobianMatrixAndResidualVector(
                                                                    current_control_vars,current_dofs,old_state_gps=current_state)

                # check residuals norm
                res_norm = jnp.linalg.norm(BC_applied_r,ord=2)
                if jnp.isnan(res_norm):
                    fol_info(
                        "\n"
                        "──────────────────── NEWTON ERROR ────────────────────\n"
                        "  Residual norm has become NaN.\n"
                        "  Possible causes:\n"
                        "    • Divergent Newton iteration\n"
                        "    • Inconsistent or ill-posed boundary conditions\n"
                        "    • Invalid / non-physical material parameters or state\n"
                        "    • Singular or severely ill-conditioned stiffness matrix\n"
                        "    • Severely distorted mesh (element quality breakdown)\n"
                        "───────────────────────────────────────────────────────\n"
                    )
                    raise ValueError("Residual norm contains NaN values.")

                # linear solve and calculate nomrs
                delta_dofs = self.LinearSolve(BC_applied_jac,BC_applied_r,current_dofs)
                delta_norm = jnp.linalg.norm(delta_dofs,ord=2)

                newton_converged = (
                    res_norm < self.nonlinear_solver_settings["abs_tol"] or
                    delta_norm < self.nonlinear_solver_settings["rel_tol"] or
                    i == self.nonlinear_solver_settings["maxiter"]
                )

                indent = " " * 5
                fol_info(
                    f"\n"
                    f"{indent} ───────── Load Step {load_step} ─────────\n"
                    f"{indent}   Newton Iteration : {i} (max = {self.nonlinear_solver_settings['maxiter']})\n"
                    f"{indent}   Residual Norm    : {res_norm:.3e} (abs_tol = {self.nonlinear_solver_settings['abs_tol']:.3e})\n"
                    f"{indent}   Δ DOFs Norm      : {delta_norm:.3e} (rel_tol = {self.nonlinear_solver_settings['rel_tol']:.3e})\n"
                    f"{indent}   Converged        : {'True' if newton_converged else 'False'}\n"
                    f"{indent}────────────────────────────────────────────"
                )

                solution_history_dict[load_step]["res_norm"].append(res_norm)
                solution_history_dict[load_step]["delta_dofs_norm"].append(delta_norm)

                if newton_converged:
                    break

                # if not converged update
                current_dofs = current_dofs.at[:].add(delta_dofs)
                current_state = new_state

            if return_all_steps:
                # Initialize on first load step
                if load_step == 1:
                    load_steps_solutions = jnp.copy(current_dofs)
                    load_steps_states = jnp.copy(current_state[None, ...])
                else:
                    load_steps_solutions = jnp.vstack([load_steps_solutions, current_dofs])
                    load_steps_states = jnp.vstack([load_steps_states, current_state[None, ...]])
            else:
                # Always only return the last step
                load_steps_solutions = jnp.copy(current_dofs)
                load_steps_states = jnp.copy(current_state)

            self.PlotHistoryDict(solution_history_dict)

        return load_steps_solutions, load_steps_states, solution_history_dict

