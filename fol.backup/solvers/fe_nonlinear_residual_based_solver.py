"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: July, 2024
 License: FOL/LICENSE
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from  .fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *
from fol.loss_functions.fe_loss import FiniteElementLoss

class FiniteElementNonLinearResidualBasedSolver(FiniteElementLinearResidualBasedSolver):
    """
    Nonlinear residual-based finite element solver using incremental Newton–Raphson.

    This solver extends :class:`~fol.solvers.fe_linear_residual_based_solver.FiniteElementLinearResidualBasedSolver`
    to support nonlinear problems by repeatedly assembling the tangent (Jacobian)
    matrix and residual vector and applying Newton–Raphson updates to the global
    degrees of freedom (DOFs).

    Loading is applied incrementally according to
    ``self.nonlinear_solver_settings["load_incr"]``. For each load step, the solver
    iterates until convergence is detected or the maximum number of Newton
    iterations is reached. Convergence history (residual norms and update norms)
    is stored per load step and can optionally be plotted via :meth:`PlotHistoryDict`.

    Typical applications include problems with geometric nonlinearities, nonlinear
    material behavior (e.g., hyperelasticity, plasticity, damage), or nonlinear
    boundary-condition effects.
    """

    @print_with_timestamp_and_execution_time
    def __init__(self, fe_solver_name: str, fe_loss_function: FiniteElementLoss, fe_solver_settings:dict={}, history_plot_settings:dict={}) -> None:
        super().__init__(fe_solver_name,fe_loss_function,fe_solver_settings)
        self.nonlinear_solver_settings = {"rel_tol":1e-8,"abs_tol":1e-8,"maxiter":20,"load_incr":5}
        self.default_history_plot_settings = {"plot":False,"criteria":["res_norm","delta_dofs_norm"],"save_directory":"."}
        self.history_plot_settings = history_plot_settings

    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:
        """
        Initialize the solver and merge nonlinear and plotting settings.

        This method calls the base class initialization and updates the
        nonlinear solver settings from ``fe_solver_settings["nonlinear_solver_settings"]``
        if provided. It also merges the user-specified history plotting settings
        with the default plotting configuration.
        """
        super().Initialize()
        if "nonlinear_solver_settings" in self.fe_solver_settings.keys():
            self.nonlinear_solver_settings = UpdateDefaultDict(self.nonlinear_solver_settings,
                                                                self.fe_solver_settings["nonlinear_solver_settings"])
        self.history_plot_settings = UpdateDefaultDict(self.default_history_plot_settings,self.history_plot_settings)

    def PlotHistoryDict(self,history_dict:dict):

        if self.history_plot_settings["plot"]:
            plot_dict = {key: [] for key in self.history_plot_settings["criteria"]}
            for plot_criterion,plot_values in plot_dict.items():
                for step,step_dict in history_dict.items():
                    plot_values.extend(step_dict[plot_criterion])

            plt.figure(figsize=(10, 5))
            for key,value in plot_dict.items():
                plt.semilogy(value, marker='o', markersize=6, linewidth=1.5, label=f"{key}")

            plt.title("Newton-Raphson History with Load Stepping")
            plt.xlabel("Cumulative Iteration")
            plt.ylabel("Log Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.history_plot_settings["save_directory"],"newton_raphson_history.png"), bbox_inches='tight')
            plt.close()

    @print_with_timestamp_and_execution_time
    def Solve(self,current_control_vars:jnp.array,current_dofs_np:jnp.array):
        """
        Solve the nonlinear FE system using incremental Newton–Raphson iterations.

        The load is applied over ``load_incr`` steps. For each load step, Dirichlet
        boundary conditions are applied at the current load factor, the Jacobian
        and residual are assembled via the loss function, and a Newton update is
        computed using :meth:`LinearSolve`. The DOFs are updated until convergence
        is detected or the iteration limit is reached. Convergence metrics are
        recorded per load step and can be plotted via :meth:`PlotHistoryDict`.

        Convergence is considered achieved when at least one of the following
        conditions is satisfied:
        - Residual norm < ``abs_tol``
        - Update norm < ``rel_tol``
        - Iteration count reaches ``maxiter``

        Args:
            current_control_vars (jax.numpy.ndarray):
                Control variables passed to the loss function (e.g., material or
                design parameters).
            current_dofs_np (jax.numpy.ndarray):
                Initial DOF vector. Converted to a JAX array internally.

        Returns:
            jax.numpy.ndarray:
                Converged DOF vector at the end of the final load step.

        Raises:
            ValueError:
                If the residual norm becomes NaN during Newton iterations.
        """
        current_dofs = jnp.asarray(current_dofs_np)
        current_control_vars = jnp.asarray(current_control_vars)
        num_load_steps = self.nonlinear_solver_settings["load_incr"]
        convergence_history = {}
        for load_step in range(1,num_load_steps+1):
            load_step_value = (load_step)/num_load_steps
            # increment load
            current_dofs = self.fe_loss_function.ApplyDirichletBCOnDofVector(current_dofs,load_step_value)
            newton_converged = False
            convergence_history[load_step] = {"res_norm":[],"delta_dofs_norm":[]}
            for i in range(1,self.nonlinear_solver_settings["maxiter"]+1):
                BC_applied_jac,BC_applied_r = self.fe_loss_function.ComputeJacobianMatrixAndResidualVector(
                                                                    current_control_vars,current_dofs)

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

                convergence_history[load_step]["res_norm"].append(res_norm)
                convergence_history[load_step]["delta_dofs_norm"].append(delta_norm)

                if newton_converged:
                    break

                # if not converged update
                current_dofs = current_dofs.at[:].add(delta_dofs)

            self.PlotHistoryDict(convergence_history)

        return current_dofs