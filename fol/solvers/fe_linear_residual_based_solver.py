"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/LICENSE
"""
import jax.numpy as jnp
from  .fe_solver import FiniteElementSolver
from fol.tools.decoration_functions import *

class FiniteElementLinearResidualBasedSolver(FiniteElementSolver):
    """
    Linear residual-based finite element solver.

    This solver advances the solution of a (typically) linear finite element
    system using a residual-based update. Given the current state (DOFs) and
    control variables, it:

    1) Applies Dirichlet boundary conditions to the current DOF vector.
    2) Assembles the Jacobian matrix and residual vector of the FE loss/residual.
    3) Solves a linear system to obtain an increment (update) in DOFs.
    4) Returns the updated DOF vector.

    The assembly and boundary-condition operations are delegated to the
    underlying loss function (``self.fe_loss_function``), while the linear
    solve is handled by :meth:`FiniteElementSolver.LinearSolve`.

    Linear Residual Solver Formulation
    ----------------------------------
    Let ``u`` be the vector of degrees of freedom (state) and ``m`` the control
    variables. After enforcing Dirichlet boundary conditions on ``u``, the loss
    function provides:

    - ``J``: the Jacobian matrix (e.g., tangent stiffness matrix)
    - ``r``: the residual vector

    The solver computes an increment ``Δu`` from a linear system of the form::

        J(u, m) Δu = r(u, m)

    and updates the state as::

        u_new = u_cu + Δu

    where ``u_cu`` is the given/current DOF vector after applying Dirichlet boundary
    conditions.

    Notes
    -----
    - This class is intended for *linear* (or linearized) problems where a
      single residual-based correction is sufficient per solve call.
    - Sign conventions for the residual (whether the system is ``J Δu = r`` or
      ``J Δu = -r``) are assumed to be consistent with the implementation of
      :meth:`FiniteElementSolver.LinearSolve` and the loss function assembly.

    """
    @print_with_timestamp_and_execution_time
    def Solve(self,current_control_vars:jnp.array,current_dofs:jnp.array):
        """
        Assemble and solve a linear residual-based update for the DOFs.

        This method applies Dirichlet boundary conditions to the current DOFs,
        then uses the FE loss function to compute the Jacobian matrix and
        residual vector. It solves the resulting linear system to obtain the DOF
        increment and returns the updated DOF vector.

        Parameters
        ----------
        current_control_vars : jnp.array
            Current values of the control variables (design/parameters) used to
            assemble the system.
        current_dofs : jnp.array
            Current degrees of freedom (state vector) serving as the starting
            point for the update.

        Returns
        -------
        jnp.array
            Updated degrees of freedom after applying Dirichlet boundary
            conditions and adding the computed increment.
        """
        BC_applied_dofs = self.fe_loss_function.ApplyDirichletBCOnDofVector(current_dofs)
        BC_applied_jac,BC_applied_r = self.fe_loss_function.ComputeJacobianMatrixAndResidualVector(
                                            current_control_vars,BC_applied_dofs)

        delta_dofs = self.LinearSolve(BC_applied_jac,BC_applied_r,BC_applied_dofs)

        return BC_applied_dofs + delta_dofs







