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
    """
    Adjoint finite element solver for gradient and sensitivity analysis.

    This class solves the adjoint system associated with a forward finite
    element problem. The adjoint formulation is typically used in
    PDE-constrained optimization, inverse problems, and sensitivity analysis
    to efficiently compute gradients of a scalar objective (loss) function
    with respect to control variables.

    The adjoint system is constructed using a
    :class:`FiniteElementResponse` object, which provides the loss functional,
    the adjoint Jacobian matrix, and the adjoint right-hand side vector.

    The class inherits from :class:`FiniteElementSolver` and reuses its
    linear system solution infrastructure.

    Mathematical Formulation
    ------------------------
    Let the forward problem be defined as:

        R(u, m) = 0

    where:
        u : state (degrees of freedom),
        m : control variables.

    Let the scalar objective (loss) be:

        J(u, m)

    The adjoint variable λ satisfies the adjoint equation:

        (∂R/∂u)ᵀ λ = - ∂J/∂u

    where:
        ∂R/∂u : Jacobian of the residual with respect to the state,
        ∂J/∂u : gradient of the loss with respect to the state.

    Boundary conditions are assumed to be applied directly to the adjoint
    Jacobian and right-hand side prior to solving the system.

    Parameters
    ----------
    adj_fe_solver_name : str
        Name of the adjoint finite element solver instance.
    fe_response : FiniteElementResponse
        Object responsible for defining the loss functional and assembling
        the adjoint Jacobian matrix and right-hand side vector.
    adj_fe_solver_settings : dict, optional
        Dictionary containing solver-specific settings such as linear solver
        type, tolerances, or preconditioning options. Default is an empty
        dictionary.

    Attributes
    ----------
    fe_response : FiniteElementResponse
        Reference to the finite element response object used to assemble
        adjoint quantities.
    """
    @print_with_timestamp_and_execution_time
    def __init__(self, adj_fe_solver_name: str, fe_response: FiniteElementResponse, adj_fe_solver_settings:dict={}) -> None:
        """
        Initialize the adjoint finite element solver.

        This constructor initializes the base finite element solver using
        the loss functional provided by the finite element response object
        and stores a reference to the response for adjoint assembly.

        Parameters
        ----------
        adj_fe_solver_name : str
            Name identifier for the adjoint solver.
        fe_response : FiniteElementResponse
            Finite element response object containing the loss definition
            and adjoint assembly routines.
        adj_fe_solver_settings : dict, optional
            Solver configuration dictionary. Default is an empty dictionary.
        """
        super().__init__(adj_fe_solver_name,fe_response.fe_loss,adj_fe_solver_settings)
        self.fe_response = fe_response

    @print_with_timestamp_and_execution_time
    def Solve(self,current_control_vars:jnp.array,current_dofs:jnp.array,current_adjoint_dofs:jnp.array):
        """
        Assemble and solve the adjoint linear system.

        This method constructs the adjoint Jacobian matrix and right-hand
        side vector based on the current control variables and forward
        solution. The resulting linear system is then solved for the
        adjoint degrees of freedom.

        The right-hand side vector is multiplied by ``-1`` to ensure
        consistency with the solver sign convention used in
        :meth:`LinearSolve`.

        Parameters
        ----------
        current_control_vars : jnp.array
            Current values of the control variables.
        current_dofs : jnp.array
            Current forward solution degrees of freedom.
        current_adjoint_dofs : jnp.array
            Initial guess or previous values of the adjoint degrees of
            freedom.

        Returns
        -------
        jnp.array
            The updated adjoint degrees of freedom obtained by solving
            the adjoint linear system.
        """
        BC_applied_jac,BC_applied_rhs = self.fe_response.ComputeAdjointJacobianMatrixAndRHSVector(current_control_vars,current_dofs)
        # here we need to multiply by -1 since the solver later mutiplies by -1
        BC_applied_rhs *= -1
        return self.LinearSolve(BC_applied_jac,BC_applied_rhs,current_adjoint_dofs)







