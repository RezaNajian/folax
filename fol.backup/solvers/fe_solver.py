"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/LICENSE
"""
import scipy
import jax.numpy as jnp
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *
from fol.loss_functions.fe_loss import FiniteElementLoss
from  .solver import Solver
from jax.experimental.sparse import BCOO
from jax.scipy.sparse.linalg import bicgstab
from jax.experimental.sparse.linalg import spsolve
try:
    from petsc4py import PETSc
    petsc_available = True
except ImportError:
    petsc_available = False


class FiniteElementSolver(Solver):
    """
    Base solver for finite element systems driven by a :class:`FiniteElementLoss`.

    This class provides a common interface for solvers that operate on FE loss
    functions and require assembling and solving linear systems of the form::

        K(u, p) * delta = -R(u, p)

    where ``K`` is a sparse tangent (Jacobian) matrix and ``R`` is a residual
    vector produced by a :class:`fol.loss_functions.fe_loss.FiniteElementLoss.ComputeJacobianMatrixAndResidualVector`
    instance.

    The class mainly manages linear-solver selection and configuration. It does
    not define a problem-specific ``Solve`` routine; derived solver classes
    implement their own ``Solve`` method (e.g., linear residual-based solvers,
    nonlinear Newton solvers, adjoint solvers) and call ``LinearSolve`` to solve
    the assembled linear system.

    Args:
        fe_solver_name (str):
            Name identifier for the solver instance.
        fe_loss_function (FiniteElementLoss):
            Finite element loss object that provides problem-specific residuals
            and tangent matrices.
        fe_solver_settings (dict, optional):
            Solver configuration dictionary. If it contains the key
            ``"linear_solver_settings"``, the settings are merged with defaults.

    Attributes:
        fe_loss_function (FiniteElementLoss):
            Loss object used to assemble residuals and tangent matrices.
        fe_solver_settings (dict):
            User-provided solver settings dictionary.
        linear_solver_settings (dict):
            Linear solver configuration with default keys:
            ``"solver"``, ``"tol"``, ``"atol"``, ``"maxiter"``,
            and ``"pre-conditioner"``.
        LinearSolve (callable):
            Function handle assigned in :meth:`Initialize` to the selected
            linear solve routine.

    Notes:
        Supported linear solver backends are selected via
        ``linear_solver_settings["solver"]``. The available options are:
        ``"JAX-bicgstab"``, ``"JAX-direct"``, and PETSc-based Krylov methods
        ``"PETSc-bcgsl"``, ``"PETSc-tfqmr"``, ``"PETSc-minres"``,
        ``"PETSc-gmres"``. If PETSc is requested but unavailable, the solver
        falls back to ``"JAX-bicgstab"``.
    """
    @print_with_timestamp_and_execution_time
    def __init__(self, fe_solver_name: str, fe_loss_function: FiniteElementLoss, fe_solver_settings:dict={}) -> None:
        """
        Construct the solver and set default linear solver settings.

        Args:
            fe_solver_name (str):
                Name identifier for the solver instance.
            fe_loss_function (FiniteElementLoss):
                Finite element loss object used by derived solvers to assemble
                the tangent matrix and residual vector.
            fe_solver_settings (dict, optional):
                Solver configuration dictionary. Default is an empty dict.

        Returns:
            None
        """
        super().__init__(fe_solver_name)
        self.fe_loss_function = fe_loss_function
        self.fe_solver_settings = fe_solver_settings
        self.linear_solver_settings = {"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                       "maxiter":1000,"pre-conditioner":"ilu"}

    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:
        """
        Initialize and select the linear solver backend.

        This method merges user-provided ``linear_solver_settings`` (if present)
        with defaults and assigns the ``LinearSolve`` callable to one of the
        available backends.

        Returns:
            None

        Raises:
            ValueError:
                If an unknown solver name is provided in
                ``linear_solver_settings["solver"]``.
        """
        if "linear_solver_settings" in self.fe_solver_settings.keys():
            self.linear_solver_settings = UpdateDefaultDict(self.linear_solver_settings,
                                                            self.fe_solver_settings["linear_solver_settings"])

        linear_solver = self.linear_solver_settings["solver"]
        available_linear_solver = ["PETSc-bcgsl","PETSc-tfqmr","PETSc-minres","PETSc-gmres",
                                   "JAX-direct","JAX-bicgstab"]

        if linear_solver=="JAX-direct":
            self.LinearSolve = self.JaxDirectLinearSolver
        elif linear_solver=="JAX-bicgstab":
            self.LinearSolve = self.JaxBicgstabLinearSolver
        elif linear_solver in ["PETSc-bcgsl","PETSc-tfqmr","PETSc-minres","PETSc-gmres"]:
            if petsc_available:
                self.LinearSolve = self.PETScLinearSolver
                self.PETSc_ksp_type = linear_solver.split('-')[1]
            else:
                fol_warning(f"petsc4py is not available, falling back to the defualt iterative solver: JAX-bicgstab ")
                self.LinearSolve = self.JaxBicgstabLinearSolver
        else:
            fol_error(f"linear solver {linear_solver} does exist, available options are {available_linear_solver}")

    @print_with_timestamp_and_execution_time
    def JaxBicgstabLinearSolver(self,tangent_matrix:BCOO,residual_vector:jnp.array,dofs_vector:jnp.array):
        """
        Solve the linear system using JAX BiCGSTAB.

        The solver computes a DOF increment by solving::

            tangent_matrix * delta_dofs = -residual_vector

        using the iterative BiCGSTAB method from ``jax.scipy.sparse.linalg``.
        The optional initial guess is taken as ``dofs_vector``.

        Args:
            tangent_matrix (jax.experimental.sparse.BCOO):
                Global tangent (Jacobian) matrix in BCOO sparse format.
            residual_vector (jax.numpy.ndarray):
                Global residual vector.
            dofs_vector (jax.numpy.ndarray):
                Initial guess for the iterative solver.

        Returns:
            jax.numpy.ndarray:
                Increment vector ``delta_dofs`` solving the linear system.
        """
        delta_dofs, info = bicgstab(tangent_matrix,
                                    -residual_vector,
                                    x0=dofs_vector,
                                    tol=self.linear_solver_settings["tol"],
                                    atol=self.linear_solver_settings["atol"],
                                    maxiter=self.linear_solver_settings["maxiter"])
        return delta_dofs

    @print_with_timestamp_and_execution_time
    def JaxDirectLinearSolver(self,tangent_matrix:BCOO,residual_vector:jnp.array,dofs_vector:jnp.array):
        """
        Solve the linear system using a direct sparse solve.

        The input BCOO matrix is converted to a SciPy CSR sparse matrix and
        solved using ``jax.experimental.sparse.linalg.spsolve`` with the right
        hand side ``-residual_vector``.

        Args:
            tangent_matrix (jax.experimental.sparse.BCOO):
                Global tangent (Jacobian) matrix in BCOO sparse format.
            residual_vector (jax.numpy.ndarray):
                Global residual vector.
            dofs_vector (jax.numpy.ndarray):
                Unused by the direct solver (kept for a consistent interface).

        Returns:
            jax.numpy.ndarray:
                Increment vector ``delta_dofs`` solving the linear system.
        """
        A_sp_scipy = scipy.sparse.csr_array((tangent_matrix.data, (tangent_matrix.indices[:,0],tangent_matrix.indices[:,1])),
                                            shape=tangent_matrix.shape)

        delta_dofs = spsolve(data=A_sp_scipy.data, indices=A_sp_scipy.indices,
                             indptr=A_sp_scipy.indptr, b=-residual_vector,
                             tol=self.linear_solver_settings["tol"])

        return delta_dofs

    @print_with_timestamp_and_execution_time
    def PETScLinearSolver(self,tangent_matrix:BCOO,residual_vector:jnp.array,dofs_vector:jnp.array):
        """
        Solve the linear system using PETSc KSP.

        The input BCOO matrix is converted to a SciPy CSR matrix and then
        wrapped into a PETSc AIJ matrix. The right hand side is set to
        ``-residual_vector``. The KSP type is selected from the solver name
        provided in :meth:`Initialize` (e.g., ``gmres``, ``minres``, ``tfqmr``).
        The PETSc preconditioner type is set via
        ``linear_solver_settings["pre-conditioner"]``.

        Args:
            tangent_matrix (jax.experimental.sparse.BCOO):
                Global tangent (Jacobian) matrix in BCOO sparse format.
            residual_vector (jax.numpy.ndarray):
                Global residual vector.
            dofs_vector (jax.numpy.ndarray):
                Unused by PETSc (kept for a consistent interface).

        Returns:
            numpy.ndarray:
                Increment vector ``delta_dofs`` returned by PETSc.

        Raises:
            RuntimeError:
                If PETSc is not available in the current environment.
        """
        A_sp_scipy = scipy.sparse.csr_array((tangent_matrix.data, (tangent_matrix.indices[:,0],tangent_matrix.indices[:,1])),
                                            shape=tangent_matrix.shape)


        A = PETSc.Mat().createAIJ(size=A_sp_scipy.shape, csr=(A_sp_scipy.indptr.astype(PETSc.IntType, copy=False),
                                                       A_sp_scipy.indices.astype(PETSc.IntType, copy=False), A_sp_scipy.data))

        rhs = PETSc.Vec().createSeq(len(residual_vector))
        rhs.setValues(range(len(residual_vector)), np.array(-residual_vector))
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setFromOptions()
        ksp.setType(self.PETSc_ksp_type)
        ksp.pc.setType(self.linear_solver_settings["pre-conditioner"])

        if self.PETSc_ksp_type == 'tfqmr':
            ksp.pc.setFactorSolverType('mumps')

        delta_dofs = PETSc.Vec().createSeq(len(residual_vector))
        ksp.solve(rhs, delta_dofs)

        return delta_dofs.getArray()

    def Finalize(self) -> None:
        pass





