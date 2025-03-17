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
    """FE-based solver class.

    """
    @print_with_timestamp_and_execution_time
    def __init__(self, fe_solver_name: str, fe_loss_function: FiniteElementLoss, fe_solver_settings:dict={}) -> None:
        super().__init__(fe_solver_name)
        self.fe_loss_function = fe_loss_function
        self.fe_solver_settings = fe_solver_settings
        self.linear_solver_settings = {"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                       "maxiter":1000,"pre-conditioner":"ilu"}

    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:

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
        delta_dofs, info = bicgstab(tangent_matrix,
                                    -residual_vector,
                                    x0=dofs_vector,
                                    tol=self.linear_solver_settings["tol"],
                                    atol=self.linear_solver_settings["atol"],
                                    maxiter=self.linear_solver_settings["maxiter"])
        return delta_dofs
    
    @print_with_timestamp_and_execution_time
    def JaxDirectLinearSolver(self,tangent_matrix:BCOO,residual_vector:jnp.array,dofs_vector:jnp.array):
        A_sp_scipy = scipy.sparse.csr_array((tangent_matrix.data, (tangent_matrix.indices[:,0],tangent_matrix.indices[:,1])),
                                            shape=tangent_matrix.shape)
        
        delta_dofs = spsolve(data=A_sp_scipy.data, indices=A_sp_scipy.indices, 
                             indptr=A_sp_scipy.indptr, b=-residual_vector,
                             tol=self.linear_solver_settings["tol"])
        
        return delta_dofs
    
    @print_with_timestamp_and_execution_time
    def PETScLinearSolver(self,tangent_matrix:BCOO,residual_vector:jnp.array,dofs_vector:jnp.array):
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





