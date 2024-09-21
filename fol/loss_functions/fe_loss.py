"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from  .loss import Loss
import jax
import jax.numpy as jnp
import warnings
from jax import jit,grad
from functools import partial
from abc import abstractmethod
from fol.tools.decoration_functions import *
from jax.experimental import sparse
from fol.mesh_input_output.mesh import Mesh
from fol.tools.fem_utilities import *

class FiniteElementLoss(Loss):
    """FE-based losse

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name)
        self.loss_settings = loss_settings
        self.dofs = self.loss_settings["ordered_dofs"]
        self.element_type = self.loss_settings["element_type"]
        self.fe_mesh = fe_mesh
        if "dirichlet_bc_dict" not in self.loss_settings.keys():
            fol_error("dirichlet_bc_dict should provided in the loss settings !")

    def __CreateDofsDict(self, dofs_list:list, dirichlet_bc_dict:dict):
        number_dofs_per_node = len(dofs_list)
        dirichlet_indices = []
        dirichlet_values = [] 
        dirichlet_dofs_boundary_dict = {}       
        for dof_index,dof in enumerate(dofs_list):
            dirichlet_dofs_boundary_dict[dof] = {}
            for boundary_name,boundary_value in dirichlet_bc_dict[dof].items():
                boundary_node_ids = jnp.array(self.fe_mesh.GetNodeSet(boundary_name))
                dirichlet_bc_indices = number_dofs_per_node*boundary_node_ids + dof_index

                boundary_start_index = len(dirichlet_indices)
                dirichlet_indices.extend(dirichlet_bc_indices.tolist())
                boundary_end_index = len(dirichlet_indices)

                dirichlet_dofs_boundary_dict[dof][boundary_name] = jnp.arange(boundary_start_index,boundary_end_index)

                dirichlet_bc_values = boundary_value * jnp.ones(dirichlet_bc_indices.size)
                dirichlet_values.extend(dirichlet_bc_values.tolist())
        
        self.dirichlet_indices = jnp.array(dirichlet_indices)
        self.dirichlet_values = jnp.array(dirichlet_values)
        all_indices = jnp.arange(number_dofs_per_node*self.fe_mesh.GetNumberOfNodes())
        self.non_dirichlet_indices = jnp.setdiff1d(all_indices, self.dirichlet_indices)

    def Initialize(self) -> None:
        self.number_dofs_per_node = len(self.dofs)
        self.total_number_of_dofs = len(self.dofs) * self.fe_mesh.GetNumberOfNodes()
        self.__CreateDofsDict(self.dofs,self.loss_settings["dirichlet_bc_dict"])
        self.number_of_unknown_dofs = self.non_dirichlet_indices.size

        # create full solution vector
        self.solution_vector = jnp.zeros(self.total_number_of_dofs)
        # apply dirichlet bcs
        self.solution_vector = self.solution_vector.at[self.dirichlet_indices].set(self.dirichlet_values)

        # now prepare gauss integration
        if "num_gp" in self.loss_settings.keys():
            self.num_gp = self.loss_settings["num_gp"]
            if self.num_gp == 1:
                g_points,g_weights = GaussQuadrature().one_point_GQ
            elif self.num_gp == 2:
                g_points,g_weights = GaussQuadrature().two_point_GQ
            elif self.num_gp == 3:
                g_points,g_weights = GaussQuadrature().three_point_GQ
            elif self.num_gp == 4:
                g_points,g_weights = GaussQuadrature().four_point_GQ
            else:
                raise ValueError(f" number gauss points {self.num_gp} is not supported ! ")
        else:
            g_points,g_weights = GaussQuadrature().one_point_GQ
            self.loss_settings["num_gp"] = 1
            self.num_gp = 1

        if not "compute_dims" in self.loss_settings.keys():
            raise ValueError(f"compute_dims must be provided in the loss settings of {self.GetName()}! ")

        self.dim = self.loss_settings["compute_dims"]

        if self.dim==1:
            self.g_points = jnp.array([[xi] for xi in g_points]).flatten()
            self.g_weights = jnp.array([[w_i] for w_i in g_weights]).flatten()
        elif self.dim==2:
            self.g_points = jnp.array([[xi, eta] for xi in g_points for eta in g_points]).flatten()
            self.g_weights = jnp.array([[w_i , w_j] for w_i in g_weights for w_j in g_weights]).flatten()
        elif self.dim==3:
            self.g_points = jnp.array([[xi,eta,zeta] for xi in g_points for eta in g_points for zeta in g_points]).flatten()
            self.g_weights = jnp.array([[w_i,w_j,w_k] for w_i in g_weights for w_j in g_weights for w_k in g_weights]).flatten()

        @jit
        def ConstructFullDofVector(known_dofs: jnp.array,unknown_dofs: jnp.array):
            solution_vector = jnp.zeros(self.total_number_of_dofs)
            solution_vector = self.solution_vector.at[self.non_dirichlet_indices].set(unknown_dofs)
            return solution_vector

        @jit
        def ConstructFullDofVectorParametricLearning(known_dofs: jnp.array,unknown_dofs: jnp.array):
            solution_vector = jnp.zeros(self.total_number_of_dofs)
            solution_vector = self.solution_vector.at[self.dirichlet_indices].set(known_dofs)
            solution_vector = self.solution_vector.at[self.non_dirichlet_indices].set(unknown_dofs)
            return solution_vector  

        if self.loss_settings.get("parametric_boundary_learning"):
            self.full_dof_vector_function = ConstructFullDofVectorParametricLearning
        else:
            self.full_dof_vector_function = ConstructFullDofVector

        self.__initialized = True

    @partial(jit, static_argnums=(0,))
    def GetFullDofVector(self,known_dofs: jnp.array,unknown_dofs: jnp.array) -> jnp.array:
        return self.full_dof_vector_function(known_dofs,unknown_dofs)

    def Finalize(self) -> None:
        pass

    def GetNumberOfUnknowns(self):
        return self.number_of_unknown_dofs
    
    def GetTotalNumberOfDOFs(self):
        return self.total_number_of_dofs

    @abstractmethod
    def ComputeElementEnergy(self):
        pass

    @abstractmethod
    def ComputeElementResiduals(self):
        pass

    @abstractmethod
    def ComputeElementStiffness(self):
        pass

    @abstractmethod
    def ComputeElementResidualsAndStiffness(self):
        pass

    @abstractmethod
    def ComputeElementEnergyVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,P):
        pass

    @abstractmethod
    def ComputeElementResidualsVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,P):
        pass

    @abstractmethod
    def ComputeElementResidualsAndStiffnessVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,P):
        pass

    @partial(jit, static_argnums=(0,))
    def ComputeElementsEnergies(self,total_control_vars,total_primal_vars):
        # parallel calculation of energies
        return jax.vmap(self.ComputeElementEnergyVmapCompatible,(0,None,None,None,None,None,None)) \
                        (self.fe_mesh.GetElementsIds(self.element_type),self.fe_mesh.GetElementsNodes(self.element_type),
                        self.fe_mesh.GetNodesX(),self.fe_mesh.GetNodesY(),self.fe_mesh.GetNodesZ(),
                        total_control_vars,total_primal_vars)

    @partial(jit, static_argnums=(0,))
    def ComputeTotalEnergy(self,total_control_vars,total_primal_vars):
        return jnp.sum(self.ComputeElementsEnergies(total_control_vars,total_primal_vars)) 

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def ComputeResidualVectorAD(self,total_control_vars: jnp.array,total_primal_vars: jnp.array):
        return jax.grad(self.ComputeTotalEnergy,argnums=1)(total_control_vars,total_primal_vars)

    # NOTE: this function should not be jitted since it is tested and gets much slower
    @print_with_timestamp_and_execution_time
    def ComputeResidualVector(self,total_control_vars: jnp.array,total_primal_vars: jnp.array):
        # parallel calculation of residuals
        elements_residuals = jnp.squeeze(jax.vmap(self.ComputeElementResidualsVmapCompatible,(0,None,None,None,None,None,None)) \
                                     (self.fe_mesh.GetElementsIds(self.element_type),self.fe_mesh.GetElementsNodes(self.element_type),
                                      self.fe_mesh.GetNodesX(),self.fe_mesh.GetNodesY(),self.fe_mesh.GetNodesZ(),
                                      total_control_vars,total_primal_vars))
        
        residuals_vector = jnp.zeros((self.total_number_of_dofs))
        for dof_idx in range(self.number_dofs_per_node):
            residuals_vector = residuals_vector.at[self.number_dofs_per_node*self.fe_mesh.GetElementsNodes(self.element_type)+dof_idx].add(jnp.squeeze(elements_residuals[:,dof_idx::self.number_dofs_per_node]))
        return residuals_vector
    
    @abstractmethod
    def ComputeJacobianIndices(self,nodes_ids:jnp.array):
        pass

    # NOTE: this function should not be jitted since it is tested and gets much slower
    @print_with_timestamp_and_execution_time
    def ComputeJacobianMatrixAndResidualVector(self,total_control_vars: jnp.array,total_primal_vars: jnp.array, apply_dirichlet_bc:bool=True):
        elements_residuals, elements_stiffness = jax.vmap(self.ComputeElementResidualsAndStiffnessVmapCompatible,(0,None,None,None,None,None,None)) \
                                                            (self.fe_mesh.GetElementsIds(self.element_type),self.fe_mesh.GetElementsNodes(self.element_type),
                                                            self.fe_mesh.GetNodesX(),self.fe_mesh.GetNodesY(),self.fe_mesh.GetNodesZ(),
                                                            total_control_vars,total_primal_vars)
        
        # first compute the global residual vector
        residuals_vector = jnp.zeros((self.total_number_of_dofs))
        for dof_idx in range(self.number_dofs_per_node):
            residuals_vector = residuals_vector.at[self.number_dofs_per_node*self.fe_mesh.GetElementsNodes(self.element_type)+dof_idx].add(jnp.squeeze(elements_residuals[:,dof_idx::self.number_dofs_per_node]))

        # second compute the global jacobian matrix  
        jacobian_data = jnp.ravel(elements_stiffness)

        indices = jax.vmap(self.ComputeJacobianIndices)(self.fe_mesh.GetElementsNodes(self.element_type)) # Get the indices
        indices = indices.reshape(-1,2)
        
        if apply_dirichlet_bc:
            # extract non-dirichlet BCs
            mask = ~jnp.isin(indices[:, 0], self.dirichlet_indices)
            jacobian_data = jacobian_data[mask]
            indices = indices[mask]
            # add dirichlet BCs
            diag_dr_bc_indices = jnp.hstack([self.dirichlet_indices.reshape(-1,1), self.dirichlet_indices.reshape(-1,1)])
            indices = jnp.vstack([indices,diag_dr_bc_indices])
            jacobian_data = jnp.append(jacobian_data, jnp.ones((len(self.dirichlet_indices))))

            # apply dirichlet BCs on residual vector  
            residuals_vector = residuals_vector.at[self.dirichlet_indices].set(0)
        
        K = sparse.BCOO((jacobian_data,indices),shape=(self.total_number_of_dofs,self.total_number_of_dofs))
        return K.sum_duplicates(), residuals_vector

    # NOTE this function return residual and jacobian matrix by differentiating the linear energy functions 
    # which only hold for linear problems   
    @print_with_timestamp_and_execution_time
    def ComputeJacobianMatrixAndResidualVectorAD(self,total_control_vars: jnp.array,total_primal_vars: jnp.array, apply_dirichlet_bc:bool=True):
        @jit
        def JitComputeJacobianMatrixAndResidualVectorAD(total_control_vars,total_primal_vars):
            residuals_AD_func = jax.grad(self.ComputeTotalEnergy,argnums=1)
            jacobian_AD_func = jax.jacfwd(residuals_AD_func,argnums=1)
            return jnp.squeeze(0.5*jacobian_AD_func(total_control_vars,total_primal_vars)),0.5*residuals_AD_func(total_control_vars,total_primal_vars)

        jacobian_AD, residuals_AD = JitComputeJacobianMatrixAndResidualVectorAD(total_control_vars,total_primal_vars)

        if apply_dirichlet_bc:
            # apply dirichlet BCs on residual vector  
            residuals_AD = residuals_AD.at[self.dirichlet_indices].set(0)
            # apply dirichlet BCs on jacobian matrix  
            jacobian_AD = jacobian_AD.at[self.dirichlet_indices,:].set(0)
            jacobian_AD = jacobian_AD.at[self.dirichlet_indices,self.dirichlet_indices].set(1)

        return jacobian_AD, residuals_AD

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,2,))
    def ApplyDirichletBC(self,full_dof_vector:jnp.array,load_increment:float=1.0):
        return full_dof_vector.at[self.dirichlet_indices].set(load_increment*self.dirichlet_values)
            


