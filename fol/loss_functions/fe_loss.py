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

        if self.initialized:
            return

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

        self.initialized = True

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
    def ComputeElement(self,
                       elem_xyz:jnp.array,
                       elem_controls:jnp.array,
                       elem_dofs:jnp.array) -> tuple[float, jnp.array, jnp.array]:
        pass

    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergy(self,
                             elem_xyz:jnp.array,
                             elem_controls:jnp.array,
                             elem_dofs:jnp.array) -> float:
        return self.ComputeElement(elem_xyz,elem_controls,elem_dofs)[0]
    
    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergyVmapCompatible(self,
                                           element_id:jnp.integer,
                                           elements_nodes:jnp.array,
                                           xyz:jnp.array,
                                           full_control_vector:jnp.array,
                                           full_dof_vector:jnp.array):
        return self.ComputeElementEnergy(xyz[elements_nodes[element_id],:],
                                         full_control_vector[elements_nodes[element_id]],
                                         full_dof_vector[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                         jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeElementsEnergies(self,total_control_vars:jnp.array,total_primal_vars:jnp.array):
        # parallel calculation of energies
        return jax.vmap(self.ComputeElementEnergyVmapCompatible,(0,None,None,None,None)) \
                        (self.fe_mesh.GetElementsIds(self.element_type),
                        self.fe_mesh.GetElementsNodes(self.element_type),
                        self.fe_mesh.GetNodesCoordinates(),
                        total_control_vars,
                        total_primal_vars)

    @partial(jit, static_argnums=(0,))
    def ComputeTotalEnergy(self,total_control_vars:jnp.array,total_primal_vars:jnp.array):
        return jnp.sum(self.ComputeElementsEnergies(total_control_vars,total_primal_vars)) 

    @partial(jit, static_argnums=(0,))
    def ComputeElementJacobianIndices(self,nodes_ids:jnp.array):
        nodes_ids *= self.number_dofs_per_node
        nodes_ids += jnp.arange(self.number_dofs_per_node).reshape(-1,1)
        indices_dof = nodes_ids.T.flatten()
        rows,cols = jnp.meshgrid(indices_dof,indices_dof,indexing='ij')#rows and columns
        indices = jnp.vstack((rows.ravel(),cols.ravel())).T #indices in global stiffness matrix
        return indices

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,2,))
    def ApplyDirichletBCOnDofVector(self,full_dof_vector:jnp.array,load_increment:float=1.0):
        return full_dof_vector.at[self.dirichlet_indices].set(load_increment*self.dirichlet_values)

    @partial(jit, static_argnums=(0,))
    def ApplyDirichletBCOnElementResidualAndJacobian(self,
                                                     elem_res:jnp.array,
                                                     elem_jac:jnp.array,
                                                     elem_BC_vec:jnp.array,
                                                     elem_mask_BC_vec:jnp.array):

        BC_matrix = jnp.zeros((elem_jac.shape))
        BC_matrix = jnp.fill_diagonal(BC_matrix, elem_BC_vec, inplace=False)

        mask_BC_matrix = jnp.zeros((elem_jac.shape))
        mask_BC_matrix = jnp.fill_diagonal(mask_BC_matrix, elem_mask_BC_vec, inplace=False)

        masked_diag_entries = jnp.diag(mask_BC_matrix @ elem_jac @ mask_BC_matrix)
        mask_BC_matrix = jnp.zeros((elem_jac.shape))
        mask_BC_matrix = jnp.fill_diagonal(mask_BC_matrix, masked_diag_entries, inplace=False)

        return   BC_matrix @ elem_res, BC_matrix @ elem_jac + mask_BC_matrix

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualAndJacobian(self,
                                          elem_xyz:jnp.array,
                                          elem_controls:jnp.array,
                                          elem_dofs:jnp.array,
                                          elem_BC:jnp.array,
                                          elem_mask_BC:jnp.array):
        _,re,ke = self.ComputeElement(elem_xyz,elem_controls,elem_dofs)
        return self.ApplyDirichletBCOnElementResidualAndJacobian(re,ke,elem_BC,elem_mask_BC)

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualAndJacobianVmapCompatible(self,element_id:jnp.integer,
                                                        elements_nodes:jnp.array,
                                                        xyz:jnp.array,
                                                        full_control_vector:jnp.array,
                                                        full_dof_vector:jnp.array,
                                                        full_dirichlet_BC_vec:jnp.array,
                                                        full_mask_dirichlet_BC_vec:jnp.array):
        return self.ComputeElementResidualAndJacobian(xyz[elements_nodes[element_id],:],
                                                      full_control_vector[elements_nodes[element_id]],
                                                      full_dof_vector[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      full_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      full_mask_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeSingleLoss(self,full_control_params:jnp.array,unknown_dofs:jnp.array):
        elems_energies = self.ComputeElementsEnergies(full_control_params.reshape(-1,1),
                                                      self.GetFullDofVector(full_control_params,
                                                                            unknown_dofs))
        # some extra calculation for reporting and not traced
        avg_elem_energy = jax.lax.stop_gradient(jnp.mean(elems_energies))
        max_elem_energy = jax.lax.stop_gradient(jnp.max(elems_energies))
        min_elem_energy = jax.lax.stop_gradient(jnp.min(elems_energies))
        return jnp.sum(elems_energies)**2,(min_elem_energy,max_elem_energy,avg_elem_energy)
    
    # NOTE: this function should not be jitted since it is tested and gets much slower
    @print_with_timestamp_and_execution_time
    def ComputeJacobianMatrixAndResidualVector(self,total_control_vars: jnp.array,total_primal_vars: jnp.array):
        
        BC_vector = jnp.ones((self.total_number_of_dofs))
        BC_vector = BC_vector.at[self.dirichlet_indices].set(0)
        mask_BC_vector = jnp.zeros((self.total_number_of_dofs))
        mask_BC_vector = mask_BC_vector.at[self.dirichlet_indices].set(1)
        
        elements_residuals, elements_stiffness = jax.vmap(self.ComputeElementResidualAndJacobianVmapCompatible,(0,None,None,None,None,None,None)) \
                                                            (self.fe_mesh.GetElementsIds(self.element_type),
                                                             self.fe_mesh.GetElementsNodes(self.element_type),
                                                             self.fe_mesh.GetNodesCoordinates(),
                                                             total_control_vars,
                                                             total_primal_vars,
                                                             BC_vector,
                                                             mask_BC_vector)

        # first compute the global residual vector
        residuals_vector = jnp.zeros((self.total_number_of_dofs))
        for dof_idx in range(self.number_dofs_per_node):
            residuals_vector = residuals_vector.at[self.number_dofs_per_node*self.fe_mesh.GetElementsNodes(self.element_type)+dof_idx].add(jnp.squeeze(elements_residuals[:,dof_idx::self.number_dofs_per_node]))

        # second compute the global jacobian matrix  
        jacobian_data = jnp.ravel(elements_stiffness)
        jacobian_indices = jax.vmap(self.ComputeElementJacobianIndices)(self.fe_mesh.GetElementsNodes(self.element_type)) # Get the indices
        jacobian_indices = jacobian_indices.reshape(-1,2)
        
        sparse_jacobian = sparse.BCOO((jacobian_data,jacobian_indices),shape=(self.total_number_of_dofs,self.total_number_of_dofs))
        
        return sparse_jacobian, residuals_vector