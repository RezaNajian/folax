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
from fol.computational_models.fe_model import FiniteElementModel
from fol.tools.fem_utilities import *

class FiniteElementLoss(Loss):
    """FE-based losse

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model: FiniteElementModel, ordered_dofs: list, loss_settings: dict={}):
        super().__init__(name)
        self.fe_model = fe_model
        self.dofs = ordered_dofs
        self.loss_settings = loss_settings
        self.number_of_dirichlet_dofs = 0
        self.number_of_unknown_dofs = 0
        for dof in self.dofs:
            if not dof in self.fe_model.GetDofsDict().keys():
                raise ValueError(f"No boundary conditions found for dof {dof} in dofs_dict of fe model ! ")
            if not "non_dirichlet_nodes_ids" in self.fe_model.GetDofsDict()[dof].keys():
                raise ValueError(f"No non_dirichlet_nodes_ids found for dof {dof} in dofs_dict of fe model ! ")
            if not "dirichlet_nodes_ids" in self.fe_model.GetDofsDict()[dof].keys():
                raise ValueError(f"No dirichlet_nodes_ids found for dof {dof} in dofs_dict of fe model ! ")
            if not "dirichlet_nodes_dof_value" in self.fe_model.GetDofsDict()[dof].keys():
                raise ValueError(f"No dirichlet_nodes_dof_value found for dof {dof} in dofs_dict of fe model ! ")

            self.number_of_dirichlet_dofs += self.fe_model.GetDofsDict()[dof]["dirichlet_nodes_ids"].shape[-1]
            self.number_of_unknown_dofs += self.fe_model.GetDofsDict()[dof]["non_dirichlet_nodes_ids"].shape[-1]

        if len(self.dofs) * self.fe_model.GetNumberOfNodes() != self.number_of_dirichlet_dofs + self.number_of_unknown_dofs:
            raise ValueError(f"number of dirichlet dofs: {self.number_of_dirichlet_dofs} + number of unknown dofs:" \
                             + f"{self.number_of_unknown_dofs} do not match with number of dofs: {len(self.dofs)} * number of nodes: {self.fe_model.GetNumberOfNodes()} ")

        self.total_number_of_dofs = len(self.dofs) * self.fe_model.GetNumberOfNodes()
        self.number_dofs_per_node = len(self.dofs)

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
            fol_warning(f"number of gauss points is set to 1 for loss {self.GetName()}!")

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

    def Initialize(self) -> None:
        pass

    def Finalize(self) -> None:
        pass

    def GetNumberOfUnknowns(self):
        return self.number_of_unknown_dofs

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

    @partial(jit, static_argnums=(0,))
    def ComputeElementsEnergies(self,total_control_vars,total_primal_vars):
        # parallel calculation of energies
        return jax.vmap(self.ComputeElementEnergyVmapCompatible,(0,None,None,None,None,None,None)) \
                        (self.fe_model.GetElementsIds(),self.fe_model.GetElementsNodes()
                        ,self.fe_model.GetNodesX(),self.fe_model.GetNodesY(),self.fe_model.GetNodesZ(),
                        total_control_vars,total_primal_vars)

    # @partial(jit, static_argnums=(0,))
    @print_with_timestamp_and_execution_time
    def ComputeResiduals(self,total_control_vars,total_primal_vars):
        # parallel calculation of residuals
        elements_residuals = jax.vmap(self.ComputeElementResidualsVmapCompatible,(0,None,None,None,None,None,None)) \
                                                (self.fe_model.GetElementsIds(),self.fe_model.GetElementsNodes()
                                                ,self.fe_model.GetNodesX(),self.fe_model.GetNodesY(),self.fe_model.GetNodesZ(),
                                                total_control_vars,total_primal_vars)

        problem_size = self.number_dofs_per_node*self.fe_model.GetNumberOfNodes()
        residuals = jnp.zeros(problem_size)
        for elem_idx, element_nodes in enumerate(self.fe_model.GetElementsNodes()):
            dof_idx = ((self.number_dofs_per_node*element_nodes)[:, jnp.newaxis] +jnp.arange(3)).reshape(-1)
            residuals = residuals.at[dof_idx].add(jnp.squeeze(elements_residuals[elem_idx]))

        return residuals
    
    # @partial(jit, static_argnums=(0,))
    @print_with_timestamp_and_execution_time
    def ComputeResidualsAndStiffness(self,total_control_vars,total_primal_vars):
        # parallel calculation of residuals
        elements_residuals, elements_stiffness = jax.vmap(self.ComputeElementResidualsAndStiffnessVmapCompatible,(0,None,None,None,None,None,None)) \
                                                (self.fe_model.GetElementsIds(),self.fe_model.GetElementsNodes()
                                                ,self.fe_model.GetNodesX(),self.fe_model.GetNodesY(),self.fe_model.GetNodesZ(),
                                                total_control_vars,total_primal_vars)

        problem_size = self.number_dofs_per_node*self.fe_model.GetNumberOfNodes()
        residuals = jnp.zeros(problem_size)
        stiffness = jnp.zeros((problem_size,problem_size))
        for elem_idx, element_nodes in enumerate(self.fe_model.GetElementsNodes()):
            dof_idx = ((self.number_dofs_per_node*element_nodes)[:, jnp.newaxis] +jnp.arange(self.number_dofs_per_node)).reshape(-1)
            residuals = residuals.at[dof_idx].add(jnp.squeeze(elements_residuals[elem_idx]))
            stiffness = stiffness.at[dof_idx[:, None],dof_idx].add(elements_stiffness[elem_idx])

        return residuals,stiffness

    @partial(jit, static_argnums=(0,))
    def ComputeTotalEnergy(self,total_control_vars,total_primal_vars):
        return jnp.sum(self.ComputeElementsEnergies(total_control_vars,total_primal_vars))
    
    @partial(jit, static_argnums=(0,))
    def Compute_DR_DC(self,total_control_vars,total_primal_vars):
        return jax.jacfwd(self.Compute_R,argnums=0)(total_control_vars,total_primal_vars)
    
    @partial(jit, static_argnums=(0,))
    def ExtendUnknowDOFsWithBC(self,unknown_dofs):
        full_dofs = jnp.zeros(self.total_number_of_dofs)
        unknown_dof_start_index = 0
        for dof_index,dof in enumerate(self.dofs):
            # apply drichlet dofs
            dirichlet_indices = self.number_dofs_per_node*self.fe_model.GetDofsDict()[dof]["dirichlet_nodes_ids"] + dof_index
            full_dofs = full_dofs.at[dirichlet_indices].set(self.fe_model.GetDofsDict()[dof]["dirichlet_nodes_dof_value"])
            # apply non-drichlet dofs
            non_dirichlet_indices = self.number_dofs_per_node*self.fe_model.GetDofsDict()[dof]["non_dirichlet_nodes_ids"] + dof_index
            unknown_dof_end_index = unknown_dof_start_index + non_dirichlet_indices.shape[-1]
            full_dofs = full_dofs.at[non_dirichlet_indices].set(unknown_dofs[unknown_dof_start_index:unknown_dof_end_index])
            unknown_dof_start_index = unknown_dof_end_index
        return full_dofs
    
    @partial(jit, static_argnums=(0,))
    def ApplyBCOnR(self,full_residual_vector):
        for dof_index,dof in enumerate(self.dofs):
            # set residuals on drichlet dofs to zero
            dirichlet_indices = self.number_dofs_per_node*self.fe_model.GetDofsDict()[dof]["dirichlet_nodes_ids"] + dof_index
            full_residual_vector = full_residual_vector.at[dirichlet_indices].set(0.0)
        return full_residual_vector
    
    @partial(jit, static_argnums=(0,))
    def ApplyBCOnMatrix(self,full_matrix):
        for dof_index,dof in enumerate(self.dofs):
            dirichlet_indices = self.number_dofs_per_node*self.fe_model.GetDofsDict()[dof]["dirichlet_nodes_ids"] + dof_index
            full_matrix = full_matrix.at[dirichlet_indices,:].set(0)
            full_matrix = full_matrix.at[dirichlet_indices,dirichlet_indices].set(1)
        return full_matrix

    @partial(jit, static_argnums=(0,2,))
    def ApplyBCOnDOFs(self,full_dof_vector,load_increment=1):
        full_dof_vector = full_dof_vector.reshape(-1)
        for dof_index,dof in enumerate(self.dofs):
            # set values on drichlet dofs 
            dirichlet_indices = self.number_dofs_per_node*self.fe_model.GetDofsDict()[dof]["dirichlet_nodes_ids"] + dof_index
            full_dof_vector = full_dof_vector.at[dirichlet_indices].set(load_increment*self.fe_model.GetDofsDict()[dof]["dirichlet_nodes_dof_value"])
        return full_dof_vector
            


