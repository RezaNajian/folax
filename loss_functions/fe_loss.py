"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from  .loss import Loss
import jax
import jax.numpy as jnp
from jax import jit,grad,jit
from functools import partial
from abc import abstractmethod

class FiniteElementLoss(Loss):
    """FE-based losse

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model, ordered_dofs:list):
        super().__init__(name)
        self.fe_model = fe_model
        self.dofs = ordered_dofs
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
    def ComputeElementEnergyVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,P):
        pass

    @partial(jit, static_argnums=(0,))
    def ComputeElementsEnergies(self,total_control_vars,total_primal_vars):
        # parallel calculation of energies
        return jax.vmap(self.ComputeElementEnergyVmapCompatible,(0,None,None,None,None,None,None)) \
                        (self.fe_model.GetElementsIds(),self.fe_model.GetElementsNodes()
                        ,self.fe_model.GetNodesX(),self.fe_model.GetNodesY(),self.fe_model.GetNodesZ(),
                        total_control_vars,total_primal_vars)

    @partial(jit, static_argnums=(0,))
    def ComputeTotalEnergy(self,total_control_vars,total_primal_vars):
        return jnp.sum(self.ComputeElementsEnergies(total_control_vars,total_primal_vars))
    
    @partial(jit, static_argnums=(0,))
    def Compute_R(self,total_control_vars,total_primal_vars):
        return grad(self.ComputeTotalEnergy,argnums=1)(total_control_vars,total_primal_vars)
    
    @partial(jit, static_argnums=(0,))
    def Compute_DR_DP(self,total_control_vars,total_primal_vars):
        return jax.jacfwd(self.Compute_R,argnums=1)(total_control_vars,total_primal_vars)
    
    @partial(jit, static_argnums=(0,))
    def Compute_DR_DC(self,total_control_vars,total_primal_vars):
        return jax.jacfwd(self.Compute_R,argnums=0)(total_control_vars,total_primal_vars)
    
    @partial(jit, static_argnums=(0,))
    def ExtendUnknowDOFsWithBC(self,unknown_dofs):
        full_dofs = jnp.zeros(self.total_number_of_dofs)
        for dof_index,dof in enumerate(self.dofs):
            # apply drichlet dofs
            dirichlet_indices = self.number_dofs_per_node*self.fe_model.GetDofsDict()[dof]["dirichlet_nodes_ids"] + dof_index
            full_dofs = full_dofs.at[dirichlet_indices].set(self.fe_model.GetDofsDict()[dof]["dirichlet_nodes_dof_value"])
            # apply non-drichlet dofs
            non_dirichlet_indices = self.number_dofs_per_node*self.fe_model.GetDofsDict()[dof]["non_dirichlet_nodes_ids"] + dof_index
            full_dofs = full_dofs.at[non_dirichlet_indices].set(unknown_dofs[dof_index::self.number_dofs_per_node])
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

    @partial(jit, static_argnums=(0,))
    def ApplyBCOnDOFs(self,full_dof_vector):
        full_dof_vector = full_dof_vector.reshape(-1)
        for dof_index,dof in enumerate(self.dofs):
            # set values on drichlet dofs 
            dirichlet_indices = self.number_dofs_per_node*self.fe_model.GetDofsDict()[dof]["dirichlet_nodes_ids"] + dof_index
            full_dof_vector = full_dof_vector.at[dirichlet_indices].set(self.fe_model.GetDofsDict()[dof]["dirichlet_nodes_dof_value"])
        return full_dof_vector
            


