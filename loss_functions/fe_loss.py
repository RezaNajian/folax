"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/License.txt
"""
from  .loss import Loss
import jax
from jax import jit,grad,jit
from functools import partial
from abc import abstractmethod

class FiniteElementLoss(Loss):
    """FE-based losse

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model, dofs:list):
        super().__init__(name)
        self.fe_model = fe_model
        self.dofs = dofs
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
    
    @partial(jit, static_argnums=(0,))
    def compute_R(self,total_control_vars,total_primal_vars):
        return grad(self.compute_total_energy,argnums=1)(total_control_vars,total_primal_vars)
    
    @partial(jit, static_argnums=(0,))
    def compute_DR_DP(self,total_control_vars,total_primal_vars):
        return jax.jacfwd(self.compute_R,argnums=1)(total_control_vars,total_primal_vars)
    
    @partial(jit, static_argnums=(0,))
    def compute_DR_DC(self,total_control_vars,total_primal_vars):
        return jax.jacfwd(self.compute_R,argnums=0)(total_control_vars,total_primal_vars)

