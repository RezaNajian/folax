from typing import Optional
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from jax import jit,grad,vmap,jit,jacfwd,jacrev
from functools import partial
from abc import ABC, abstractmethod

class ThermalLoss(FiniteElementLoss):
    """FE-based Thermal loss

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model):
        super().__init__(name,fe_model)
        if not "T" in self.fe_model.GetDofsDict().keys():
            raise ValueError("No boundary conditions found for temperature T in dofs_dict of fe model ! ")
        if not "non_dirichlet_nodes_ids" in self.fe_model.GetDofsDict()["T"].keys():
            raise ValueError("No non_dirichlet_nodes_ids found in dofs_dict of fe model ! ")
        if not "dirichlet_nodes_ids" in self.fe_model.GetDofsDict()["T"].keys():
            raise ValueError("No dirichlet_nodes_ids found in dofs_dict of fe model ! ")
        if not "dirichlet_nodes_dof_value" in self.fe_model.GetDofsDict()["T"].keys():
            raise ValueError("No dirichlet_nodes_dof_value found in dofs_dict of fe model ! ")
        
        self.number_of_dirichlet_nodes = self.fe_model.GetDofsDict()["T"]["dirichlet_nodes_ids"].shape[-1]
        self.number_of_unknowns = self.fe_model.GetDofsDict()["T"]["non_dirichlet_nodes_ids"].shape[-1]

    def GetNumberOfUnknowns(self):
        return self.number_of_unknowns

    @partial(jit, static_argnums=(0,1,2,5,))
    def ComputeElementEnergy(self,xe,ye,Ke,Te,body_force=0):
        gauss_points = [-1 / jnp.sqrt(3), 1 / jnp.sqrt(3)]
        gauss_weights = [1, 1]
        Te = Te.reshape(-1,1)
        Fe = jnp.zeros((4,1))
        elem_stiffness = jnp.zeros((4, 4))
        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                Nf = jnp.array([0.25 * (1 - xi) * (1 - eta), 
                                0.25 * (1 + xi) * (1 - eta), 
                                0.25 * (1 + xi) * (1 + eta), 
                                0.25 * (1 - xi) * (1 + eta)])
                conductivity_at_gauss = jnp.dot(Nf, Ke.squeeze())
                dN_dxi = jnp.array([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)]) * 0.25
                dN_deta = jnp.array([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi]) * 0.25
                J = jnp.dot(jnp.array([dN_dxi, dN_deta]), jnp.array([xe, ye]).T)
                detJ = jnp.linalg.det(J)
                B = jnp.array([dN_dxi, dN_deta])
                elem_stiffness += conductivity_at_gauss * jnp.dot(B.T, B) * detJ * gauss_weights[i] * gauss_weights[j]  
                Fe += gauss_weights[i] * gauss_weights[j] * detJ * body_force *  Nf.reshape(-1,1) 
        
        return  Te.T @ (elem_stiffness@Te-Fe)
    
    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergyVmapCompatible(self,element_id,elements_nodes,X,Y,K,T):
        return self.ComputeElementEnergy(X[elements_nodes[element_id]],
                                           Y[elements_nodes[element_id]],
                                           K[elements_nodes[element_id]],
                                           T[elements_nodes[element_id]])

    @partial(jit, static_argnums=(0,))
    def ComputeSingleLoss(self,K,T):

        K = K.reshape(-1,1)
        full_T = self.ApplyBC(T)

        elems_ids = self.fe_model.GetElementsIds()
        elems_nodes = self.fe_model.GetElementsNodes()
        X,Y,_ = self.fe_model.GetNodesCoordinates()
        # parallel calculation of energies
        elems_energies = jax.vmap(self.ComputeElementEnergyVmapCompatible,
                                  (0,None,None,None,None,None))(elems_ids,elems_nodes,X,Y,K,full_T)
        
        # some extra calculation for reporting and not traced
        avg_elem_energy = jax.lax.stop_gradient(jnp.mean(elems_energies))
        max_elem_energy = jax.lax.stop_gradient(jnp.max(elems_energies))
        min_elem_energy = jax.lax.stop_gradient(jnp.min(elems_energies))

        return jnp.sum(elems_energies),(0,max_elem_energy,avg_elem_energy)
    
    @partial(jit, static_argnums=(0,))
    def ApplyBC(self,T):
        # apply drichlet BCs and return full_T
        full_T = jnp.zeros(self.fe_model.GetNumberOfNodes())
        full_T = full_T.at[self.fe_model.GetDofsDict()["T"]["dirichlet_nodes_ids"]] \
        .set(self.fe_model.GetDofsDict()["T"]["dirichlet_nodes_dof_value"])
        full_T = full_T.at[self.fe_model.GetDofsDict()["T"]["non_dirichlet_nodes_ids"]].set(T)
        return full_T.reshape(-1,1)

