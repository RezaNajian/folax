"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

class ThermalLoss3D(FiniteElementLoss):
    """FE-based 3D Thermal loss

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model):
        super().__init__(name,fe_model,["T"])

    @partial(jit, static_argnums=(0,1,2,5,))
    def ComputeElementEnergy(self,xe,ye,ze,Ke,Te,body_force=0):
        gauss_points = [-1 / jnp.sqrt(3), 1 / jnp.sqrt(3)]
        gauss_weights = [1, 1]
        Te = Te.reshape(-1,1)
        Fe = jnp.zeros((8,1))
        elem_stiffness = jnp.zeros((8, 8))
        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                for k, zeta in enumerate(gauss_points):
                    Nf = jnp.array([(1 - xi) * (1 - eta) * (1 - zeta), 
                                    (1 + xi) * (1 - eta) * (1 - zeta), 
                                    (1 + xi) * (1 + eta) * (1 - zeta), 
                                    (1 - xi) * (1 + eta) * (1 - zeta),
                                    (1 - xi) * (1 - eta) * (1 + zeta),
                                    (1 + xi) * (1 - eta) * (1 + zeta),
                                    (1 + xi) * (1 + eta) * (1 + zeta),
                                    (1 - xi) * (1 + eta) * (1 + zeta)
                                    ]) * 0.125 

                    conductivity_at_gauss = jnp.dot(Nf, Ke.squeeze())
                    dN_dxi = jnp.array([-(1 - eta) * (1 - zeta), (1 - eta) * (1 - zeta), (1 + eta) * (1 - zeta), -(1 + eta) * (1 - zeta),
                                        -(1 - eta) * (1 + zeta), (1 - eta) * (1 + zeta), (1 + eta) * (1 + zeta), -(1 + eta) * (1 + zeta)]) * 0.125
                    dN_deta = jnp.array([-(1 - xi) * (1 - zeta), -(1 + xi) * (1 - zeta), (1 + xi) * (1 - zeta), (1 - xi) * (1 - zeta),
                                         -(1 - xi) * (1 + zeta), -(1 + xi) * (1 + zeta), (1 + xi) * (1 + zeta), (1 - xi) * (1 + zeta)]) * 0.125
                    dN_dzeta = jnp.array([-(1 - xi) * (1 - eta),-(1 + xi) * (1 - eta),-(1 + xi) * (1 + eta),-(1 - xi) * (1 + eta),
                                           (1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) * 0.125
                    J = jnp.dot(jnp.array([dN_dxi, dN_deta,dN_dzeta]), jnp.array([xe, ye, ze]).T)
                    detJ = jnp.linalg.det(J)
                    invJ = jnp.linalg.inv(J)
                    B = jnp.array([dN_dxi, dN_deta, dN_dzeta])
                    B = jnp.dot(invJ,B)
                    elem_stiffness += conductivity_at_gauss * jnp.dot(B.T, B) * detJ * gauss_weights[i] * gauss_weights[j] * gauss_weights[k]  
                    Fe += gauss_weights[i] * gauss_weights[j] * gauss_weights[k] * detJ * body_force *  Nf.reshape(-1,1) 
        
        return  Te.T @ (elem_stiffness@Te-Fe)
    
    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergyVmapCompatible(self,element_id,elements_nodes,X,Y,Z,K,T):
        return self.ComputeElementEnergy(X[elements_nodes[element_id]],
                                         Y[elements_nodes[element_id]],
                                         Z[elements_nodes[element_id]],
                                         K[elements_nodes[element_id]],
                                         T[elements_nodes[element_id]])

    @partial(jit, static_argnums=(0,))
    def ComputeSingleLoss(self,K,T):
        elems_energies = self.ComputeElementsEnergies(K.reshape(-1,1),self.ExtendUnknowDOFsWithBC(T))
        # some extra calculation for reporting and not traced
        avg_elem_energy = jax.lax.stop_gradient(jnp.mean(elems_energies))
        max_elem_energy = jax.lax.stop_gradient(jnp.max(elems_energies))
        min_elem_energy = jax.lax.stop_gradient(jnp.min(elems_energies))
        return jnp.sum(elems_energies),(0,max_elem_energy,avg_elem_energy)
    

