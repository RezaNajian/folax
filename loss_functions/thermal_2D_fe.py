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

class ThermalLoss2D(FiniteElementLoss):
    """FE-based 2D Thermal loss

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model):
        super().__init__(name,fe_model,["T"])

    @partial(jit, static_argnums=(0,1,2,5,))
    def ComputeElement(self,xyze,Ke,Te,body_force):
        xye = jnp.array([xyze[::3], xyze[1::3]]).T
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
                J = jnp.dot(jnp.array([dN_dxi, dN_deta]), xye)
                detJ = jnp.linalg.det(J)
                invJ = jnp.linalg.inv(J)
                B = jnp.array([dN_dxi, dN_deta])
                B = jnp.dot(invJ,B)
                elem_stiffness += conductivity_at_gauss * jnp.dot(B.T, B) * detJ * gauss_weights[i] * gauss_weights[j]  
                Fe += gauss_weights[i] * gauss_weights[j] * detJ * body_force *  Nf.reshape(-1,1) 
        
        return  (Te.T @ (elem_stiffness @ Te - Fe))[0,0], 2 * (elem_stiffness @ Te - Fe), 2 * elem_stiffness
    
    def ComputeElementEnergy(self,xyze,de,uvwe,body_force=0.0):
        return self.ComputeElement(xyze,de,uvwe,body_force)[0]

    def ComputeElementResidualsAndStiffness(self,xyze,de,uvwe,body_force=0.0):
        _,re,ke = self.ComputeElement(xyze,de,uvwe,body_force)
        return re,ke

    def ComputeElementResiduals(self,xyze,de,uvwe,body_force=0.0):
        return self.ComputeElement(xyze,de,uvwe,body_force)[1]
    
    def ComputeElementStiffness(self,xyze,de,uvwe,body_force=0.0):
        return self.ComputeElement(xyze,de,uvwe,body_force)[2]

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualsVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,UV):
        return self.ComputeElementResiduals(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     UV[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualsAndStiffnessVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,UV):
        return self.ComputeElementResidualsAndStiffness(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     UV[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergyVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,UV):
        return self.ComputeElementEnergy(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     UV[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeSingleLoss(self,full_control_params,unknown_dofs):
        elems_energies = self.ComputeElementsEnergies(full_control_params.reshape(-1,1),
                                                      self.ExtendUnknowDOFsWithBC(unknown_dofs))
        # some extra calculation for reporting and not traced
        avg_elem_energy = jax.lax.stop_gradient(jnp.mean(elems_energies))
        max_elem_energy = jax.lax.stop_gradient(jnp.max(elems_energies))
        min_elem_energy = jax.lax.stop_gradient(jnp.min(elems_energies))
        return jnp.sum(elems_energies),(0,max_elem_energy,avg_elem_energy)
