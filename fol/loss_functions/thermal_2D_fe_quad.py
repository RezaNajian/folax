"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from  .fe_loss import FiniteElementLoss
from fol.tools.fem_utilities import *
from fol.computational_models.fe_model import FiniteElementModel

class ThermalLoss2D(FiniteElementLoss):
    """FE-based 2D Thermal loss

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model: FiniteElementModel, loss_settings: dict):
        super().__init__(name,fe_model,["T"],loss_settings)
        self.shape_function = QuadShapeFunction()
        self.dim = 2

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,Ke,Te,body_force):
        xye = jnp.array([xyze[::3], xyze[1::3]])
        Te = Te.reshape(-1,1)
        @jit
        def compute_at_gauss_point(xi,eta,total_weight):
            Nf = self.shape_function.evaluate(xi,eta)
            conductivity_at_gauss = jnp.dot(Nf, Ke.squeeze())
            dN_dxi = self.shape_function.derivatives(xi,eta)
            J = jnp.dot(dN_dxi.T, xye.T)
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            B = jnp.dot(invJ,dN_dxi.T)
            gp_stiffness = conductivity_at_gauss * jnp.dot(B.T, B) * detJ * total_weight
            gp_f = total_weight * detJ * body_force *  Nf.reshape(-1,1) 
            return gp_stiffness,gp_f
        @jit
        def vmap_compatible_compute_at_gauss_point(gp_index):
            return compute_at_gauss_point(self.g_points[self.dim*gp_index],
                                          self.g_points[self.dim*gp_index+1],
                                          self.g_weights[self.dim*gp_index] * self.g_weights[self.dim*gp_index+1])

        k_gps,f_gps = jax.vmap(vmap_compatible_compute_at_gauss_point,(0))(jnp.arange(self.num_gp**self.dim))
        Se = jnp.sum(k_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        element_residuals = jax.lax.stop_gradient(Se @ Te - Fe)
        return  ((Te.T @ element_residuals)[0,0]), 2 * (Se @ Te - Fe), 2 * Se
    
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
