"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/License.txt
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from jax import jit,grad
from functools import partial
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.computational_models.fe_model import FiniteElementModel

class ThermalLoss3DTetra(FiniteElementLoss):
    """FE-based Thermal loss

    This is the base class for the loss functions require FE formulation.

    """
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, fe_model: FiniteElementModel, loss_settings: dict={}):
        super().__init__(name,fe_model,["T"],{**loss_settings,"compute_dims":3})
        self.shape_function = TetrahedralShapeFunction()

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,te,body_force):
        xyze = jnp.array([xyze[::3], xyze[1::3], xyze[2::3]])
        @jit
        def compute_at_gauss_point(xi,eta,zeta,total_weight):
            Nf = self.shape_function.evaluate(xi,eta,zeta)
            conductivity_at_gauss = jnp.dot(Nf, de.squeeze()) * (1 + 
                                    self.loss_settings["beta"]*(jnp.dot(Nf,te.squeeze()))**self.loss_settings["c"])
            dN_dxi = self.shape_function.derivatives(xi,eta,zeta)
            J = jnp.dot(dN_dxi.T, xyze.T)
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
                                          self.g_points[self.dim*gp_index+2],
                                          self.g_weights[self.dim*gp_index] * 
                                          self.g_weights[self.dim*gp_index+1]* 
                                          self.g_weights[self.dim*gp_index+2])

        k_gps,f_gps = jax.vmap(vmap_compatible_compute_at_gauss_point,(0))(jnp.arange(self.num_gp**self.dim))
        Se = jnp.sum(k_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        element_residuals = jax.lax.stop_gradient(Se @ te - Fe)
        return  ((te.T @ element_residuals)[0,0]), 2 * (Se @ te - Fe), 2 * Se

    def ComputeElementEnergy(self,xyze,de,te,body_force=jnp.zeros((1,1))):
        return self.ComputeElement(xyze,de,te,body_force)[0]

    def ComputeElementResidualsAndStiffness(self,xyze,de,te,body_force=jnp.zeros((1,1))):
        _,re,ke = self.ComputeElement(xyze,de,te,body_force)
        return re,ke

    def ComputeElementResiduals(self,xyze,de,te,body_force=jnp.zeros((1,1))):
        return self.ComputeElement(xyze,de,te,body_force)[1]
    
    def ComputeElementStiffness(self,xyze,de,te,body_force=jnp.zeros((1,1))):
        return self.ComputeElement(xyze,de,te,body_force)[2]

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualsVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,T):
        return self.ComputeElementResiduals(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     T[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualsAndStiffnessVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,T):
        return self.ComputeElementResidualsAndStiffness(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     T[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergyVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,T):
        return self.ComputeElementEnergy(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     T[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeSingleLoss(self,full_control_params,unknown_dofs):
        elems_energies = self.ComputeElementsEnergies(full_control_params.reshape(-1,1),
                                                      self.ExtendUnknowDOFsWithBC(unknown_dofs))
        # some extra calculation for reporting and not traced
        avg_elem_energy = jax.lax.stop_gradient(jnp.mean(elems_energies))
        max_elem_energy = jax.lax.stop_gradient(jnp.max(elems_energies))
        min_elem_energy = jax.lax.stop_gradient(jnp.min(elems_energies))
        return jnp.abs(jnp.sum(elems_energies)),(min_elem_energy,max_elem_energy,avg_elem_energy)
 