"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from jax import jit,grad
from functools import partial
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.mesh_input_output.mesh import Mesh

class ThermalLoss3D(FiniteElementLoss):
    """FE-based 3D Thermal loss

    This is the base class for the loss functions require FE formulation.

    """

    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T"],  
                               "element_type":"hexahedron"},fe_mesh)
        self.shape_function = HexahedralShapeFunction()

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,te,body_force=0):
        @jit
        def compute_at_gauss_point(xi,eta,zeta,total_weight):
            Nf = self.shape_function.evaluate(xi,eta,zeta)
            conductivity_at_gauss = jnp.dot(Nf, de.squeeze()) * (1 + 
                                    self.loss_settings["beta"]*(jnp.dot(Nf,te.squeeze()))**self.loss_settings["c"])
            dN_dxi = self.shape_function.derivatives(xi,eta,zeta)
            J = jnp.dot(dN_dxi.T, xyze)
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
        return  ((te.T @ element_residuals)[0,0]), (Se @ te - Fe), Se