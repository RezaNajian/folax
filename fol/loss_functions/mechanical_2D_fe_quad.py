"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/LICENSE
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh

class MechanicalLoss2D(FiniteElementLoss):
    """FE-based Mechanical loss

    This is the base class for the loss functions require FE formulation.

    """

    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["Ux","Uy"],  
                               "element_type":"quad"},fe_mesh)
        if "material_dict" not in self.loss_settings.keys():
            fol_error("material_dict should provided in the loss settings !")

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:  
        if self.initialized and not reinitialize:
            return
        super().Initialize() 
        self.shape_function = QuadShapeFunction()
        self.e = self.loss_settings["material_dict"]["young_modulus"]
        self.v = self.loss_settings["material_dict"]["poisson_ratio"]
        self.D = jnp.array([[1,self.v,0],[self.v,1,0],[0,0,(1-self.v)/2]]) * (self.e/(1-self.v**2))

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,uve,body_force=jnp.zeros((2,1))):
        @jit
        def compute_at_gauss_point(xi,eta,total_weight):
            N = self.shape_function.evaluate(xi,eta)
            e_at_gauss = jnp.dot(N, de.squeeze())
            dN_dxi = self.shape_function.derivatives(xi,eta)
            J = jnp.dot(dN_dxi.T, xyze[:,0:2])
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            dN_dX = jnp.dot(invJ,dN_dxi.T)
            B = jnp.zeros((3, 2 * 4))
            Nf = jnp.zeros((2, 8))
            indices = jnp.arange(4)
            B = B.at[0, 2 * indices].set(dN_dX[0, indices])
            B = B.at[1, 2 * indices + 1].set(dN_dX[1, indices])
            B = B.at[2, 2 * indices].set(dN_dX[1, indices])
            B = B.at[2, 2 * indices + 1].set(dN_dX[0, indices])       
            Nf = Nf.at[0, 2 * indices].set(N)
            Nf = Nf.at[1, 2 * indices + 1].set(N)     
            gp_stiffness = total_weight * detJ * e_at_gauss * (B.T @ self.D @ B)
            gp_f = total_weight * detJ * jnp.dot(jnp.transpose(Nf), body_force)
            return gp_stiffness,gp_f
        @jit
        def vmap_compatible_compute_at_gauss_point(gp_index):
            return compute_at_gauss_point(self.g_points[self.dim*gp_index],
                                          self.g_points[self.dim*gp_index+1],
                                          self.g_weights[self.dim*gp_index] * self.g_weights[self.dim*gp_index+1])

        k_gps,f_gps = jax.vmap(vmap_compatible_compute_at_gauss_point,(0))(jnp.arange(self.num_gp**self.dim))
        Se = jnp.sum(k_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        element_residuals = jax.lax.stop_gradient(Se @ uve - Fe)
        return  ((uve.T @ element_residuals)[0,0]), (Se @ uve - Fe), Se
