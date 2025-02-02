"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from  .fe_loss_transient_hetero import FiniteElementLossTransientHetero
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh

class ThermalTransientLoss2DQuad(FiniteElementLossTransientHetero):
    """FE-based 2D Thermal loss

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T"],  
                               "element_type":"quad"},fe_mesh)
        
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:  
        if self.initialized and not reinitialize:
            return
        super().Initialize() 
        self.shape_function = QuadShapeFunction()
        self.rho = self.loss_settings["material_dict"]["rho"]
        self.cp =  self.loss_settings["material_dict"]["cp"]
        self.dt =  self.loss_settings["material_dict"]["dt"]

    @partial(jit, static_argnums=(0,))
    def ComputeElementHetero(self,xyze,Te_c,Ke,Te_n,body_force=0):
        Te_c = Te_c.reshape(-1,1)
        Te_n = Te_n.reshape(-1,1)
        @jit
        def compute_at_gauss_point(xi,eta,total_weight):
            Nf = self.shape_function.evaluate(xi,eta)
            #conductivity_at_gauss = jnp.dot(Nf, Ke.squeeze())
            dN_dxi = self.shape_function.derivatives(xi,eta)
            J = jnp.dot(dN_dxi.T, xyze[:,0:2])
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            B = jnp.dot(invJ,dN_dxi.T)
            T_at_gauss_n = jnp.dot(Nf, Te_n)
            T_at_gauss_c = jnp.dot(Nf, Te_c)
            gp_stiffness =  jnp.dot(B.T, B) * detJ * total_weight #* conductivity_at_gauss
            gp_mass = self.rho * self.cp* jnp.outer(Nf, Nf) * detJ * total_weight
            gp_f = total_weight * detJ * body_force *  Nf.reshape(-1,1) 
            gp_t = self.rho * self.cp * 0.5/(self.dt)*total_weight * detJ *(T_at_gauss_n-T_at_gauss_c)**2
            return gp_stiffness,gp_mass, gp_f, gp_t
        @jit
        def vmap_compatible_compute_at_gauss_point(gp_index):
            return compute_at_gauss_point(self.g_points[self.dim*gp_index],
                                          self.g_points[self.dim*gp_index+1],
                                          self.g_weights[self.dim*gp_index] * self.g_weights[self.dim*gp_index+1])

        k_gps,m_gps,f_gps,t_gps = jax.vmap(vmap_compatible_compute_at_gauss_point,(0))(jnp.arange(self.num_gp**self.dim))
        Se = jnp.sum(k_gps, axis=0)
        Me = jnp.sum(m_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        Te = jnp.sum(t_gps)
        # element_residual = jax.lax.stop_gradient((Me+self.dt*Se)@Te_n - Me@Te_c) 

        return 0.5*Te_n.T@Se@Te_n + Te, (Me+self.dt*Se)@Te_n - Me@Te_c, (Me+self.dt*Se)
