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
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh

class AllenCahnLoss3DHex(FiniteElementLoss):
    """FE-based 3D Phase-field loss

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T"],  
                               "element_type":"hexahedron"},fe_mesh)
        
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:  
        if self.initialized and not reinitialize:
            return
        super().Initialize() 
        self.shape_function =  HexahedralShapeFunction()
        self.dt =  self.loss_settings["material_dict"]["dt"]
        self.epsilon =  self.loss_settings["material_dict"]["epsilon"]  

    # @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,Te_c,Te_n,body_force=0):
        Te_c = Te_c.reshape(-1,1)
        Te_n = Te_n.reshape(-1,1)
        @jit
        def compute_at_gauss_point(xi,eta,zeta,total_weight):      
            Nf = self.shape_function.evaluate(xi,eta,zeta)
            # conductivity_at_gauss = jnp.dot(Nf, Ke.squeeze())
            dN_dxi = self.shape_function.derivatives(xi,eta,zeta)
            J = jnp.dot(dN_dxi.T, xyze)
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            B = jnp.dot(invJ,dN_dxi.T)
            T_at_gauss_n = jnp.dot(Nf, Te_n)
            T_at_gauss_c = jnp.dot(Nf, Te_c)
            source_term = 0.25*(T_at_gauss_n*T_at_gauss_n - 1)**2
            Dsource_term = (T_at_gauss_n*T_at_gauss_n - 1)*T_at_gauss_n
            gp_stiffness =  jnp.dot(B.T, B) * detJ * total_weight 
            gp_mass =jnp.outer(Nf, Nf) * detJ * total_weight  
            gp_f = source_term * detJ * total_weight
            gp_f_res = Nf.reshape(-1,1)*Dsource_term * detJ * total_weight 
            gp_t = total_weight * detJ *0.5/(self.dt)*(T_at_gauss_n-T_at_gauss_c)**2
            gp_Df = jnp.outer(Nf, Nf) * (3 * T_at_gauss_n**2 - 1) *  detJ * total_weight
            return gp_stiffness,gp_mass, gp_f,gp_f_res, gp_t, gp_Df

        @jit
        def vmap_compatible_compute_at_gauss_point(gp_index):
            return compute_at_gauss_point(self.g_points[self.dim*gp_index],
                                          self.g_points[self.dim*gp_index+1],
                                          self.g_points[self.dim*gp_index+2],
                                          self.g_weights[self.dim*gp_index] * 
                                          self.g_weights[self.dim*gp_index+1]* 
                                          self.g_weights[self.dim*gp_index+2])

        k_gps,m_gps,f_gps,f_res_gps,t_gps, df_gps = jax.vmap(vmap_compatible_compute_at_gauss_point,(0))(jnp.arange(self.num_gp**self.dim))
        Se = jnp.sum(k_gps, axis=0)
        Me = jnp.sum(m_gps, axis=0)
        Fe = jnp.sum(f_gps)
        Fe_res = jnp.sum(f_res_gps)
        Te = jnp.sum(t_gps)
        dFe = jnp.sum(df_gps,axis=0)

        element_residual = jax.lax.stop_gradient((Me+self.dt*Se)@Te_n - (Me@Te_c- self.dt/(self.epsilon**2)*Fe_res))
        element_tangent = (Me+self.dt*Se - self.dt/(self.epsilon**2)*dFe)
        element_energy = 0.5*Te_n.T@Se@Te_n + 1/(self.epsilon**2)*Fe + Te

        return  element_energy, ((Me+self.dt*Se)@Te_n - (Me@Te_c- 1/(self.epsilon**2)*self.dt*Fe_res)), (element_tangent)
    
    
    def ComputeElementHetero(self, *args):
        pass
    
