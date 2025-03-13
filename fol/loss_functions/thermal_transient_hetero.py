"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/LICENSE
"""
from  .fe_loss_transient_hetero import FiniteElementLossTransientHetero
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh

class ThermalTransientLossHetero(FiniteElementLossTransientHetero):

    def Initialize(self) -> None:  
        super().Initialize() 
        if "material_dict" not in self.loss_settings.keys():
            fol_error("material_dict should provided in the loss settings !")
        if self.dim == 2:
            self.CalculateNMatrix = self.CalculateNMatrix2D
            self.CalculateBMatrix = self.CalculateBMatrix2D
            self.rho = self.loss_settings["material_dict"]["rho"]
            self.cp =  self.loss_settings["material_dict"]["cp"]
            self.dt =  self.loss_settings["material_dict"]["dt"]
            self.body_force = jnp.zeros((2,1))
            if "body_foce" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_foce"])
        else:
            self.CalculateNMatrix = self.CalculateNMatrix3D
            self.CalculateBMatrix = self.CalculateBMatrix3D
            self.rho = self.loss_settings["material_dict"]["rho"]
            self.cp =  self.loss_settings["material_dict"]["cp"]
            self.dt =  self.loss_settings["material_dict"]["dt"]
            self.body_force = jnp.zeros((3,1))
            if "body_foce" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_foce"])

    @partial(jit, static_argnums=(0,))
    def CalculateBMatrix2D(self,DN_DX:jnp.array) -> jnp.array:
        B = jnp.zeros((3, 2 * DN_DX.shape[0]))
        indices = jnp.arange(DN_DX.shape[0])
        B = B.at[0, 2 * indices].set(DN_DX[indices,0])
        B = B.at[1, 2 * indices + 1].set(DN_DX[indices,1])
        B = B.at[2, 2 * indices].set(DN_DX[indices,1])
        B = B.at[2, 2 * indices + 1].set(DN_DX[indices,0])  
        return B

    @partial(jit, static_argnums=(0,))
    def CalculateBMatrix3D(self,DN_DX:jnp.array) -> jnp.array:
        B = jnp.zeros((6,3*DN_DX.shape[0]))
        index = jnp.arange(DN_DX.shape[0]) * 3
        B = B.at[0, index + 0].set(DN_DX[:,0])
        B = B.at[1, index + 1].set(DN_DX[:,1])
        B = B.at[2, index + 2].set(DN_DX[:,2])
        B = B.at[3, index + 0].set(DN_DX[:,1])
        B = B.at[3, index + 1].set(DN_DX[:,0])
        B = B.at[4, index + 1].set(DN_DX[:,2])
        B = B.at[4, index + 2].set(DN_DX[:,1])
        B = B.at[5, index + 0].set(DN_DX[:,2])
        B = B.at[5, index + 2].set(DN_DX[:,0])
        return B
    
    @partial(jit, static_argnums=(0,))
    def CalculateNMatrix2D(self,N_vec:jnp.array) -> jnp.array:
        N_mat = jnp.zeros((2, 2 * N_vec.size))
        indices = jnp.arange(N_vec.size)   
        N_mat = N_mat.at[0, 2 * indices].set(N_vec)
        N_mat = N_mat.at[1, 2 * indices + 1].set(N_vec)    
        return N_mat
    
    @partial(jit, static_argnums=(0,))
    def CalculateNMatrix3D(self,N_vec:jnp.array) -> jnp.array:
        N_mat = jnp.zeros((3,3*N_vec.size))
        N_mat = N_mat.at[0,0::3].set(N_vec)
        N_mat = N_mat.at[1,1::3].set(N_vec)
        N_mat = N_mat.at[2,2::3].set(N_vec)
        return N_mat
    
    @partial(jit, static_argnums=(0,))
    def ComputeElementHetero(self,xyze,Te_c,Ke,Te_n,body_force=0):
        Te_c = Te_c.reshape(-1,1)
        Te_n = Te_n.reshape(-1,1)
        Ke = Ke.reshape(-1,1)
        @jit
        def compute_at_gauss_point(gp_point,gp_weight):
            N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
            N_mat = self.CalculateNMatrix(N_vec)
            ke_at_gauss = jnp.dot(N_vec, Ke.squeeze())
            DN_DX = self.fe_element.ShapeFunctionsLocalGradients(gp_point)
            # B_mat = self.CalculateBMatrix(DN_DX)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            B_mat = jnp.dot(invJ,DN_DX.T)
            T_at_gauss_n = jnp.dot(N_vec.reshape(1,-1), Te_n)
            T_at_gauss_c = jnp.dot(N_vec.reshape(1,-1), Te_c)
            gp_stiffness =  B_mat.T@B_mat * detJ * gp_weight * ke_at_gauss
            gp_mass = self.rho * self.cp* jnp.outer(N_vec, N_vec) * detJ * gp_weight 
            gp_f = gp_weight * detJ * N_mat.reshape(-1,1) #* body_force
            gp_t = self.rho * self.cp * 0.5/(self.dt)*gp_weight  * detJ *(T_at_gauss_n-T_at_gauss_c)**2
            return gp_stiffness,gp_mass, gp_f, gp_t

        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        k_gps,m_gps,f_gps,t_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        Se = jnp.sum(k_gps, axis=0)
        Me = jnp.sum(m_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        Te = jnp.sum(t_gps)
        # element_residual = jax.lax.stop_gradient((Me+self.dt*Se)@Te_n - Me@Te_c) 

        return 0.5*Te_n.T@Se@Te_n + Te, (Me+self.dt*Se)@Te_n - Me@Te_c, (Me+self.dt*Se)


class ThermalTransientLoss2DQuad(ThermalTransientLossHetero):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T"],  
                               "element_type":"quad"},fe_mesh)
        
class ThermalTransientLoss2DTri(ThermalTransientLossHetero):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T"],  
                               "element_type":"triangle"},fe_mesh)

class ThermalTransientLoss3DHexa(ThermalTransientLossHetero):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T"],   
                               "element_type":"hexahedron"},fe_mesh)

