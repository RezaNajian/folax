"""
 Authors: Kianoosh Taghikhani, https://github.com/kianoosh1989
 Date: May, 2024
 License: FOL/LICENSE
"""
from  .mechanical_3D_fe_tetra import MechanicalLoss3DTetra
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *

class NonLinearMechanicalLoss3DTetra(MechanicalLoss3DTetra):
    """FE-based Mechanical loss

    This is the base class for the loss functions require FE formulation.

    """
    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None: 
        super().Initialize() 
        self.shape_function = TetrahedralShapeFunction()
        self.material_model = NeoHookianModel()

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,uvwe,body_force=jnp.zeros((3,1))):
        @jit
        def compute_at_gauss_point(xi,eta,zeta,total_weight):
            Nf = self.shape_function.evaluate(xi,eta,zeta)
            e_at_gauss = jnp.dot(Nf, de.squeeze())
            k_at_gauss = e_at_gauss / (3 * (1 - 2*self.v))
            mu_at_gauss = e_at_gauss / (2 * (1 + self.v))
            dN_dxi = self.shape_function.derivatives(xi,eta,zeta)
            J = jnp.dot(dN_dxi.T, xyze)
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            dN_dX = jnp.dot(invJ,dN_dxi.T)
            uvweT = jnp.array([uvwe[::3].squeeze(),uvwe[1::3].squeeze(),uvwe[2::3].squeeze()]).T
            H = jnp.dot(dN_dX,uvweT).T
            F = H + jnp.eye(H.shape[0])
            xsi,S,C = self.material_model.evaluate(F,k_at_gauss,mu_at_gauss)
            num_elem_nodes = Nf.size
            indices = jnp.arange(num_elem_nodes)
            B = jnp.zeros((6, 3 * num_elem_nodes))
            
            B = B.at[0, 3 * indices].set(F[0, 0] * dN_dX[0, indices])
            B = B.at[0, 3 * indices + 1].set(F[1, 0] * dN_dX[0, indices])
            B = B.at[0, 3 * indices + 2].set(F[2, 0] * dN_dX[0, indices])
            B = B.at[1, 3 * indices].set(F[0, 1] * dN_dX[1, indices])
            B = B.at[1, 3 * indices + 1].set(F[1, 1] * dN_dX[1, indices])
            B = B.at[1, 3 * indices + 2].set(F[2, 1] * dN_dX[1, indices])
            B = B.at[2, 3 * indices].set(F[0, 2] * dN_dX[2, indices])
            B = B.at[2, 3 * indices + 1].set(F[1, 2] * dN_dX[2, indices])
            B = B.at[2, 3 * indices + 2].set(F[2, 2] * dN_dX[2, indices])
            B = B.at[3, 3 * indices].set(F[0, 1] * dN_dX[2, indices] + F[0, 2] * dN_dX[1, indices])
            B = B.at[3, 3 * indices + 1].set(F[1, 1] * dN_dX[2, indices] + F[1, 2] * dN_dX[1, indices])
            B = B.at[3, 3 * indices + 2].set(F[2, 1] * dN_dX[2, indices] + F[2, 2] * dN_dX[1, indices])
            B = B.at[4, 3 * indices].set(F[0, 0] * dN_dX[2, indices] + F[0, 2] * dN_dX[0, indices])
            B = B.at[4, 3 * indices + 1].set(F[1, 0] * dN_dX[2, indices] + F[1, 2] * dN_dX[0, indices])
            B = B.at[4, 3 * indices + 2].set(F[2, 0] * dN_dX[2, indices] + F[2, 2] * dN_dX[0, indices])
            B = B.at[5, 3 * indices].set(F[0, 0] * dN_dX[1, indices] + F[0, 1] * dN_dX[0, indices])
            B = B.at[5, 3 * indices + 1].set(F[1, 0] * dN_dX[1, indices] + F[1, 1] * dN_dX[0, indices])
            B = B.at[5, 3 * indices + 2].set(F[2, 0] * dN_dX[1, indices] + F[2, 1] * dN_dX[0, indices])

            N = jnp.zeros((3,uvwe.size))
            N = N.at[0,0::3].set(Nf)
            N = N.at[0,1::3].set(Nf)
            N = N.at[0,2::3].set(Nf)
            gp_stiffness = total_weight * detJ * (B.T @ (C @ B))
            gp_f = total_weight * detJ * (N.T @ body_force)
            gp_fint = total_weight * detJ * jnp.dot(B.T,S)
            gp_energy = total_weight * detJ * xsi
            return gp_energy,gp_stiffness,gp_f,gp_fint
        @jit
        def vmap_compatible_compute_at_gauss_point(gp_index):
            return compute_at_gauss_point(self.g_points[self.dim*gp_index],
                                          self.g_points[self.dim*gp_index+1],
                                          self.g_points[self.dim*gp_index+2],
                                          self.g_weights[self.dim*gp_index] * 
                                          self.g_weights[self.dim*gp_index+1]* 
                                          self.g_weights[self.dim*gp_index+2])

        E_gps,k_gps,f_gps,fint_gps = jax.vmap(vmap_compatible_compute_at_gauss_point,(0))(jnp.arange(self.num_gp**self.dim))
        Se = jnp.sum(k_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        Fint = jnp.sum(fint_gps, axis=0)
        Ee = jnp.sum(E_gps, axis=0)
        return  Ee, Fint - Fe, Se

 