"""
 Authors: Kianoosh Taghikhani, https://github.com/kianoosh1989
 Date: July, 2024
 License: FOL/License.txt
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial
from fol.tools.fem_utilities import *
from fol.computational_models.fe_model import FiniteElementModel
from fol.tools.usefull_functions import TensorToVoigt,FourthTensorToVoigt,Neo_Hooke

class MechanicalLoss2D(FiniteElementLoss):
    """FE-based Mechanical loss

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model: FiniteElementModel, loss_settings: dict={}):
        super().__init__(name,fe_model,["Ux","Uy"],{**loss_settings,"compute_dims":2})
        self.shape_function = QuadShapeFunction()
        self.material_model = NeoHookianModel()
        self.e = self.loss_settings["young_modulus"]
        self.v = self.loss_settings["poisson_ratio"]        

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,uve,body_force):
        xye = jnp.array([xyze[::3], xyze[1::3]])
        @jit
        def compute_at_gauss_point(xi,eta,total_weight):
            N = self.shape_function.evaluate(xi,eta)
            e_at_gauss = jnp.dot(N, de.squeeze())
            k_at_gauss = e_at_gauss / (3 * (1 - 2*self.v))
            mu_at_gauss = e_at_gauss / (2 * (1 + self.v))
            dN_dxi = self.shape_function.derivatives(xi,eta)
            J = jnp.dot(dN_dxi.T, xye.T)
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            dN_dX = jnp.dot(invJ,dN_dxi.T)
            uveT = jnp.array([uve[::2].squeeze(),uve[1::2].squeeze()]).T
            H = jnp.dot(dN_dX,uveT).T
            F = H + jnp.eye(H.shape[0])
            xsi,S,C = self.material_model.evaluate(F,k_at_gauss,mu_at_gauss)
            Nf = jnp.zeros((2, 8))
            indices = np.arange(4)
            B = jnp.zeros((3, 8))
            B = B.at[0, 2 * indices].set(F[0, 0] * dN_dX[0, indices])
            B = B.at[0, 2 * indices + 1].set(F[1, 0] * dN_dX[0, indices])
            B = B.at[1, 2 * indices].set(F[0, 1] * dN_dX[1, indices])
            B = B.at[1, 2 * indices + 1].set(F[1, 1] * dN_dX[1, indices])
            B = B.at[2, 2 * indices].set(F[0, 1] * dN_dX[0, indices] + F[0, 0] * dN_dX[1, indices])
            B = B.at[2, 2 * indices + 1].set(F[1, 1] * dN_dX[0, indices] + F[1, 0] * dN_dX[1, indices])
      
            Nf = Nf.at[0, 2 * indices].set(N)
            Nf = Nf.at[1, 2 * indices + 1].set(N)     
            gp_stiffness = total_weight * detJ * (B.T @ C @ B)
            gp_f = total_weight * detJ * jnp.dot(jnp.transpose(Nf), body_force)
            gp_fint = total_weight * detJ * jnp.dot(B.T,S)
            gp_energy = total_weight * detJ * xsi
            return gp_energy,gp_stiffness,gp_f,gp_fint
        @jit
        def vmap_compatible_compute_at_gauss_point(gp_index):
            return compute_at_gauss_point(self.g_points[self.dim*gp_index],
                                          self.g_points[self.dim*gp_index+1],
                                          self.g_weights[self.dim*gp_index] * self.g_weights[self.dim*gp_index+1])

        E_gps,k_gps,f_gps,fint_gps = jax.vmap(vmap_compatible_compute_at_gauss_point,(0))(jnp.arange(self.num_gp**self.dim))
        Se = jnp.sum(k_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        Fint = jnp.sum(fint_gps, axis=0)
        Ee = jnp.sum(E_gps, axis=0)
        return  Ee, Fint - Fe, Se

    def ComputeElementEnergy(self,xyze,de,uvwe,body_force=jnp.zeros((2,1))):
        return self.ComputeElement(xyze,de,uvwe,body_force)[0]
    
    def ComputeElementResiduals(self,xyze,de,uvwe,body_force=jnp.zeros((2,1))):
        return self.ComputeElement(xyze,de,uvwe,body_force)[1]
    
    def ComputeElementResidualsAndStiffness(self,xyze,de,uvwe,body_force=jnp.zeros((2,1))):
        _,re,ke = self.ComputeElement(xyze,de,uvwe,body_force)
        return re,ke

    def ComputeElementStiffness(self,xyze,de,uvwe,body_force=jnp.zeros((2,1))):
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
