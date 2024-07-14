"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/License.txt
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *
from fol.computational_models.fe_model import FiniteElementModel

class MechanicalLoss3DTetra(FiniteElementLoss):
    """FE-based Mechanical loss

    This is the base class for the loss functions require FE formulation.

    """
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, fe_model: FiniteElementModel, loss_settings: dict={}):
        super().__init__(name,fe_model,["Ux","Uy","Uz"],{**loss_settings,"compute_dims":3})

        # construction of the constitutive matrix
        young_modulus = 1 # TODO should moved to the inputs
        poisson_ratio = 0.3 # TODO should moved to the inputs
        c1 = young_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        c2 = c1 * (1.0 - poisson_ratio)
        c3 = c1 * poisson_ratio
        c4 = c1 * 0.5 * (1.0 - 2.0 * poisson_ratio)
        D = jnp.zeros((6,6))
        D = D.at[0,0].set(c2)
        D = D.at[0,1].set(c3)
        D = D.at[0,2].set(c3)
        D = D.at[1,0].set(c3)
        D = D.at[1,1].set(c2)
        D = D.at[1,2].set(c3)
        D = D.at[2,0].set(c3)
        D = D.at[2,1].set(c3)
        D = D.at[2,2].set(c2)
        D = D.at[3,3].set(c4)
        D = D.at[4,4].set(c4)
        D = D.at[5,5].set(c4)
        self.D = D

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,uvwe,body_force):
        xyze = jnp.array([xyze[::3], xyze[1::3], xyze[2::3]]).T
        num_elem_nodes = 4
        gauss_points = [0]
        gauss_weights = [2]
        fe = jnp.zeros((uvwe.size,1))
        ke = jnp.zeros((uvwe.size, uvwe.size))
        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                for k, zeta in enumerate(gauss_points):
                    Nf = jnp.array([1 - xi - eta - zeta, xi, eta, zeta])
                    e_at_gauss = jnp.dot(Nf, de.squeeze())
                    dN_dxi = jnp.array([-1, 1, 0, 0])
                    dN_deta = jnp.array([-1, 0, 1, 0])
                    dN_dzeta = jnp.array([-1, 0, 0, 1])
                    
                    J = jnp.dot(jnp.array([dN_dxi, dN_deta,dN_dzeta]), xyze)
                    detJ = jnp.linalg.det(J)
                    invJ = jnp.linalg.inv(J)

                    dN_dX = jnp.array([dN_dxi, dN_deta, dN_dzeta])
                    dN_dX = jnp.dot(invJ,dN_dX)
                    B = jnp.zeros((6,uvwe.size))
                    index = 0
                    for i_node in range(num_elem_nodes):
                        B = B.at[0, index + 0].set(dN_dX[0, i_node])
                        B = B.at[1, index + 1].set(dN_dX[1, i_node])
                        B = B.at[2, index + 2].set(dN_dX[2, i_node])
                        B = B.at[3, index + 0].set(dN_dX[1, i_node])
                        B = B.at[3, index + 1].set(dN_dX[0, i_node])
                        B = B.at[4, index + 1].set(dN_dX[2, i_node])
                        B = B.at[4, index + 2].set(dN_dX[1, i_node])
                        B = B.at[5, index + 0].set(dN_dX[2, i_node])
                        B = B.at[5, index + 2].set(dN_dX[0, i_node])
                        index += 3

                    N = jnp.zeros((3,uvwe.size))
                    N = N.at[0,0::3].set(Nf)
                    N = N.at[0,1::3].set(Nf)
                    N = N.at[0,2::3].set(Nf)

                    ke = ke + gauss_weights[i] * gauss_weights[j] * gauss_weights[k] * detJ * e_at_gauss * (B.T @ (self.D @ B))
                    fe = fe + gauss_weights[i] * gauss_weights[j] * gauss_weights[k] * detJ * (N.T @ body_force)

        return jnp.abs((uvwe.T @ (ke @ uvwe - fe))[0,0]), 2 * (ke @ uvwe - fe), 2 * ke

    def ComputeElementEnergy(self,xyze,de,uvwe,body_force=jnp.zeros((3,1))):
        return self.ComputeElement(xyze,de,uvwe,body_force)[0]

    def ComputeElementResidualsAndStiffness(self,xyze,de,uvwe,body_force=jnp.zeros((3,1))):
        _,re,ke = self.ComputeElement(xyze,de,uvwe,body_force)
        return re,ke

    def ComputeElementResiduals(self,xyze,de,uvwe,body_force=jnp.zeros((3,1))):
        return self.ComputeElement(xyze,de,uvwe,body_force)[1]
    
    def ComputeElementStiffness(self,xyze,de,uvwe,body_force=jnp.zeros((3,1))):
        return self.ComputeElement(xyze,de,uvwe,body_force)[2]

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualsVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,UVW):
        return self.ComputeElementResiduals(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     UVW[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualsAndStiffnessVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,UVW):
        return self.ComputeElementResidualsAndStiffness(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     UVW[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergyVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,UVW):
        return self.ComputeElementEnergy(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     UVW[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
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

