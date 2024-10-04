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

class MechanicalLoss3D(FiniteElementLoss):
    """FE-based Mechanical loss

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["Ux","Uy","Uz"],  
                               "element_type":"hexahedron"},fe_mesh)
        if "material_dict" not in self.loss_settings.keys():
            fol_error("material_dict should provided in the loss settings !")

    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:  
        super().Initialize() 
        self.shape_function = HexahedralShapeFunction()
        # construction of the constitutive matrix
        self.e = self.loss_settings["material_dict"]["young_modulus"]
        self.v = self.loss_settings["material_dict"]["poisson_ratio"]
        c1 = self.e / ((1.0 + self.v) * (1.0 - 2.0 * self.v))
        c2 = c1 * (1.0 - self.v)
        c3 = c1 * self.v
        c4 = c1 * 0.5 * (1.0 - 2.0 * self.v)
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
    def ComputeElement(self,xyze,de,uvwe,body_force=jnp.zeros((3,1))):
        num_elem_nodes = 8
        gauss_points = [-1 / jnp.sqrt(3), 1 / jnp.sqrt(3)]
        gauss_weights = [1, 1]
        fe = jnp.zeros((uvwe.size,1))
        ke = jnp.zeros((uvwe.size, uvwe.size))
        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                for k, zeta in enumerate(gauss_points):
                    Nf = jnp.array([(1 - xi) * (1 - eta) * (1 - zeta), 
                                    (1 + xi) * (1 - eta) * (1 - zeta), 
                                    (1 + xi) * (1 + eta) * (1 - zeta), 
                                    (1 - xi) * (1 + eta) * (1 - zeta),
                                    (1 - xi) * (1 - eta) * (1 + zeta),
                                    (1 + xi) * (1 - eta) * (1 + zeta),
                                    (1 + xi) * (1 + eta) * (1 + zeta),
                                    (1 - xi) * (1 + eta) * (1 + zeta)
                                    ]) * 0.125 
                    e_at_gauss = jnp.dot(Nf, de.squeeze())
                    dN_dxi = jnp.array([-(1 - eta) * (1 - zeta), (1 - eta) * (1 - zeta), (1 + eta) * (1 - zeta), -(1 + eta) * (1 - zeta),
                                        -(1 - eta) * (1 + zeta), (1 - eta) * (1 + zeta), (1 + eta) * (1 + zeta), -(1 + eta) * (1 + zeta)]) * 0.125
                    dN_deta = jnp.array([-(1 - xi) * (1 - zeta), -(1 + xi) * (1 - zeta), (1 + xi) * (1 - zeta), (1 - xi) * (1 - zeta),
                                         -(1 - xi) * (1 + zeta), -(1 + xi) * (1 + zeta), (1 + xi) * (1 + zeta), (1 - xi) * (1 + zeta)]) * 0.125
                    dN_dzeta = jnp.array([-(1 - xi) * (1 - eta),-(1 + xi) * (1 - eta),-(1 + xi) * (1 + eta),-(1 - xi) * (1 + eta),
                                           (1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) * 0.125
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

        return (uvwe.T @ (ke @ uvwe - fe))[0,0], 2 * (ke @ uvwe - fe), 2 * ke
