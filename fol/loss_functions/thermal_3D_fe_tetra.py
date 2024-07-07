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

class ThermalLoss3DTetra(FiniteElementLoss):
    """FE-based Thermal loss

    This is the base class for the loss functions require FE formulation.

    """
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, fe_model, loss_settings: dict):
        super().__init__(name,fe_model,["T"])
        self.loss_settings = loss_settings

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,te,body_force):
        xyze = jnp.array([xyze[::3], xyze[1::3], xyze[2::3]]).T
        gauss_points = [0]
        gauss_weights = [2]
        fe = jnp.zeros((te.size,1))
        ke = jnp.zeros((te.size, te.size))
        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                for k, zeta in enumerate(gauss_points):
                    Nf = jnp.array([1 - xi - eta - zeta, xi, eta, zeta])

                    conductivity_at_gauss = jnp.dot(Nf, de.squeeze()) * (1 + 
                                            self.loss_settings["beta"]*(jnp.dot(Nf,te.squeeze()))**self.loss_settings["c"])

                    dN_dxi = jnp.array([-1, 1, 0, 0])
                    dN_deta = jnp.array([-1, 0, 1, 0])
                    dN_dzeta = jnp.array([-1, 0, 0, 1])
                    
                    J = jnp.dot(jnp.array([dN_dxi, dN_deta,dN_dzeta]), xyze)
                    detJ = jnp.linalg.det(J)
                    invJ = jnp.linalg.inv(J)

                    B = jnp.array([dN_dxi, dN_deta, dN_dzeta])
                    B = jnp.dot(invJ,B)

                    ke += conductivity_at_gauss * jnp.dot(B.T, B) * detJ * gauss_weights[i] * gauss_weights[j] * gauss_weights[k]  
                    fe += gauss_weights[i] * gauss_weights[j] * gauss_weights[k] * detJ * body_force *  Nf.reshape(-1,1) 

        element_residuals = jax.lax.stop_gradient(ke @ te - fe)

        return ((te.T @ element_residuals)[0,0]), (ke @ te - fe), ke

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
 