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

class MechanicalLoss(FiniteElementLoss):
    """FE-based Mechanical loss

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model):
        super().__init__(name,fe_model,["Ux","Uy"])
        self.number_of_unknown_ux = self.fe_model.GetDofsDict()["Ux"]["non_dirichlet_nodes_ids"].shape[-1]
        self.number_of_unknown_uy = self.fe_model.GetDofsDict()["Uy"]["non_dirichlet_nodes_ids"].shape[-1]

    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergy(self,xe,ye,de,ue,ve,body_force=0):

        num_elem_nodes = ue.size
        uve = jnp.zeros((ue.size+ve.size))
        uve = uve.at[0::2].set(ue.reshape(-1))
        uve = uve.at[1::2].set(ve.reshape(-1))
        uve = uve.reshape(-1,1)

        gauss_points = [-1 / jnp.sqrt(3), 1 / jnp.sqrt(3)]
        gauss_weights = [1, 1]

        v = 0.3
        ei = 1
        ee = ei/(1-v**2)
        dd = jnp.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]])
        dd = dd*ee

        fe = jnp.zeros((uve.size,1))
        ke = jnp.zeros((uve.size, uve.size))
        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                Nf = jnp.array([0.25 * (1 - xi) * (1 - eta), 
                                0.25 * (1 + xi) * (1 - eta), 
                                0.25 * (1 + xi) * (1 + eta), 
                                0.25 * (1 - xi) * (1 + eta)])
                e_at_gauss = jnp.dot(Nf, de.squeeze())
                dN_dxi = jnp.array([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)]) * 0.25
                dN_deta = jnp.array([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi]) * 0.25

                J = jnp.dot(jnp.array([dN_dxi, dN_deta]), jnp.array([xe, ye]).T)
                detJ = jnp.linalg.det(J)
                invJ = jnp.linalg.inv(J)

                B = jnp.zeros((2, num_elem_nodes))

                for m in range(num_elem_nodes):
                    dN_dx = jnp.dot(invJ, jnp.array([dN_dxi[m], dN_deta[m]]).reshape(-1, 1))
                    B = B.at[0, 1 * m].set(dN_dx[0, 0])
                    B = B.at[1, 1 * m].set(dN_dx[1, 0])
                b = jnp.array([[B[0,0],0,B[0,1],0,B[0,2],0,B[0,3],0],[0,B[1,0],0,B[1,1],0,B[1,2],0,B[1,3]],[B[1,0],B[0,0],B[1,1],B[0,1],B[1,2],B[0,2],B[1,3],B[0,3]]])
                bT = jnp.dot(b.T,dd)
                nf = jnp.array([[Nf[0], 0, Nf[1], 0, Nf[2], 0, Nf[3], 0],[0, Nf[0], 0, Nf[1], 0, Nf[2], 0, Nf[3]]])
                ke = ke + gauss_weights[i] * gauss_weights[j] * detJ * e_at_gauss * jnp.dot(bT, b )
                fe = fe + gauss_weights[i] * gauss_weights[j] * detJ * jnp.dot(jnp.transpose(nf), body_force)

        return  uve.T @ (ke @ uve - fe)
    
    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergyVmapCompatible(self,element_id,elements_nodes,X,Y,C,UV):
        return self.ComputeElementEnergy(X[elements_nodes[element_id]],
                                         Y[elements_nodes[element_id]],
                                         C[elements_nodes[element_id]],
                                         UV[self.number_dofs_per_node*elements_nodes[element_id]],
                                         UV[self.number_dofs_per_node*elements_nodes[element_id]+1])

    @partial(jit, static_argnums=(0,))
    def ComputeSingleLoss(self,full_control_params,unknown_dofs):

        full_control_params = full_control_params.reshape(-1,1)
        full_dofs = self.ApplyBC(unknown_dofs)

        elems_ids = self.fe_model.GetElementsIds()
        elems_nodes = self.fe_model.GetElementsNodes()
        X,Y,_ = self.fe_model.GetNodesCoordinates()
        # parallel calculation of energies
        elems_energies = jax.vmap(self.ComputeElementEnergyVmapCompatible,
                                  (0,None,None,None,None,None))(elems_ids,elems_nodes,X,Y,full_control_params,full_dofs)
        
        # some extra calculation for reporting and not traced
        avg_elem_energy = jax.lax.stop_gradient(jnp.mean(elems_energies))
        max_elem_energy = jax.lax.stop_gradient(jnp.max(elems_energies))
        min_elem_energy = jax.lax.stop_gradient(jnp.min(elems_energies))

        return jnp.sum(elems_energies),(0,max_elem_energy,avg_elem_energy)
    
    @partial(jit, static_argnums=(0,))
    def ApplyBC(self,unknown_dofs):
        # apply drichlet BCs and return vector dofs
        full_dofs = jnp.zeros(self.total_number_of_dofs)

        # apply Ux BCs
        full_dofs = full_dofs.at[self.number_dofs_per_node*self.fe_model.GetDofsDict()["Ux"]["dirichlet_nodes_ids"]] \
                    .set(self.fe_model.GetDofsDict()["Ux"]["dirichlet_nodes_dof_value"])

        full_dofs = full_dofs.at[self.number_dofs_per_node*self.fe_model.GetDofsDict()["Uy"]["dirichlet_nodes_ids"]+1] \
                    .set(self.fe_model.GetDofsDict()["Uy"]["dirichlet_nodes_dof_value"])

        full_dofs = full_dofs.at[self.number_dofs_per_node*self.fe_model.GetDofsDict()["Ux"]["non_dirichlet_nodes_ids"]] \
                    .set(unknown_dofs[0:self.number_of_unknown_ux])
        
        full_dofs = full_dofs.at[self.number_dofs_per_node*self.fe_model.GetDofsDict()["Uy"]["non_dirichlet_nodes_ids"]+1] \
                    .set(unknown_dofs[self.number_of_unknown_ux:self.number_of_unknown_dofs])

        return full_dofs.reshape(-1,1)

