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

class MechanicalLoss2D(FiniteElementLoss):
    """FE-based Mechanical loss

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model):
        super().__init__(name,fe_model,["Ux","Uy"])


    def TensorToVoigt(self,tensor):
        voigt = jnp.zeros((3,1))
        voigt = voigt.at[1,1].set(tensor[1,1])
        voigt = voigt.at[2,1].set(tensor[2,2])
        voigt = voigt.at[3,1].set(tensor[1,2])
        return voigt
    
    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,uve,body_force):

        num_elem_nodes = 4
        xye = jnp.array([xyze[::3], xyze[1::3]]).T

        gauss_points = [-1 / jnp.sqrt(3), 1 / jnp.sqrt(3)]
        gauss_weights = [1, 1]

        v = 0.3
        ei = 1
        ee = ei/(1-v**2)
        dd = jnp.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]])
        dd = dd*ee

        xsi = jnp.full((), 0.0)
        W_int = jnp.zeros(())
        fe = jnp.zeros((uve.size,1))
        ke = jnp.zeros((uve.size, uve.size))
        C_voigt = jnp.zeros((3,1))
        invC_voigt = jnp.zeros((3,1))
        Se_voigt = jnp.zeros((3,1))
        C_tangent = jnp.zeros((3,3))
        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                Nf = jnp.array([0.25 * (1 - xi) * (1 - eta), 
                                0.25 * (1 + xi) * (1 - eta), 
                                0.25 * (1 + xi) * (1 + eta), 
                                0.25 * (1 - xi) * (1 + eta)])
                e_at_gauss = jnp.dot(Nf, de.squeeze())
                dN_dxi = jnp.array([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)]) * 0.25
                dN_deta = jnp.array([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi]) * 0.25

                J = jnp.dot(jnp.array([dN_dxi, dN_deta]), xye)
                detJ = jnp.linalg.det(J)
                invJ = jnp.linalg.inv(J)

                mu = e_at_gauss / (2 * (1 + v))
                lambdaa = e_at_gauss * v / ((1 + v) * (1 - 2 * v))
                dN_dX = jnp.dot(invJ,jnp.array([dN_dxi, dN_deta]))
                
                ue = uve[::2].squeeze()
                ve = uve[1::2].squeeze()
                uveT = jnp.array([ue,ve]).T
                H = jnp.dot(dN_dX,uveT).T
                F = H + jnp.eye(2)
                C = jnp.dot(F.T,F)
                C_voigt = self.TensorToVoigt(C)
                C_voigt = C_voigt.at[3,1].multiply(2)
                invC = jnp.linalg.inv(C)
                invC_voigt = invC_voigt.at[3,1].multiply(2)
                
                detF = jnp.linalg.det(C)**0.5
                
                xsie = xsie + 0.5*mu * (jnp.trace(C)- 2) - mu * jnp.log(detF) + 0.5*lambdaa*(jnp.log(detF)**2)
                xsi = xsi + xsie * gauss_weights[i] * gauss_weights[j] * detJ

                

        return  xsi

    def ComputeElementEnergy(self,xyze,de,uvwe,body_force=jnp.zeros((2,1))):
        return self.ComputeElement(xyze,de,uvwe,body_force)[0]
    
    def ComputeElementResiduals(self,xyze,de,uvwe,body_force=jnp.zeros((2,1))):
        return self.ComputeElement(xyze,de,uvwe,body_force)[1]
    
    def ComputeElementResidualsAndStiffness(self,xyze,de,uve,body_force=jnp.zeros((2,1))):
        num_elem_nodes = 4
        xye = jnp.array([xyze[::3], xyze[1::3]]).T

        gauss_points = [-1 / jnp.sqrt(3), 1 / jnp.sqrt(3)]
        gauss_weights = [1, 1]

        v = 0.3
        ei = 1
        ee = ei/(1-v**2)
        dd = jnp.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]])
        dd = dd*ee

        xsi = jnp.full((), 0.0)
        W_int = jnp.zeros(())
        fe = jnp.zeros((uve.size,1))
        ke = jnp.zeros((uve.size, uve.size))
        C_voigt = jnp.zeros((3,1))
        invC_voigt = jnp.zeros((3,1))
        Se_voigt = jnp.zeros((3,1))
        C_tangent = jnp.zeros((3,3))
        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                Nf = jnp.array([0.25 * (1 - xi) * (1 - eta), 
                                0.25 * (1 + xi) * (1 - eta), 
                                0.25 * (1 + xi) * (1 + eta), 
                                0.25 * (1 - xi) * (1 + eta)])
                e_at_gauss = jnp.dot(Nf, de.squeeze())
                dN_dxi = jnp.array([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)]) * 0.25
                dN_deta = jnp.array([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi]) * 0.25

                J = jnp.dot(jnp.array([dN_dxi, dN_deta]), xye)
                detJ = jnp.linalg.det(J)
                invJ = jnp.linalg.inv(J)

                mu = e_at_gauss / (2 * (1 + v))
                lambdaa = e_at_gauss * v / ((1 + v) * (1 - 2 * v))
                dN_dX = jnp.dot(invJ,jnp.array([dN_dxi, dN_deta]))
                
                ue = uve[::2].squeeze()
                ve = uve[1::2].squeeze()
                uveT = jnp.array([ue,ve]).T
                H = jnp.dot(dN_dX,uveT).T
                F = H + jnp.eye(2)
                C = jnp.dot(F.T,F)
                C_voigt = self.TensorToVoigt(C)
                C_voigt = C_voigt.at[3,1].multiply(2)
                invC = jnp.linalg.inv(C)
                invC_voigt = invC_voigt.at[3,1].multiply(2)
                
                detF = jnp.linalg.det(C)**0.5
                
                Se = 0.5*mu*(jnp.eye(2)-invC) + 0.5*lambdaa*jnp.log(detF)*invC
                Se_voigt = self.TensorToVoigt(Se)

                C_tangent = 0.5*(mu - lambdaa*jnp.log(detF) + 0.5*lambdaa)*jnp.dot(invC_voigt,invC_voigt.T)
                

                zero = jnp.zeros((1,4))
                ds = dN_dX[0:1,:]
                b1 = jnp.concatenate((F[1,1]*dN_dX[:1,:], zero),axis=1)
                b2 = jnp.concatenate((zero, F[2,2]*dN_dX[1:,:]),axis=1)
                b3 = jnp.concatenate((0.5*F[1,2]*dN_dX[1:,:], 0.5*F[2,1]*dN_dX[:1,:]),axis=1)
                BB = jnp.zeros((3, 2*num_elem_nodes))
                for m in range(num_elem_nodes):
                    BB = BB.at[0,2*m:2*m+2].set([F[1,1]*dN_dX[0,m], F[2,1]*dN_dX[0,m]])
                    BB = BB.at[1,2*m:2*m+2].set([F[1,2]*dN_dX[1,m], F[2,2]*dN_dX[1,m]])
                    BB = BB.at[2,2*m:2*m+2].set([F[1,2]*dN_dX[0,m]+F[1,1]*dN_dX[1,m], [F[2,2]*dN_dX[0,m]+F[2,1]*dN_dX[1,m]]])
                
                # jax.debug.print('degbu parameter: {d}',d=b)
                # b = jnp.array([[F[1,1]*dN_dX[0,:], zero],
                #                 [zero, F[2,2]*dN_dX[1,:]],
                #                 [0.5*F[1,2]*dN_dX[1,:], 0.5*F[2,1]*dN_dX[0,:]]])

                # B = jnp.zeros((2, num_elem_nodes))

                # for m in range(num_elem_nodes):
                #     dN_dx = jnp.dot(invJ, jnp.array([dN_dxi[m], dN_deta[m]]).reshape(-1, 1))
                #     B = B.at[0, 1 * m].set(dN_dx[0, 0])
                #     B = B.at[1, 1 * m].set(dN_dx[1, 0])
                # b = jnp.array([[B[0,0],0,B[0,1],0,B[0,2],0,B[0,3],0],[0,B[1,0],0,B[1,1],0,B[1,2],0,B[1,3]],[B[1,0],B[0,0],B[1,1],B[0,1],B[1,2],B[0,2],B[1,3],B[0,3]]])
                nf = jnp.array([[Nf[0], 0, Nf[1], 0, Nf[2], 0, Nf[3], 0],[0, Nf[0], 0, Nf[1], 0, Nf[2], 0, Nf[3]]])
                
                
                fe = fe + gauss_weights[i] * gauss_weights[j] * detJ * jnp.dot(jnp.transpose(nf), body_force)
                W_int = W_int + jnp.dot(b.T,Se_voigt) * gauss_weights[i] * gauss_weights[j] * detJ
                ke = ke + jnp.dot(jnp.dot(b.T,C_tangent),b) * gauss_weights[i] * gauss_weights[j] * detJ
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
