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
from fol.tools.usefull_functions import TensorToVoigt,FourthTensorToVoigt,Neo_Hooke

class MechanicalLoss2D(FiniteElementLoss):
    """FE-based Mechanical loss

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model):
        super().__init__(name,fe_model,["Ux","Uy"])

    
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

        ScalarShape = jnp.full((), 0.0)
        ForceShape = jnp.zeros((uve.size,1))
        StiffnessShape = jnp.zeros((uve.size, uve.size))
        voigtShape = jnp.zeros((3,1))
        tensorShape = jnp.zeros((3,3))

        xsi = jnp.zeros_like(ScalarShape)
        W_int = jnp.zeros_like(ForceShape)
        fe = jnp.zeros_like(ForceShape)
        ke = jnp.zeros_like(StiffnessShape)
        C_voigt = jnp.zeros_like(voigtShape)
        invC_voigt = jnp.zeros_like(voigtShape)
        Se_voigt = jnp.zeros_like(voigtShape)
        C_tangent = jnp.zeros_like(tensorShape)

        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                Nf = jnp.array([0.25 * (1 - xi) * (1 - eta), 
                                0.25 * (1 + xi) * (1 - eta), 
                                0.25 * (1 + xi) * (1 + eta), 
                                0.25 * (1 - xi) * (1 + eta)])
                e_at_gauss = jnp.dot(Nf, de.squeeze())
                dN_dxi = jnp.array([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)]) * 0.25
                dN_deta = jnp.array([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi]) * 0.25

                Jacobian = jnp.dot(jnp.array([dN_dxi, dN_deta]), xye)
                detJ = jnp.linalg.det(Jacobian)
                # if jnp.abs(detJ) < 1e-12:
                #     raise ValueError("Jacobian determinant is too small, possible degenerate element.")
                invJ = jnp.linalg.inv(Jacobian)

                # mu = e_at_gauss / (2 * (1 + v))
                mu = 0.5673
                lambdaa = e_at_gauss * v / ((1 + v) * (1 - 2*v))
                # k = e_at_gauss / (3 * (1 - 2*v)) # Bulk Modulus
                k = 2000

                dN_dX = jnp.dot(invJ,jnp.array([dN_dxi, dN_deta]))
                ue = uve[::2].squeeze()
                ve = uve[1::2].squeeze()
                uveT = jnp.array([ue,ve]).T

                H = jnp.dot(dN_dX,uveT).T
                F = H + jnp.eye(H.shape[0])
                xsie, Se, C_tangent_fourth = Neo_Hooke(F,k,mu)
                Se_voigt = TensorToVoigt(Se)
                C_tangent = FourthTensorToVoigt(C_tangent_fourth)

                # To test with a simple model taken from Chat-GPT:
                # Se, C_tangent_fourth = neo_hookean_stress_tangent(F, mu, lambdaa)
                # Se_voigt = TensorToVoigt(Se)
                # C_tangent = FourthTensorToVoigt(C_tangent_fourth)

                # Linear To check:
                # E = 0.5*(C - jnp.eye(2))
                # Ee_voigt = TensorToVoigt(E)
                # Ee_voigt = Ee_voigt.at[2,0].multiply(2)
                # Se_voigt = jnp.dot(dd,Ee_voigt)
                # W_int = W_int + jnp.dot(BB.T,Se_voigt) * gauss_weights[i] * gauss_weights[j] * detJ
                # ke = ke + jnp.dot(jnp.dot(BB.T,dd),BB) * gauss_weights[i] * gauss_weights[j] * detJ
                
                BB = jnp.zeros((3, 2*num_elem_nodes))
                for m in range(num_elem_nodes):
                    BB = BB.at[0,2*m:2*m+2].set([F[0,0]*dN_dX[0,m], F[1,0]*dN_dX[0,m]])
                    BB = BB.at[1,2*m:2*m+2].set([F[0,1]*dN_dX[1,m], F[1,1]*dN_dX[1,m]])
                    BB = BB.at[2,2*m:2*m+2].set([F[0,1]*dN_dX[0,m]+F[0,0]*dN_dX[1,m], F[1,1]*dN_dX[0,m]+F[1,0]*dN_dX[1,m]])
                
                nf = jnp.array([[Nf[0], 0, Nf[1], 0, Nf[2], 0, Nf[3], 0],[0, Nf[0], 0, Nf[1], 0, Nf[2], 0, Nf[3]]])
                

                xsi = xsi + xsie * gauss_weights[i] * gauss_weights[j] * detJ
                fe = fe + gauss_weights[i] * gauss_weights[j] * detJ * jnp.dot(jnp.transpose(nf), body_force)
                W_int = W_int + jnp.dot(BB.T,Se_voigt) * gauss_weights[i] * gauss_weights[j] * detJ
                ke = ke + jnp.dot(jnp.dot(BB.T,C_tangent),BB) * gauss_weights[i] * gauss_weights[j] * detJ
                

        return  xsi, W_int - fe, ke

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
