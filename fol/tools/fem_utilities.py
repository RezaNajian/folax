#
# Authors: Reza Najian Asl, https://github.com/RezaNajian
# Date: July, 2024
# License: FOL/LICENSE
#

"""
fem_utilities.py
A module for performing basic finite element operations.
"""
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
import jax

class ConstantsMeta(type):
    def __setattr__(self, key, value):
        raise AttributeError("Cannot modify a constant value")

class GaussQuadrature(metaclass=ConstantsMeta):

    """
    Gauss Quadrature class for integration.

    """

    @property
    def one_point_GQ(self):
        # 1-point Gauss quadrature
        points = jnp.array([0.0])
        weights = jnp.array([2.0])
        return points,weights

    @property
    def two_point_GQ(self):
        # 2-point Gauss quadrature
        points = jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])
        weights = jnp.array([1.0, 1.0])
        return points,weights
    
    @property
    def three_point_GQ(self):
        # 3-point Gauss quadrature
        points = jnp.array([-jnp.sqrt(3/5), 0.0, jnp.sqrt(3/5)])
        weights = jnp.array([5/9, 8/9, 5/9])
        return points,weights
    
    @property
    def four_point_GQ(self):
        # 4-point Gauss quadrature
        points = jnp.array([-jnp.sqrt((3+2*jnp.sqrt(6/5))/7), -jnp.sqrt((3-2*jnp.sqrt(6/5))/7), jnp.sqrt((3-2*jnp.sqrt(6/5))/7), jnp.sqrt((3+2*jnp.sqrt(6/5))/7)])
        weights = jnp.array([(18-jnp.sqrt(30))/36, (18+jnp.sqrt(30))/36, (18+jnp.sqrt(30))/36, (18-jnp.sqrt(30))/36])
        return points,weights


class ShapeFunction:
    """
    Base class for shape functions of finite elements.
    """

    def evaluate(self, xi, eta, zeta=None):
        """
        Evaluate the shape function at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.
        zeta (float, optional): Local coordinate in the zeta direction for 3D elements.

        Returns:
        jnp.ndarray: Values of shape functions at given local coordinates.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def derivatives(self, xi, eta, zeta=None):
        """
        Evaluate the derivatives of the shape functions at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.
        zeta (float, optional): Local coordinate in the zeta direction for 3D elements.

        Returns:
        jnp.ndarray: Derivatives of shape functions at given local coordinates.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class QuadShapeFunction(ShapeFunction):
    """
    Shape functions for a quadrilateral element.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, xi, eta, zeta=None):
        """
        Evaluate the shape function for a quadrilateral element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.

        Returns:
        jnp.ndarray: Values of shape functions at given local coordinates.
        """
        N = jnp.array([
            (1 - xi) * (1 - eta) / 4,
            (1 + xi) * (1 - eta) / 4,
            (1 + xi) * (1 + eta) / 4,
            (1 - xi) * (1 + eta) / 4
        ])
        return N
    
    @partial(jit, static_argnums=(0,))
    def derivatives(self, xi, eta, zeta=None):
        """
        Evaluate the derivatives of the shape functions for a quadrilateral element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.

        Returns:
        jnp.ndarray: Derivatives of shape functions at given local coordinates.
        """
        dN_dxi = jnp.array([
            [-(1 - eta) / 4, -(1 - xi) / 4],
            [(1 - eta) / 4, -(1 + xi) / 4],
            [(1 + eta) / 4, (1 + xi) / 4],
            [-(1 + eta) / 4, (1 - xi) / 4]
        ])
        return dN_dxi

class TriangleShapeFunction(ShapeFunction):
    """
    Shape functions for a triangular element.
    """

    def evaluate(self, xi, eta, zeta=None):
        """
        Evaluate the shape function for a triangular element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.

        Returns:
        jnp.ndarray: Values of shape functions at given local coordinates.
        """
        N = jnp.array([
            1 - xi - eta,
            xi,
            eta
        ])
        return N
    
    def derivatives(self, xi, eta, zeta=None):
        """
        Evaluate the derivatives of the shape functions for a triangular element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.

        Returns:
        jnp.ndarray: Derivatives of shape functions at given local coordinates.
        """
        dN_dxi = jnp.array([
            [-1, -1],
            [1, 0],
            [0, 1]
        ])
        return dN_dxi

class TetrahedralShapeFunction(ShapeFunction):
    """
    Shape functions for a tetrahedral element.
    """

    def evaluate(self, xi, eta, zeta):
        """
        Evaluate the shape function for a tetrahedral element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.
        zeta (float): Local coordinate in the zeta direction.

        Returns:
        jnp.ndarray: Values of shape functions at given local coordinates.
        """
        N = jnp.array([
            1 - xi - eta - zeta,
            xi,
            eta,
            zeta
        ])
        return N
    
    def derivatives(self, xi, eta, zeta):
        """
        Evaluate the derivatives of the shape functions for a tetrahedral element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.
        zeta (float): Local coordinate in the zeta direction.

        Returns:
        jnp.ndarray: Derivatives of shape functions at given local coordinates.
        """
        dN_dxi = jnp.array([
            [-1, -1, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        return dN_dxi

class HexahedralShapeFunction(ShapeFunction):
    """
    Shape functions for a hexahedral element.
    """

    def evaluate(self, xi, eta, zeta):
        """
        Evaluate the shape function for a hexahedral element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.
        zeta (float): Local coordinate in the zeta direction.

        Returns:
        jnp.ndarray: Values of shape functions at given local coordinates.
        """
        N = jnp.array([
            (1 - xi) * (1 - eta) * (1 - zeta) / 8,
            (1 + xi) * (1 - eta) * (1 - zeta) / 8,
            (1 + xi) * (1 + eta) * (1 - zeta) / 8,
            (1 - xi) * (1 + eta) * (1 - zeta) / 8,
            (1 - xi) * (1 - eta) * (1 + zeta) / 8,
            (1 + xi) * (1 - eta) * (1 + zeta) / 8,
            (1 + xi) * (1 + eta) * (1 + zeta) / 8,
            (1 - xi) * (1 + eta) * (1 + zeta) / 8
        ])
        return N
    
    def derivatives(self, xi, eta, zeta):
        """
        Evaluate the derivatives of the shape functions for a hexahedral element at given local coordinates.

        Parameters:
        xi (float): Local coordinate in the xi direction.
        eta (float): Local coordinate in the eta direction.
        zeta (float): Local coordinate in the zeta direction.

        Returns:
        jnp.ndarray: Derivatives of shape functions at given local coordinates.
        """
        dN_dxi = jnp.array([
            [-(1 - eta) * (1 - zeta) / 8, -(1 - xi) * (1 - zeta) / 8, -(1 - xi) * (1 - eta) / 8],
            [(1 - eta) * (1 - zeta) / 8, -(1 + xi) * (1 - zeta) / 8, -(1 + xi) * (1 - eta) / 8],
            [(1 + eta) * (1 - zeta) / 8, (1 + xi) * (1 - zeta) / 8, -(1 + xi) * (1 + eta) / 8],
            [-(1 + eta) * (1 - zeta) / 8, (1 - xi) * (1 - zeta) / 8, -(1 - xi) * (1 + eta) / 8],
            [-(1 - eta) * (1 + zeta) / 8, -(1 - xi) * (1 + zeta) / 8, (1 - xi) * (1 - eta) / 8],
            [(1 - eta) * (1 + zeta) / 8, -(1 + xi) * (1 + zeta) / 8, (1 + xi) * (1 - eta) / 8],
            [(1 + eta) * (1 + zeta) / 8, (1 + xi) * (1 + zeta) / 8, (1 + xi) * (1 + eta) / 8],
            [-(1 + eta) * (1 + zeta) / 8, (1 - xi) * (1 + zeta) / 8, (1 - xi) * (1 + eta) / 8]
        ])
        return dN_dxi

class MaterialModel:
    """
    Base class for Material models.
    """

    def evaluate(self, F, *args, **kwargs):
        """
        Evaluate the stress and tangent operator at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        F (ndarray): Deformation gradient.
        args (float): Optional material constants

        Returns:
        jnp.ndarray: Values of stress and tangent operator at given local coordinates.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def fourth_order_identity_tensor(self, dim=3):
        """
        Calculate fourth identity matrix
        """
        # I = jnp.zeros((dim, dim, dim, dim))
        # I = jnp.einsum('ik,jl->ijkl',jnp.eye(dim),jnp.eye(dim))
        I4 = jnp.zeros((dim, dim, dim, dim))
        for i in range(dim):
            for j in range(dim):
                I4 = I4.at[i, j, i, j].set(1)
        return I4
    
    def diad_special(self, A, B, dim):
        """
        Calculate a specific tensor diad: Cijkl = (1/2)*(Aik * Bjl + Ail * Bjk)
        """
        P = jnp.zeros((dim, dim, dim, dim))
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        P = P.at[i,j,k,l].add(0.5* (A[i,k]*B[j,l] + A[i,l]*B[j,k]))
        return P
    
    def TensorToVoigt(self, tensor):
        """
        Convert a tensor to a vector
        """
        if tensor.size == 4:
            voigt = jnp.zeros((3,1))
            voigt = voigt.at[0,0].set(tensor[0,0])
            voigt = voigt.at[1,0].set(tensor[1,1])
            voigt = voigt.at[2,0].set(tensor[0,1])
            return voigt
        elif tensor.size == 9:
            voigt = jnp.zeros((6,1))
            voigt = voigt.at[0,0].set(tensor[0,0])
            voigt = voigt.at[1,0].set(tensor[1,1])
            voigt = voigt.at[2,0].set(tensor[2,2])
            voigt = voigt.at[3,0].set(tensor[0,1])
            voigt = voigt.at[4,0].set(tensor[0,2])
            voigt = voigt.at[5,0].set(tensor[1,2])
            return voigt

    def FourthTensorToVoigt(self,Cf):
        """
        Convert a fouth-order tensor to a second-order tensor
        """
        if Cf.size == 16:
            C = jnp.zeros((3,3))
            C = C.at[0,0].set(Cf[0,0,0,0])
            C = C.at[0,1].set(Cf[0,0,1,1])
            C = C.at[0,2].set(Cf[0,0,0,1])
            C = C.at[1,0].set(C[0,1])
            C = C.at[1,1].set(Cf[1,1,1,1])
            C = C.at[1,2].set(Cf[1,1,0,1])
            C = C.at[2,0].set(C[0,2])
            C = C.at[2,1].set(C[1,2])
            C = C.at[2,2].set(Cf[0,1,0,1])
            return C
        elif Cf.size == 81: 
            C = jnp.zeros((6, 6))
            indices = [
                (0, 0), (1, 1), (2, 2), 
                (0, 1), (0, 2), (1, 2)
                ]
            
            for I, (i, j) in enumerate(indices):
                for J, (k, l) in enumerate(indices):
                    C = C.at[I, J].set(Cf[i, j, k, l])
            
            return C


class NeoHookianModel3D(MaterialModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, F, k, mu):
        """
        Evaluate the stress and tangent operator at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        F (ndarray): Deformation gradient.
        args (float): Optional material constants

        Returns:
        jnp.ndarray: Values of stress and tangent operator at given local coordinates.
        """
        # Supporting functions:

        C = jnp.dot(F.T,F)
        invC = jnp.linalg.inv(C)
        J = jnp.sqrt(jnp.linalg.det(C))
        ph = 0.5*k*(J-(1/J))
        dp_dJ = (k/4)*(2 + 2*J**(-2))

        # Strain Energy
        xsie_vol = (k/4)*(J**2 - 2*jnp.log(J) -1)
        I1_bar = (J**(-2/3))*jnp.trace(C)
        xsie_iso = 0.5*mu*(I1_bar - 3)
        xsie = xsie_vol + xsie_iso

        # Stress Tensor
        S_vol = J*ph*invC
        I_fourth = self.fourth_order_identity_tensor(C.shape[0])
        P = I_fourth - (1/3)*jnp.einsum('ij,kl->ijkl', invC, C)
        S_bar = mu*jnp.eye(C.shape[0])
        S_iso = (J**(-2/3))*jnp.einsum('ijkl,kl->ij',P,S_bar)
        Se = S_vol + S_iso

        # Tangent Stiffness
        C_ = jnp.einsum('ij,kl->ijkl',jnp.zeros(C.shape),jnp.zeros(C.shape))
        P_double_C = jnp.einsum('ijkl,klpq->ijpq',P,C_)
        P_bar = self.diad_special(invC,invC,invC.shape[0]) - (1/3)*jnp.einsum('ij,kl->ijkl',invC,invC)
        C_vol = (J*ph + dp_dJ*(J**2))*jnp.einsum('ij,kl->ijkl',invC,invC) - 2*J*ph*self.diad_special(invC,invC,invC.shape[0])
        C_iso = jnp.einsum('ijkl,pqkl->ijpq',P_double_C,P) + \
                (2/3)*(J**(-2/3))*jnp.vdot(S_bar,C)*P_bar - \
                (2/3)*(jnp.einsum('ij,kl->ijkl',invC,S_iso) + jnp.einsum('ij,kl->ijkl',S_iso,invC))
        C_tangent_fourth = C_vol + C_iso
        Se_voigt = self.TensorToVoigt(Se)
        C_tangent = self.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent
    
class NeoHookianModel2D(MaterialModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, F, k, mu):
        """
        Evaluate the stress and tangent operator at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        F (ndarray): Deformation gradient.
        args (float): Optional material constants

        Returns:
        jnp.ndarray: Values of stress and tangent operator at given local coordinates.
        """
        C = jnp.dot(F.T,F)
        invC = jnp.linalg.inv(C)
        J = jnp.sqrt(jnp.linalg.det(C))
        ph = 0.5*k*(J-(1/J))
        dp_dJ = 0.5*k*(1 + J**(-2))

        # Strain Energy
        xsie_vol = (k/4)*(J**2 - 2*jnp.log(J) -1)
        I1_bar = (J**(-1))*jnp.trace(C)
        xsie_iso = 0.5*mu*(I1_bar - 2)
        xsie = xsie_vol + xsie_iso

        # Stress Tensor
        S_vol = J*ph*invC
        I_fourth = self.fourth_order_identity_tensor(C.shape[0])
        P = I_fourth - (1/2)*jnp.einsum('ij,kl->ijkl', invC, C)
        S_bar = mu*jnp.eye(C.shape[0])
        S_iso = (J**(-1))*jnp.einsum('ijkl,kl->ij',P,S_bar)
        Se = S_vol + S_iso

        # Tangent Stiffness
        C_ = jnp.einsum('ij,kl->ijkl',jnp.zeros(C.shape),jnp.zeros(C.shape))
        P_double_C = jnp.einsum('ijkl,klpq->ijpq',P,C_)
        P_bar = self.diad_special(invC,invC,invC.shape[0]) - (1/2)*jnp.einsum('ij,kl->ijkl',invC,invC)
        C_vol = (J*ph + dp_dJ*(J**2))*jnp.einsum('ij,kl->ijkl',invC,invC) - 2*J*ph*self.diad_special(invC,invC,invC.shape[0])
        C_iso = jnp.einsum('ijkl,pqkl->ijpq',P_double_C,P) + \
                (2/2)*(J**(-2/2))*jnp.vdot(S_bar,C)*P_bar - \
                (2/2)*(jnp.einsum('ij,kl->ijkl',invC,S_iso) + jnp.einsum('ij,kl->ijkl',S_iso,invC))
        C_tangent_fourth = C_vol + C_iso
        Se_voigt = self.TensorToVoigt(Se)
        C_tangent = self.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent

class CompressibleNeoHookeanMaterial(MaterialModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, F, mu, lamda):
        """
        Evaluate the stress and tangent operator at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        F (ndarray): Deformation gradient.
        args (float): Optional material constants

        Returns:
        jnp.ndarray: Values of stress and tangent operator at given local coordinates.
        """
        # Supporting functions:

        C = jnp.dot(F.T,F)
        invC = jnp.linalg.inv(C)
        J = jnp.linalg.det(F)
        I_c = jnp.linalg.trace(C)

        # Strain Energy
        xsie = 0.5*mu*(I_c - 3) - mu*jnp.log(J) + 0.5*lamda*(jnp.log(J)**2)

        # Stress Tensor
        Se = mu*(jnp.eye(C.shape[0]) - invC) + lamda*(jnp.log(J))*invC

        # Fourth order tensor I:
        I_fourth = jnp.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        I_fourth = I_fourth.at[i,j,k,l].add(0.5*(invC[i,k]*invC[j,l] + invC[i,l]*invC[j,k]))

        C_tangent_fourth = lamda*jnp.einsum('ij,kl->ijkl', invC,invC,) + 2*(mu - lamda*jnp.log(J))*I_fourth
        Se_voigt = self.TensorToVoigt(Se)
        C_tangent = self.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent
    
class IncompressibleNeoHookianModel(MaterialModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, F, k, mu):
        """
        Evaluate the stress and tangent operator at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        F (ndarray): Deformation gradient.
        args (float): Optional material constants

        Returns:
        jnp.ndarray: Values of stress and tangent operator at given local coordinates.
        """
        
        C = jnp.dot(F.T,F)
        invC = jnp.linalg.inv(C)
        J = jnp.linalg.det(F)
        C_ = (jnp.linalg.det(C)**(-1/3))*C

        IC = jnp.linalg.trace(C)
        IIIC = jnp.linalg.det(C)
        IC_ = jnp.linalg.trace(C_)

        U = 0.5*k*(J - 1)**2
        # U = 0.25*k*(J**2 - 2*jnp.log(J) - 1)
        p = k*(J - 1)   # p = dU/dJ
        # p = 0.5*k*(J - (1/J))

        # Fourth order tensor I:
        I_fourth = jnp.zeros((3,3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        I_fourth = I_fourth.at[i,j,k,l].add(0.5*(invC[i,k]*invC[j,l] + invC[i,l]*invC[j,k]))

        tang_C_ = 2*mu*(IIIC**(-1/3))*((1/3)*IC*I_fourth - (1/3)*jnp.einsum('ij,kl->ijkl',jnp.eye(C.shape[0]),invC) -
                                        (1/3)*jnp.einsum('ij,kl->ijkl',invC,jnp.eye(C.shape[0])) + 
                                        (1/9)*IC*jnp.einsum('ij,kl->ijkl',invC,invC))
        tang_Cp = p*J*(jnp.einsum('ij,kl->ijkl',invC,invC) - 2*I_fourth)
        tang_Ck = k*(J**2)*jnp.einsum('ij,kl->ijkl',invC,invC)
        
        # Strain Energy
        xsie_vol = U
        xsie_iso = 0.5*mu*(IC_ - 3)
        xsie = xsie_vol + xsie_iso
        # xsie = xsie_iso

        # Stress Tensor
        Se = mu*(IIIC**(-1/3))*(jnp.eye(C.shape[0]) - (1/3)*IC*invC) + p*J*invC
        # Se = mu*(IIIC**(-1/3))*(jnp.eye(3) - (1/3)*IC*invC) + p*J*invC
        # tang_C_ = 2.*mu*(IIIC**(-1./3.))*((1./3.)*IC*I_fourth - 
        #                                   (1./3.)*jnp.einsum('ij,kl->ijkl',jnp.eye(3),invC) -
        #                                   (1./3.)*jnp.einsum('ij,kl->ijkl',invC,jnp.eye(3)) +
        #                                   (1./9.)*IC*jnp.einsum('ij,kl->ijkl',invC,invC))
        # tang_Cp = p*J*(jnp.einsum('ij,kl->ijkl',invC,invC) - 2*I_fourth)

        C_tangent_fourth = tang_C_ + tang_Cp + tang_Ck
        Se_voigt = self.TensorToVoigt(Se)
        C_tangent = self.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent

# class IncompressibleNeoHookianModelAD(MaterialModel):
#     """
#     Material model.
#     """
#     @partial(jit, static_argnums=(0,))
#     def evaluate(self, F, k, mu):
#         """
#         Evaluate the stress and tangent operator at given local coordinates.
#         This method should be overridden by subclasses.

#         Parameters:
#         F (ndarray): Deformation gradient.
#         args (float): Optional material constants

#         Returns:
#         jnp.ndarray: Values of stress and tangent operator at given local coordinates.
#         """
        

#         def compute_energy(C, J, k, mu):
#             """ Compute strain energy function for nearly incompressible Neo-Hookean material. """
#             C_ = (J**(-2/3)) * C  # Deviatoric part of C
#             IC_ = jnp.trace(C_)   # Modified first invariant
#             psi = 0.5 * mu * (IC_ - 3) + 0.5 * k * (J - 1)**2
#             return psi

#         @jit
#         def compute_stress(C, J, k, mu):
#             """Compute the Second Piola-Kirchhoff stress tensor."""
#             return jax.grad(compute_energy, 0)(C, J, k, mu)  # Directly computes d(psi)/dC

#         # @jit
#         def compute_tangent(C, J, k, mu):
#             """Compute the material tangent operator."""
#             return jax.jacfwd(compute_stress, 0)(C, J, k, mu)  # Computes dS/dC

#         C = jnp.dot(F.T,F)
#         invC = jnp.linalg.inv(C)
#         J = jnp.linalg.det(F)

#         psi = compute_energy(C, J, k, mu)
#         stress = compute_stress(C, J, k, mu)
#         print("stress shape is: ", stress.shape)
#         jax.debug.print("stress vector: {output}", output=stress)
#         tangent = compute_tangent(C, J, k, mu)
#         print("tangent shape is: ", tangent.shape)
#         jax.debug.print("tangent vector: {output}", output=tangent)

#         Se_voigt = self.TensorToVoigt(stress)
#         C_tangent = self.FourthTensorToVoigt(tangent) 
#         return psi, Se_voigt, C_tangent