#
# Authors: Reza Najian Asl, https://github.com/RezaNajian
# Date: July, 2024
# License: FOL/LICENSE
#

"""
fem_utilities.py
A module for performing basic finite element operations.
"""
import jax.numpy as jnp
from jax import jit
from functools import partial

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
        eye = jnp.eye(dim)
        I4 = jnp.einsum('ik,jl->ijkl', eye, eye)
        return I4
    
    def diad_special(self, A, B, dim):
        """
        Calculate a specific tensor diad: Cijkl = (1/2)*(A[i,k] * B[j,l] + A[i,l] * B[j,k])
        """
        P = 0.5* (jnp.einsum('ik,jl->ijkl',A,B) + jnp.einsum('il,jk->ijkl',A,B))
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
            voigt = voigt.at[3,0].set(tensor[1,2])
            voigt = voigt.at[4,0].set(tensor[0,2])
            voigt = voigt.at[5,0].set(tensor[0,1])
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
                (1, 2), (0, 2), (0, 1)
                ]
            
            for I, (i, j) in enumerate(indices):
                for J, (k, l) in enumerate(indices):
                    C = C.at[I, J].set(Cf[i, j, k, l])
            
            return C


class NeoHookianModel(MaterialModel):
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
        J = jnp.linalg.det(F)
        ph = 0.5*k*(J-(1/J))
        #dp_dJ = (k/4)*(2 + 2*J**(-2))
        dp_dJ = 0.5*k*(1 + J**(-2))

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

        C_ = jnp.einsum('ij,kl->ijkl',jnp.zeros(C.shape),jnp.zeros(C.shape))
        P_double_C = jnp.einsum('ijkl,klpq->ijpq',P,C_)
        P_bar = self.diad_special(invC,invC,invC.shape[0]) - (1/3)*jnp.einsum('ij,kl->ijkl',invC,invC)
        C_vol = (J*ph + dp_dJ*J**2)*jnp.einsum('ij,kl->ijkl',invC,invC) - 2*J*ph*self.diad_special(invC,invC,invC.shape[0])
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
        # Supporting functions:

        C = jnp.dot(F.T,F)
        invC = jnp.linalg.inv(C)
        J = jnp.linalg.det(F)
        p = 0.5*k*(J-(1/J))
        dp_dJ = 0.5*k*(1 + J**(-2))

        # Strain Energy
        xsie_vol = (k/4)*(J**2 - 2*jnp.log(J) -1)
        I1_bar = (J**(-2/2))*jnp.trace(C)
        xsie_iso = 0.5*mu*(I1_bar - 2)
        xsie = xsie_vol + xsie_iso

        # Stress Tensor
        S_vol = J*p*invC
        I_fourth = self.fourth_order_identity_tensor(C.shape[0])
        P = I_fourth - (1/2)*jnp.einsum('ij,kl->ijkl', invC, C)
        S_bar = mu*jnp.eye(C.shape[0])
        S_iso = (J**(-2/2))*jnp.einsum('ijkl,kl->ij',P,S_bar)
        Se = S_vol + S_iso

        C_ = jnp.einsum('ij,kl->ijkl',jnp.zeros(C.shape),jnp.zeros(C.shape))
        P_double_C = jnp.einsum('ijkl,klpq->ijpq',P,C_)
        P_bar = self.diad_special(invC,invC,invC.shape[0]) - (1/2)*jnp.einsum('ij,kl->ijkl',invC,invC)
        C_vol = (J*p + dp_dJ*J**2)*jnp.einsum('ij,kl->ijkl',invC,invC) - 2*J*p*self.diad_special(invC,invC,invC.shape[0])
        C_iso = jnp.einsum('ijkl,pqkl->ijpq',P_double_C,P) + \
                (2/2)*(J**(-2/2))*jnp.vdot(S_bar,C)*P_bar - \
                (2/2)*(jnp.einsum('ij,kl->ijkl',invC,S_iso) + jnp.einsum('ij,kl->ijkl',S_iso,invC))
        C_tangent_fourth = C_vol + C_iso
        Se_voigt = self.TensorToVoigt(Se)
        C_tangent = self.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent
    
    
class MooneyRivlinModel(MaterialModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, F:jnp.array, c10:float=0.2588 ,c01:float=-0.0449 ,kappa:float=2000):
        """
        Evaluate the stress and tangent operator at given local coordinates.
        This method should be overridden by subclasses.

        Parameters:
        F (ndarray): Deformation gradient.
        args (float): Optional material constants

        Returns:
        jnp.ndarray: Values of stress and tangent operator at given local coordinates.
        """
        c10, c01 = c10, c01

        C = jnp.dot(F.T,F)
        E_mat = 0.5*(C - jnp.eye(3)) 
        E_voigt = jnp.zeros((6,1))
        E_voigt = E_voigt.at[0,0].set(E_mat[0,0])
        E_voigt = E_voigt.at[1,0].set(E_mat[1,1])
        E_voigt = E_voigt.at[2,0].set(E_mat[2,2])
        E_voigt = E_voigt.at[3,0].set(E_mat[0,1])
        E_voigt = E_voigt.at[4,0].set(E_mat[1,2])
        E_voigt = E_voigt.at[5,0].set(E_mat[0,2])

        one_tensor_voigt = jnp.array([[1],[1],[1],[0],[0],[0]])
        C_cauchy = 2 * E_voigt + one_tensor_voigt
        I1 = C_cauchy[0,0] + C_cauchy[1,0] + C_cauchy[2,0]
        I2 = C_cauchy[0,0] * C_cauchy[1,0] + C_cauchy[0,0] * C_cauchy[2,0] + C_cauchy[1,0] * C_cauchy[2,0] \
                - C_cauchy[3,0] * C_cauchy[3,0] - C_cauchy[4,0] * C_cauchy[4,0] - C_cauchy[5,0] * C_cauchy[5,0]
        I3 = (C_cauchy[0,0]*C_cauchy[1,0] - C_cauchy[3,0]*C_cauchy[3,0])*C_cauchy[2,0] +\
                (C_cauchy[3,0]*C_cauchy[5,0] - C_cauchy[0,0]*C_cauchy[4,0])*C_cauchy[4,0] +\
                (C_cauchy[3,0]*C_cauchy[4,0] - C_cauchy[1,0]*C_cauchy[5,0])*C_cauchy[4,0]
        
        J1 = I1 * I3**(-1/3)
        J2 = I2 * I3**(-2/3)
        J3 = I3 ** (1/2)
        J3M1 = J3 - 1.
        
        I1E = 2*one_tensor_voigt
        I2E = 2 * jnp.array([[C_cauchy[1,0] + C_cauchy[2,0]], 
                               [C_cauchy[2,0] + C_cauchy[0,0]],
                               [C_cauchy[0,0] + C_cauchy[1,0]],
                               [-C_cauchy[3,0]], [-C_cauchy[4,0]], [-C_cauchy[6,0]]])
        I3E = 2 * jnp.array([[C_cauchy[1,0]*C_cauchy[2,0] - C_cauchy[4,0]*C_cauchy[4,0]],
                              [ C_cauchy[2,0]*C_cauchy[0,0] - C_cauchy[5,0]*C_cauchy[5,0]],
                               [C_cauchy[0,0]*C_cauchy[1,0] - C_cauchy[3,0]*C_cauchy[3,0]],
                               [C_cauchy[4,0]*C_cauchy[5,0] - C_cauchy[2,0]*C_cauchy[3,0]],
                               [C_cauchy[5,0]*C_cauchy[3,0] - C_cauchy[0,0]*C_cauchy[4,0]],
                               [C_cauchy[3,0]*C_cauchy[4,0] - C_cauchy[1,0]*C_cauchy[5,0]]])
        
        J1E = I3**(-1/3) * I1E - (1/3)*I1*(I3**(-4/3))*I3E
        J2E = I3**(-2/3) * I2E - (2/3)*I2*(I3**(-5/3))*I3E
        J3E = (1/2)*I3**(-1/2)*I3E

        S = c10*J1E + c01*J2E + kappa*J3M1*J3E

        # Material stiffness
        D = jnp.zeros((6, 6))

        # Define matrices
        I2EE = jnp.array([
            [0, 4, 4, 0, 0, 0],
            [4, 0, 4, 0, 0, 0],
            [4, 4, 0, 0, 0, 0],
            [0, 0, 0, -2, 0, 0],
            [0, 0, 0, 0, -2, 0],
            [0, 0, 0, 0, 0, -2]
        ])

        I3EE = jnp.array([
            [0,   4*C_cauchy[3,0], 4*C_cauchy[2,0],   0,  -4*C_cauchy[5,0],   0],
            [4*C_cauchy[3,0],   0, 4*C_cauchy[1,0],   0,     0, -4*C_cauchy[6,0]],
            [4*C_cauchy[2,0], 4*C_cauchy[1,0],   0,   0, -4*C_cauchy[4,0],   0],
            [0,     0,   0, -2*C_cauchy[3,0], 2*C_cauchy[6,0], 2*C_cauchy[5,0]],
            [-4*C_cauchy[5,0], 0, -4*C_cauchy[4,0], 2*C_cauchy[6,0], -2*C_cauchy[1,0], 2*C_cauchy[4,0]],
            [0, -4*C_cauchy[6,0], 0, 2*C_cauchy[5,0], 2*C_cauchy[4,0], -2*C_cauchy[2,0]]
        ])

        # Scalars
        W1 = (2/3) * I3**(-1/2)
        W4 = (4/3) * I3**(-1/2)
        W2 = (8/9) * I1 * I3**(-4/3)
        W3 = (1/3) * I1 * I3**(-4/3)
        W5 = (8/9) * I2 * I3**(-5/3)
        W6 = I3**(-2/3)
        W7 = (2/3) * I2 * I3**(-5/3)
        W8 = I3**(-1/2)
        W9 = (1/2) * I3**(-1/2)

        # Second-order tensors
        J1EE = -W1 * (J1E @ J3E.T + J3E @ J1E.T) + W2 * (J3E @ J3E.T) - W3 * I3EE
        J2EE = -W4 * (J2E @ J3E.T + J3E @ J2E.T) + W5 * (J3E @ J3E.T) + W6 * I2EE - W7 * I3EE
        J3EE = -W8 * (J3E @ J3E.T) + W9 * I3EE

        # Final stiffness matrix
        D = c10 * J1EE + c01 * J2EE + kappa * (J3E @ J3E.T) + kappa * J3M1 * J3EE

        Xsie = c10*(J1 - 3) + c01*(J2 - 3) + (kappa/2)*(J3 - 1)**2
        return Xsie, S, D
    
class MooneyRivlinModelPaper(MaterialModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, F:jnp.array, c10:float=0.2588 ,c01:float=-0.0449 ,kappa:float=2000):
        
        c10 = c10
        c01 = c01
        kappa = kappa

        C = jnp.dot(F.T,F)
        invC = jnp.linalg.inv(C)
        J = jnp.linalg.det(F)
        # p = (kappa/4)*(2*J-2*J**(-1))
        p_h = 0.5*kappa*(J-(1/J))
        dp_dJ = (kappa/4)*(2 + 2*J**(-2))

        # Strain Energy
        xsie_vol = (kappa/4)*(J**2 - 2*jnp.log(J) -1)
        I1_bar = (J**(-2/3))*jnp.trace(C)
        C2 = jnp.einsum('ij,jk->ik',C,C)
        I2_bar = 0.5*(I1_bar**2 - (J**(-4/3)*jnp.trace(C2)))
        xsie_iso = c10*(I1_bar - 3) + c01*(I2_bar - 3)
        xsie = xsie_vol + xsie_iso

        # Stress Tensor
        S_vol = J*p_h*invC
        I_fourth = self.fourth_order_identity_tensor(C.shape[0])
        P = I_fourth - (1/3)*jnp.einsum('ij,kl->ijkl', invC, C)
        S_bar = 2*(c10 + c01*I1_bar)*jnp.eye(C.shape[0]) - 2*c01*J**(-2/3)*C
        S_iso = (J**(-2/3))*jnp.einsum('ijkl,kl->ij',P,S_bar)
        Se = S_vol + S_iso

        C_ = 4*J**(-4/3)*(c01*jnp.einsum('ij,kl->ijkl',jnp.eye(3),jnp.eye(3)) - c01*I_fourth)
        P_double_C = jnp.einsum('ijkl,klpq->ijpq',P,C_)
        P_bar = self.diad_special(invC,invC,invC.shape[0]) - (1/3)*jnp.einsum('ij,kl->ijkl',invC,invC)
        C_vol = (J*p_h + dp_dJ*(J**2))*jnp.einsum('ij,kl->ijkl',invC,invC) - 2*J*p_h*self.diad_special(invC,invC,invC.shape[0])
        C_iso = jnp.einsum('ijkl,pqkl->ijpq',P_double_C,P) + \
                (2/3)*(J**(-2/3))*jnp.einsum('ij,ij->',S_bar,C)*P_bar - \
                (2/3)*(jnp.einsum('ij,kl->ijkl',invC,S_iso) + jnp.einsum('ij,kl->ijkl',S_iso,invC))
        C_tangent_fourth = C_vol + C_iso
        Se_voigt = self.TensorToVoigt(Se)
        C_tangent = self.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent
    

class SaintVenant(MaterialModel):
    """
    Material model.
    """
    @partial(jit, static_argnums=(0,))
    def evaluate(self, F, lambda_, mu):
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

        E = 0.5*(F.T @ F - jnp.eye(F.shape[0]))
        xsie = 0.5*lambda_*(jnp.linalg.trace(E) ** 2) + mu*jnp.linalg.trace(E @ E)

        I_fourth = self.fourth_order_identity_tensor(F.shape[0])
        C_tangent_fourth = lambda_ * jnp.einsum('ij,kl->jikl',jnp.eye(F.shape[0]),jnp.eye(F.shape[0])) +\
                            2 * mu * I_fourth
        Se = jnp.einsum('ijkl,kl->ij',C_tangent_fourth,E)
        Se_voigt = self.TensorToVoigt(Se)
        C_tangent = self.FourthTensorToVoigt(C_tangent_fourth)
        return xsie, Se_voigt, C_tangent