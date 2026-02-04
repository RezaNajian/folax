"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/LICENSE
"""
from  .fe_loss import FiniteElementLoss
from .fe_kinematics import b_matrix_2d, b_matrix_3d, d_matrix_2d, d_matrix_3d, n_matrix_2d, n_matrix_3d
import jax
import jax.numpy as jnp
from functools import partial
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh

class MechanicalLoss(FiniteElementLoss):
    """
    Mechanical loss functional for linear elastic finite element problems.

    This class defines a scalar loss functional for small-strain continuum
    mechanics problems. The total loss value represents a domain-integrated
    measure of mechanical equilibrium and is assembled by summing element-level
    contributions over all finite elements in the mesh.

    For each element, the scalar contribution to the total loss is defined as
    the inner product of the element residual vector with the corresponding
    element degrees of freedom. The global loss is obtained by accumulating
    these scalar contributions across the computational domain, resulting in
    a weighted residual formulation.

    Element-level stiffness matrices and force vectors are evaluated using
    Gaussian quadrature. The constitutive response is linear elastic and is
    defined by Young's modulus and Poisson's ratio provided in the loss
    settings. Both two-dimensional and three-dimensional problems are
    supported.

    The loss formulation is compatible with adjoint-based sensitivity analysis
    and gradient-based optimization, as the scalar loss is constructed
    consistently from the element residuals and state variables.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing problem and material definitions. Must
            include ``material_dict`` with keys ``"young_modulus"`` and
            ``"poisson_ratio"``. May optionally include ``"body_foce"`` to
            specify a constant body force vector.
        fe_mesh (Mesh):
            Finite element mesh over which the loss is defined.

    Attributes:
        D (jax.numpy.ndarray):
            Constitutive matrix in Voigt notation (3x3 in 2D, 6x6 in 3D).
        body_force (jax.numpy.ndarray):
            Body force vector with shape ``(2, 1)`` in 2D or ``(3, 1)`` in 3D.
        CalculateNMatrix (callable):
            Dimension-dependent shape-function matrix constructor.
        CalculateBMatrix (callable):
            Dimension-dependent strain-displacement matrix constructor.
    """
    def Initialize(self) -> None:
        super().Initialize()
        if "material_dict" not in self.loss_settings.keys():
            fol_error("material_dict should provided in the loss settings !")
        if self.dim == 2:
            self.CalculateNMatrix = n_matrix_2d
            self.CalculateBMatrix = b_matrix_2d
            self.D = d_matrix_2d(self.loss_settings["material_dict"]["young_modulus"],
                                 self.loss_settings["material_dict"]["poisson_ratio"])
            self.body_force = jnp.zeros((2,1))
            if "body_foce" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_foce"])
        else:
            self.CalculateNMatrix = n_matrix_3d
            self.CalculateBMatrix = b_matrix_3d
            self.D = d_matrix_3d(self.loss_settings["material_dict"]["young_modulus"],
                                 self.loss_settings["material_dict"]["poisson_ratio"])
            self.body_force = jnp.zeros((3,1))
            if "body_foce" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_foce"])

    def ComputeElement(self,xyze,de,uvwe):
        """
        Compute element-level contributions to the mechanical loss.

        This method evaluates the element stiffness matrix and force vector
        using Gaussian quadrature. The element residual is defined as
        ``Se @ uvwe - Fe``. The scalar element contribution represents this
        element's contribution to the total scalar loss value of the problem.

        The total loss for the mechanical problem is obtained by summing the
        scalar contributions of all elements over the computational domain.
        Each element contribution is computed as the inner product of the
        element residual vector with the element degrees of freedom, resulting
        in a weighted residual formulation in which the element DOFs act as
        weighting functions.

        Args:
            xyze:
                Element nodal coordinates.
            de:
                Element parameter field values at nodes (e.g., density or
                heterogeneity coefficients).
            uvwe:
                Element DOF vector arranged consistently with the element type
                and ordered degrees of freedom.

        Returns:
            Tuple[float, jax.numpy.ndarray, jax.numpy.ndarray]:
                - Scalar element loss contribution defined as
                  ``uvwe.T @ (Se @ uvwe - Fe)``.
                - Element residual vector ``Se @ uvwe - Fe``.
                - Element stiffness matrix ``Se``.
        """
        def compute_at_gauss_point(gp_point,gp_weight):
            N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
            N_mat = self.CalculateNMatrix(N_vec)
            e_at_gauss = jnp.dot(N_vec, de.squeeze())
            DN_DX = self.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point)
            B_mat = self.CalculateBMatrix(DN_DX)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            gp_stiffness = gp_weight * detJ * e_at_gauss * (B_mat.T @ (self.D @ B_mat))
            gp_f = gp_weight * detJ * (N_mat.T @ self.body_force)
            return gp_stiffness,gp_f

        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        k_gps,f_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        Se = jnp.sum(k_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        element_residuals = jax.lax.stop_gradient(Se @ uvwe - Fe)
        return  ((uvwe.T @ element_residuals)[0,0]), (Se @ uvwe - Fe), Se

class MechanicalLoss3DTetra(MechanicalLoss):
    """
    Mechanical loss for 3D tetrahedral finite elements.

    This class configures :class:`MechanicalLoss` for three-dimensional
    problems discretized with tetrahedral elements. The displacement field
    consists of three components (``Ux``, ``Uy``, ``Uz``) per node, and the
    mechanical loss is assembled using the weighted residual formulation
    defined in the base class.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing material parameters and optional body-force
            definitions. Must include ``material_dict`` with elastic constants.
        fe_mesh (Mesh):
            Finite element mesh associated with the loss.
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["Ux","Uy","Uz"],
                               "element_type":"tetra"},fe_mesh)

class MechanicalLoss3DHexa(MechanicalLoss):
    """
    Mechanical loss for 3D hexahedral finite elements.

    This class configures :class:`MechanicalLoss` for three-dimensional
    problems discretized with hexahedral elements. The displacement field
    consists of three components (``Ux``, ``Uy``, ``Uz``) per node, and the
    mechanical loss is assembled using the weighted residual formulation
    defined in the base class.

    If the number of Gauss points is not specified in the loss settings,
    a default value of ``num_gp = 2`` is used.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing material parameters, optional body-force
            definitions, and optional integration settings.
        fe_mesh (Mesh):
            Finite element mesh associated with the loss.
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["Ux","Uy","Uz"],
                               "element_type":"hexahedron"},fe_mesh)

class MechanicalLoss2DTri(MechanicalLoss):
    """
    Mechanical loss for 2D triangular finite elements.

    This class configures :class:`MechanicalLoss` for two-dimensional
    problems discretized with triangular elements. The displacement field
    consists of two components (``Ux``, ``Uy``) per node, and the mechanical
    loss is assembled using the weighted residual formulation defined in the
    base class.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing material parameters and optional body-force
            definitions.
        fe_mesh (Mesh):
            Finite element mesh associated with the loss.
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["Ux","Uy"],
                               "element_type":"triangle"},fe_mesh)

class MechanicalLoss2DQuad(MechanicalLoss):
    """
    Mechanical loss for 2D quadrilateral finite elements.

    This class configures :class:`MechanicalLoss` for two-dimensional
    problems discretized with quadrilateral elements. The displacement field
    consists of two components (``Ux``, ``Uy``) per node, and the mechanical
    loss is assembled using the weighted residual formulation defined in the
    base class.

    If the number of Gauss points is not specified in the loss settings,
    a default value of ``num_gp = 2`` is used.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing material parameters, optional body-force
            definitions, and optional integration settings.
        fe_mesh (Mesh):
            Finite element mesh associated with the loss.
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["Ux","Uy"],
                               "element_type":"quad"},fe_mesh)
