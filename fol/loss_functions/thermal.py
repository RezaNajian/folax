"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/LICENSE
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from functools import partial
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *

class ThermalLoss(FiniteElementLoss):
    """
    Thermal loss functional for steady heat conduction finite element problems.

    This class defines a scalar loss functional for heat conduction problems
    solved with finite elements. The total loss value represents a
    domain-integrated measure of thermal equilibrium and is assembled by
    summing element-level contributions over all finite elements in the mesh.

    For each element, the scalar contribution to the total loss is defined as
    the inner product of the element residual vector with the corresponding
    element temperature degrees of freedom. The global loss is obtained by
    accumulating these scalar contributions across the computational domain,
    resulting in a weighted residual formulation.

    Element-level stiffness matrices and source vectors are evaluated using
    Gaussian quadrature. The element conductivity can be nonlinear in the
    temperature field according to the parameters ``beta`` and ``c`` in the
    loss settings, using a conductivity of the form:

    ``k(T) = d(x) * (1 + beta * T^c)``

    where ``d(x)`` is the element parameter field interpolated from ``de`` at
    Gauss points.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing problem settings and optional nonlinear
            conductivity parameters. Supported keys include ``"beta"`` and
            ``"c"`` for the temperature-dependent conductivity factor. The
            element discretization settings (dimension, element type, and
            ordered DOFs) are typically provided by the specialized subclasses.
        fe_mesh (Mesh):
            Finite element mesh over which the loss is defined.

    Attributes:
        thermal_loss_settings (dict):
            Dictionary containing nonlinear conductivity parameters with keys
            ``"beta"`` and ``"c"``.
    """
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        if self.initialized and not reinitialize:
            return
        super().Initialize()
        self.thermal_loss_settings = {"beta":0,"c":1}
        if "beta" in self.loss_settings.keys():
            self.thermal_loss_settings["beta"] = self.loss_settings["beta"]
        if "c" in self.loss_settings.keys():
            self.thermal_loss_settings["c"] = self.loss_settings["c"]

    def ComputeElement(self,xyze,de,te,body_force=0):
        """
        Compute element-level contributions to the thermal loss.

        This method evaluates the element stiffness matrix and source vector
        using Gaussian quadrature. The element residual is defined as
        ``Se @ te - Fe``. The scalar element contribution represents this
        element's contribution to the total scalar loss value of the problem.

        The total loss for the thermal problem is obtained by summing the
        scalar contributions of all elements over the computational domain.
        Each element contribution is computed as the inner product of the
        element residual vector with the element temperature degrees of
        freedom, resulting in a weighted residual formulation.

        Args:
            xyze:
                Element nodal coordinates.
            de:
                Element parameter field values at nodes used to construct the
                conductivity field (interpolated to Gauss points).
            te:
                Element temperature DOFs arranged consistently with the
                element type and ordered DOFs.
            body_force:
                Element volumetric heat source term. This value is treated as a
                constant over the element and integrated against the shape
                functions. Default is 0.

        Returns:
            Tuple[float, jax.numpy.ndarray, jax.numpy.ndarray]:
                - Scalar element loss contribution defined as
                  ``te.T @ (Se @ te - Fe)``.
                - Element residual vector ``Se @ te - Fe``.
                - Element stiffness matrix ``Se``.
        """
        def compute_at_gauss_point(gp_point,gp_weight,te):
            te = jax.lax.stop_gradient(te)
            N = self.fe_element.ShapeFunctionsValues(gp_point)
            conductivity_at_gauss = jnp.dot(N, de.squeeze()) * (1 +
                                    self.thermal_loss_settings["beta"]*(jnp.dot(N,te.squeeze()))**self.thermal_loss_settings["c"])
            DN_DX = self.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            gp_stiffness = conductivity_at_gauss * (DN_DX @ DN_DX.T) * detJ * gp_weight
            gp_f = gp_weight * detJ * body_force *  N.reshape(-1,1)
            return gp_stiffness,gp_f
        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        k_gps,f_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0,None))(gp_points,gp_weights,te)
        Se = jnp.sum(k_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        def compute_elem_res(Se,te ,Fe):
            te = jax.lax.stop_gradient(te)
            return (Se @ te - Fe)
        element_residuals = compute_elem_res(Se,te ,Fe)
        return  ((te.T @ element_residuals)[0,0]), (Se @ te - Fe), Se

class ThermalLoss3DTetra(ThermalLoss):
    """
    Thermal loss for 3D tetrahedral finite elements.

    This class configures :class:`ThermalLoss` for three-dimensional problems
    discretized with tetrahedral elements. The temperature field has a single
    DOF per node (``T``), and the thermal loss is assembled using the weighted
    residual formulation defined in the base class.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing thermal settings and optional nonlinear
            conductivity parameters (e.g., ``"beta"`` and ``"c"``).
        fe_mesh (Mesh):
            Finite element mesh associated with the loss.
    """
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T"],
                               "element_type":"tetra"},fe_mesh)

class ThermalLoss3DHexa(ThermalLoss):
    """
    Thermal loss for 3D hexahedral finite elements.

    This class configures :class:`ThermalLoss` for three-dimensional problems
    discretized with hexahedral elements. The temperature field has a single
    DOF per node (``T``), and the thermal loss is assembled using the weighted
    residual formulation defined in the base class.

    If the number of Gauss points is not specified in the loss settings, a
    default value of ``num_gp = 2`` is used.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing thermal settings, optional nonlinear
            conductivity parameters, and optional integration settings.
        fe_mesh (Mesh):
            Finite element mesh associated with the loss.
    """
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T"],
                               "element_type":"hexahedron"},fe_mesh)

class ThermalLoss2DQuad(ThermalLoss):
    """
    Thermal loss for 2D quadrilateral finite elements.

    This class configures :class:`ThermalLoss` for two-dimensional problems
    discretized with quadrilateral elements. The temperature field has a single
    DOF per node (``T``), and the thermal loss is assembled using the weighted
    residual formulation defined in the base class.

    If the number of Gauss points is not specified in the loss settings, a
    default value of ``num_gp = 2`` is used.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing thermal settings, optional nonlinear
            conductivity parameters, and optional integration settings.
        fe_mesh (Mesh):
            Finite element mesh associated with the loss.
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T"],
                               "element_type":"quad"},fe_mesh)

class ThermalLoss2DTri(ThermalLoss):
    """
    Thermal loss for 2D triangular finite elements.

    This class configures :class:`ThermalLoss` for two-dimensional problems
    discretized with triangular elements. The temperature field has a single
    DOF per node (``T``), and the thermal loss is assembled using the weighted
    residual formulation defined in the base class.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing thermal settings and optional nonlinear
            conductivity parameters (e.g., ``"beta"`` and ``"c"``).
        fe_mesh (Mesh):
            Finite element mesh associated with the loss.
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T"],
                               "element_type":"triangle"},fe_mesh)