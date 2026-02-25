"""
Authors: Yusuke Yamazaki; Reza Najian Asl (https://github.com/RezaNajian)
Date: April, 2024
License: FOL/LICENSE
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from functools import partial
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh

class AllenCahnLoss(FiniteElementLoss):
    """
    Allen–Cahn phase-field energy functional with implicit time discretization.

    This class defines an energy-based loss functional for Allen–Cahn type
    phase-field evolution problems. The total loss value represents the total
    discrete free energy of the system and is assembled by summing
    element-level energy contributions over all finite elements in the mesh.

    For each element, a scalar energy contribution is computed using Gaussian
    quadrature based on the current and previous phase-field values. The total
    energy of the system is obtained by accumulating these element energies
    across the computational domain.

    In addition to the scalar energy contribution, the class provides the
    element residual vector and Jacobian matrix corresponding to the discrete
    nonlinear system derived from the energy functional. These quantities are
    used for implicit time integration and Newton-based solution procedures.

    Material and time-integration parameters are provided via
    ``loss_settings["material_dict"]``. The interface width is controlled by
    ``epsilon``, and the time step size is given by ``dt``. The formulation
    supports both two-dimensional and three-dimensional problems.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing problem settings. Must include
            ``material_dict`` with keys ``"rho"``, ``"cp"``, ``"dt"``, and
            ``"epsilon"``. Element discretization settings (dimension, element
            type, ordered DOFs) are typically provided by the specialized
            subclasses.
        fe_mesh (Mesh):
            Finite element mesh over which the energy functional is defined.

    Attributes:
        rho (float):
            Density parameter from ``material_dict``.
        cp (float):
            Heat-capacity-like parameter from ``material_dict``.
        dt (float):
            Time step size used for implicit time discretization.
        epsilon (float):
            Interface-width parameter controlling the phase-field thickness.
        body_force (jax.numpy.ndarray):
            Body force vector with shape ``(2, 1)`` in 2D or ``(3, 1)`` in 3D.
    """
    def Initialize(self) -> None:
        super().Initialize()
        if "material_dict" not in self.loss_settings.keys():
            fol_error("material_dict should provided in the loss settings !")
        if self.dim == 2:
            self.rho = self.loss_settings["material_dict"]["rho"]
            self.cp =  self.loss_settings["material_dict"]["cp"]
            self.dt =  self.loss_settings["material_dict"]["dt"]
            self.epsilon =  self.loss_settings["material_dict"]["epsilon"]
            self.body_force = jnp.zeros((2,1))
            if "body_foce" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_foce"])
        else:
            self.rho = self.loss_settings["material_dict"]["rho"]
            self.cp =  self.loss_settings["material_dict"]["cp"]
            self.dt =  self.loss_settings["material_dict"]["dt"]
            self.epsilon =  self.loss_settings["material_dict"]["epsilon"]
            self.body_force = jnp.zeros((3,1))
            if "body_foce" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_foce"])

    def ComputeElement(self,xyze,phi_e_c,phi_e_n,body_force=0):
        """
        Compute element-level energy, residual, and Jacobian contributions.

        This method evaluates the discrete Allen–Cahn free energy contribution
        of a single finite element using Gaussian quadrature. The returned
        scalar value represents the energy contribution of this element to the
        total system energy. The total energy of the problem is obtained by
        summing these element energy contributions over all elements in the
        mesh.

        In addition to the scalar energy, the method computes the element
        residual vector and the element Jacobian matrix corresponding to the
        discrete nonlinear system derived from the energy functional. These
        quantities are used in implicit time integration and Newton-based
        solution procedures.

        The formulation depends on the phase-field values at the current time
        step and the previous time step. The time-discrete contribution
        enforces temporal consistency through the time step size ``dt``.

        Args:
            xyze:
                Element nodal coordinates.
            phi_e_c:
                Element phase-field degrees of freedom at the previous time
                step (committed state). Expected shape is compatible with the
                element node count.
            phi_e_n:
                Element phase-field degrees of freedom at the current time
                step (unknown state). Expected shape is compatible with the
                element node count.
            body_force:
                Optional element source term. This argument is included for
                extensibility and defaults to 0.

        Returns:
            Tuple[jax.numpy.ndarray, jax.numpy.ndarray, jax.numpy.ndarray]:
                - Scalar element free energy contribution.
                - Element residual vector for the discrete Allen–Cahn system.
                - Element Jacobian matrix associated with the residual.

        Raises:
            ValueError:
                If invalid numerical values are encountered during the
                evaluation of the element energy or residual.
        """
        phi_e_c = phi_e_c.reshape(-1,1)
        phi_e_n = phi_e_n.reshape(-1,1)

        def compute_at_gauss_point(gp_point,gp_weight):
            N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
            DN_DX = self.fe_element.ShapeFunctionsLocalGradients(gp_point)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            B_mat = jnp.dot(invJ,DN_DX.T)
            phi_at_gauss_n = jnp.dot(N_vec.reshape(1,-1), phi_e_n)
            phi_at_gauss_c = jnp.dot(N_vec.reshape(1,-1), phi_e_c)
            source_term = 0.25*(phi_at_gauss_n*phi_at_gauss_n - 1)**2
            Dsource_term = (phi_at_gauss_n*phi_at_gauss_n - 1)*phi_at_gauss_n
            gp_stiffness =  B_mat.T@B_mat * detJ * gp_weight
            gp_mass = jnp.outer(N_vec, N_vec) * detJ * gp_weight
            gp_f_res = N_vec.reshape(-1,1)*Dsource_term * detJ * gp_weight
            gp_f = source_term * detJ * gp_weight
            gp_t = 0.5/(self.dt)*gp_weight  * detJ *(phi_at_gauss_n-phi_at_gauss_c)**2
            gp_Df = jnp.outer(N_vec, N_vec) * (3 * phi_at_gauss_n**2 - 1) *  detJ * gp_weight
            return gp_stiffness,gp_mass, gp_f, gp_f_res, gp_t, gp_Df

        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        k_gps,m_gps,f_gps,f_res_gps,t_gps, df_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        Se = jnp.sum(k_gps, axis=0)
        Me = jnp.sum(m_gps, axis=0)
        Fe = jnp.sum(f_gps)
        Fe_res = jnp.sum(f_res_gps,axis=0)
        Te = jnp.sum(t_gps)
        dFe = jnp.sum(df_gps, axis=0)

        return 0.5*phi_e_n.T@Se@phi_e_n + 1/(self.epsilon**2)*Fe + Te, ((Me+self.dt*Se)@phi_e_n - (Me@phi_e_c- 1/(self.epsilon**2)*self.dt*Fe_res)), (Me+self.dt*Se - self.dt/(self.epsilon**2)*dFe)

class AllenCahnLoss2DQuad(AllenCahnLoss):
    """
    Allen–Cahn loss for 2D quadrilateral finite elements.

    This class configures :class:`AllenCahnLoss` for two-dimensional problems
    discretized with quadrilateral elements. The phase-field has a single DOF
    per node (``Phi``), and element contributions are assembled using the base
    formulation.

    If the number of Gauss points is not specified in the loss settings, a
    default value of ``num_gp = 2`` is used.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing ``material_dict`` and optional settings such
            as integration parameters.
        fe_mesh (Mesh):
            Finite element mesh associated with the loss.
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["Phi"],
                               "element_type":"quad"},fe_mesh)

class AllenCahnLoss2DTri(AllenCahnLoss):
    """
    Allen–Cahn loss for 2D triangular finite elements.

    This class configures :class:`AllenCahnLoss` for two-dimensional problems
    discretized with triangular elements. The phase-field has a single DOF per
    node (``Phi``), and element contributions are assembled using the base
    formulation.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing ``material_dict`` and optional settings.
        fe_mesh (Mesh):
            Finite element mesh associated with the loss.
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["Phi"],
                               "element_type":"triangle"},fe_mesh)
class AllenCahnLoss3DHexa(AllenCahnLoss):
    """
    Allen–Cahn loss for 3D hexahedral finite elements.

    This class configures :class:`AllenCahnLoss` for three-dimensional problems
    discretized with hexahedral elements. The phase-field has a single DOF per
    node (``Phi``), and element contributions are assembled using the base
    formulation.

    If the number of Gauss points is not specified in the loss settings, a
    default value of ``num_gp = 2`` is used.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing ``material_dict`` and optional settings such
            as integration parameters.
        fe_mesh (Mesh):
            Finite element mesh associated with the loss.
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["Phi"],
                               "element_type":"hexahedron"},fe_mesh)

