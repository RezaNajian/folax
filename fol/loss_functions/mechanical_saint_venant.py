"""
 Authors: Kianoosh Taghikhani, https://github.com/kianoosh1989
 Date: July, 2024
 License: FOL/LICENSE
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *
from fol.constitutive_material_models.saint_venant import SaintVenant

class SaintVenantMechanicalLoss(FiniteElementLoss):
    """
    Saint Venant–Kirchhoff mechanical energy loss for finite deformation elasticity.

    This class defines an energy-based loss functional for mechanical problems
    governed by a Saint Venant–Kirchhoff constitutive model. The total loss
    value represents the total strain energy of the structure and is assembled
    by summing element-level energy contributions over all finite elements in
    the mesh.

    For each element, the strain energy density is evaluated at Gauss points
    and integrated to obtain an element energy contribution. The global energy
    is obtained by accumulating these element energies across the computational
    domain.

    In addition to the scalar energy contribution, this loss provides the
    element residual vector (internal minus external forces) and the element
    tangent matrix (material stiffness plus geometric stiffness) required for
    Newton-based solution procedures.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Configuration dictionary. Must include ``material_dict`` with keys
            ``"young_modulus"`` and ``"poisson_ratio"``. Optional entries
            include ``"body_foce"`` and parametric heterogeneity settings via
            ``"heterogeneity_field_name"`` and ``"heterogeneity_default_value"``.
            Element discretization settings (dimension, element type, ordered
            DOFs) are typically provided by specialized subclasses.
        fe_mesh (Mesh):
            Finite element mesh over which the energy functional is defined.

    Attributes:
        e (float):
            Young's modulus used by the constitutive model.
        v (float):
            Poisson's ratio used by the constitutive model.
        material_model (SaintVenant):
            Constitutive material model used to compute energy density, stress,
            and consistent tangent.
        body_force (jax.numpy.ndarray):
            Body force vector with shape ``(2, 1)`` in 2D or ``(3, 1)`` in 3D.
    """
    default_material_settings = {"young_modulus":1.0,
                                 "poisson_ratio":"0.3",
                                 "heterogeneity_field_name":"K",
                                 "heterogeneity_default_value":1.0}

    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        loss_settings["material_dict"] = UpdateDefaultDict(self.default_material_settings,loss_settings["material_dict"])
        super().__init__(name,loss_settings,fe_mesh)

    def Initialize(self) -> None:
        super().Initialize()
        if "material_dict" not in self.loss_settings.keys():
            fol_error("material_dict should provided in the loss settings !")
        self.e = self.loss_settings["material_dict"]["young_modulus"]
        self.v = self.loss_settings["material_dict"]["poisson_ratio"]
        self.material_model = SaintVenant()

        if self.dim == 2:
            self.CalculateNMatrix = self.CalculateNMatrix2D
            self.CalculateKinematics = self.CalculateKinematics2D
            if self.element_type == "quad":
                self.CalculateGeometricStiffness = self.CalculateQuadGeometricStiffness2D
            elif self.element_type == "triangle":
                self.CalculateGeometricStiffness = self.CalculateTriangleGeometricStiffness2D
            self.body_force = jnp.zeros((2,1))
            if "body_foce" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_foce"])

        if self.dim == 3:
            self.CalculateNMatrix = self.CalculateNMatrix3D
            self.CalculateKinematics = self.CalculateKinematics3D
            if self.element_type == "tetra":
                self.CalculateGeometricStiffness = self.CalculateTetraGeometricStiffness3D
            elif self.element_type == "hexahedron":
                 self.CalculateGeometricStiffness = self.CalculateHexaGeometricStiffness3D
            self.body_force = jnp.zeros((3,1))
        if "body_foce" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_foce"])

        if self.loss_settings.get("parametric_boundary_learning"):
            heterogeneity_field = self.loss_settings["material_dict"]["heterogeneity_field_name"]
            heterogeneity_default_value = self.loss_settings["material_dict"]["heterogeneity_default_value"]
            if not self.fe_mesh.HasPointData(heterogeneity_field):
                self.fe_mesh[heterogeneity_field] = heterogeneity_default_value * np.ones(self.fe_mesh.GetNumberOfNodes())

            self.heterogeneity_field = jnp.asarray(self.fe_mesh[heterogeneity_field])
            self.get_param_function = lambda x: self.heterogeneity_field

    @partial(jit, static_argnums=(0,))
    def CalculateKinematics2D(self,DN_DX_T:jnp.array,uve:jnp.array) -> jnp.array:
        num_nodes = DN_DX_T.shape[1]
        uveT = jnp.array([uve[::2].squeeze(),uve[1::2].squeeze()]).T
        H = jnp.dot(DN_DX_T,uveT).T
        F = H + jnp.eye(H.shape[0])
        indices = np.arange(num_nodes)
        B = jnp.zeros((3, 2*num_nodes))
        B = B.at[0, 2 * indices].set(F[0, 0] * DN_DX_T[0, indices])
        B = B.at[0, 2 * indices + 1].set(F[1, 0] * DN_DX_T[0, indices])
        B = B.at[1, 2 * indices].set(F[0, 1] * DN_DX_T[1, indices])
        B = B.at[1, 2 * indices + 1].set(F[1, 1] * DN_DX_T[1, indices])
        B = B.at[2, 2 * indices].set(F[0, 1] * DN_DX_T[0, indices] + F[0, 0] * DN_DX_T[1, indices])
        B = B.at[2, 2 * indices + 1].set(F[1, 1] * DN_DX_T[0, indices] + F[1, 0] * DN_DX_T[1, indices])
        return H,F,B

    @partial(jit, static_argnums=(0,))
    def CalculateKinematics3D(self,DN_DX_T:jnp.array,uvwe:jnp.array) -> jnp.array:
        num_nodes = DN_DX_T.shape[1]
        uvweT = jnp.array([uvwe[::3].squeeze(),uvwe[1::3].squeeze(),uvwe[2::3].squeeze()]).T
        H = jnp.dot(DN_DX_T,uvweT).T
        F = H + jnp.eye(H.shape[0])
        indices = jnp.arange(num_nodes)
        B = jnp.zeros((6, 3 * num_nodes))

        B = B.at[0, 3 * indices].set(F[0, 0] * DN_DX_T[0, indices])
        B = B.at[0, 3 * indices + 1].set(F[1, 0] * DN_DX_T[0, indices])
        B = B.at[0, 3 * indices + 2].set(F[2, 0] * DN_DX_T[0, indices])
        B = B.at[1, 3 * indices].set(F[0, 1] * DN_DX_T[1, indices])
        B = B.at[1, 3 * indices + 1].set(F[1, 1] * DN_DX_T[1, indices])
        B = B.at[1, 3 * indices + 2].set(F[2, 1] * DN_DX_T[1, indices])
        B = B.at[2, 3 * indices].set(F[0, 2] * DN_DX_T[2, indices])
        B = B.at[2, 3 * indices + 1].set(F[1, 2] * DN_DX_T[2, indices])
        B = B.at[2, 3 * indices + 2].set(F[2, 2] * DN_DX_T[2, indices])
        B = B.at[3, 3 * indices].set(F[0, 1] * DN_DX_T[2, indices] + F[0, 2] * DN_DX_T[1, indices])
        B = B.at[3, 3 * indices + 1].set(F[1, 1] * DN_DX_T[2, indices] + F[1, 2] * DN_DX_T[1, indices])
        B = B.at[3, 3 * indices + 2].set(F[2, 1] * DN_DX_T[2, indices] + F[2, 2] * DN_DX_T[1, indices])
        B = B.at[4, 3 * indices].set(F[0, 0] * DN_DX_T[2, indices] + F[0, 2] * DN_DX_T[0, indices])
        B = B.at[4, 3 * indices + 1].set(F[1, 0] * DN_DX_T[2, indices] + F[1, 2] * DN_DX_T[0, indices])
        B = B.at[4, 3 * indices + 2].set(F[2, 0] * DN_DX_T[2, indices] + F[2, 2] * DN_DX_T[0, indices])
        B = B.at[5, 3 * indices].set(F[0, 0] * DN_DX_T[1, indices] + F[0, 1] * DN_DX_T[0, indices])
        B = B.at[5, 3 * indices + 1].set(F[1, 0] * DN_DX_T[1, indices] + F[1, 1] * DN_DX_T[0, indices])
        B = B.at[5, 3 * indices + 2].set(F[2, 0] * DN_DX_T[1, indices] + F[2, 1] * DN_DX_T[0, indices])

        return H,F,B

    @partial(jit, static_argnums=(0,))
    def CalculateNMatrix2D(self,N_vec:jnp.array) -> jnp.array:
        N_mat = jnp.zeros((2, 2 * N_vec.size))
        indices = jnp.arange(N_vec.size)
        N_mat = N_mat.at[0, 2 * indices].set(N_vec)
        N_mat = N_mat.at[1, 2 * indices + 1].set(N_vec)
        return N_mat

    @partial(jit, static_argnums=(0,))
    def CalculateNMatrix3D(self,N_vec:jnp.array) -> jnp.array:
        N_mat = jnp.zeros((3,3*N_vec.size))
        N_mat = N_mat.at[0,0::3].set(N_vec)
        N_mat = N_mat.at[1,1::3].set(N_vec)
        N_mat = N_mat.at[2,2::3].set(N_vec)
        return N_mat

    @partial(jit, static_argnums=(0,))
    def CalculateQuadGeometricStiffness2D(self,DN_DX_T:jnp.array,S:jnp.array) -> jnp.array:
        """
        Compute the geometric stiffness matrix for a quadratic element.
        Args:
            DN_DX_T: (2, num_nodes), shape function derivatives w.r.t spatial coordinates at Gauss point
            S: (3,1), stress vector in Voigt notation at Gauss point
        Returns:
            gp_geo_stiffness: (2*num_nodes, 2*num_nodes), geometric stiffness matrix
        """
        S_mat = jnp.zeros((2,2))
        S_mat = S_mat.at[0,0].set(S[0,0])
        S_mat = S_mat.at[0,1].set(S[2,0])
        S_mat = S_mat.at[1,0].set(S[2,0])
        S_mat = S_mat.at[1,1].set(S[1,0])

        num_nodes = DN_DX_T.shape[1]
        gp_geo_stiffness = jnp.zeros((2*num_nodes,2*num_nodes))

        def geo_stiffness_entry(i, j, DN_DX_T, S_mat):
            val = DN_DX_T[:, i].T @ (S_mat @ DN_DX_T[:, j])
            return jnp.eye(2) * val  # Returns a (2, 2) block

        # Vectorize over i and j
        vmap_j = jax.vmap(lambda j: jax.vmap(lambda i: geo_stiffness_entry(i, j, DN_DX_T, S_mat))(jnp.arange(num_nodes)), in_axes=0)
        blocks = vmap_j(jnp.arange(num_nodes))  # Shape: (4, 4, 2, 2)

        # Rearrange blocks into full (8, 8) matrix
        # Vectorized reshape instead of jnp.block
        gp_geo_stiffness = blocks.transpose(0, 2, 1, 3).reshape(2*num_nodes, 2*num_nodes)

        return gp_geo_stiffness

    @partial(jit, static_argnums=(0,))
    def CalculateTriangleGeometricStiffness2D(self,DN_DX_T:jnp.array,S:jnp.array) -> jnp.array:
        """
        Compute the geometric stiffness matrix for a triangle element.
        Args:
            DN_DX_T: (2, num_nodes), shape function derivatives w.r.t spatial coordinates at Gauss point
            S: (3,1), stress vector in Voigt notation at Gauss point
        Returns:
            gp_geo_stiffness: (2*num_nodes, 2*num_nodes), geometric stiffness matrix
        """
        S_mat = jnp.zeros((2,2))
        S_mat = S_mat.at[0,0].set(S[0,0])
        S_mat = S_mat.at[0,1].set(S[2,0])
        S_mat = S_mat.at[1,0].set(S[2,0])
        S_mat = S_mat.at[1,1].set(S[1,0])

        num_nodes = DN_DX_T.shape[1]
        gp_geo_stiffness = jnp.zeros((2*num_nodes,2*num_nodes))

        def geo_stiffness_entry(i, j, DN_DX_T, S_mat):
            val = DN_DX_T[:, i].T @ (S_mat @ DN_DX_T[:, j])
            return jnp.eye(2) * val  # Returns a (2, 2) block

        # Vectorize over i and j
        vmap_j = jax.vmap(lambda j: jax.vmap(lambda i: geo_stiffness_entry(i, j, DN_DX_T, S_mat))(jnp.arange(num_nodes)), in_axes=0)
        blocks = vmap_j(jnp.arange(num_nodes))  # Shape: (3, 3, 2, 2)

        # Rearrange blocks into full (8, 8) matrix
        # Vectorized reshape instead of jnp.block
        gp_geo_stiffness = blocks.transpose(0, 2, 1, 3).reshape(2*num_nodes, 2*num_nodes)

        return gp_geo_stiffness

    @partial(jit, static_argnums=(0,))
    def CalculateTetraGeometricStiffness3D(self,DN_DX_T:jnp.array,S:jnp.array) -> jnp.array:
        """
        Compute the geometric stiffness matrix for a tetra element.
        Args:
            DN_DX_T: (3, num_nodes), shape function derivatives w.r.t spatial coordinates at Gauss point
            S: (6,1), stress vector in Voigt notation at Gauss point
        Returns:
            gp_geo_stiffness: (3*num_nodes, 3*num_nodes), geometric stiffness matrix
        """
        S_mat = jnp.zeros((3,3))
        S_mat = S_mat.at[0,0].set(S[0,0])
        S_mat = S_mat.at[0,1].set(S[5,0])
        S_mat = S_mat.at[0,2].set(S[4,0])
        S_mat = S_mat.at[1,0].set(S[5,0])
        S_mat = S_mat.at[1,1].set(S[1,0])
        S_mat = S_mat.at[1,2].set(S[3,0])
        S_mat = S_mat.at[2,0].set(S[4,0])
        S_mat = S_mat.at[2,1].set(S[3,0])
        S_mat = S_mat.at[2,2].set(S[2,0])
        num_nodes = DN_DX_T.shape[1]
        gp_geo_stiffness = jnp.zeros((3*num_nodes,3*num_nodes))

        def geo_stiffness_entry(i, j, DN_DX_T, S_mat):
            val = DN_DX_T[:, i].T @ (S_mat @ DN_DX_T[:, j])
            return jnp.eye(3) * val  # Returns a (3, 3) block

        # Vectorize over i and j
        vmap_j = jax.vmap(lambda j: jax.vmap(lambda i: geo_stiffness_entry(i, j, DN_DX_T, S_mat))(jnp.arange(num_nodes)), in_axes=0)
        blocks = vmap_j(jnp.arange(num_nodes))  # Shape: (4, 4, 3, 3)

        # Rearrange blocks into full (12, 12) matrix
        # Vectorized reshape instead of jnp.block
        gp_geo_stiffness = blocks.transpose(0, 2, 1, 3).reshape(3*num_nodes, 3*num_nodes)

        return gp_geo_stiffness

    @partial(jit, static_argnums=(0,))
    def CalculateHexaGeometricStiffness3D(self,DN_DX_T:jnp.array,S:jnp.array) -> jnp.array:
        """
        Compute the geometric stiffness matrix for a hexahedral element.
        Args:
            DN_DX_T: (3, num_nodes), shape function derivatives w.r.t spatial coordinates at Gauss point
            S: (6,1), stress vector in Voigt notation at Gauss point
        Returns:
            gp_geo_stiffness: (3*num_nodes, 3*num_nodes), geometric stiffness matrix
        """
        S_mat = jnp.zeros((3,3))
        S_mat = S_mat.at[0,0].set(S[0,0])
        S_mat = S_mat.at[0,1].set(S[5,0])
        S_mat = S_mat.at[0,2].set(S[4,0])
        S_mat = S_mat.at[1,0].set(S[5,0])
        S_mat = S_mat.at[1,1].set(S[1,0])
        S_mat = S_mat.at[1,2].set(S[3,0])
        S_mat = S_mat.at[2,0].set(S[4,0])
        S_mat = S_mat.at[2,1].set(S[3,0])
        S_mat = S_mat.at[2,2].set(S[2,0])
        num_nodes = DN_DX_T.shape[1]
        gp_geo_stiffness = jnp.zeros((3*num_nodes,3*num_nodes))

        def geo_stiffness_entry(i, j, DN_DX_T, S_mat):
            val = DN_DX_T[:, i].T @ (S_mat @ DN_DX_T[:, j])
            return jnp.eye(3) * val  # Returns a (3, 3) block

        # Vectorize over i and j
        vmap_j = jax.vmap(lambda j: jax.vmap(lambda i: geo_stiffness_entry(i, j, DN_DX_T, S_mat))(jnp.arange(num_nodes)), in_axes=0)
        blocks = vmap_j(jnp.arange(num_nodes))  # Shape: (8, 8, 3, 3)

        # Rearrange blocks into full (24, 24) matrix
        # Vectorized reshape instead of jnp.block
        gp_geo_stiffness = blocks.transpose(0, 2, 1, 3).reshape(3*num_nodes, 3*num_nodes)

        return gp_geo_stiffness

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,uvwe):
        """
        Compute element-level energy, residual, and tangent contributions.

        This method evaluates the element strain energy by Gaussian quadrature
        using the Saint Venant–Kirchhoff constitutive model. The returned scalar
        value represents the strain energy contribution of this element to the
        total energy of the system. The total loss value is obtained by summing
        these element energies over all elements in the mesh.

        The method also computes the element residual vector as the difference
        between internal and external nodal force vectors and returns the
        element tangent matrix including both material and geometric stiffness
        contributions.

        Args:
            xyze:
                Element nodal coordinates.
            de:
                Element parameter field values at nodes used to interpolate
                material parameters to Gauss points.
            uvwe:
                Element displacement DOF vector arranged consistently with the
                element type and ordered degrees of freedom.

        Returns:
            Tuple[jax.numpy.ndarray, jax.numpy.ndarray, jax.numpy.ndarray]:
                - Scalar element strain energy contribution.
                - Element residual vector defined as ``Fint - Fe``.
                - Element tangent matrix including material and geometric
                  stiffness contributions.
        """
        @jit
        def compute_at_gauss_point(gp_point,gp_weight):

            N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
            N_mat = self.CalculateNMatrix(N_vec)
            DN_DX_T = self.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point).T
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)

            e_at_gauss = jnp.dot(N_vec, de.squeeze())
            k_at_gauss = e_at_gauss / (3 * (1 - 2*self.v))
            mu_at_gauss = e_at_gauss / (2 * (1 + self.v))
            lambda_ = e_at_gauss * self.v / ((1 + self.v) * (1 - 2*self.v))

            H,F,B = self.CalculateKinematics(DN_DX_T,uvwe)

            xsi,S,C = self.material_model.evaluate(F,lambda_,mu_at_gauss)
            gp_geo_stiffness = self.CalculateGeometricStiffness(DN_DX_T,S)


            gp_stiffness = gp_weight * detJ * (B.T @ C @ B)
            gp_geo_stiffness = gp_weight * detJ * gp_geo_stiffness  # will be added to gp_stiffness
            gp_f = gp_weight * detJ * N_mat.T @ self.body_force
            gp_fint = gp_weight * detJ * jnp.dot(B.T,S)
            gp_energy = gp_weight * detJ * xsi
            return gp_energy,gp_stiffness + gp_geo_stiffness,gp_f,gp_fint

        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        E_gps,k_gps,f_gps,fint_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        Se = jnp.sum(k_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        Fint = jnp.sum(fint_gps, axis=0)
        Ee = jnp.sum(E_gps, axis=0)
        return  Ee, Fint - Fe, Se

class SaintVenantMechanicalLoss2DQuad(SaintVenantMechanicalLoss):
    """
    Saint Venant–Kirchhoff mechanical energy loss for 2D quadrilateral elements.

    This class configures :class:`SaintVenantMechanicalLoss` for two-dimensional
    problems discretized with quadrilateral elements. The displacement field
    consists of two components (``Ux``, ``Uy``) per node, and the total energy
    is assembled by summing element energy contributions defined in the base
    class.

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
                               "ordered_dofs": ["Ux","Uy"],
                               "element_type":"quad"},fe_mesh)

class SaintVenantMechanicalLoss2DTri(SaintVenantMechanicalLoss):
    """
    Saint Venant–Kirchhoff mechanical energy loss for 2D triangular elements.

    This class configures :class:`SaintVenantMechanicalLoss` for two-dimensional
    problems discretized with triangular elements. The displacement field
    consists of two components (``Ux``, ``Uy``) per node, and the total energy
    is assembled by summing element energy contributions defined in the base
    class.

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
                               "ordered_dofs": ["Ux","Uy"],
                               "element_type":"triangle"},fe_mesh)

class SaintVenantMechanicalLoss3DTetra(SaintVenantMechanicalLoss):
    """
    Saint Venant–Kirchhoff mechanical energy loss for 3D tetrahedral elements.

    This class configures :class:`SaintVenantMechanicalLoss` for three-dimensional
    problems discretized with tetrahedral elements. The displacement field
    consists of three components (``Ux``, ``Uy``, ``Uz``) per node, and the total
    energy is assembled by summing element energy contributions defined in the
    base class.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Dictionary containing ``material_dict`` and optional settings.
        fe_mesh (Mesh):
            Finite element mesh associated with the loss.
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["Ux","Uy","Uz"],
                               "element_type":"tetra"},fe_mesh)

class SaintVenantMechanicalLoss3DHexa(SaintVenantMechanicalLoss):
    """
    Saint Venant–Kirchhoff mechanical energy loss for 3D hexahedral elements.

    This class configures :class:`SaintVenantMechanicalLoss` for three-dimensional
    problems discretized with hexahedral elements. The displacement field
    consists of three components (``Ux``, ``Uy``, ``Uz``) per node, and the total
    energy is assembled by summing element energy contributions defined in the
    base class.

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
                               "ordered_dofs": ["Ux","Uy","Uz"],
                               "element_type":"hexahedron"},fe_mesh)