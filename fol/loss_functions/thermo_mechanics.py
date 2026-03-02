"""
Authors: Yusuke Yamazaki; Reza Najian Asl (https://github.com/RezaNajian)
Date: Feb, 2026
License: FOL/LICENSE
"""
from  .fe_loss import FiniteElementLoss,ELEMENT_TYPE_NUM_NODES
from .fe_kinematics import b_matrix_2d, b_matrix_3d, d_matrix_2d, d_matrix_3d, n_matrix_2d, n_matrix_3d
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *
from jax.experimental.sparse import BCOO
from jax.scipy.sparse.linalg import cg

class ThermoMechanicsLoss(FiniteElementLoss):
    """
    Coupled thermo-mechanical finite-element loss for conductivity (thermal) and
    linear-elasticity (mechanical) with temperature-dependent material response.

    This class computes, at the element level, the total loss, residual vector, and
    Jacobian matrix for a coupled system where temperature ``T`` and displacement
    components (``Ux, Uy`` in 2D; ``Ux, Uy, Uz`` in 3D) are solved together. The
    implementation supports 2D and 3D formulations and relies on the mesh and
    element integration rules provided by the underlying finite-element framework.

    Args:
        name (str): Identifier for the loss instance, typically used for logging,
            diagnostics, or registration within a solver or optimization pipeline.
        loss_settings (dict): Dictionary containing material, thermal, numerical,
            Dirichlet boundary-condition, and element-specific settings. Expected entries
            include sub-dictionaries such as ``material_dict``, ``thermal_dict``, and
            ``dirichlet_bc_dict``, as well as finite-element configuration parameters.
        fe_mesh (Mesh): Finite-element mesh object defining nodal coordinates,
            element connectivity, integration rules, and shape function data.

    Returns:
        None

    Attributes:
        thermal_loss_settings (dict): Runtime thermal configuration 
                                      including material response parameters and initial temperature.
        elem_t_local_ids (jnp.ndarray): Local element DOF indices corresponding to temperature.
        elem_uvw_local_ids (jnp.ndarray): Local element DOF indices corresponding to displacement.
        CalculateNMatrix (callable): Dimension-specific N-matrix constructor.
        CalculateBMatrix (callable): Dimension-specific B-matrix constructor.
        D (jnp.ndarray): Constitutive matrix for linear elasticity in 2D or 3D.
        body_force (jnp.ndarray): Body force vector in global coordinates.
        thermal_st_vec (jnp.ndarray): Thermal strain selector vector for the active dimension. 
                                      Under the assumption of an isotropic material, 
                                      this should only has influence on the volumetric part of the thermal strain 
                                      and is set to 1 for normal strains and 0 for shear strains.
    """
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:  
        """
        Initialize thermo-mechanical operators, constitutive data, and DOF mappings.

        This method prepares dimension-dependent FE matrices (N, B, D), thermal settings,
        and local index mappings for temperature and displacement degrees of freedom.
        Initialization is idempotent unless ``reinitialize=True`` is provided.

        Args:
            reinitialize (bool): If True, forces re-initialization even if the instance
                was previously initialized.

        Returns:
            None
        """
        if self.initialized and not reinitialize:
            return
        super().Initialize() 

        self.thermal_loss_settings = {"k1":0.5,"k2":2.0,"k3":20.0,"k4":0.5,
                                      "T0":jnp.zeros((self.fe_mesh.GetNumberOfNodes()))}
        self.mechanical_loss_settings = {"e1":1.0,"e2":-0.6}

        if "k1" in self.loss_settings["thermal_dict"].keys():
            self.thermal_loss_settings["k1"] = self.loss_settings["thermal_dict"]["k1"]
        if "k2" in self.loss_settings["thermal_dict"].keys():
            self.thermal_loss_settings["k2"] = self.loss_settings["thermal_dict"]["k2"]
        if "k3" in self.loss_settings["thermal_dict"].keys():
            self.thermal_loss_settings["k3"] = self.loss_settings["thermal_dict"]["k3"]
        if "k4" in self.loss_settings["thermal_dict"].keys():
            self.thermal_loss_settings["k4"] = self.loss_settings["thermal_dict"]["k4"]
        if "e1" in self.loss_settings["mechanical_dict"].keys():
            self.mechanical_loss_settings["e1"] = self.loss_settings["mechanical_dict"]["e1"]
        if "e2" in self.loss_settings["mechanical_dict"].keys():
            self.mechanical_loss_settings["e2"] = self.loss_settings["mechanical_dict"]["e2"]
        if "T0" in self.loss_settings["material_dict"].keys():
            self.thermal_loss_settings["T0"] = self.loss_settings["material_dict"]["T0"]
            
        if self.dim == 2:
            self.CalculateNMatrix = n_matrix_2d
            self.CalculateBMatrix = b_matrix_2d
            self.D = d_matrix_2d(self.loss_settings["material_dict"]["young_modulus"],
                                 self.loss_settings["material_dict"]["poisson_ratio"])
            self.body_force = jnp.zeros((2,1))
            self.thermal_st_vec = jnp.array([[1.0], [1.0], [0.0]]) 
            if "body_force" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_force"])
        else:
            self.CalculateNMatrix = n_matrix_3d
            self.CalculateBMatrix = b_matrix_3d
            self.D = d_matrix_3d(self.loss_settings["material_dict"]["young_modulus"],
                                 self.loss_settings["material_dict"]["poisson_ratio"])
            self.body_force = jnp.zeros((3,1))
            self.thermal_st_vec = jnp.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
            if "body_force" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_force"])

        num_elem_nodes = ELEMENT_TYPE_NUM_NODES[self.element_type]
        elem_dof_local_ids = jnp.arange(num_elem_nodes*self.number_dofs_per_node)
        self.elem_t_local_ids = elem_dof_local_ids[::self.number_dofs_per_node]
        self.elem_uvw_local_ids = elem_dof_local_ids[(jnp.arange(elem_dof_local_ids.shape[0]) % self.number_dofs_per_node) != 0]

    def ComputeElementThermal(self,xyze,de,te,body_force=0):
        """
        Compute thermal contribution to element loss, residual, and tangent matrix.

        The thermal loss is based on the temperature gradient energy integrated over
        Gauss points. A temperature-dependent conductivity is evaluated at each Gauss
        point from nodal controls ``de`` and nodal temperatures ``te``.

        Args:
            xyze: Element nodal coordinates array shaped like ``(nnode, dim)``.
            de: Element nodal control values for thermal conductivity/stiffness, typically
                shaped like ``(nnode,)`` or compatible and reshaped internally.
            te: Element nodal temperatures, shaped like ``(nnode,)`` or ``(nnode, 1)``.
            body_force: Optional scalar source term used to build an element thermal load
                vector. Defaults to 0.

        Returns:
            tuple:
                A 3-tuple ``(loss, residual, tangent)`` where
                ``loss`` is the scalar thermal loss contribution for the element,
                ``residual`` is the element thermal residual vector (shape ``(nnode, 1)``),
                and ``tangent`` is the thermal tangent matrix (shape ``(nnode, nnode)``).
        """
        de = jax.lax.stop_gradient(de.reshape(-1,1))
        te = te.reshape(-1,1)
        def compute_at_gauss_point(gp_point,gp_weight):
            N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
            temp_at_gauss = jnp.dot(N_vec,te.squeeze())
            conductivity_at_gauss = jnp.dot(N_vec.reshape(1,-1), de) \
            * (self.thermal_loss_settings["k1"]+1/self.thermal_loss_settings["k2"]*\
               (1/(1+jnp.exp(self.thermal_loss_settings["k3"]*(temp_at_gauss-self.thermal_loss_settings["k4"])))))
            DN_DX = self.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            grad_T = DN_DX.T @ te
            gp_loss_t = 0.5*(grad_T.T@grad_T) * conductivity_at_gauss * detJ *gp_weight
            dk_dT = jnp.dot(N_vec.reshape(1,-1), de)*\
                (-self.thermal_loss_settings["k3"]/self.thermal_loss_settings["k2"]\
                 *jnp.exp(self.thermal_loss_settings["k3"]*(temp_at_gauss-self.thermal_loss_settings["k4"])))\
                /((1+jnp.exp(self.thermal_loss_settings["k3"]*(temp_at_gauss-self.thermal_loss_settings["k4"])))**2)
            gp_stiffness = conductivity_at_gauss * (DN_DX @ DN_DX.T) * detJ * gp_weight 
            gp_stiffness_thermal = dk_dT* ((DN_DX@grad_T)@N_vec.reshape(1,-1)) * detJ * gp_weight 
            gp_f = gp_weight * detJ * body_force *  N_vec.reshape(-1,1) 
            return gp_loss_t, gp_stiffness, gp_stiffness_thermal, gp_f
        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        loss_t_gps, s_gps, st_gps, f_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        Loss_t_e = jnp.sum(loss_t_gps)
        Se = jnp.sum(s_gps, axis=0)
        Se_t = jnp.sum(st_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        def compute_elem_res(Se,te,Fe):
            te = jax.lax.stop_gradient(te)
            return (Se @ te - Fe)
        element_residuals = compute_elem_res(Se,te ,Fe)
        return  te.T@jax.lax.stop_gradient(element_residuals), element_residuals, Se + Se_t
    
    def ComputeElementMechanical(self,xyze,de,te,se,te_init,body_force=0):
        # # @Yusuke: Please clean the function from commented lines and hard coded stuff
        """
        Compute mechanical contribution to element loss, residual, and tangents.

        The mechanical loss is based on linear-elastic strain energy, corrected by a
        thermal strain term derived from the difference between current temperature
        and initial temperature. The elastic modulus is evaluated from nodal controls
        ``de`` and may depend on temperature.

        Args:
            xyze: Element nodal coordinates array shaped like ``(nnode, dim)``.
            de: Element nodal control values for mechanical stiffness (e.g., Young's
                modulus scale), shaped like ``(nnode,)`` or compatible.
            te: Element nodal temperatures used for thermal strain and temperature-dependent
                stiffness, shaped like ``(nnode,)`` or ``(nnode, 1)``.
            se: Element nodal displacement DOFs arranged as a vector shaped like
                ``(nnode * ndof_u, 1)``.
            te_init: Element nodal initial temperatures used as the reference state,
                shaped like ``(nnode,)`` or ``(nnode, 1)``.
            body_force: Optional body force vector applied in the mechanical equilibrium
                equation. Defaults to 0.

        Returns:
            tuple:
                A 4-tuple ``(loss, residual, dres_du, dres_dT)`` where
                ``loss`` is the scalar mechanical loss contribution for the element,
                ``residual`` is the mechanical residual vector (shape ``(nnode*ndof_u, 1)``),
                ``dres_du`` is the mechanical tangent w.r.t. displacement
                (shape ``(nnode*ndof_u, nnode*ndof_u)``),
                and ``dres_dT`` is the coupling tangent w.r.t. temperature
                (shape ``(nnode*ndof_u, nnode)`` when assembled with local IDs).
        """
        # Mechanics loss
        ke = de.reshape(-1,1)
        se = se.reshape(-1,1)
        te = jax.lax.stop_gradient(te.reshape(-1,1))
        te_init = jax.lax.stop_gradient(te_init.reshape(-1,1))
        def compute_at_gauss_point(gp_point,gp_weight):
            # Mechanical part
            N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
            N_mat = self.CalculateNMatrix(N_vec)
            DN_DX = self.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point)
            B_mat = self.CalculateBMatrix(DN_DX)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            temp_at_gauss = jnp.dot(N_vec,te.squeeze())
            total_strain_at_gauss = B_mat@se
            thermal_strain_vec = 1.0\
                * (temp_at_gauss - jnp.dot(N_vec, te_init.squeeze())) * self.thermal_st_vec
            elastic_strain = total_strain_at_gauss - thermal_strain_vec
            e_at_gauss = jnp.dot(N_vec, ke.squeeze())* \
                (self.mechanical_loss_settings["e1"]+self.mechanical_loss_settings["e2"]*temp_at_gauss)
            
            gp_loss_m = 0.5 * elastic_strain.T @ self.D @ elastic_strain *e_at_gauss * detJ * gp_weight
            gp_stiffness = gp_weight * detJ * e_at_gauss * (B_mat.T @ (self.D @ B_mat))
            gp_lhs = gp_weight * detJ * e_at_gauss * B_mat.T @ self.D @ elastic_strain
            gp_f = gp_weight * detJ * (N_mat.T @ self.body_force)
            de_dT = jnp.dot(N_vec, ke.squeeze())*(self.mechanical_loss_settings["e2"]) 
            gp_stiffness_thermal = gp_weight * detJ * B_mat.T @ self.D @\
                (elastic_strain*de_dT - e_at_gauss*(1.0) *self.thermal_st_vec).reshape(-1,1) @ N_vec.reshape(1,-1)
            return gp_loss_m, gp_stiffness, gp_f, gp_stiffness_thermal,gp_lhs
        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        loss_m_gps,s_gps, f_gps, st_gps, lhs_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        Loss_m_e = jnp.sum(loss_m_gps)
        Se = jnp.sum(s_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        Se_t = jnp.sum(st_gps, axis=0)
        LHSe = jnp.sum(lhs_gps,axis=0)
        def compute_elem_res(Se,se ,Fe, Fe_thermal):
            return (Se @ se - Fe - Fe_thermal)
        element_residuals = LHSe - Fe
        return  se.T@jax.lax.stop_gradient(LHSe - Fe), element_residuals, Se, Se_t
    
    def ComputeElement(self,xyze,de,tuvwe,t0e):
        """
        Compute coupled element loss, residual, and Jacobian for thermo-mechanics.

        This method combines thermal and mechanical element contributions into a single
        element loss ``l_e``, residual vector ``re`` and Jacobian matrix ``ke`` using
        the local DOF ordering encoded by ``elem_t_local_ids`` and ``elem_uvw_local_ids``.

        Args:
            xyze: Element nodal coordinates array shaped like ``(nnode, dim)``.
            de: Element nodal control values shared by thermal/mechanical models, shaped
                like ``(nnode,)`` or compatible.
            tuvwe: Element nodal DOF vector including temperature and displacement DOFs,
                shaped like ``(nnode * ndofs_per_node, 1)``.
            t0e: Element nodal initial temperature vector shaped like ``(nnode,)`` or
                ``(nnode, 1)``.

        Returns:
            tuple:
                A 3-tuple ``(loss, residual, jacobian)`` where
                ``loss`` is the scalar coupled element loss,
                ``residual`` is the coupled residual vector (shape ``(ndof_elem, 1)``),
                and ``jacobian`` is the coupled element Jacobian (shape ``(ndof_elem, ndof_elem)``).
        """
        # Compute thermal contribution:
        # l_t           -> scalar thermal loss
        # r_t           -> thermal residual vector (temperature DOFs only)
        # d_r_t_d_t     -> thermal tangent matrix w.r.t. temperature
        l_t, r_t, d_r_t_d_t = self.ComputeElementThermal(xyze,
                                                         de,
                                                         tuvwe[self.elem_t_local_ids])
        
        # Compute mechanical contribution:
        # l_m           -> scalar mechanical loss
        # r_m           -> mechanical residual vector (displacement DOFs only)
        # d_r_m_d_uvw   -> mechanical tangent matrix w.r.t. displacement
        # d_r_m_d_t     -> coupling tangent matrix w.r.t. temperature        
        l_m, r_m, d_r_m_d_uvw, d_r_m_d_t = self.ComputeElementMechanical(xyze,
                                                                         de,
                                                                         tuvwe[self.elem_t_local_ids],
                                                                         tuvwe[self.elem_uvw_local_ids],
                                                                         t0e)

        # Initialize full element residual and Jacobian with zeros
        # Size is based on total number of element DOFs
        re = jnp.zeros((tuvwe.shape[0],1))
        ke = jnp.zeros((tuvwe.shape[0],tuvwe.shape[0]))

        # Assemble mechanical stiffness block (displacement-displacement)
        ke = ke.at[jnp.ix_(self.elem_uvw_local_ids, self.elem_uvw_local_ids)].add(d_r_m_d_uvw)
        # Assemble thermo-mechanical coupling block (displacement-temperature)
        ke = ke.at[jnp.ix_(self.elem_uvw_local_ids, self.elem_t_local_ids)].add(d_r_m_d_t)
        # Assemble thermal stiffness block (temperature-temperature)
        ke = ke.at[jnp.ix_(self.elem_t_local_ids, self.elem_t_local_ids)].add(d_r_t_d_t)

        # Assemble thermal residual into global element residual
        re = re.at[jnp.ix_(self.elem_t_local_ids)].add(r_t)
        # Assemble mechanical residual into global element residual
        re = re.at[jnp.ix_(self.elem_uvw_local_ids)].add(r_m)

         # Total element loss is the sum of thermal and mechanical losses
        l_e = l_t + l_m

        return l_e, re, ke
    
    def ComputeElementResidualAndJacobian(self,
                                          elem_xyz:jnp.array,
                                          elem_controls:jnp.array,
                                          elem_dofs:jnp.array,
                                          elem_t0:jnp.array,
                                          elem_BC:jnp.array,
                                          elem_mask_BC:jnp.array,
                                          transpose_jac:bool):
        """
        Compute element residual and Jacobian with Dirichlet boundary conditions applied.

        The element Jacobian can optionally be transposed (useful for certain assembly
        or solver conventions). Dirichlet constraints are applied using the element BC
        values and masks via ``ApplyDirichletBCOnElementResidualAndJacobian``.

        Args:
            elem_xyz (jnp.array): Element nodal coordinates shaped like ``(nnode, dim)``.
            elem_controls (jnp.array): Element nodal controls shaped like ``(nnode,)`` or compatible.
            elem_dofs (jnp.array): Element DOF vector shaped like ``(ndof_elem, 1)``.
            elem_t0 (jnp.array): Element initial temperature values shaped like ``(nnode,)`` or ``(nnode, 1)``.
            elem_BC (jnp.array): Element Dirichlet BC vector shaped like ``(ndof_elem, 1)``.
            elem_mask_BC (jnp.array): Element mask for Dirichlet BC entries shaped like ``(ndof_elem, 1)``.
            transpose_jac (bool): If True, returns the transposed element Jacobian.

        Returns:
            tuple:
                A 2-tuple ``(residual, jacobian)`` after Dirichlet BC application, where
                ``residual`` has shape ``(ndof_elem, 1)`` and ``jacobian`` has shape
                ``(ndof_elem, ndof_elem)``.
        """
        _,re,ke = self.ComputeElement(elem_xyz,elem_controls,elem_dofs,elem_t0)

       # Convert transpose_jac (bool) to an integer index (0 = False, 1 = True)
        index = jnp.asarray(transpose_jac, dtype=jnp.int32)

        # Define the two branches for switch
        branches = [
            lambda _: ke,                  # Case 0: No transpose
            lambda _: jnp.transpose(ke)    # Case 1: Transpose ke
        ]

        # Apply the switch operation
        ke = jax.lax.switch(index, branches, None)

        return self.ApplyDirichletBCOnElementResidualAndJacobian(re,ke,elem_BC,elem_mask_BC)

    def ComputeElementResidualAndJacobianVmapCompatible(self,element_id:jnp.integer,
                                                        elements_nodes:jnp.array,
                                                        xyz:jnp.array,
                                                        full_control_vector:jnp.array,
                                                        full_dof_vector:jnp.array,
                                                        full_dirichlet_BC_vec:jnp.array,
                                                        full_mask_dirichlet_BC_vec:jnp.array,
                                                        transpose_jac:bool):
        """
        Vectorization-friendly wrapper for element residual/Jacobian computation.

        This method gathers element-local data (coordinates, controls, DOFs, and BCs)
        from full-field arrays using ``element_id`` and ``elements_nodes`` and then
        calls :meth:`ComputeElementResidualAndJacobian`. It is designed to be compatible
        with ``jax.vmap`` over ``element_id``.

        Args:
            element_id (jnp.integer): Index of the element to evaluate.
            elements_nodes (jnp.array): Connectivity array mapping element indices to node indices,
                shaped like ``(nelem, nnode)``.
            xyz (jnp.array): Global nodal coordinates shaped like ``(nnode_global, dim)``.
            full_control_vector (jnp.array): Global nodal control values shaped like ``(nnode_global,)`` or compatible.
            full_dof_vector (jnp.array): Global DOF vector shaped like ``(nnode_global * ndofs_per_node, 1)``.
            full_dirichlet_BC_vec (jnp.array): Global Dirichlet BC vector shaped like the global DOF vector.
            full_mask_dirichlet_BC_vec (jnp.array): Global mask for Dirichlet BC entries shaped like the global DOF vector.
            transpose_jac (bool): If True, returns the transposed element Jacobian.

        Returns:
            tuple:
                A 2-tuple ``(residual, jacobian)`` for the selected element after BC application.
        """
        return self.ComputeElementResidualAndJacobian(xyz[elements_nodes[element_id],:],
                                                      full_control_vector[elements_nodes[element_id]],
                                                      full_dof_vector[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      self.thermal_loss_settings["T0"][elements_nodes[element_id]],
                                                      full_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      full_mask_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      transpose_jac)
    
    def ComputeElementEnergyVmapCompatible(self,
                                           element_id:jnp.integer,
                                           elements_nodes:jnp.array,
                                           xyz:jnp.array,
                                           full_control_vector:jnp.array,
                                           full_dof_vector:jnp.array):
        """
        Vmap-compatible wrapper to compute an element scalar contribution.

        This helper extracts element nodal coordinates, controls, and DOFs from
        global arrays using ``element_id`` and then calls :meth:`ComputeElementEnergy`.

        Args:
            element_id (jax.numpy.integer):
                Element index.
            elements_nodes (jax.numpy.ndarray):
                Connectivity array mapping element ids to node ids.
            xyz (jax.numpy.ndarray):
                Nodal coordinates array.
            full_control_vector (jax.numpy.ndarray):
                Global control/parameter values per node.
            full_dof_vector (jax.numpy.ndarray):
                Global DOF vector.

        Returns:
            float:
                Scalar element contribution.
        """
        return self.ComputeElement(xyz[elements_nodes[element_id],:],
                                         full_control_vector[elements_nodes[element_id]],
                                         full_dof_vector[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                         jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                         self.thermal_loss_settings["T0"][elements_nodes[element_id]])[0]
    
    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def ComputeHeatFlux(self,nodal_conductivity: jnp.array, nodal_temperature: jnp.array):
        """
        Compute the global nodal heat flux vector for a scalar heat conduction problem
        in arbitrary spatial dimension.

        The heat flux is computed from the temperature field using Fourier's law
        ``q = -k ∇T``. Temperature gradients are evaluated at Gauss points within
        each finite element, the resulting Gauss-point heat flux is projected to
        element nodal values using an L2 (least-squares) projection, and the element
        nodal contributions are assembled into a global nodal heat flux vector.

        **Integration point requirement**
        
        For the L2 projection to be well-posed, the number of Gauss points must
        be greater than or equal to 2.

        Using only one Gauss point leads to a rank-deficient projection matrix:

            N_g^T N_g

        which becomes singular (or ill-conditioned), making the inversion unstable
        or impossible. At least two integration points are required to ensure that
        the projection matrix has sufficient rank and produces a stable nodal
        reconstruction of the heat flux field.

        If fewer than 2 Gauss points are detected, a warning is issued using
        ``fol_warning``. The integration order is temporarily increased to 2
        for the duration of this computation and restored to its original
        setting afterward.
        
        The implementation is dimension-agnostic and works for any spatial dimension
        ``dim = self.dim``, provided that the finite element gradients and node
        coordinates are defined consistently in that dimension.

        Args:
            nodal_conductivity (jnp.array):
                Nodal scalar thermal conductivity values.
                Shape ``(num_nodes,)``.

            nodal_temperature (jnp.array):
                Nodal temperature values.
                Shape ``(num_nodes,)``.

        Returns:
            jnp.array:
                Nodal heat flux field.
                Shape ``(num_nodes, dim)``, where each row contains the spatial
                components of the heat flux at a node.
        """

        # -------------------------------------------------
        # Check number of Gauss points
        # -------------------------------------------------
        if self.loss_settings["num_gp"]<2:
            fol_warning(
                "Number of Gauss points is less than 2. "
                "Setting integration points to 2 temporarily."
            )

            # Set integration rule temporarily to order 2
            self.fe_element.SetGaussIntegrationMethod("GI_GAUSS_2")

        nodes_coords = jnp.asarray(self.fe_mesh.GetNodesCoordinates())
        k = jnp.asarray(nodal_conductivity)
        T = jnp.asarray(nodal_temperature)

        def ComputeElementGPHeatFlux(xyze, ke, te):
            te = te.reshape(-1, 1)
            def compute_at_gauss_point(gp_point, gp_weight):
                DN_DX = self.fe_element.ShapeFunctionsGlobalGradients(xyze, gp_point)
                N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
                J = self.fe_element.Jacobian(xyze, gp_point)
                detJ = jnp.linalg.det(J)
                Jw = jnp.abs(detJ) * gp_weight                
                temp_at_gauss = jnp.dot(N_vec, te.squeeze())
                conductivity_at_gauss = jnp.dot(N_vec.reshape(1,-1), ke) \
                * (self.thermal_loss_settings["k1"]+1/self.thermal_loss_settings["k2"]*\
               (1/(1+jnp.exp(self.thermal_loss_settings["k3"]*(temp_at_gauss-self.thermal_loss_settings["k4"])))))
                temp_grad = DN_DX.T @ te
                q = -conductivity_at_gauss * temp_grad  # Fourier's law
                return N_vec,q.squeeze(), Jw
            gp_points, gp_weights = self.fe_element.GetIntegrationData()
            N_g,q_g,Jw = jax.vmap(compute_at_gauss_point, in_axes=(0, 0))(gp_points, gp_weights)
            N_g_T_N_g = N_g.T @ (N_g*Jw[:, jnp.newaxis])  
            N_g_T_q_g = N_g.T @ (q_g*Jw[:, jnp.newaxis])
            return jnp.linalg.inv(N_g_T_N_g) @ N_g_T_q_g

        def ComputeElementNodalHeatFlux(element_nodes):
            return ComputeElementGPHeatFlux(nodes_coords[element_nodes, :], 
                                            k[element_nodes], 
                                            T[element_nodes]).reshape(-1,1)

        element_fluxes = jax.vmap(ComputeElementNodalHeatFlux)(self.fe_mesh.GetElementsNodes(self.element_type))
        nodal_flux_vector = jnp.zeros((self.fe_mesh.GetNumberOfNodes()*self.dim))
        contribution_count = jnp.zeros(self.fe_mesh.GetNumberOfNodes()*self.dim)
        for dim_idx in range(self.dim):
            nodal_flux_vector = nodal_flux_vector.at[self.dim*self.fe_mesh.GetElementsNodes(self.element_type)+dim_idx].add(jnp.squeeze(element_fluxes[:,dim_idx::self.dim]))
            contribution_count = contribution_count.at[self.dim*self.fe_mesh.GetElementsNodes(self.element_type)+dim_idx].add(1)
        nodal_flux_vector = nodal_flux_vector / contribution_count  # Average contributions at shared nodes
        # -------------------------------------------------
        # Restore original integration rule if modified
        # -------------------------------------------------
        if self.loss_settings["num_gp"]<2:
            self.fe_element.SetGaussIntegrationMethod("GI_GAUSS_1")

        return nodal_flux_vector.reshape(-1,self.dim)
    
    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))    
    def ComputeStress(self,
                      nodal_conductivity: jnp.array, 
                      nodal_tuvw: jnp.array):
        """
        Compute the global nodal mechanical stress field for a thermo-mechanical
        linear elasticity problem in arbitrary spatial dimension.

        The Cauchy stress tensor is evaluated at Gauss integration points using
        the constitutive relation

            σ = D (ε_total − ε_thermal)

        where the total strain is obtained from the displacement field through

            ε_total = B u

        and the thermal strain is computed from the temperature difference with
        respect to the reference temperature. The constitutive matrix ``D`` is
        defined for isotropic linear elasticity and may be scaled by a
        temperature-dependent degradation factor evaluated at Gauss points.

        Gauss-point stresses are projected to element nodal stresses using an
        L2 (least-squares) projection. The element nodal contributions are then
        assembled into a global nodal stress vector.
        
        **Integration point requirement**

        For the L2 projection to be well-posed, the number of Gauss points must
        be greater than or equal to 2.

        Using only one Gauss point leads to a rank-deficient projection matrix:

            N_g^T N_g

        which becomes singular (or ill-conditioned), making the inversion unstable
        or impossible. At least two integration points are required to ensure that
        the projection matrix has sufficient rank and produces a stable nodal
        reconstruction of the stress field.

        If fewer than 2 Gauss points are detected, a warning is issued using
        ``fol_warning``. The integration rule is temporarily increased to second
        order for the duration of this computation and restored to its original
        setting afterward.

        The implementation is dimension-agnostic and works for any spatial
        dimension ``dim = self.dim``. The number of independent stress components
        is determined as

            num_stress_comps = dim (dim + 1) / 2

        corresponding to 3 components in 2D and 6 components in 3D.

        Args:
            nodal_conductivity (jnp.array):
                Nodal scalar degradation or stiffness-scaling field.
                Shape ``(num_nodes,)``. This field is interpolated to Gauss
                points and scales the constitutive response.

            nodal_tuvw (jnp.array):
                Nodal thermo-mechanical degrees of freedom, containing
                temperature and displacement components per node.
                Shape ``(num_nodes * number_dofs_per_node,)``.

        Returns:
            jnp.array:
                Nodal stress field in Voigt notation.
                Shape ``(num_nodes, num_stress_comps)``, where each row
                contains the independent stress components at a node.
        """
        # -------------------------------------------------
        # Check number of Gauss points
        # -------------------------------------------------
        if self.loss_settings["num_gp"]<2:
            fol_warning(
                "Number of Gauss points is less than 2. "
                "Setting integration points to 2 temporarily."
            )

            # Set integration rule temporarily to order 2
            self.fe_element.SetGaussIntegrationMethod("GI_GAUSS_2")

        nodes_coords = jnp.asarray(self.fe_mesh.GetNodesCoordinates())
        k = jnp.asarray(nodal_conductivity)
        tuvw = jnp.asarray(nodal_tuvw)

        def ComputeElementGPStress(xyze,ke,tuvwe,te_init):
            ke = ke.reshape(-1,1)
            te = tuvwe[self.elem_t_local_ids].reshape(-1,1)
            uvwe = tuvwe[self.elem_uvw_local_ids].reshape(-1,1)
            te_init = te_init.reshape(-1,1)
            def compute_at_gauss_point(gp_point,gp_weight):
                N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
                DN_DX = self.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point)
                B_mat = self.CalculateBMatrix(DN_DX)
                temp_at_gauss = jnp.dot(N_vec,te.squeeze())

                J = self.fe_element.Jacobian(xyze, gp_point)
                detJ = jnp.linalg.det(J)
                Jw = jnp.abs(detJ) * gp_weight
                total_strain_at_gauss = B_mat@uvwe
                thermal_strain_vec = 1.0\
                    * (temp_at_gauss - jnp.dot(N_vec, te_init.squeeze())) * self.thermal_st_vec
                elastic_strain = total_strain_at_gauss - thermal_strain_vec
                k_at_gauss = jnp.dot(N_vec, ke.squeeze()) * \
                            (self.mechanical_loss_settings["e1"]+self.mechanical_loss_settings["e2"]*temp_at_gauss)
                elastic_stress = k_at_gauss * (self.D @ elastic_strain)
                return N_vec,elastic_stress.squeeze(),Jw
            gp_points, gp_weights = self.fe_element.GetIntegrationData()
            N_g,s_g,Jw_g = jax.vmap(compute_at_gauss_point, in_axes=(0, 0))(gp_points, gp_weights)
            N_g_T_N_g = N_g.T @ (N_g*Jw_g[:, jnp.newaxis])
            N_g_T_s_g = N_g.T @ (s_g*Jw_g[:, jnp.newaxis])
            return jnp.linalg.inv(N_g_T_N_g) @ N_g_T_s_g

        def ComputeElementNodalStress(element_nodes):
            return ComputeElementGPStress(nodes_coords[element_nodes, :], 
                                          k[element_nodes], 
                                          tuvw[((self.number_dofs_per_node*element_nodes)[:, jnp.newaxis] +
                                            jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                          self.thermal_loss_settings["T0"][element_nodes],
                                          ).reshape(-1,1)
        
        elements_nodal_stresses = jax.vmap(ComputeElementNodalStress)(self.fe_mesh.GetElementsNodes(self.element_type))

        num_stress_comps = int(self.dim*(self.dim+1)/2)

        nodal_stress_vector = jnp.zeros((self.fe_mesh.GetNumberOfNodes()*num_stress_comps))
        contribution_count = jnp.zeros(self.fe_mesh.GetNumberOfNodes()*num_stress_comps)
        for dim_idx in range(num_stress_comps):
            nodal_stress_vector = nodal_stress_vector.at[num_stress_comps*self.fe_mesh.GetElementsNodes(self.element_type)+dim_idx].add(jnp.squeeze(elements_nodal_stresses[:,dim_idx::num_stress_comps]))
            contribution_count = contribution_count.at[num_stress_comps*self.fe_mesh.GetElementsNodes(self.element_type)+dim_idx].add(1)
        nodal_stress_vector = nodal_stress_vector / contribution_count  # Average contributions at shared nodes
        # -------------------------------------------------
        # Restore original integration rule if modified
        # -------------------------------------------------
        if self.loss_settings["num_gp"]<2:
            self.fe_element.SetGaussIntegrationMethod("GI_GAUSS_1")

        return nodal_stress_vector.reshape(-1,num_stress_comps)

class ThermoMechanicsLoss3DTetra(ThermoMechanicsLoss):
    """
    3D thermo-mechanical finite-element loss using tetrahedral elements.

    This class specializes :class:`ThermoMechanicsLoss` for three-dimensional
    problems with tetrahedral elements and ordered degrees of freedom
    ``["T", "Ux", "Uy", "Uz"]``.

    Args:
        name (str): Identifier for the loss instance used for logging,
            diagnostics, or solver registration.
        loss_settings (dict): Dictionary containing material, thermal, numerical,
            Dirichlet boundary-condition, and element-specific settings. Expected
            entries include ``material_dict``, ``thermal_dict``, and
            ``dirichlet_bc_dict``.
        fe_mesh (Mesh): Finite-element mesh defining nodes, connectivity, and
            integration data for tetrahedral elements.

    Returns:
        None
    """
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T", "Ux","Uy","Uz"],                                           
                               "element_type":"tetra"},fe_mesh)
        
class ThermoMechanicsLoss3DHexa(ThermoMechanicsLoss):
    """
    3D thermo-mechanical finite-element loss using hexahedral elements.

    This class specializes :class:`ThermoMechanicsLoss` for three-dimensional
    problems with hexahedral elements and ordered degrees of freedom
    ``["T", "Ux", "Uy", "Uz"]``. If the number of Gauss points is not specified
    in ``loss_settings``, it defaults to 2.

    Args:
        name (str): Identifier for the loss instance used for logging,
            diagnostics, or solver registration.
        loss_settings (dict): Dictionary containing material, thermal, numerical,
            Dirichlet boundary-condition, and element-specific settings. Expected
            entries include ``material_dict``, ``thermal_dict``, and
            ``dirichlet_bc_dict``.
        fe_mesh (Mesh): Finite-element mesh defining nodes, connectivity, and
            integration data for hexahedral elements.

    Returns:
        None
    """
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T", "Ux","Uy","Uz"],                               
                               "element_type":"hexahedron"},fe_mesh)

class ThermoMechanicsLoss2DQuad(ThermoMechanicsLoss):
    """
    2D thermo-mechanical finite-element loss using quadrilateral elements.

    This class specializes :class:`ThermoMechanicsLoss` for two-dimensional
    problems with quadrilateral elements and ordered degrees of freedom
    ``["T", "Ux", "Uy"]``. If the number of Gauss points is not specified
    in ``loss_settings``, it defaults to 2.

    Args:
        name (str): Identifier for the loss instance used for logging,
            diagnostics, or solver registration.
        loss_settings (dict): Dictionary containing material, thermal, numerical,
            Dirichlet boundary-condition, and element-specific settings. Expected
            entries include ``material_dict``, ``thermal_dict``, and
            ``dirichlet_bc_dict``.
        fe_mesh (Mesh): Finite-element mesh defining nodes, connectivity, and
            integration data for quadrilateral elements.

    Returns:
        None
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T", "Ux","Uy"], 
                               "element_type":"quad"},fe_mesh)
        
class ThermoMechanicsLoss2DTri(ThermoMechanicsLoss):
    """
    2D thermo-mechanical finite-element loss using triangular elements.

    This class specializes :class:`ThermoMechanicsLoss` for two-dimensional
    problems with triangular elements and ordered degrees of freedom
    ``["T", "Ux", "Uy"]``.

    Args:
        name (str): Identifier for the loss instance used for logging,
            diagnostics, or solver registration.
        loss_settings (dict): Dictionary containing material, thermal, numerical,
            Dirichlet boundary-condition, and element-specific settings. Expected
            entries include ``material_dict``, ``thermal_dict``, and
            ``dirichlet_bc_dict``.
        fe_mesh (Mesh): Finite-element mesh defining nodes, connectivity, and
            integration data for triangular elements.

    Returns:
        None
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T", "Ux","Uy"],  
                               "element_type":"triangle"},fe_mesh)
