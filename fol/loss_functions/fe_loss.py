"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""
from  .loss import Loss
import jax
import jax.numpy as jnp
import warnings
from jax import jit,grad
from functools import partial
from abc import abstractmethod
from fol.tools.decoration_functions import *
from jax.experimental import sparse
from fol.mesh_input_output.mesh import Mesh
from fol.tools.fem_utilities import *
from fol.geometries import fe_element_dict

class FiniteElementLoss(Loss):
    """
    Base class for finite element loss functionals.

    This class provides the common infrastructure required to define loss
    functionals on a finite element mesh. Concrete loss functions in
    ``fol.loss_functions`` are typically derived from this class and implement
    the abstract element routine :meth:`ComputeElement`.

    The base class handles:

    - DOF bookkeeping and global indexing.
    - Dirichlet boundary condition indexing and application.
    - Element integration rule selection (Gauss points).
    - Element-wise evaluation and vectorized mapping over elements.
    - Assembly of a global residual vector and sparse Jacobian matrix.
    - Consistent interfaces for energy-based and weighted-residual losses.

    A derived class must implement :meth:`ComputeElement`, which returns an
    element scalar contribution, an element residual vector, and an element
    Jacobian/tangent matrix. The scalar contribution may represent an element
    energy (energy-based losses) or a scalar constructed from the element
    residual and element DOFs (weighted-residual losses). The base class does
    not enforce a specific interpretation of the scalar contribution; it only
    provides consistent aggregation across the mesh.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Configuration dictionary. The following keys are required:
            ``"ordered_dofs"``, ``"element_type"``, ``"compute_dims"``,
            and ``"dirichlet_bc_dict"``. Optional keys include ``"num_gp"``,
            ``"loss_function_exponent"``, and ``"parametric_boundary_learning"``.
        fe_mesh (Mesh):
            Finite element mesh over which the loss is defined.
    """
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        """
        Create a finite element loss instance.

        Args:
            name (str):
                Name identifier for the loss instance.
            loss_settings (dict):
                Loss configuration dictionary. Must include
                ``ordered_dofs``, ``element_type``, and ``dirichlet_bc_dict``.
            fe_mesh (Mesh):
                Finite element mesh associated with this loss.

        Raises:
            ValueError:
                If required entries are missing from ``loss_settings``.
        """
        super().__init__(name)
        self.loss_settings = loss_settings
        self.dofs = self.loss_settings["ordered_dofs"]
        self.element_type = self.loss_settings["element_type"]
        self.fe_mesh = fe_mesh
        if "dirichlet_bc_dict" not in self.loss_settings.keys():
            fol_error("dirichlet_bc_dict should provided in the loss settings !")

    def __CreateDofsDict(self, dofs_list:list, dirichlet_bc_dict:dict):
        number_dofs_per_node = len(dofs_list)
        dirichlet_indices = []
        dirichlet_values = []
        for dof_index,dof in enumerate(dofs_list):
            for boundary_name,boundary_value in dirichlet_bc_dict[dof].items():
                boundary_node_ids = jnp.array(self.fe_mesh.GetNodeSet(boundary_name))
                dirichlet_bc_indices = number_dofs_per_node*boundary_node_ids + dof_index
                dirichlet_indices.append(dirichlet_bc_indices)

                dirichlet_bc_values = boundary_value * jnp.ones_like(dirichlet_bc_indices)
                dirichlet_values.append(dirichlet_bc_values)

        if len(dirichlet_indices) != 0:
            self.dirichlet_indices = jnp.concatenate(dirichlet_indices)
            self.dirichlet_values = jnp.concatenate(dirichlet_values)
        else:
            self.dirichlet_indices = jnp.array([], dtype=jnp.int32)
            self.dirichlet_values = jnp.array([])

        all_indices = jnp.arange(number_dofs_per_node*self.fe_mesh.GetNumberOfNodes())
        self.non_dirichlet_indices = jnp.setdiff1d(all_indices, self.dirichlet_indices)

    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize FE bookkeeping, integration rules, and batching configuration.

        This method prepares all data required for fast element-wise evaluation
        and assembly. It is designed to be called once before evaluation but can
        be forced to re-run by setting ``reinitialize=True``.

        The initialization performs:

        - DOF counts and global indexing.
        - Construction of Dirichlet and non-Dirichlet index sets from
          ``loss_settings["dirichlet_bc_dict"]``.
        - Selection and configuration of the FE element implementation from
          ``element_type`` and Gauss integration settings from ``num_gp``.
        - Definition of a full-DOF reconstruction function that applies
          Dirichlet values, optionally using parametric boundary learning.
        - Determination of an element batch size that evenly divides the number
          of elements to support efficient batched assembly.
        - Configuration of an exponent applied to the scalar loss aggregation.

        Args:
            reinitialize (bool, optional):
                If ``True``, re-run initialization even if the instance has
                already been initialized. Default is ``False``.

        Returns:
            None

        Raises:
            ValueError:
                If ``compute_dims`` is not provided or if an unsupported Gauss
                integration rule is requested.
        """
        if self.initialized and not reinitialize:
            return

        self.number_dofs_per_node = len(self.dofs)
        self.total_number_of_dofs = len(self.dofs) * self.fe_mesh.GetNumberOfNodes()
        self.__CreateDofsDict(self.dofs,self.loss_settings["dirichlet_bc_dict"])
        self.number_of_unknown_dofs = self.non_dirichlet_indices.size

        # fe element
        self.fe_element = fe_element_dict[self.element_type]

        # now prepare gauss integration
        if "num_gp" in self.loss_settings.keys():
            self.num_gp = self.loss_settings["num_gp"]
            if self.num_gp == 1:
                self.fe_element.SetGaussIntegrationMethod("GI_GAUSS_1")
            elif self.num_gp == 2:
                self.fe_element.SetGaussIntegrationMethod("GI_GAUSS_2")
            elif self.num_gp == 3:
                self.fe_element.SetGaussIntegrationMethod("GI_GAUSS_3")
            else:
                raise ValueError(f" number gauss points {self.num_gp} is not supported ! ")
        else:
            self.fe_element.SetGaussIntegrationMethod("GI_GAUSS_1")
            self.loss_settings["num_gp"] = 1
            self.num_gp = 1

        if not "compute_dims" in self.loss_settings.keys():
            raise ValueError(f"compute_dims must be provided in the loss settings of {self.GetName()}! ")

        self.dim = self.loss_settings["compute_dims"]

        def ConstructFullDofVector(known_dofs: jnp.array,all_dofs: jnp.array):
            return all_dofs.at[:,self.dirichlet_indices].set(self.dirichlet_values)

        def ConstructFullDofVectorParametricLearning(known_dofs: jnp.array,all_dofs: jnp.array):
            return all_dofs.at[:,self.dirichlet_indices].set(known_dofs)

        if self.loss_settings.get("parametric_boundary_learning"):
            self.full_dof_vector_function = ConstructFullDofVectorParametricLearning
        else:
            self.full_dof_vector_function = ConstructFullDofVector

        # set identity function for parametric learning
        self.get_param_function = lambda x: x

        # element batching
        num_cuts = 20
        num_elements = self.fe_mesh.GetNumberOfElements(self.element_type)
        element_batch_size = num_elements if num_elements < num_cuts else int(jnp.floor(num_elements / num_cuts))
        self.adjusted_batch_size = next(i for i in range(element_batch_size, 0, -1) if num_elements % i == 0)
        self.num_element_batches = int(num_elements/self.adjusted_batch_size)
        if self.adjusted_batch_size != element_batch_size:
            fol_info(f"for the proper batching of elements, the batch size is changed from {element_batch_size} to {self.adjusted_batch_size}")
        else:
            fol_info(f"element batch size is {self.adjusted_batch_size}")

        # set scalar-valued loss function exponent
        if "loss_function_exponent" in self.loss_settings:
            self.loss_function_exponent = self.loss_settings["loss_function_exponent"]
        else:
            self.loss_function_exponent = 1.0

        self.initialized = True

    def GetFullDofVector(self,known_dofs: jnp.array,unknown_dofs: jnp.array) -> jnp.array:
        """
        Construct a full DOF vector including Dirichlet values.

        Args:
            known_dofs (jax.numpy.ndarray):
                Values used for Dirichlet DOFs when parametric boundary learning
                is enabled.
            unknown_dofs (jax.numpy.ndarray):
                DOF array that will be completed by inserting Dirichlet values.

        Returns:
            jax.numpy.ndarray:
                Full DOF vector with Dirichlet entries applied.
        """
        return self.full_dof_vector_function(known_dofs,unknown_dofs)

    def GetParametersVectors(self,param_vector: jnp.array) -> jnp.array:
        """
        Map an input parameter vector to the control variables used by elements.

        By default this is the identity mapping. Derived classes may override the
        mapping by setting ``self.get_param_function`` during initialization.

        Args:
            param_vector (jax.numpy.ndarray):
                Input parameter vector.

        Returns:
            jax.numpy.ndarray:
                Control vector used by element routines.
        """
        return self.get_param_function(param_vector)

    def Finalize(self) -> None:
        pass

    def GetNumberOfUnknowns(self):
        """
        Return the number of unknown DOFs (non-Dirichlet).

        Returns:
            int:
                Number of non-Dirichlet DOFs.
        """
        return self.number_of_unknown_dofs

    def GetTotalNumberOfDOFs(self):
        """
        Return the total number of DOFs including Dirichlet DOFs.

        Returns:
            int:
                Total number of DOFs in the FE system.
        """
        return self.total_number_of_dofs

    def GetDOFs(self):
        """
        Return the ordered DOF names per node.

        Returns:
            list:
                DOF names in the order used by the loss.
        """
        return self.dofs

    @abstractmethod
    def ComputeElement(self,
                       elem_xyz:jnp.array,
                       elem_controls:jnp.array,
                       elem_dofs:jnp.array) -> tuple[float, jnp.array, jnp.array]:
        """
        Compute element-level scalar, residual, and Jacobian contributions.

        Derived classes must implement this method. The returned scalar is used
        by the base class to build element energies and batch losses, while the
        residual and Jacobian are used to assemble global vectors and matrices.

        Args:
            elem_xyz (jax.numpy.ndarray):
                Element nodal coordinates.
            elem_controls (jax.numpy.ndarray):
                Element control variables (for example material field values)
                associated with the element nodes.
            elem_dofs (jax.numpy.ndarray):
                Element DOF vector arranged consistently with the element type
                and ordered degrees of freedom.

        Returns:
            Tuple[float, jax.numpy.ndarray, jax.numpy.ndarray]:
                - Scalar element contribution (energy or scalar loss term).
                - Element residual vector.
                - Element Jacobian/tangent matrix.
        """
        pass

    def ComputeElementEnergy(self,
                             elem_xyz:jnp.array,
                             elem_controls:jnp.array,
                             elem_dofs:jnp.array) -> float:
        """
        Convenience wrapper returning the scalar element contribution.

        Args:
            elem_xyz (jax.numpy.ndarray):
                Element nodal coordinates.
            elem_controls (jax.numpy.ndarray):
                Element control variables.
            elem_dofs (jax.numpy.ndarray):
                Element DOF vector.

        Returns:
            float:
                Scalar element contribution returned by :meth:`ComputeElement`.
        """
        return self.ComputeElement(elem_xyz,elem_controls,elem_dofs)[0]

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
        return self.ComputeElementEnergy(xyz[elements_nodes[element_id],:],
                                         full_control_vector[elements_nodes[element_id]],
                                         full_dof_vector[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                         jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    def ComputeElementsEnergies(self,total_control_vars:jnp.array,total_primal_vars:jnp.array):
        """
        Compute scalar element contributions for all elements.

        Args:
            total_control_vars (jax.numpy.ndarray):
                Global control/parameter vector (nodal values).
            total_primal_vars (jax.numpy.ndarray):
                Global DOF vector (nodal unknowns).

        Returns:
            jax.numpy.ndarray:
                Array of per-element scalar contributions.
        """
        return jax.vmap(self.ComputeElementEnergyVmapCompatible,(0,None,None,None,None)) \
                        (self.fe_mesh.GetElementsIds(self.element_type),
                        self.fe_mesh.GetElementsNodes(self.element_type),
                        self.fe_mesh.GetNodesCoordinates(),
                        total_control_vars,
                        total_primal_vars)

    def ComputeTotalEnergy(self,total_control_vars:jnp.array,total_primal_vars:jnp.array):
        """
        Compute the total scalar loss value by summing element contributions.

        Args:
            total_control_vars (jax.numpy.ndarray):
                Global control/parameter vector (nodal values).
            total_primal_vars (jax.numpy.ndarray):
                Global DOF vector (nodal unknowns).

        Returns:
            jax.numpy.ndarray:
                Total scalar loss value assembled over all elements.
        """
        return jnp.sum(self.ComputeElementsEnergies(total_control_vars,total_primal_vars))

    def ComputeElementJacobianIndices(self,nodes_ids:jnp.array):
        """
        Compute global (row, col) index pairs for an element Jacobian block.

        This method maps element node ids to global DOF indices and returns the
        2D index pairs needed to assemble an element matrix into a global sparse
        matrix.

        Args:
            nodes_ids (jax.numpy.ndarray):
                Element node ids.

        Returns:
            jax.numpy.ndarray:
                Array of shape ``(ndofs_elem * ndofs_elem, 2)`` containing global
                (row, col) index pairs for the element Jacobian block.
        """
        nodes_ids *= self.number_dofs_per_node
        nodes_ids += jnp.arange(self.number_dofs_per_node).reshape(-1,1)
        indices_dof = nodes_ids.T.flatten()
        rows,cols = jnp.meshgrid(indices_dof,indices_dof,indexing='ij')#rows and columns
        indices = jnp.vstack((rows.ravel(),cols.ravel())).T #indices in global stiffness matrix
        return indices

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,2,))
    def ApplyDirichletBCOnDofVector(self,full_dof_vector:jnp.array,load_increment:float=1.0):
        """
        Apply Dirichlet boundary conditions on a global DOF vector.

        Args:
            full_dof_vector (jax.numpy.ndarray):
                Global DOF vector to be modified in-place (functional update).
            load_increment (float, optional):
                Scaling factor applied to the prescribed Dirichlet values.
                Default is ``1.0``.

        Returns:
            jax.numpy.ndarray:
                Global DOF vector with Dirichlet entries updated.
        """
        return full_dof_vector.at[self.dirichlet_indices].set(load_increment*self.dirichlet_values)

    def ApplyDirichletBCOnElementResidualAndJacobian(self,
                                                     elem_res:jnp.array,
                                                     elem_jac:jnp.array,
                                                     elem_BC_vec:jnp.array,
                                                     elem_mask_BC_vec:jnp.array):

        """
        Apply Dirichlet boundary conditions to an element residual and Jacobian.

        This method modifies the element residual and Jacobian so that Dirichlet
        DOFs are enforced at the element level. The inputs ``elem_BC_vec`` and
        ``elem_mask_BC_vec`` are diagonal vectors constructed from global BC
        vectors restricted to the element.

        Args:
            elem_res (jax.numpy.ndarray):
                Element residual vector.
            elem_jac (jax.numpy.ndarray):
                Element Jacobian/tangent matrix.
            elem_BC_vec (jax.numpy.ndarray):
                Element BC diagonal vector used to enforce prescribed DOFs.
            elem_mask_BC_vec (jax.numpy.ndarray):
                Element mask diagonal vector used to preserve diagonal entries
                for constrained DOFs.

        Returns:
            Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
                Element residual and Jacobian after applying Dirichlet
                modifications.
        """

        BC_matrix = jnp.zeros((elem_jac.shape))
        BC_matrix = jnp.fill_diagonal(BC_matrix, elem_BC_vec, inplace=False)

        mask_BC_matrix = jnp.zeros((elem_jac.shape))
        mask_BC_matrix = jnp.fill_diagonal(mask_BC_matrix, elem_mask_BC_vec, inplace=False)

        masked_diag_entries = jnp.diag(mask_BC_matrix @ elem_jac @ mask_BC_matrix)
        mask_BC_matrix = jnp.zeros((elem_jac.shape))
        mask_BC_matrix = jnp.fill_diagonal(mask_BC_matrix, masked_diag_entries, inplace=False)

        return   BC_matrix @ elem_res, BC_matrix @ elem_jac + mask_BC_matrix

    def ComputeElementResidualAndJacobian(self,
                                          elem_xyz:jnp.array,
                                          elem_controls:jnp.array,
                                          elem_dofs:jnp.array,
                                          elem_BC:jnp.array,
                                          elem_mask_BC:jnp.array,
                                          transpose_jac:bool):
        """
        Vmap-compatible wrapper for residual/Jacobian evaluation of one element.

        This helper extracts element nodal coordinates, controls, DOFs, and
        element-restricted BC vectors from global arrays and calls
        :meth:`ComputeElementResidualAndJacobian`.

        Args:
            element_id (jax.numpy.integer):
                Element index.
            elements_nodes (jax.numpy.ndarray):
                Element connectivity array.
            xyz (jax.numpy.ndarray):
                Nodal coordinates.
            full_control_vector (jax.numpy.ndarray):
                Global control values per node.
            full_dof_vector (jax.numpy.ndarray):
                Global DOF vector.
            full_dirichlet_BC_vec (jax.numpy.ndarray):
                Global BC enforcement vector.
            full_mask_dirichlet_BC_vec (jax.numpy.ndarray):
                Global BC mask vector.
            transpose_jac (bool):
                If ``True``, transpose element Jacobian before BC application.

        Returns:
            Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
                Element residual and Jacobian after BC application.
        """
        _,re,ke = self.ComputeElement(elem_xyz,elem_controls,elem_dofs)

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
        return self.ComputeElementResidualAndJacobian(xyz[elements_nodes[element_id],:],
                                                      full_control_vector[elements_nodes[element_id]],
                                                      full_dof_vector[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      full_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      full_mask_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                                      transpose_jac)

    def ComputeBatchLoss(self,batch_params:jnp.array,batch_dofs:jnp.array):
        """
        Compute the finite element loss over a batch of parameter samples and
        their associated solution fields.

        This method evaluates the finite element loss for multiple input samples
        in parallel. Each sample consists of a parameter (control) vector and
        its corresponding primal solution (DOF) vector. For every sample,
        Dirichlet boundary conditions are applied to construct a full DOF
        vector, and the scalar element contributions returned by
        ``ComputeElement`` are summed over all elements to form the sample loss
        value.

        If ``loss_function_exponent`` is specified in ``loss_settings``, the
        summed scalar value is raised to that exponent for each sample. The
        returned batch loss is the mean of the per-sample loss values.

        Args:
            batch_params (jax.numpy.ndarray):
                Batch of parameter or control vectors, one per sample. These
                vectors are mapped to element-level control fields using
                ``GetParametersVectors``. The input is reshaped internally to
                ``(batch_size, -1)``.
            batch_dofs (jax.numpy.ndarray):
                Batch of primal solution vectors (DOF fields), one per sample.
                Dirichlet DOFs are inserted using ``GetFullDofVector``. The
                input is reshaped internally to ``(batch_size, -1)``.

        Returns:
            Tuple[jax.numpy.ndarray, Tuple[jax.numpy.ndarray, jax.numpy.ndarray, jax.numpy.ndarray]]:
                A tuple containing the mean loss value across the batch and a
                tuple with the minimum, maximum, and mean per-sample loss
                values.
        """

        batch_params = jnp.atleast_2d(batch_params)
        batch_params = batch_params.reshape(batch_params.shape[0], -1)
        batch_dofs = jnp.atleast_2d(batch_dofs)
        batch_dofs = batch_dofs.reshape(batch_dofs.shape[0], -1)
        BC_applied_batch_dofs = self.GetFullDofVector(batch_params,batch_dofs)

        def ComputeSingleLoss(params,dofs):
            return jnp.sum(self.ComputeElementsEnergies(self.GetParametersVectors(params),dofs))**self.loss_function_exponent

        batch_energies = jax.vmap(ComputeSingleLoss)(batch_params,BC_applied_batch_dofs)

        return jnp.mean(batch_energies),(jnp.min(batch_energies),jnp.max(batch_energies),jnp.mean(batch_energies))

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def ComputeJacobianMatrixAndResidualVector(self,total_control_vars: jnp.array,total_primal_vars: jnp.array,transpose_jacobian:bool=False,new_implementation:bool=False):
        """
        Assemble the global sparse Jacobian matrix and residual vector.

        This method evaluates element residuals and Jacobians in batches,
        applies Dirichlet boundary conditions at element level, and assembles a
        global residual vector and a global sparse Jacobian matrix in BCOO
        format.

        Args:
            total_control_vars (jax.numpy.ndarray):
                Global control/parameter vector (nodal values).
            total_primal_vars (jax.numpy.ndarray):
                Global DOF vector (nodal unknowns).
            transpose_jacobian (bool, optional):
                If ``True``, element Jacobians are transposed before assembly.
                Default is ``False``.
            new_implementation (bool, optional):
                Placeholder flag for alternative implementations. Default is
                ``False``.

        Returns:
            Tuple[jax.experimental.sparse.BCOO, jax.numpy.ndarray]:
                - Sparse global Jacobian matrix.
                - Global residual vector.
        """
        BC_vector = jnp.ones((self.total_number_of_dofs))
        BC_vector = BC_vector.at[self.dirichlet_indices].set(0)
        mask_BC_vector = jnp.zeros((self.total_number_of_dofs))
        mask_BC_vector = mask_BC_vector.at[self.dirichlet_indices].set(1)

        num_nodes_per_elem = len(self.fe_mesh.GetElementsNodes(self.element_type)[0])
        element_matrix_size = self.number_dofs_per_node * num_nodes_per_elem
        elements_jacobian_flat = jnp.zeros(self.fe_mesh.GetNumberOfElements(self.element_type)*element_matrix_size*element_matrix_size)

        template_element_indices = jnp.arange(0,self.adjusted_batch_size)
        template_elem_res_indices = jnp.arange(0,element_matrix_size,self.number_dofs_per_node)
        template_elem_jac_indices = jnp.arange(0,self.adjusted_batch_size*element_matrix_size*element_matrix_size)

        residuals_vector = jnp.zeros((self.total_number_of_dofs))

        def fill_arrays(batch_index,batch_arrays):
            glob_res_vec,elem_jac_vec = batch_arrays

            batch_element_indices = (batch_index * self.adjusted_batch_size) + template_element_indices
            batch_elem_jac_indices = (batch_index * self.adjusted_batch_size * element_matrix_size**2) + template_elem_jac_indices

            batch_elements_residuals, batch_elements_stiffness = jax.vmap(self.ComputeElementResidualAndJacobianVmapCompatible,(0,None,None,None,None,None,None,None)) \
                                                                (batch_element_indices,
                                                                self.fe_mesh.GetElementsNodes(self.element_type),
                                                                self.fe_mesh.GetNodesCoordinates(),
                                                                total_control_vars,
                                                                total_primal_vars,
                                                                BC_vector,
                                                                mask_BC_vector,
                                                                transpose_jacobian)

            elem_jac_vec = elem_jac_vec.at[batch_elem_jac_indices].set(batch_elements_stiffness.ravel())

            @jax.jit
            def fill_res_vec(dof_idx,glob_res_vec):
                glob_res_vec = glob_res_vec.at[self.number_dofs_per_node*self.fe_mesh.GetElementsNodes(self.element_type)[batch_element_indices]+dof_idx].add(jnp.squeeze(batch_elements_residuals[:,template_elem_res_indices+dof_idx]))
                return glob_res_vec

            glob_res_vec = jax.lax.fori_loop(0, self.number_dofs_per_node, fill_res_vec, (glob_res_vec))

            return (glob_res_vec,elem_jac_vec)

        residuals_vector, elements_jacobian_flat = jax.lax.fori_loop(0, self.num_element_batches, fill_arrays, (residuals_vector, elements_jacobian_flat))

        # second compute the global jacobian matrix
        jacobian_indices = jax.vmap(self.ComputeElementJacobianIndices)(self.fe_mesh.GetElementsNodes(self.element_type)) # Get the indices
        jacobian_indices = jacobian_indices.reshape(-1,2)

        sparse_jacobian = sparse.BCOO((elements_jacobian_flat,jacobian_indices),shape=(self.total_number_of_dofs,self.total_number_of_dofs))

        return sparse_jacobian, residuals_vector