"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: Sep, 2025
 License: FOL/LICENSE
"""
try:
    import fol_ffi_functions
    _HAS_FOL_FFI_LIB = True
except ImportError:
    _HAS_FOL_FFI_LIB = False

from  fol.loss_functions.fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *
from jax.experimental import sparse

if _HAS_FOL_FFI_LIB:
    from fol_ffi_functions import kr_small_displacement_element
    jax.ffi.register_ffi_target("compute_nodal_residuals", kr_small_displacement_element.compute_nodal_residuals(), platform="CUDA")
    jax.ffi.register_ffi_type_id("compute_nodal_residuals", kr_small_displacement_element.type_id(), platform="CUDA")
    jax.ffi.register_ffi_target("compute_elements", kr_small_displacement_element.compute_elements(), platform="CUDA")

class KratosSmallDisplacement3DTetra(FiniteElementLoss):
    """
    Kratos-based 3D small-displacement tetrahedral loss accelerated via JAX FFI.

    This class provides a finite element loss for linear small-displacement
    elasticity in 3D using tetrahedral elements. It derives from
    :class:`fol.loss_functions.fe_loss.FiniteElementLoss` and reuses the base
    class infrastructure for DOF bookkeeping, Dirichlet boundary conditions,
    and sparse assembly.

    In contrast to standard Python/JAX element implementations, this class
    delegates element-level computations to custom CUDA kernels exposed through
    ``fol_ffi_functions`` and registered with ``jax.ffi``. The kernels compute
    element residuals and element stiffness matrices, which are then processed
    to enforce Dirichlet boundary conditions and assembled into a global sparse
    Jacobian matrix and residual vector.

    The total loss value is computed in an energy-consistent manner using the
    dot product of the global displacement vector with the global nodal
    residual vector produced by the FFI kernel.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Configuration dictionary. User-provided settings may include
            ``"material_dict"`` with keys ``"poisson_ratio"`` and
            ``"young_modulus"`` and the standard FE settings expected by the
            base class. The element discretization is fixed to 3D tetrahedra
            with displacement DOFs.
        fe_mesh (Mesh):
            Finite element mesh containing node coordinates and tetrahedral
            connectivity.

    Raises:
        RuntimeError:
            If ``fol_ffi_functions`` is not available.

    Notes:
        This class relies on CUDA FFI targets registered at import time. It is
        expected to run on platforms where the registered targets are available.
        Element-level routines are not implemented in Python for this class.
    """
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not _HAS_FOL_FFI_LIB:
            fol_error(" fol_ffi_functions is not available, install by running install script under ffi_functions folder!")
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["Ux","Uy","Uz"],
                               "element_type":"tetra"},fe_mesh)

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize FE bookkeeping and material parameters.

        This method calls the base class initialization and then prepares the
        material parameters used by the CUDA kernels. If ``material_dict`` is
        provided in ``loss_settings``, it is used directly; otherwise default
        values are applied.

        Args:
            reinitialize (bool, optional):
                If ``True``, re-run initialization even if already initialized.
                Default is ``False``.

        Returns:
            None
        """
        if self.initialized and not reinitialize:
            return
        super().Initialize()

        self.material_settings = {"poisson_ratio":0.3,"young_modulus":1.0}
        if "material_dict" in self.loss_settings.keys():
            self.material_settings = self.loss_settings["material_dict"]

    def ComputeElement(self,xyze,de,te,body_force=0):
        """
        Element-level computation is not available for this FFI-based loss.

        The element residual and stiffness contributions are computed in CUDA
        through JAX FFI. The Python element routine is intentionally not
        implemented for this class.

        Args:
            xyze:
                Element nodal coordinates.
            de:
                Element control values.
            te:
                Element DOF vector.
            body_force:
                Body force term (unused).

        Returns:
            None

        Raises:
            RuntimeError:
                Always raised because the method is not implemented.
        """
        fol_error(" is not implemented for KratosSmallDisplacement3DTetra!")

    def ComputeTotalEnergy(self,total_control_vars:jnp.array,total_primal_vars:jnp.array):
        """
        Compute the total energy-consistent scalar loss value for the system.

        This method calls the FFI kernel ``compute_nodal_residuals`` to evaluate
        the global nodal residual vector for the provided displacement field.
        The returned scalar value is computed as the dot product of the global
        displacement vector with the global residual vector.

        Args:
            total_control_vars (jax.numpy.ndarray):
                Control variables. This argument is accepted for API
                compatibility and is not used by the current FFI kernels.
            total_primal_vars (jax.numpy.ndarray):
                Global displacement (DOF) vector with shape
                ``(total_number_of_dofs,)`` or compatible. The vector is
                reshaped internally to a batch of size 1.

        Returns:
            jax.numpy.ndarray:
                Scalar loss value computed from the displacement field and the
                nodal residuals.
        """
        total_primal_vars = total_primal_vars.reshape(1,-1)
        batch_size = 1
        nodal_res_type = jax.ShapeDtypeStruct((batch_size,self.number_dofs_per_node*self.fe_mesh.GetNumberOfNodes()), total_primal_vars.dtype)
        nodal_res = jax.ffi.ffi_call("compute_nodal_residuals", nodal_res_type, vmap_method="legacy_vectorized")(self.fe_mesh.GetNodesCoordinates(),
                                                                                self.fe_mesh.GetElementsNodes(self.element_type),
                                                                                jnp.array([self.material_settings["poisson_ratio"],
                                                                                           self.material_settings["young_modulus"]]),total_primal_vars)
        return (total_primal_vars @ nodal_res.T)[0,0]

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def ComputeJacobianMatrixAndResidualVector(self,total_control_vars: jnp.array,total_primal_vars: jnp.array,transpose_jacobian:bool=False):
        """
        Assemble the global sparse Jacobian matrix and residual vector using FFI.

        This method calls the FFI kernel ``compute_elements`` to compute element
        stiffness matrices and element residual vectors for all tetrahedral
        elements. Dirichlet boundary conditions are applied at element level
        using the base class masking strategy, and the processed contributions
        are assembled into:

        - a global residual vector, and
        - a global sparse Jacobian matrix in BCOO format.

        Args:
            total_control_vars (jax.numpy.ndarray):
                Control variables. This argument is accepted for API
                compatibility and is not used by the current FFI kernels.
            total_primal_vars (jax.numpy.ndarray):
                Global displacement (DOF) vector.
            transpose_jacobian (bool, optional):
                If ``True``, element stiffness matrices are transposed before
                applying boundary conditions and assembly. Default is ``False``.

        Returns:
            Tuple[jax.experimental.sparse.BCOO, jax.numpy.ndarray]:
                A tuple containing the global sparse Jacobian matrix and the
                global residual vector.
        """
        BC_vector = jnp.ones((self.total_number_of_dofs))
        BC_vector = BC_vector.at[self.dirichlet_indices.astype(jnp.int32)].set(0)
        mask_BC_vector = jnp.zeros((self.total_number_of_dofs))
        mask_BC_vector = mask_BC_vector.at[self.dirichlet_indices.astype(jnp.int32)].set(1)

        total_num_elements = self.fe_mesh.GetNumberOfElements(self.element_type)
        num_nodes_per_elem = len(self.fe_mesh.GetElementsNodes(self.element_type)[0])
        element_matrix_size = self.number_dofs_per_node * num_nodes_per_elem

        lhs_type = jax.ShapeDtypeStruct((total_num_elements,element_matrix_size,element_matrix_size), total_primal_vars.dtype)
        rhs_type = jax.ShapeDtypeStruct((total_num_elements,element_matrix_size), total_primal_vars.dtype)
        lhs,res = jax.ffi.ffi_call("compute_elements", (lhs_type, rhs_type))(self.fe_mesh.GetNodesCoordinates(),
                                                                            self.fe_mesh.GetElementsNodes(self.element_type),
                                                                            jnp.array([self.material_settings["poisson_ratio"],
                                                                                        self.material_settings["young_modulus"]]),
                                                                            total_primal_vars)

        def Proccess(ke:jnp.array,
                     re:jnp.array,
                     elem_BC:jnp.array,
                     elem_mask_BC:jnp.array):

            index = jnp.asarray(transpose_jacobian, dtype=jnp.int32)
            # Define the two branches for switch
            branches = [
                lambda _: ke,                  # Case 0: No transpose
                lambda _: jnp.transpose(ke)    # Case 1: Transpose ke
            ]
            # Apply the switch operation
            ke = jax.lax.switch(index, branches, None)
            return self.ApplyDirichletBCOnElementResidualAndJacobian(re,ke,elem_BC,elem_mask_BC)


        def ProccessVmapCompatible(element_id:jnp.integer,
                                    elements_nodes:jnp.array,
                                    elements_stiffness:jnp.array,
                                    elements_residuals:jnp.array,
                                    full_dirichlet_BC_vec:jnp.array,
                                    full_mask_dirichlet_BC_vec:jnp.array):

            return Proccess(elements_stiffness[element_id],
                                 elements_residuals[element_id],
                                full_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                jnp.arange(self.number_dofs_per_node))].reshape(-1,1),
                                full_mask_dirichlet_BC_vec[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

        elements_residuals, elements_stiffness = jax.vmap(ProccessVmapCompatible, (0,None,None,None,None,None))(self.fe_mesh.GetElementsIds(self.element_type),
                                                                                                            self.fe_mesh.GetElementsNodes(self.element_type),
                                                                                                            lhs,
                                                                                                            res,
                                                                                                            BC_vector,
                                                                                                            mask_BC_vector)


        # first compute the global residual vector
        residuals_vector = jnp.zeros((self.total_number_of_dofs))
        for dof_idx in range(self.number_dofs_per_node):
            residuals_vector = residuals_vector.at[self.number_dofs_per_node*self.fe_mesh.GetElementsNodes(self.element_type)+dof_idx].add(jnp.squeeze(elements_residuals[:,dof_idx::self.number_dofs_per_node]))

        # second compute the global jacobian matrix
        jacobian_data = jnp.ravel(elements_stiffness)
        jacobian_indices = jax.vmap(self.ComputeElementJacobianIndices)(self.fe_mesh.GetElementsNodes(self.element_type)) # Get the indices
        jacobian_indices = jacobian_indices.reshape(-1,2)

        sparse_jacobian = sparse.BCOO((jacobian_data,jacobian_indices),shape=(self.total_number_of_dofs,self.total_number_of_dofs))

        return sparse_jacobian, residuals_vector
