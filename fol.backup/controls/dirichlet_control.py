"""
 Authors: Kianoosh Taghikhani, https://github.com/Kianoosh1989
 Date: August, 2025
 License: FOL/LICENSE
"""
from  fol.controls.control import Control
import jax.numpy as jnp
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.fe_loss import FiniteElementLoss
from fol.tools.decoration_functions import *

class DirichletControl(Control):
    """
    Dirichlet boundary-condition control for finite element problems.

    This control maps a user-defined set of *learnable* Dirichlet boundary DOFs
    to the global Dirichlet arrays owned by a :class:`fol.loss_functions.fe_loss.FiniteElementLoss`.
    It provides a JAX-friendly mechanism to overwrite selected Dirichlet values
    from a vector of control variables (e.g., during optimization or inverse problems).

    The control settings define which DOF components (e.g., ``"Ux"``, ``"Uy"``, ``"T"``)
    are controlled on which boundary node sets. During :meth:`Initialize`, the class:

    1. Ensures the associated FE loss is initialized.
    2. Reads the global Dirichlet indices and values from the FE loss.
    3. Builds a flattened index map (into the FE loss Dirichlet arrays) for the
       subset of Dirichlet DOFs that are controlled.
    4. Stores segment offsets and sizes so the control vector can be broadcast
       to the corresponding controlled DOFs efficiently.

    Args:
        control_name (str):
            Name identifier for the control instance.
        control_settings (dict):
            Control configuration dictionary. Must contain the key
            ``"learning_boundary"`` mapping DOF names to a list of boundary node-set
            names. Example::

                {
                    "learning_boundary": {
                        "Ux": ["left", "right"],
                        "Uy": ["top"]
                    }
                }

        fe_mesh (Mesh):
            Finite element mesh used to resolve boundary node sets via
            :meth:`fol.mesh_input_output.mesh.Mesh.GetNodeSet`.
        fe_loss (FiniteElementLoss):
            FE loss providing the global Dirichlet index/value arrays and the DOF
            ordering used to compute DOF indices per node.

    Attributes:
        settings (dict):
            Control configuration dictionary.
        fe_mesh (Mesh):
            Mesh reference used for boundary node set queries.
        loss_function (FiniteElementLoss):
            Associated FE loss object.
        dirichlet_values (jax.numpy.ndarray):
            Copy of the FE loss Dirichlet values array (one value per Dirichlet DOF).
        dirichlet_indices (jax.numpy.ndarray):
            Copy of the FE loss Dirichlet global DOF indices.
        learning_dirichlet_indices (jax.numpy.ndarray):
            Indices into ``dirichlet_indices`` / ``dirichlet_values`` for the subset
            of Dirichlet DOFs that are controlled.
        learning_dirichlet_starts (jax.numpy.ndarray):
            Segment start offsets for each (dof, boundary) block appended into
            ``learning_dirichlet_indices`` during initialization.
        learning_dirichlet_sizes (jax.numpy.ndarray):
            Segment sizes (number of controlled Dirichlet DOFs) for each (dof, boundary)
            block appended into ``learning_dirichlet_indices``.
        num_control_vars (int):
            Number of DOF components listed in ``control_settings["learning_boundary"]``.
        num_controlled_vars (int):
            Total number of individual Dirichlet DOFs affected by this controller.
        initialized (bool):
            Whether :meth:`Initialize` has been executed.

    Notes:
        - The control vector passed to :meth:`ComputeControlledVariables` is expanded
          by repeating each control entry according to ``learning_dirichlet_sizes``.
          This implies the control vector is interpreted as one scalar per controlled
          (dof, boundary) block, broadcast to all nodes in that block.
        - This class does not modify the FE loss object directly. It returns an updated
          Dirichlet value array that can be used by downstream code.
    """

    def __init__(self,control_name: str,control_settings: dict, fe_mesh: Mesh,fe_loss:FiniteElementLoss):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh
        self.loss_function = fe_loss

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize index mappings for learnable Dirichlet DOFs.

        This method constructs the internal flattened index structures that map
        controlled Dirichlet DOFs to entries of the FE loss Dirichlet arrays.

        It populates:
        - ``learning_dirichlet_indices``: flattened indices into the FE loss Dirichlet list
        - ``learning_dirichlet_starts``: start offsets per (dof, boundary) block
        - ``learning_dirichlet_sizes``: sizes per (dof, boundary) block

        Args:
            reinitialize (bool, optional):
                If ``True``, forces rebuilding the mappings even if already initialized.
                Defaults to ``False``.

        Returns:
            None

        Raises:
            KeyError:
                If ``"learning_boundary"`` is missing from ``control_settings``.
            ValueError:
                If a DOF name in ``"learning_boundary"`` is not present in
                ``fe_loss.GetDOFs()``.
        """

        if self.initialized and not reinitialize:
            return

        self.loss_function.Initialize()

        self.dirichlet_values = self.loss_function.dirichlet_values
        self.dirichlet_indices = self.loss_function.dirichlet_indices

        self.learning_dirichlet_starts = []
        self.learning_dirichlet_sizes = []
        self.learning_dirichlet_indices = jnp.array([], dtype=jnp.int32)
        for learning_dof,learning_boundaryies in self.settings["learning_boundary"].items():
            dof_index = self.loss_function.GetDOFs().index(learning_dof)
            for learning_boundary_name in learning_boundaryies:
                learning_boundary_node_ids = jnp.asarray(self.fe_mesh.GetNodeSet(learning_boundary_name))
                learning_dirichlet_bc_indices = self.loss_function.number_dofs_per_node*learning_boundary_node_ids + dof_index
                learning_dirichlet_bc_idx = jnp.where(self.dirichlet_indices[:, None] == learning_dirichlet_bc_indices)[0]
                self.learning_dirichlet_starts.append(self.learning_dirichlet_indices.size)
                self.learning_dirichlet_sizes.append(learning_dirichlet_bc_indices.size)
                self.learning_dirichlet_indices = jnp.hstack([self.learning_dirichlet_indices,learning_dirichlet_bc_idx])

        self.learning_dirichlet_starts = jnp.array(self.learning_dirichlet_starts)
        self.learning_dirichlet_sizes = jnp.array(self.learning_dirichlet_sizes)

        self.num_control_vars = len(self.settings["learning_boundary"].keys())
        self.num_controlled_vars = self.learning_dirichlet_indices.size
        self.initialized = True

    def ComputeControlledVariables(self,variable_vector:jnp.array):
        """
        Compute an updated Dirichlet value array from control variables.

        The returned array has the same shape as ``dirichlet_values``. Only the
        entries corresponding to ``learning_dirichlet_indices`` are overwritten.

        Broadcasting rule:
            Each entry in ``variable_vector`` is repeated according to
            ``learning_dirichlet_sizes`` and assigned to the corresponding segment
            in ``learning_dirichlet_indices``.

        Args:
            variable_vector (jax.numpy.ndarray):
                Control variables to assign. Expected to be a 1D array whose length
                matches the number of (dof, boundary) blocks created during
                initialization (i.e., ``len(learning_dirichlet_sizes)``).

        Returns:
            jax.numpy.ndarray:
                Updated Dirichlet values array with controlled entries overwritten.

        Raises:
            ValueError:
                If ``Initialize`` has not been called, or if ``variable_vector`` has
                an incompatible length for the configured segments.
        """
        dirichlet_values = jnp.copy(self.dirichlet_values)
        values_per_index = jnp.repeat(variable_vector, self.learning_dirichlet_sizes)
        return dirichlet_values.at[self.learning_dirichlet_indices].set(values_per_index)

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        """
        Finalize the control.

        This method is provided for API consistency with the base
        :class:`fol.controls.control.Control`. The default implementation performs
        no action.

        Returns:
            None
        """
        pass
