"""
 Authors: Kianoosh Taghikhani, https://github.com/kianoosh1989
 Date: October, 2024
 License: FOL/LICENSE
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,vmap
import numpy as np
from functools import partial
from fol.mesh_input_output.mesh import Mesh
from fol.tools.decoration_functions import *

class VoronoiControl2D(Control):
    """
    Voronoi-based parametric control in two dimensions.

    This control maps a vector of control variables to a nodal scalar field
    using a Voronoi tessellation defined by a set of seed points in the
    two-dimensional domain. Each mesh node is assigned the value associated
    with the nearest seed point, measured by Euclidean distance in the
    (x, y) plane.

    The control variables consist of:
    - x-coordinates of the Voronoi seed points,
    - y-coordinates of the Voronoi seed points,
    - scalar values associated with each seed.

    For each mesh node, the controlled value is selected from the seed whose
    coordinates are closest to the node coordinates. This produces a
    piecewise-constant field over the mesh, with discontinuities along Voronoi
    cell boundaries.

    The number of control variables is three times the number of seeds, and
    the number of controlled variables equals the number of mesh nodes.

    Args:
        control_name (str):
            Name identifier for the control instance.
        control_settings (dict):
            Dictionary defining the Voronoi parameterization. Required entries
            include ``"number_of_seeds"`` and ``"E_values"``. The length of
            ``"E_values"`` must match ``number_of_seeds``.
        fe_mesh (Mesh):
            Finite element mesh providing nodal coordinates.
    """
    def __init__(self,control_name: str,control_settings, fe_mesh: Mesh):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize Voronoi seed configuration and control dimensions.

        This method reads the number of Voronoi seeds and their associated
        values from ``control_settings`` and sets the number of control and
        controlled variables. Initialization is performed once unless
        ``reinitialize`` is set to ``True``.

        Args:
            reinitialize (bool, optional):
                If ``True``, forces reinitialization even if already initialized.
                Default is ``False``.

        Returns:
            None

        Raises:
            ValueError:
                If ``E_values`` is not provided as a list or tuple, or if its
                length is incompatible with ``number_of_seeds``.
        """
        if self.initialized and not reinitialize:
            return

        self.number_of_seeds = self.settings["number_of_seeds"]
        if not isinstance(self.settings["E_values"],tuple) and not isinstance(self.settings["E_values"],list):
            raise(ValueError("'E values' should be either tuple or list"))
        self.E_values = self.settings["E_values"]

        # number 3 stands for the following: x coordinates array, y coordinates array, and K values
        self.num_control_vars = self.number_of_seeds * 3
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()

        self.initialized = True

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self, variable_vector: jnp.array):
        """
        Compute the nodal field induced by the Voronoi control.

        This method interprets the input control vector as concatenated arrays
        of seed x-coordinates, seed y-coordinates, and seed values. For each
        mesh node, the nearest seed point is identified using Euclidean
        distance, and the corresponding seed value is assigned to the node.

        Args:
            variable_vector (jax.numpy.ndarray):
                One-dimensional control vector of length
                ``3 * number_of_seeds``. The first third contains x-coordinates
                of the seeds, the second third contains y-coordinates, and the
                final third contains the scalar values associated with each
                seed.

        Returns:
            jax.numpy.ndarray:
                One-dimensional array of controlled values evaluated at mesh
                nodes, with length equal to the number of mesh nodes.

        """
        x_coord = variable_vector[:self.number_of_seeds]
        y_coord = variable_vector[self.number_of_seeds:2 * self.number_of_seeds]
        k_values = variable_vector[2 * self.number_of_seeds:]
        X = self.fe_mesh.GetNodesX()
        Y = self.fe_mesh.GetNodesY()
        K = jnp.zeros((self.num_controlled_vars))
        seed_points = jnp.vstack((x_coord, y_coord)).T
        grid_points = jnp.vstack([X.ravel(), Y.ravel()]).T

        # Calculate Euclidean distance between each grid point and each seed point
        def euclidean_distance(grid_point, seed_points):
            return jnp.sqrt(jnp.sum((grid_point - seed_points) ** 2, axis=1))

        # Iterate over grid points and assign the value from the nearest seed point
        def assign_value_to_grid(grid_point):
            distances = euclidean_distance(grid_point, seed_points)
            nearest_seed_idx = jnp.argmin(distances)
            return k_values[nearest_seed_idx]
        assign_value_to_grid_vmap_compatible = vmap(assign_value_to_grid,in_axes= 0)(grid_points)
        K = assign_value_to_grid_vmap_compatible
        return K

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass