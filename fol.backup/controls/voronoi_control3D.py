"""
 Authors: Kianoosh Taghikhani, https://github.com/kianoosh1989
 Date: October, 2024
 License: FOL/License.txt
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial
from fol.mesh_input_output.mesh import Mesh
from fol.tools.decoration_functions import *

class VoronoiControl3D(Control):
    """
    Voronoi-based parametric control in three dimensions.

    This control maps a vector of design variables to a nodal scalar field
    defined over a three-dimensional finite element mesh. The mapping is based
    on a Voronoi tessellation induced by a set of seed points in the
    three-dimensional space.

    The control variables represent:
    - x-coordinates of Voronoi seed points,
    - y-coordinates of Voronoi seed points,
    - z-coordinates of Voronoi seed points,
    - one scalar value associated with each seed.

    For each mesh node with coordinates ``(X_i, Y_i, Z_i)``, the controlled
    value is taken from the seed point that is closest to the node in Euclidean
    distance. This produces a piecewise-constant field over the mesh, with
    discontinuities along Voronoi cell boundaries.

    The number of control variables is four times the number of seeds, and the
    number of controlled variables equals the number of mesh nodes.

    Args:
        control_name (str):
            Name identifier for the control instance.
        control_settings (dict):
            Dictionary defining the Voronoi parameterization. Expected keys
            include ``"number_of_seeds"`` and ``"E_values"``. The entry
            ``"E_values"`` is validated during initialization but is not used
            directly in the control mapping.
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

        This method reads the number of Voronoi seeds from
        ``control_settings`` and sets the number of control variables and
        controlled variables. Initialization is performed once unless
        ``reinitialize`` is set to ``True``.

        The number of control variables is defined as:

        - ``4 * number_of_seeds`` (x-coordinates, y-coordinates, z-coordinates,
          and one scalar value per seed)

        The number of controlled variables equals the number of mesh nodes.

        Args:
            reinitialize (bool, optional):
                If ``True``, forces reinitialization even if already initialized.
                Default is ``False``.

        Returns:
            None

        Raises:
            ValueError:
                If ``control_settings["E_values"]`` is not a tuple or a list.
        """
        if self.initialized and not reinitialize:
            return

        self.number_of_seeds = self.settings["number_of_seeds"]
        if not isinstance(self.settings["E_values"],tuple) and not isinstance(self.settings["E_values"],list):
            raise(ValueError("'E values' should be either tuple or list"))
        self.E_values = self.settings["E_values"]
        # 4 stands for the following: x coordinates array, y, z, and E values
        self.num_control_vars = self.number_of_seeds * 4
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()

        self.initialized = True

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self, variable_vector: jnp.array):
        """
        Compute the nodal field induced by the 3D Voronoi control.

        The input control vector is interpreted as a concatenation of:

        - x-coordinates of the Voronoi seeds,
        - y-coordinates of the Voronoi seeds,
        - z-coordinates of the Voronoi seeds,
        - scalar values associated with each seed.

        For each mesh node coordinate ``(X_i, Y_i, Z_i)``, the nearest seed
        point is identified using Euclidean distance, and the corresponding
        seed value is assigned to the node.

        Args:
            variable_vector (jax.numpy.ndarray):
                One-dimensional control vector of length
                ``4 * number_of_seeds``.

        Returns:
            jax.numpy.ndarray:
                One-dimensional array of controlled values evaluated at mesh
                nodes, with length equal to the number of mesh nodes.

        Notes:
            This method does not explicitly validate the length of
            ``variable_vector`` and does not explicitly raise exceptions.
            If the input vector has an incompatible length, indexing or shape
            errors may occur during JAX tracing or execution.
        """
        x_coord = variable_vector[:self.number_of_seeds]
        y_coord = variable_vector[self.number_of_seeds:2 * self.number_of_seeds]
        z_coord = variable_vector[2 * self.number_of_seeds:3 * self.number_of_seeds]
        k_values = variable_vector[3 * self.number_of_seeds:]
        X = self.fe_mesh.GetNodesX()
        Y = self.fe_mesh.GetNodesY()
        Z = self.fe_mesh.GetNodesZ()
        seed_points = jnp.vstack((jnp.vstack((x_coord, y_coord)),z_coord)).T
        grid_points = jnp.vstack([jnp.vstack([X.ravel(), Y.ravel()]), Z.ravel()]).T
        K = jnp.zeros((self.num_controlled_vars))

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