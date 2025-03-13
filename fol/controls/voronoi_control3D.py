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
    def __init__(self,control_name: str,control_settings, fe_mesh: Mesh):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:

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