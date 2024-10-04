"""
 Authors: Kianoosh Taghikhani, https://github.com/kianoosh1989
 Date: July, 2024
 License: FOL/LICENSE
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,pure_callback,ShapeDtypeStruct
from scipy.spatial import Voronoi, KDTree
import numpy as np
from functools import partial
from fol.mesh_input_output.mesh import Mesh
from fol.tools.decoration_functions import *

class VoronoiControl(Control):

    def __init__(self,control_name: str,control_settings, fe_mesh: Mesh):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh

    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:
        self.numberof_seeds = self.settings["numberof_seeds"]
        if isinstance(self.settings["k_rangeof_values"],tuple):
            start, end = self.settings["k_rangeof_values"]
            self.k_rangeof_values = range(start,end)
        if isinstance(self.settings["k_rangeof_values"],list):
            self.k_rangeof_values = list(self.settings["k_rangeof_values"])

        # The number 3 stands for the following: x coordinates array, y coordinates array, and K values
        self.num_control_vars = self.numberof_seeds * 3 
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
    
    def compute_K_host(self,x_coord,y_coord,k_values):
        N = int(self.num_controlled_vars**0.5)
        x = jnp.linspace(0, 1, N)
        y = jnp.linspace(0, 1, N)
        X, Y = jnp.meshgrid(x, y)
        K = np.zeros((N**2))
        seed_points = np.vstack((x_coord,y_coord)).T.astype(float)
        if seed_points.shape[0] < 4:
            raise ValueError("At least 4 seed points are required to create a Voronoi diagram.")
        if x_coord.shape[-1] != self.numberof_seeds or y_coord.shape[-1] != self.numberof_seeds or k_values.shape[-1] != self.numberof_seeds:
            raise ValueError("Number of coordinates should be equal to number of seed points!")
        
        # Add a small perturbation to the seed points to avoid coplanar issues
        seed_points += np.random.normal(scale=1e-8, size=seed_points.shape)
        vor = Voronoi(seed_points, qhull_options='QJ')
        tree = KDTree(seed_points)
        grid_points = np.vstack([X.ravel(),Y.ravel()]).T
        # Find the nearest seed point for each grid point
        _, regions = tree.query(grid_points)
        # Assign the feature value based on the nearest seed point
        for i, region in enumerate(regions):
            K.ravel()[i] =  k_values[region]
        return K
    
    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self,variable_vector:jnp.array):
        variable_vector = variable_vector.reshape(1,-1)
        N = int(self.num_controlled_vars**0.5)
        x_coord = variable_vector[1,:self.numberof_seeds]
        y_coord = variable_vector[1,self.numberof_seeds:2*self.numberof_seeds]
        k_values = variable_vector[1,2*self.numberof_seeds:]
        K = jnp.zeros((N**2))
        result_shape = ShapeDtypeStruct(K.shape, K.dtype)
        return pure_callback(self.compute_K_host, result_shape,x_coord,y_coord,k_values)
    
    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass