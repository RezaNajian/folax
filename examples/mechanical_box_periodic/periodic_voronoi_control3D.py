"""Periodic three-dimensional Voronoi material control."""

from functools import partial

import jax.numpy as jnp
from jax import jit, vmap

from fol.controls.control import Control
from fol.tools.decoration_functions import print_with_timestamp_and_execution_time


class VoronoiControl3D(Control):
    """Map periodic 3-D Voronoi seeds and their values to a nodal field.

    A control vector contains four consecutive blocks, each with
    ``number_of_seeds`` entries: seed x-, y-, and z-coordinates followed by
    the scalar material value assigned to each seed. Every mesh node receives
    the value of its nearest seed using distance on the periodic 3-D box.
    """

    def __init__(self, control_name, control_settings, fe_mesh):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh

    @print_with_timestamp_and_execution_time
    def Initialize(self, reinitialize=False):
        if self.initialized and not reinitialize:
            return

        self.number_of_seeds = int(self.settings["number_of_seeds"])
        if self.number_of_seeds < 1:
            raise ValueError("number_of_seeds must be positive.")

        field_value_range = self.settings["E_values"]
        if not isinstance(field_value_range, tuple) or len(field_value_range) != 2:
            raise ValueError(
                "E_values must be a (minimum, maximum) tuple for continuous "
                "random grain values."
            )
        if field_value_range[0] > field_value_range[1]:
            raise ValueError("E_values minimum must not exceed its maximum.")
        self.E_values = field_value_range

        for length_name in ("Lx", "Ly", "Lz"):
            if self.settings[length_name] <= 0.0:
                raise ValueError(f"{length_name} must be positive.")

        self.num_control_vars = 4 * self.number_of_seeds
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
        self.initialized = True

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self, variable_vector):
        number_of_seeds = self.number_of_seeds
        x_coordinates = variable_vector[:number_of_seeds]
        y_coordinates = variable_vector[number_of_seeds : 2 * number_of_seeds]
        z_coordinates = variable_vector[2 * number_of_seeds : 3 * number_of_seeds]
        field_values = variable_vector[3 * number_of_seeds :]

        box_lengths = jnp.asarray(
            (self.settings["Lx"], self.settings["Ly"], self.settings["Lz"])
        )
        grid_points = jnp.mod(self.fe_mesh.GetNodesCoordinates(), box_lengths)
        seed_points = jnp.mod(
            jnp.column_stack((x_coordinates, y_coordinates, z_coordinates)),
            box_lengths,
        )

        def assign_nearest_seed_value(grid_point):
            coordinate_distances = jnp.abs(grid_point - seed_points)
            periodic_distances = jnp.minimum(
                coordinate_distances, box_lengths - coordinate_distances
            )
            squared_distances = jnp.sum(periodic_distances**2, axis=1)
            return field_values[jnp.argmin(squared_distances)]

        return vmap(assign_nearest_seed_value)(grid_points)

    @print_with_timestamp_and_execution_time
    def Finalize(self):
        pass
