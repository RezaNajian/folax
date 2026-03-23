"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: January, 2025
 License: FOL/LICENSE
"""
from .geometry import Geometry
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *
from typing import Tuple
import jax.numpy as jnp
from jax import jit
from functools import partial

class Tetrahedra3D4(Geometry):

    @partial(jit, static_argnums=(0,))
    def GaussIntegration1(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        points = jnp.array([[0.25,0.25,0.25]])
        weights = jnp.array([1.00 / 6.00])
        return points, weights

    @partial(jit, static_argnums=(0,))
    def GaussIntegration2(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        points = jnp.array([[0.58541020,0.13819660,0.13819660],
                            [0.13819660,0.58541020,0.13819660],
                            [0.13819660,0.13819660,0.58541020],
                            [0.13819660,0.13819660,0.13819660]])
        weights = jnp.array([1.00 / 24.00,1.00 / 24.00,1.00 / 24.00,1.00 / 24.00])
        return points, weights

    @partial(jit, static_argnums=(0,))
    def GaussIntegration3(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        points = jnp.array([[0.015835909865720057993,0.32805469671142664734,0.32805469671142664734],
                            [0.32805469671142664734,0.015835909865720057993,0.32805469671142664734],
                            [0.32805469671142664734,0.32805469671142664734,0.015835909865720057993],
                            [0.32805469671142664734,0.32805469671142664734,0.32805469671142664734],
                            [0.67914317820120795168,0.10695227393293068277,0.10695227393293068277],
                            [0.10695227393293068277,0.67914317820120795168,0.10695227393293068277],
                            [0.10695227393293068277,0.10695227393293068277,0.67914317820120795168],
                            [0.10695227393293068277,0.10695227393293068277,0.10695227393293068277]])
        weights = jnp.array([0.02308799441864369039,0.02308799441864369039,0.02308799441864369039,0.02308799441864369039,
                             0.01857867224802297628,0.01857867224802297628,0.01857867224802297628,0.01857867224802297628])
        return points, weights

    def GaussIntegration4(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError("Quadrilateral2D4::GaussIntegration4. is not implemented.")

    def GaussIntegration5(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError("Quadrilateral2D4::GaussIntegration5. is not implemented.")
    
    @partial(jit, static_argnums=(0,))
    def ShapeFunctionsValues(self,local_coordinates:jnp.ndarray) -> jnp.ndarray:
        return jnp.array([1.0-(local_coordinates[0]+local_coordinates[1]+local_coordinates[2]),
                          local_coordinates[0],
                          local_coordinates[1],
                          local_coordinates[2]])

    @partial(jit, static_argnums=(0,))
    def ShapeFunctionsLocalGradients(self,local_coordinates:jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[-1.0, -1.0, -1.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])
    