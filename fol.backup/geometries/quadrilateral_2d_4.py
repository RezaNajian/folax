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

class Quadrilateral2D4(Geometry):

    @partial(jit, static_argnums=(0,))
    def GaussIntegration1(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        points = jnp.array([[0.00,0.00,0.00]])
        weights = jnp.array([4.00])
        return points, weights

    @partial(jit, static_argnums=(0,))
    def GaussIntegration2(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        points = jnp.array([[-1.00/jnp.sqrt(3.0),-1.00/jnp.sqrt(3.0),0.00],
                           [1.00/jnp.sqrt(3.0),-1.00/jnp.sqrt(3.0),0.00],
                           [1.00/jnp.sqrt(3.0),1.00/jnp.sqrt(3.0),0.00],
                           [-1.00/jnp.sqrt(3.0),1.00/jnp.sqrt(3.0),0.00]])
        weights = jnp.array([1.00,1.00,1.00,1.00])
        return points, weights

    @partial(jit, static_argnums=(0,))
    def GaussIntegration3(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        points = jnp.array([[-jnp.sqrt(3.00/5.00),-jnp.sqrt(3.00/5.00),0.00],
                           [0.00 , -jnp.sqrt(3.00/5.00),0.00],
                           [jnp.sqrt(3.00/5.00),-jnp.sqrt(3.00/5.00),0.00],
                           [-jnp.sqrt(3.00/5.00),0.00,0.00],
                           [0.00 , 0.00,0.00],
                           [jnp.sqrt(3.00/5.00),0.00,0.00],
                           [-jnp.sqrt(3.00/5.00),jnp.sqrt(3.00/5.00),0.00],
                           [0.00, jnp.sqrt(3.00/5.00),0.00],
                           [jnp.sqrt(3.00/5.00), jnp.sqrt(3.00/5.00),0.00]])
        weights = jnp.array([25.00/81.00,40.00/81.00,25.00/81.00,
                             40.00/81.00,64.00/81.00,40.00/81.00,
                             25.00/81.00,40.00/81.00,25.00/81.00])
        return points, weights

    def GaussIntegration4(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError("Quadrilateral2D4::GaussIntegration4. is not implemented.")

    def GaussIntegration5(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError("Quadrilateral2D4::GaussIntegration5. is not implemented.")
    
    @partial(jit, static_argnums=(0,))
    def ShapeFunctionsValues(self,local_coordinates:jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.25 * (1.0 - local_coordinates[0]) * (1.0 - local_coordinates[1]),
                          0.25 * (1.0 + local_coordinates[0]) * (1.0 - local_coordinates[1]),
                          0.25 * (1.0 + local_coordinates[0]) * (1.0 + local_coordinates[1]),
                          0.25 * (1.0 - local_coordinates[0]) * (1.0 + local_coordinates[1])])

    @partial(jit, static_argnums=(0,))
    def ShapeFunctionsLocalGradients(self,local_coordinates:jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[-0.25 * ( 1.0 - local_coordinates[1] ),-0.25 * ( 1.0 - local_coordinates[0] )],
                          [ 0.25 * ( 1.0 - local_coordinates[1] ),-0.25 * ( 1.0 + local_coordinates[0] )],
                          [ 0.25 * ( 1.0 + local_coordinates[1] ), 0.25 * ( 1.0 + local_coordinates[0] )],
                          [-0.25 * ( 1.0 + local_coordinates[1] ), 0.25 * ( 1.0 - local_coordinates[0] )]])
    
    @partial(jit, static_argnums=(0,))
    def Jacobian(self,points_coordinates:jnp.ndarray,local_coordinates:jnp.ndarray) -> jnp.ndarray:
        dN_dxi = self.ShapeFunctionsLocalGradients(local_coordinates.flatten())
        return jnp.dot(dN_dxi.T, points_coordinates[:,0:2]).T
    