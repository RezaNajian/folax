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

class Hexahedra3D8(Geometry):

    @partial(jit, static_argnums=(0,))
    def GaussIntegration1(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        points = jnp.array([[0.0,0.0,0.0]])
        weights = jnp.array([8.0])
        return points, weights

    @partial(jit, static_argnums=(0,))
    def GaussIntegration2(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        points = jnp.array([[-1.00/jnp.sqrt(3.0) , -1.00/jnp.sqrt(3.0), -1.00/jnp.sqrt(3.0)],
                            [1.00/jnp.sqrt(3.0) , -1.00/jnp.sqrt(3.0), -1.00/jnp.sqrt(3.0)],
                            [1.00/jnp.sqrt(3.0) ,  1.00/jnp.sqrt(3.0), -1.00/jnp.sqrt(3.0)],
                            [-1.00/jnp.sqrt(3.0) ,  1.00/jnp.sqrt(3.0), -1.00/jnp.sqrt(3.0)],
                            [-1.00/jnp.sqrt(3.0) , -1.00/jnp.sqrt(3.0),  1.00/jnp.sqrt(3.0)],
                            [1.00/jnp.sqrt(3.0) , -1.00/jnp.sqrt(3.0),  1.00/jnp.sqrt(3.0)],
                            [1.00/jnp.sqrt(3.0) ,  1.00/jnp.sqrt(3.0),  1.00/jnp.sqrt(3.0)],
                            [-1.00/jnp.sqrt(3.0) ,  1.00/jnp.sqrt(3.0),  1.00/jnp.sqrt(3.0)]])
        weights = jnp.array([1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00])
        return points, weights

    @partial(jit, static_argnums=(0,))
    def GaussIntegration3(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        points = jnp.array([[-jnp.sqrt(3.00/5.00) , -jnp.sqrt(3.00/5.00), -jnp.sqrt(3.00/5.00)],
                            [0.0 , -jnp.sqrt(3.00/5.00), -jnp.sqrt(3.00/5.00)],
                            [jnp.sqrt(3.00/5.00) , -jnp.sqrt(3.00/5.00), -jnp.sqrt(3.00/5.00)],
                            [-jnp.sqrt(3.00/5.00) , 0.0, -jnp.sqrt(3.00/5.00)],
                            [0.0 , 0.0, -jnp.sqrt(3.00/5.00)],
                            [jnp.sqrt(3.00/5.00) ,0.0, -jnp.sqrt(3.00/5.00)],
                            [-jnp.sqrt(3.00/5.00) ,  jnp.sqrt(3.00/5.00), -jnp.sqrt(3.00/5.00)],
                            [0.0 ,  jnp.sqrt(3.00/5.00), -jnp.sqrt(3.00/5.00)],
                            [jnp.sqrt(3.00/5.00) ,  jnp.sqrt(3.00/5.00), -jnp.sqrt(3.00/5.00)],
                            [-jnp.sqrt(3.00/5.00) , -jnp.sqrt(3.00/5.00), 0.0],
                            [0.0 , -jnp.sqrt(3.00/5.00), 0.0],
                            [jnp.sqrt(3.00/5.00) , -jnp.sqrt(3.00/5.00),0.0],
                            [-jnp.sqrt(3.00/5.00) , 0.0, 0.0],
                            [0.0 , 0.0, 0.0],
                            [jnp.sqrt(3.00/5.00) , 0.0, 0.0],
                            [-jnp.sqrt(3.00/5.00) ,  jnp.sqrt(3.00/5.00),                   0.0],
                            [0.0 ,  jnp.sqrt(3.00/5.00),                   0.0],
                            [jnp.sqrt(3.00/5.00) ,  jnp.sqrt(3.00/5.00),                   0.0],
                            [-jnp.sqrt(3.00/5.00) , -jnp.sqrt(3.00/5.00),  jnp.sqrt(3.00/5.00)],
                            [0.0 , -jnp.sqrt(3.00/5.00),  jnp.sqrt(3.00/5.00)],
                            [jnp.sqrt(3.00/5.00) , -jnp.sqrt(3.00/5.00),  jnp.sqrt(3.00/5.00)],
                            [-jnp.sqrt(3.00/5.00) ,                   0.0,  jnp.sqrt(3.00/5.00)],
                            [0.0 ,                   0.0,  jnp.sqrt(3.00/5.00)],
                            [jnp.sqrt(3.00/5.00) ,                   0.0,  jnp.sqrt(3.00/5.00)],
                            [-jnp.sqrt(3.00/5.00) ,  jnp.sqrt(3.00/5.00),  jnp.sqrt(3.00/5.00)],
                            [0.0 ,  jnp.sqrt(3.00/5.00),  jnp.sqrt(3.00/5.00)],
                            [jnp.sqrt(3.00/5.00) ,  jnp.sqrt(3.00/5.00),  jnp.sqrt(3.00/5.00)]])
        weights = jnp.array([125.00/729.00,200.00/729.00,125.00/729.00,200.00/729.00,320.00/729.00,
                             200.00/729.00,125.00/729.00,200.00/729.00,125.00/729.00,200.00/729.00,
                             320.00/729.00,200.00/729.00,320.00/729.00,512.00/729.00,320.00/729.00,
                             200.00/729.00,320.00/729.00,200.00/729.00,125.00/729.00,200.00/729.00,
                             125.00/729.00,200.00/729.00,320.00/729.00,200.00/729.00,125.00/729.00,
                             200.00/729.00,125.00/729.00])
        return points, weights

    def GaussIntegration4(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError("Quadrilateral2D4::GaussIntegration4. is not implemented.")

    def GaussIntegration5(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError("Quadrilateral2D4::GaussIntegration5. is not implemented.")
    
    @partial(jit, static_argnums=(0,))
    def ShapeFunctionsValues(self,local_coordinates:jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.125*( 1.0 - local_coordinates[0] )*( 1.0 - local_coordinates[1] )*( 1.0 - local_coordinates[2] ),
                          0.125*( 1.0 + local_coordinates[0] )*( 1.0 - local_coordinates[1] )*( 1.0 - local_coordinates[2] ),
                          0.125*( 1.0 + local_coordinates[0] )*( 1.0 + local_coordinates[1] )*( 1.0 - local_coordinates[2] ),
                          0.125*( 1.0 - local_coordinates[0] )*( 1.0 + local_coordinates[1] )*( 1.0 - local_coordinates[2] ),
                          0.125*( 1.0 - local_coordinates[0] )*( 1.0 - local_coordinates[1] )*( 1.0 + local_coordinates[2] ),
                          0.125*( 1.0 + local_coordinates[0] )*( 1.0 - local_coordinates[1] )*( 1.0 + local_coordinates[2] ),
                          0.125*( 1.0 + local_coordinates[0] )*( 1.0 + local_coordinates[1] )*( 1.0 + local_coordinates[2] ),
                          0.125*( 1.0 - local_coordinates[0] )*( 1.0 + local_coordinates[1] )*( 1.0 + local_coordinates[2] )])

    @partial(jit, static_argnums=(0,))
    def ShapeFunctionsLocalGradients(self,local_coordinates:jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[-0.125 * ( 1.0 - local_coordinates[1] ) * ( 1.0 - local_coordinates[2] ),
                           -0.125 * ( 1.0 - local_coordinates[0] ) * ( 1.0 - local_coordinates[2] ),
                           -0.125 * ( 1.0 - local_coordinates[0] ) * ( 1.0 - local_coordinates[1] )],
                          [0.125 * ( 1.0 - local_coordinates[1] ) * ( 1.0 - local_coordinates[2] ),
                           -0.125 * ( 1.0 + local_coordinates[0] ) * ( 1.0 - local_coordinates[2] ),
                           -0.125 * ( 1.0 + local_coordinates[0] ) * ( 1.0 - local_coordinates[1] )],
                          [0.125 * ( 1.0 + local_coordinates[1] ) * ( 1.0 - local_coordinates[2] ),
                           0.125 * ( 1.0 + local_coordinates[0] ) * ( 1.0 - local_coordinates[2] ),
                           -0.125 * ( 1.0 + local_coordinates[0] ) * ( 1.0 + local_coordinates[1] )],
                            [-0.125 * ( 1.0 + local_coordinates[1] ) * ( 1.0 - local_coordinates[2] ),
                            0.125 * ( 1.0 - local_coordinates[0] ) * ( 1.0 - local_coordinates[2] ),
                            -0.125 * ( 1.0 - local_coordinates[0] ) * ( 1.0 + local_coordinates[1] )],
                            [-0.125 * ( 1.0 - local_coordinates[1] ) * ( 1.0 + local_coordinates[2] ),
                            -0.125 * ( 1.0 - local_coordinates[0] ) * ( 1.0 + local_coordinates[2] ),
                            0.125 * ( 1.0 - local_coordinates[0] ) * ( 1.0 - local_coordinates[1] )],
                            [0.125 * ( 1.0 - local_coordinates[1] ) * ( 1.0 + local_coordinates[2] ),
                            -0.125 * ( 1.0 + local_coordinates[0] ) * ( 1.0 + local_coordinates[2] ),
                            0.125 * ( 1.0 + local_coordinates[0] ) * ( 1.0 - local_coordinates[1] )],
                            [0.125 * ( 1.0 + local_coordinates[1] ) * ( 1.0 + local_coordinates[2] ),
                            0.125 * ( 1.0 + local_coordinates[0] ) * ( 1.0 + local_coordinates[2] ),
                            0.125 * ( 1.0 + local_coordinates[0] ) * ( 1.0 + local_coordinates[1] )],
                            [-0.125 * ( 1.0 + local_coordinates[1] ) * ( 1.0 + local_coordinates[2] ),
                                0.125 * ( 1.0 - local_coordinates[0] ) * ( 1.0 + local_coordinates[2] ),
                                0.125 * ( 1.0 - local_coordinates[0] ) * ( 1.0 + local_coordinates[1] )]])
    