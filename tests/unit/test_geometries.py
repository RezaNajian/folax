import pytest
import unittest
import os
import numpy as np
from fol.geometries.quadrilateral_2d_4 import Quadrilateral2D4
from fol.tools.usefull_functions import *
import jax

class TestGeometries(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        self.test_quad = Quadrilateral2D4("test_quadrilateral2D4")
        self.points_coordinates = jnp.array([[50.0,0.0,0.0],
                                             [51.0,0.0,0.0],
                                             [51.0,51.0,0.0],
                                             [50.0,51.0,0.0]])

    def test_quad2D4(self):

        points,weights = self.test_quad.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[0., 0., 0.]]), rtol=1e-5, atol=1e-10)
        np.testing.assert_allclose(weights,jnp.array([4]), rtol=1e-5, atol=1e-10)
        shape_function_values = jax.vmap(self.test_quad.ShapeFunctionsValues)(points)
        np.testing.assert_allclose(shape_function_values[0],jnp.array([0.25, 0.25, 0.25, 0.25]), rtol=1e-5, atol=1e-10)
        shape_function_grads = jax.vmap(self.test_quad.ShapeFunctionsGradients)(points)
        np.testing.assert_allclose(shape_function_grads[0],jnp.array([[-0.25,-0.25],[0.25,-0.25],[0.25,0.25],
                                                                      [-0.25,0.25]]), rtol=1e-5, atol=1e-10)
        jacobian = jax.vmap(self.test_quad.Jacobian, in_axes=(None, 0))(self.points_coordinates,points)
        np.testing.assert_allclose(jacobian,jnp.array([[[0.5,0],[0,25.5]]]), rtol=1e-5, atol=1e-10)
        
        self.test_quad.SetGaussIntegrationMethod("GI_GAUSS_2")
        points,weights = self.test_quad.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[-0.57735026, -0.57735026, 0.00],
                                                    [ 0.57735026, -0.57735026, 0.00],
                                                    [ 0.57735026,  0.57735026, 0.00],
                                                    [-0.57735026,  0.57735026, 0.00]]), rtol=1e-5, atol=1e-10)
        np.testing.assert_allclose(weights,jnp.array([1, 1, 1, 1]), rtol=1e-5, atol=1e-10)
        shape_function_values = jax.vmap(self.test_quad.ShapeFunctionsValues)(points)
        np.testing.assert_allclose(shape_function_values,jnp.array([[0.622008,0.166667,0.0446582,0.166667],
                                                                    [0.166667,0.622008,0.166667,0.0446582],
                                                                    [0.0446582,0.166667,0.622008,0.166667],
                                                                    [0.166667,0.0446582,0.166667,0.622008]]), 
                                                                    rtol=1e-5, atol=1e-10)
        shape_function_grads = jax.vmap(self.test_quad.ShapeFunctionsGradients)(points)
        np.testing.assert_allclose(shape_function_grads,jnp.array([[[-0.394338,-0.394338],[0.394338,-0.105662],[0.105662,0.105662],[-0.105662,0.394338]],
                                                                   [[-0.394338,-0.105662],[0.394338,-0.394338],[0.105662,0.394338],[-0.105662,0.105662]],
                                                                   [[-0.105662,-0.105662],[0.105662,-0.394338],[0.394338,0.394338],[-0.394338,0.105662]],
                                                                   [[-0.105662,-0.394338],[0.105662,-0.105662],[0.394338,0.105662],[-0.394338,0.394338]]]), rtol=1e-5, atol=1e-10)
        jacobian = jax.vmap(self.test_quad.Jacobian, in_axes=(None, 0))(self.points_coordinates,points)
        np.testing.assert_allclose(jacobian,jnp.array([[[0.5,0.0],[0.0,25.5]],
                                                       [[0.5,0.0],[0.0,25.5]],
                                                       [[0.5,0.0],[0.0,25.5]],
                                                       [[0.5,0.0],[0.0,25.5]]]), rtol=1e-5, atol=1e-5)

        self.test_quad.SetGaussIntegrationMethod("GI_GAUSS_3")
        points,weights = self.test_quad.GetIntegrationData()
        np.testing.assert_allclose(points,jnp.array([[-0.774597 , -0.774597 , 0],
                                                    [0 , -0.774597 , 0],
                                                    [0.774597 , -0.774597 , 0],
                                                    [-0.774597 , 0 , 0],
                                                    [0 , 0 , 0],
                                                    [0.774597 , 0 , 0],
                                                    [-0.774597 , 0.774597 , 0],
                                                    [0 , 0.774597 , 0],
                                                    [0.774597 , 0.774597 , 0]]), rtol=1e-5, atol=1e-10)
        np.testing.assert_allclose(weights,jnp.array([0.308642,0.493827,0.308642,
                                                      0.493827,0.790123,0.493827,
                                                      0.308642,0.493827,0.308642]), rtol=1e-5, atol=1e-10)
        shape_function_values = jax.vmap(self.test_quad.ShapeFunctionsValues)(points)
        np.testing.assert_allclose(shape_function_values,jnp.array([[0.787298,0.1,0.0127017,0.1],
                                                                    [0.443649,0.443649,0.0563508,0.0563508],
                                                                    [0.1,0.787298,0.1,0.0127017],
                                                                    [0.443649,0.0563508,0.0563508,0.443649],
                                                                    [0.25,0.25,0.25,0.25],
                                                                    [0.0563508,0.443649,0.443649,0.0563508],
                                                                    [0.1,0.0127017,0.1,0.787298],
                                                                    [0.0563508,0.0563508,0.443649,0.443649],
                                                                    [0.0127017,0.1,0.787298,0.1]]), 
                                                                    rtol=1e-5, atol=1e-10)
        
        shape_function_grads = jax.vmap(self.test_quad.ShapeFunctionsGradients)(points)
        np.testing.assert_allclose(shape_function_grads,jnp.array([[[-0.443649,-0.443649],[0.443649,-0.0563508],[0.0563508,0.0563508],[-0.0563508,0.443649]],
                                                                      [[-0.443649,-0.25],[0.443649,-0.25],[0.0563508,0.25],[-0.0563508,0.25]],
                                                                      [[-0.443649,-0.0563508],[0.443649,-0.443649],[0.0563508,0.443649],[-0.0563508,0.0563508]],
                                                                      [[-0.25,-0.443649],[0.25,-0.0563508],[0.25,0.0563508],[-0.25,0.443649]],
                                                                      [[-0.25,-0.25],[0.25,-0.25],[0.25,0.25],[-0.25,0.25]],
                                                                      [[-0.25,-0.0563508],[0.25,-0.443649],[0.25,0.443649],[-0.25,0.0563508]],
                                                                      [[-0.0563508,-0.443649],[0.0563508,-0.0563508],[0.443649,0.0563508],[-0.443649,0.443649]],
                                                                      [[-0.0563508,-0.25],[0.0563508,-0.25],[0.443649,0.25],[-0.443649,0.25]],
                                                                      [[-0.0563508,-0.0563508],[0.0563508,-0.443649],[0.443649,0.443649],[-0.443649,0.0563508]]]), rtol=1e-5, atol=1e-10)

        jacobians = jax.vmap(self.test_quad.Jacobian, in_axes=(None, 0))(self.points_coordinates,points)
        np.testing.assert_allclose(jacobians,jnp.array([[[0.5,0.0],[0.0,25.5]],
                                                        [[0.5,0.0],[0.0,25.5]],
                                                        [[0.5,0.0],[0.0,25.5]],
                                                        [[0.5,0.0],[0.0,25.5]],
                                                        [[0.5,0.0],[0.0,25.5]],
                                                        [[0.5,0.0],[0.0,25.5]],
                                                        [[0.5,0.0],[0.0,25.5]],
                                                        [[0.5,0.0],[0.0,25.5]],
                                                        [[0.5,0.0],[0.0,25.5]]]), rtol=1e-5, atol=1e-5)

        with self.assertRaises(NotImplementedError):
            self.test_quad.SetGaussIntegrationMethod("GI_GAUSS_4")
            points,weights = self.test_quad.GetIntegrationData()
        with self.assertRaises(NotImplementedError):
            self.test_quad.SetGaussIntegrationMethod("GI_GAUSS_5")
            points,weights = self.test_quad.GetIntegrationData()
        # print(np.array2string(weights, separator=', '))

if __name__ == '__main__':
    unittest.main()