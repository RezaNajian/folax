import pytest
import unittest 
import os
import numpy as np
import jax.numpy as jnp
from fol.loss_functions.mechanical_elastoplasticity import ElastoplasticityLoss2DQuad
from fol.loss_functions.mechanical_elastoplasticity import ElastoplasticityLoss3DTetra
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *


class TestElastoplasticity(unittest.TestCase):
    """Unit tests for elastoplastic material behavior in 2D and 3D"""

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self, request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def test_tetra(self):
        """Test tetrahedral element computation"""
        
        tet_points_coordinates = jnp.array([
            [0.1, 0.1, 0.1],
            [0.28739360416666665, 0.27808503701741405, 0.05672979583333333],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.1]
        ])

        fe_mesh = Mesh("", ".")
        fe_mesh.node_ids = jnp.arange(len(tet_points_coordinates))
        fe_mesh.nodes_coordinates = tet_points_coordinates
        fe_mesh.elements_nodes = {"tetra": fe_mesh.node_ids.reshape(1, -1)}

        material_dict = {
            "young_modulus": 3.0,
            "poisson_ratio": 0.3,
            "iso_hardening_parameter_1": 0.4,
            "iso_hardening_param_2": 10.0,
            "yield_limit": 0.2
        }

        mechanical_loss_3d = ElastoplasticityLoss3DTetra(
            "mechanical_loss_3d",
            loss_settings={
                "dirichlet_bc_dict": {"Ux": {}, "Uy": {}, "Uz": {}},
                "material_dict": material_dict,
                "body_foce": jnp.array([[1], [2], [3]])
            },
            fe_mesh=fe_mesh
        )
        mechanical_loss_3d.Initialize()
        
        # Rigid body translation 
        displacements = jnp.ones((12, 1))
        
        elem_controls = jnp.array([1.0])
        
        # State: [ε_p_xx, ε_p_yy, ε_p_zz, ε_p_xy, ε_p_yz, ε_p_xz, internal_parameter]
        num_gp = 1
        state_size = 7
        element_state_gps = jnp.zeros((num_gp, state_size))

        en, gps_state, residuals, stiffness = mechanical_loss_3d.ComputeElement(
            tet_points_coordinates,
            elem_controls,
            displacements,
            element_state_gps
        )

        np.testing.assert_allclose(stiffness, jnp.array([
            [ 0.19664259,  0.05349432,  0.        , -0.2374955 , -0.03784438,  0.        ,  0.10276479,  0.01637534,  0.20824468, -0.06191189, -0.03202529, -0.20824468],
            [ 0.05349432,  0.08356421,  0.        , -0.04745976, -0.07738307,  0.        ,  0.02053594,  0.03348381,  0.08290199, -0.0265705 , -0.03966496, -0.08290199],
            [ 0.        ,  0.        ,  0.06226818,  0.        ,  0.        , -0.06997301,  0.13882978,  0.05526799,  0.03027747, -0.13882978, -0.05526799, -0.02257263],
            [-0.2374955 , -0.04745976,  0.        ,  0.29341802,  0.0232053 ,  0.        , -0.1269626 , -0.01004098, -0.25961536,  0.07104003,  0.03429545,  0.25961536],
            [-0.03784438, -0.07738307,  0.        ,  0.0232053 ,  0.08714877,  0.        , -0.01004098, -0.03770945, -0.02884615,  0.02468006,  0.02794375,  0.02884615],
            [ 0.        ,  0.        , -0.06997301,  0.        ,  0.        ,  0.08457041, -0.17307691, -0.01923077, -0.03659379,  0.17307691,  0.01923077,  0.0219964 ],
            [ 0.10276479,  0.02053594,  0.13882978, -0.12696259, -0.01004098, -0.1730769 ,  0.41351914,  0.00434475,  0.18722682, -0.38932133, -0.01483971, -0.1529797 ],
            [ 0.01637534,  0.03348381,  0.05526799, -0.01004098, -0.03770945, -0.01923077,  0.00434475,  0.37489912,  0.02080299, -0.01067911, -0.37067348, -0.05684022],
            [ 0.20824468,  0.08290199,  0.03027747, -0.25961536, -0.02884615, -0.03659379,  0.18722683,  0.02080299,  1.2708718 , -0.13585614, -0.07485884, -1.2645556 ],
            [-0.06191189, -0.0265705 , -0.13882978,  0.07104003,  0.02468006,  0.1730769 , -0.38932133, -0.01067911, -0.13585614,  0.38019317,  0.01256955,  0.10160901],
            [-0.03202529, -0.03966496, -0.05526799,  0.03429545,  0.02794376,  0.01923077, -0.01483971, -0.37067348, -0.07485884,  0.01256955,  0.38239467,  0.11089606],
            [-0.20824468, -0.08290199, -0.02257263,  0.25961536,  0.02884615,  0.0219964 , -0.1529797 , -0.05684022, -1.2645556 ,  0.10160901,  0.11089606,  1.2651317 ]
        ]), rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(residuals.flatten(), jnp.array([
            -0.00077693, -0.00155386, -0.00233078,
            -0.00077692, -0.00155385, -0.00233079,
            -0.00077693, -0.00155382, -0.00233079,
            -0.00077693, -0.00155389, -0.00233078
        ]), rtol=1e-5, atol=1e-5)

    def test_quad(self):
        """Test 2D quadrilateral element computation"""
        
        quad_points_coordinates = jnp.array([
            [3.00, 0.00, 0.00],
            [2.00, 0.75, 0.00],
            [0.75, 1.00, 0.00],
            [0.00, 0.00, 0.00]
        ])
        
        fe_mesh = Mesh("", ".")
        fe_mesh.node_ids = jnp.arange(len(quad_points_coordinates))
        fe_mesh.nodes_coordinates = quad_points_coordinates
        fe_mesh.elements_nodes = {"quad": fe_mesh.node_ids.reshape(1, -1)}

        material_dict = {
            "young_modulus": 3.0,
            "poisson_ratio": 0.3,
            "iso_hardening_parameter_1": 0.4,
            "iso_hardening_param_2": 10.0,
            "yield_limit": 0.2
        }

        mechanical_loss_2d = ElastoplasticityLoss2DQuad(
            "mechanical_loss_2d",
            loss_settings={
                "dirichlet_bc_dict": {"Ux": {}, "Uy": {}},
                "material_dict": material_dict,
                "body_foce": jnp.array([[1], [2]])
            },
            fe_mesh=fe_mesh
        )
        mechanical_loss_2d.Initialize()

        # Displacement field (Rigid body motion)
        displacements = jnp.ones((8, 1))

        # Controls: Scalar density
        elem_controls = jnp.array([1.0])
        
        # State: [ε_p_xx, ε_p_yy, ε_p_xy, internal_parameter]
        num_gp = 4 
        state_size = 4  
        element_state_gps = jnp.zeros((num_gp, state_size))

        en, gps_state, residuals, stiffness = mechanical_loss_2d.ComputeElement(
            quad_points_coordinates,
            elem_controls,
            displacements,
            element_state_gps
        )

        np.testing.assert_allclose(stiffness, jnp.array([
            [ 1.18619502e+00, -5.44507742e-01, -8.20018828e-01, -4.94042411e-03, -5.38831949e-01,  6.52743816e-01,  1.72655791e-01, -1.03295684e-01],
            [-5.44507742e-01,  2.85444641e+00, -2.93401986e-01, -3.54907894e+00,  6.52743816e-01, -1.37665224e+00,  1.85165882e-01,  2.07128477e+00],
            [-8.20018828e-01, -2.93402016e-01,  2.43876863e+00,  1.06382239e+00, -5.30999541e-01,  6.75179213e-02, -1.08775020e+00, -8.37938249e-01],
            [-4.94043157e-03, -3.54907894e+00,  1.06382239e+00,  5.43770981e+00, -2.20943674e-01,  4.64986712e-01, -8.37938190e-01, -2.35361767e+00],
            [-5.38831949e-01,  6.52743816e-01, -5.30999541e-01, -2.20943674e-01,  1.55209565e+00, -4.83330756e-01, -4.82264221e-01,  5.15305698e-02],
            [ 6.52743816e-01, -1.37665224e+00,  6.75178617e-02,  4.64986771e-01, -4.83330756e-01,  3.68972111e+00, -2.36930981e-01, -2.77805567e+00],
            [ 1.72655821e-01,  1.85165912e-01, -1.08775020e+00, -8.37938190e-01, -4.82264251e-01, -2.36930996e-01,  1.39735854e+00,  8.89703274e-01],
            [-1.03295669e-01,  2.07128477e+00, -8.37938190e-01, -2.35361767e+00,  5.15305698e-02, -2.77805567e+00,  8.89703274e-01,  3.06038880e+00]
        ]), rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(residuals.flatten(), jnp.array([
            -0.4947916 , -0.98958325, -0.3645833 , -0.7291667 ,
            -0.42708334, -0.8541666 , -0.5572916 , -1.1145833
        ]), rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()