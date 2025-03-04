import pytest
import unittest 
import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss2DQuad
from fol.controls.fourier_control import FourierControl
from fol.responses.fe_response import FiniteElementResponse
from fol.tools.usefull_functions import *
import jax.numpy as jnp
import jax

class TestSA(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def test_quad(self):
                
        model_settings = {"L":1,"N":3,
                            "Ux_left":0.0,"Ux_right":0.05,
                            "Uy_left":0.0,"Uy_right":0.05}

        fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

        bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
                "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}        

        material_dict = {"young_modulus":1,"poisson_ratio":0.3}
        mechanical_loss_2d = MechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                    "num_gp":2,
                                                                                    "material_dict":material_dict},
                                                                                    fe_mesh=fe_mesh)
        fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                    "beta":20,"min":1e-1,"max":1}
        fourier_control = FourierControl("E",fourier_control_settings,fe_mesh)


        test_response = FiniteElementResponse("test_response",response_formula="(E**2)*U[0]",fe_loss=mechanical_loss_2d,control=fourier_control)

        fe_mesh.Initialize()
        mechanical_loss_2d.Initialize()
        fourier_control.Initialize()
        test_response.Initialize()

        key = jax.random.PRNGKey(42)  
        random_FE_UV = jax.random.uniform(key, shape=(2*fe_mesh.GetNumberOfNodes(),), minval=0.0, maxval=1.0) 
        random_K = jax.random.uniform(key, shape=(fe_mesh.GetNumberOfNodes(),), minval=0.0, maxval=1.0)  

        BC_applied_jac,BC_applied_rhs = test_response.ComputeAdjointJacobianMatrixAndRHSVector(random_K,random_FE_UV)


        np.testing.assert_allclose(BC_applied_jac.todense()[8,:],jnp.array([-1.34763107e-01, -9.90961790e-02,  6.18437380e-02,
                                                                            -1.36789493e-03, -1.44184157e-01,  1.05333894e-01,
                                                                            -3.05576682e-01,  4.32263222e-03,  1.07536995e+00,
                                                                            9.85294580e-03, -3.33979428e-01, -5.33044338e-04,
                                                                            -1.15071647e-01,  8.76187980e-02,  5.15497029e-02,
                                                                            7.43124075e-03, -1.55188382e-01, -1.13562390e-01]),
                                                                            rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(BC_applied_rhs.flatten(),jnp.array([-0., -0.,-0.04356432,-0.,-0.,
                                                                        -0.,-0.,-0.,-0.07417537,-0.,
                                                                        -0.,-0.,-0.,-0.,-0.04579198,
                                                                        -0.,-0.,-0.]),
                                                                        rtol=1e-5, atol=1e-5)

        # test sensitivities
        key = jax.random.PRNGKey(65)  
        random_adj_FE_UV = jax.random.uniform(key, shape=(2*fe_mesh.GetNumberOfNodes(),), minval=0.0, maxval=1.0)

        control_derivatives = test_response.ComputeAdjointNodalControlDerivatives(random_K,random_FE_UV,random_adj_FE_UV)
        shape_derivatives = test_response.ComputeAdjointNodalShapeDerivatives(random_K,random_FE_UV,random_adj_FE_UV)

        np.testing.assert_allclose(control_derivatives.flatten(),jnp.array([0.01678335, 0.05183903, 0.03314116, 0.07139717, 0.11391172,
                                                                            0.00820228, 0.03601279, 0.08922546, 0.02700011]),rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(shape_derivatives.flatten(),jnp.array([-0.19199248,  0.03288122,  0.        ,  0.03663278,  0.0869931 ,
                                                                          0.        ,  0.05096679,  0.03322467,  0.        , -0.11208187,
                                                                          -0.18554386,  0.        , -0.00207733, -0.08008336,  0.        ,
                                                                          0.18808049, -0.09918975,  0.        , -0.0223342 ,  0.00731204,
                                                                          0.        ,  0.01165495,  0.14946726,  0.        ,  0.04115084,
                                                                          0.05493871,  0.        ]),rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()