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

        random_FE_UV = jnp.array([0.2628367 , 0.10945356, 0.46926022, 0.4095999 , 0.08672023, 0.48140895,
                                  0.4428264 , 0.9602665 , 0.03818166, 0.5736197 , 0.5380293 , 0.65560925,
                                  0.83580685, 0.7974292 , 0.98296523, 0.24858999, 0.6211263 , 0.69799936])
        
        random_K = jnp.array([0.6439377 , 0.32251573, 0.19349372, 0.8864933 , 0.7247435 , 0.19193006,
                                0.34513092, 0.2523831 , 0.6319014 ])

        BC_applied_jac,BC_applied_rhs = test_response.ComputeAdjointJacobianMatrixAndRHSVector(random_K,random_FE_UV)

        np.testing.assert_allclose(BC_applied_jac.todense()[8,:],jnp.array([-1.5933526e-01, -1.1299228e-01,  5.2224368e-02,  1.9953307e-04,
                                                                            -8.8558704e-02,  6.2255092e-02, -4.3755195e-01,  1.4239475e-03,
                                                                            1.0885024e+00,  1.4627263e-02, -2.6412827e-01,  4.2019216e-03,
                                                                            -1.3652994e-01,  9.7495511e-02,  5.6700267e-02,  8.8018682e-03,
                                                                            -1.1132296e-01, -7.6012865e-02]),rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(BC_applied_rhs.flatten(),jnp.array([-0.        , -0.        , -0.02832778, -0.        , -0.        ,
                                                                        -0.        , -0.        , -0.        , -0.08190062, -0.        ,
                                                                        -0.        , -0.        , -0.        , -0.        , -0.02557062,
                                                                        -0.        , -0.        , -0.        ]),
                                                                        rtol=1e-5, atol=1e-5)

        # test sensitivities 
        random_adj_FE_UV = jnp.array([0.87936103, 0.6449497 , 0.1879133 , 0.8177328 , 0.30477262, 0.45858467,
                                        0.5103154 , 0.37463892, 0.8313782 , 0.5277164 , 0.31288528, 0.01651657,
                                        0.5735476 , 0.9206091 , 0.8522861 , 0.92917275, 0.38319993, 0.83216035])

        control_derivatives = test_response.ComputeAdjointNodalControlDerivatives(random_K,random_FE_UV,random_adj_FE_UV)
        shape_derivatives = test_response.ComputeAdjointNodalShapeDerivatives(random_K,random_FE_UV,random_adj_FE_UV)

        np.testing.assert_allclose(control_derivatives.flatten(),jnp.array([-0.05200566, -0.0508755 , -0.02084208, -0.04955585, -0.15092644,
                                                                            -0.0703495 ,  0.0015992 , -0.00430271,  0.01975888]),rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(shape_derivatives.flatten(),jnp.array([ 0.12193067, -0.10455623,  0.        , -0.07988495, -0.16263953,
                                                                            0.        , -0.05576743, -0.00384144,  0.        , -0.22090645,
                                                                            0.33066773,  0.        , -0.23995018, -0.17812887,  0.        ,
                                                                            0.14102945, -0.06443781,  0.        ,  0.18596953,  0.2172716 ,
                                                                            0.        ,  0.16827656, -0.05117913,  0.        , -0.02069724,
                                                                            0.01684369,  0.        ]),rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()