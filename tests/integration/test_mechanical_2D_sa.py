import pytest
import unittest
import optax
from flax import nnx     
import jax 
import os
import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss2DQuad
from fol.responses.fe_response import FiniteElementResponse
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.controls.fourier_control import FourierControl
from fol.solvers.adjoint_fe_solver import AdjointFiniteElementSolver
from fol.tools.usefull_functions import *

class TestMechanical2DSensitivityAnalysis(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')    

    def setUp(self):
        # directory & save handling
        working_directory_name = str(os.path.basename(__file__).split(".")[0])
        self.test_directory = os.path.join('.', working_directory_name)
        create_clean_directory(working_directory_name)

        # problem setup
        model_settings = {"L":1,"N":5,
                            "Ux_left":0.0,"Ux_right":0.05,
                            "Uy_left":0.0,"Uy_right":0.05}

        # creation of the model
        self.fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

        # create fe-based loss function
        bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
                "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}

        material_dict = {"young_modulus":1,"poisson_ratio":0.3}
        self.mechanical_loss_2d = MechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                    "num_gp":2,
                                                                                    "material_dict":material_dict},
                                                                                    fe_mesh=self.fe_mesh)

        fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                    "beta":20,"min":1e-1,"max":1}
        self.fourier_control = FourierControl("E",fourier_control_settings,self.fe_mesh)


        self.test_response = FiniteElementResponse("test_response",response_formula="(E**2)*U[0]",fe_loss=self.mechanical_loss_2d,control=self.fourier_control)

        fe_setting = {"linear_solver_settings":{"solver":"JAX-direct","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu"},
                    "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                "maxiter":10,"load_incr":5}}
        self.linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",self.mechanical_loss_2d,fe_setting)

        adj_fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"}}
        self.adj_fe_solver = AdjointFiniteElementSolver("first_adj_fe_solver",self.test_response,adj_fe_setting)        

        self.fe_mesh.Initialize()
        self.mechanical_loss_2d.Initialize()
        self.fourier_control.Initialize()
        self.test_response.Initialize()
        self.linear_fe_solver.Initialize()
        self.adj_fe_solver.Initialize()

        key = jax.random.PRNGKey(42)
        self.random_K = jax.random.uniform(key, shape=(self.fe_mesh.GetNumberOfNodes(),), minval=0.0, maxval=1.0)        

    def test_sensitivites(self):
        FE_UV = np.array(self.linear_fe_solver.Solve(self.random_K,np.zeros(2*self.fe_mesh.GetNumberOfNodes())))  
        FE_adj_UV = np.array(self.adj_fe_solver.Solve(self.random_K, FE_UV,jnp.ones(2*self.fe_mesh.GetNumberOfNodes())))  
        control_derivatives = self.test_response.ComputeAdjointNodalControlDerivatives(self.random_K,FE_UV,FE_adj_UV)        
        shape_derivatives = self.test_response.ComputeAdjointNodalShapeDerivatives(self.random_K,FE_UV,FE_adj_UV)           
           
        np.testing.assert_allclose(control_derivatives,jnp.array([-2.5496504e-04,  2.6701926e-04,  1.3782313e-03,  1.8125116e-03,
                                                                  9.5334568e-04, -4.6650146e-04,  1.6040074e-04,  2.6520300e-03,
                                                                  4.0676566e-03,  2.3497981e-03, -3.3273635e-04,  5.8654405e-04,
                                                                  2.5705218e-03,  3.8915081e-03,  2.2413232e-03, -9.8458899e-05,
                                                                  3.8365490e-04,  1.4032004e-03,  2.4042374e-03,  1.9622538e-03,
                                                                  -6.5488985e-06,  3.4391785e-05,  3.2376559e-04,  1.1482707e-03,
                                                                  9.0321468e-04]),rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(shape_derivatives,jnp.array([-2.69951392e-03,  1.45725906e-03,  0.00000000e+00, -3.02692177e-04,
                                                                8.74764519e-05,  0.00000000e+00, -2.54532904e-04, -3.81702231e-03,
                                                                0.00000000e+00,  2.54520564e-04, -5.17783221e-03,  0.00000000e+00,
                                                                1.98083371e-03, -2.69629946e-03,  0.00000000e+00, -2.31046625e-03,
                                                                -6.59564976e-05,  0.00000000e+00, -4.86376230e-04, -2.26997538e-04,
                                                                0.00000000e+00, -1.47712964e-03, -1.62297569e-03,  0.00000000e+00,
                                                                5.13951178e-04, -1.99920242e-03,  0.00000000e+00,  4.27535316e-03,
                                                                -3.85374296e-04,  0.00000000e+00, -2.81802053e-03,  3.24317778e-04,
                                                                0.00000000e+00, -3.49429436e-04, -3.67481611e-04,  0.00000000e+00,
                                                                1.56196649e-04,  4.90568229e-04,  0.00000000e+00,  2.10749218e-03,
                                                                2.96993693e-03,  0.00000000e+00,  2.38405890e-03,  1.10230269e-03,
                                                                0.00000000e+00, -1.34471827e-03,  1.37852418e-04,  0.00000000e+00,
                                                                -5.78113832e-05,  4.59190167e-04,  0.00000000e+00,  1.02243910e-03,
                                                                2.04717927e-03,  0.00000000e+00, -8.09075951e-04,  1.66147971e-03,
                                                                0.00000000e+00,  1.09783164e-03,  5.28676668e-04,  0.00000000e+00,
                                                                2.62943613e-05, -3.93901428e-06,  0.00000000e+00, -1.29071705e-05,
                                                                5.38835593e-05,  0.00000000e+00, -1.12824026e-04,  5.28278877e-04,
                                                                0.00000000e+00, -6.48412039e-04,  2.07747309e-03,  0.00000000e+00,
                                                                -1.35060516e-04,  2.43720738e-03,  0.00000000e+00]),rtol=1e-5, atol=1e-5)  

        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)
        else:            
            FD_control_sens = self.test_response.ComputeFDNodalControlDerivatives(self.random_K,self.linear_fe_solver,fd_step_size=1e-5,fd_mode="CD")
            FD_shape_sens =  self.test_response.ComputeFDNodalShapeDerivatives(self.random_K,self.linear_fe_solver,fd_step_size=1e-5,fd_mode="CD")
            plot_mesh_vec_data(1,[FE_UV[0::2],FE_UV[1::2],
                                FE_adj_UV[0::2],FE_adj_UV[1::2]],
                            ["U","V","adj-U","adj-V"],
                            fig_title=" FEM and adj FEM solution",
                            file_name=os.path.join(self.test_directory,f"FEM-adj-UV-dist_test.png"))

            plot_mesh_vec_data(1,[control_derivatives,FD_control_sens],
                                ["df/dE","FD-df/dE"],
                                fig_title="Control Derivatives Verification",
                                file_name=os.path.join(self.test_directory,f"control_derivatives_verification.png"))   

            plot_mesh_vec_data(1,[shape_derivatives[0::3],shape_derivatives[1::3],
                                FD_shape_sens[0::3],FD_shape_sens[1::3]],
                                ["df/dx","df/dy","FD-df/dx","FD-df/dy"],
                                fig_title="Shape Derivatives Verification",
                                file_name=os.path.join(self.test_directory,f"shape_derivatives_verification.png")) 

if __name__ == '__main__':
    unittest.main()