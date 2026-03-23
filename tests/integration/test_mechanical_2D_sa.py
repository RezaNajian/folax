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
        self.random_K = np.array([0.81838834, 0.65722322, 0.44376528, 0.02480078, 0.24395847, 0.85979581,
                                    0.45638335, 0.5481745 , 0.2588321 , 0.09001696, 0.52249897, 0.95812428,
                                    0.01554692, 0.05347252, 0.416067  , 0.97464693, 0.65322578, 0.69762206,
                                    0.73933816, 0.89411318, 0.41846848, 0.20929086, 0.32102525, 0.9390955 ,
                                    0.1159023 ])

    def test_sensitivites(self):
        FE_UV = np.array(self.linear_fe_solver.Solve(self.random_K,np.zeros(2*self.fe_mesh.GetNumberOfNodes())))  
        FE_adj_UV = np.array(self.adj_fe_solver.Solve(self.random_K, FE_UV,jnp.ones(2*self.fe_mesh.GetNumberOfNodes())))  
        control_derivatives = self.test_response.ComputeAdjointNodalControlDerivatives(self.random_K,FE_UV,FE_adj_UV)        
        shape_derivatives = self.test_response.ComputeAdjointNodalShapeDerivatives(self.random_K,FE_UV,FE_adj_UV)           
           
        np.testing.assert_allclose(control_derivatives,jnp.array([-4.8715207e-05,  2.6097722e-04,  5.9457647e-04,  5.7544943e-04,
                                                                    3.9241169e-04, -7.7068304e-05,  3.4124247e-04,  1.0304382e-03,
                                                                    1.4260340e-03,  8.6203369e-04, -7.5674485e-05,  3.9009936e-04,
                                                                    8.7188167e-04,  1.6479660e-03,  1.2620580e-03, -6.8743655e-05,
                                                                    3.5264570e-04,  1.4804506e-03,  3.3365702e-03,  2.2660964e-03,
                                                                    -3.6042602e-05,  5.1012830e-05,  5.7099469e-04,  1.7231244e-03,
                                                                    1.0206304e-03]),rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(shape_derivatives,jnp.array([-1.6028261e-03,  7.5871428e-04,  0.0000000e+00,  1.3926474e-04,
                                                                -2.3926908e-04,  0.0000000e+00,  5.8070011e-04, -1.2478330e-03,
                                                                0.0000000e+00,  3.2709245e-04, -6.5526308e-04,  0.0000000e+00,
                                                                -1.7160914e-05, -3.2301893e-04,  0.0000000e+00, -1.4801667e-03,
                                                                -1.2652221e-05,  0.0000000e+00,  2.3737620e-04, -7.8161858e-05,
                                                                0.0000000e+00,  7.6834881e-04,  4.1262968e-04,  0.0000000e+00,
                                                                7.7141606e-04,  8.6311513e-05,  0.0000000e+00, -7.1701681e-05,
                                                                -2.4101028e-04,  0.0000000e+00, -1.6512836e-03,  7.0739770e-06,
                                                                0.0000000e+00,  5.7129777e-04, -1.2536603e-04,  0.0000000e+00,
                                                                9.0413116e-04, -6.6315709e-04,  0.0000000e+00, -9.2262530e-04,
                                                                -2.2200982e-03,  0.0000000e+00,  1.2135776e-03, -2.0478733e-03,
                                                                0.0000000e+00, -1.3417134e-03, -4.4193817e-05,  0.0000000e+00,
                                                                1.4025613e-04,  2.4879008e-04,  0.0000000e+00, -2.7188566e-04,
                                                                -8.8445976e-04,  0.0000000e+00, -8.5951108e-04, -3.4653391e-03,
                                                                0.0000000e+00,  3.4432765e-03, -4.2557949e-05,  0.0000000e+00,
                                                                -2.6966995e-04, -2.2628350e-04,  0.0000000e+00, -5.0078816e-05,
                                                                -3.4226963e-05,  0.0000000e+00, -7.5573236e-04,  1.2665566e-03,
                                                                0.0000000e+00,  4.2446167e-04,  5.6450609e-03,  0.0000000e+00,
                                                                -2.2684410e-04,  4.1256277e-03,  0.0000000e+00]),rtol=1e-5, atol=1e-5)  

        # print(f"control_derivatives:{np.array2string(control_derivatives, separator=', ')}")
        # print(f"shape_derivatives:{np.array2string(shape_derivatives, separator=', ')}")

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