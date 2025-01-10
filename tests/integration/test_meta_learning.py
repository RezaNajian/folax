import pytest
import unittest
import optax
from flax import nnx     
import jax 
import os
import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.deep_neural_networks.nns import MLP,HyperNetwork

class TestMetaLearning(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        test_name = 'test_meta_learning'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)

        self.fe_mesh = create_2D_square_mesh(L=1,N=5)
        bc_dict = {"Ux":{"left":0.0,"right":0.1},
                   "Uy":{"left":0.0,"right":0.1}}

        material_dict = {"young_modulus":1,"poisson_ratio":0.3}
        self.mechanical_loss = MechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "num_gp":2,
                                                                                "material_dict":material_dict},
                                                                                fe_mesh=self.fe_mesh)
        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"}}
        self.fe_solver = FiniteElementLinearResidualBasedSolver("nonlin_fe_solver",self.mechanical_loss,fe_setting)
        fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                    "beta":20,"min":1e-1,"max":1}
        self.fourier_control = FourierControl("fourier_control",fourier_control_settings,self.fe_mesh)

        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"}}
        self.linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",self.mechanical_loss,fe_setting)

        self.fourier_control.scale_min = -3.6441391950165527
        self.fourier_control.scale_max = 10.0

        self.fe_mesh.Initialize()
        self.mechanical_loss.Initialize()
        self.fourier_control.Initialize()
        self.linear_fe_solver.Initialize()

        self.coeffs_matrix = jnp.array([[0.17897024,0.24900086,0.28653019,0.2165933,0.23963302,0.18190986,
                                        0.30445688,0.17304319,0.27828901,0.26616984],
                                        [0.26987956,0.25202478,0.33906099,0.27196203,0.14363826,0.28560447,
                                        0.23743781,0.21978082,0.23560695,0.36889156],
                                        [0.39518719,0.27482211,0.25887451,0.21133597,0.40459931,0.23805859,
                                        0.35307977,0.26587458,0.3118189,0.38953973]])
        
        self.K_matrix = self.fourier_control.ComputeBatchControlledVariables(self.coeffs_matrix)

    def test_one_modulator_per_synthesizer_layer(self):

        # design synthesizer & modulator NN for hypernetwork
        synthesizer_nn = MLP(name="synthesizer_nn",
                            input_size=3,
                            output_size=2,
                            hidden_layers=[5] * 6,
                            activation_settings={"type":"sin"})

        latent_size = 10
        modulator_nn = MLP(name="modulator_nn",
                        input_size=latent_size,
                        use_bias=False) 

        hyper_network = HyperNetwork(name="hyper_nn",
                                    modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                                    coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

        # create fol optax-based optimizer
        main_loop_transform = optax.chain(optax.adam(1e-4))

        # create fol
        fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=self.fourier_control,
                                                        loss_function=self.mechanical_loss,
                                                        flax_neural_network=hyper_network,
                                                        main_loop_optax_optimizer=main_loop_transform,
                                                        checkpoint_settings={"restore_state":False,
                                                        "state_directory":self.test_directory+"/flax_state"},
                                                        working_directory=self.test_directory)
        fol.Initialize()


        train_start_id = 0
        train_end_id = 3
        fol.Train(train_set=(self.coeffs_matrix[train_start_id:train_end_id,:],),batch_size=1,
                    convergence_settings={"num_epochs":200,"relative_error":1e-100,
                                        "absolute_error":1e-100,"num_latent_itrs":3})

        for eval_id in range(train_start_id,train_end_id):
            FOL_UV = np.array(fol.Predict(self.coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
            if eval_id == 0:
                # print(np.array2string(FOL_UV.reshape((self.fe_mesh.GetNumberOfNodes(), 2))[:,0], separator=', '))
                np.testing.assert_allclose(FOL_UV.reshape((self.fe_mesh.GetNumberOfNodes(), 2))[:,0],
                                           np.array([0.        , 0.04614109, 0.06613609, 0.08705932, 0.1       , 0.        ,
                                            0.03290145, 0.05861736, 0.07996484, 0.1       , 0.        , 0.02106158,
                                            0.05036562, 0.07600654, 0.1       , 0.        , 0.02105756, 0.04292466,
                                            0.06860314, 0.1       , 0.        , 0.00895773, 0.02669284, 0.05424772,
                                            0.1       ]),
                                            rtol=1e-5, atol=1e-10)
                
            elif eval_id == 1:
                # print(np.array2string(FOL_UV.reshape((self.fe_mesh.GetNumberOfNodes(), 2))[:,0], separator=', '))
                np.testing.assert_allclose(FOL_UV.reshape((self.fe_mesh.GetNumberOfNodes(), 2))[:,0],
                                           np.array([0.        , 0.04503338, 0.06893442, 0.08841547, 0.1       , 0.        ,
                                                        0.02982597, 0.0577705 , 0.0809105 , 0.1       , 0.        , 0.02321399,
                                                        0.05095536, 0.07743262, 0.1       , 0.        , 0.02155444, 0.04399753,
                                                        0.0707641 , 0.1       , 0.        , 0.01015904, 0.02734981, 0.05567839,
                                                        0.1       ]),
                                            rtol=1e-5, atol=1e-10)
            elif eval_id == 2:
                # print(np.array2string(FOL_UV.reshape((self.fe_mesh.GetNumberOfNodes(), 2))[:,0], separator=', '))
                np.testing.assert_allclose(FOL_UV.reshape((self.fe_mesh.GetNumberOfNodes(), 2))[:,0],
                                           np.array([0.        , 0.0432228 , 0.06919706, 0.0886727 , 0.1       , 0.        ,
                                                    0.02928586, 0.05743678, 0.08082657, 0.1       , 0.        , 0.02355807,
                                                    0.05034218, 0.07730627, 0.1       , 0.        , 0.02110366, 0.04351436,
                                                    0.07099995, 0.1       , 0.        , 0.00986826, 0.02702353, 0.05540785,
                                                    0.1       ]),
                                            rtol=1e-5, atol=1e-10)

            # the rest is for debugging purposes
            self.fe_mesh['U_FOL'] = FOL_UV
            # solve FE here
            FE_UV = np.array(self.linear_fe_solver.Solve(self.K_matrix[eval_id],np.zeros(2*self.fe_mesh.GetNumberOfNodes())))  
            self.fe_mesh['U_FE'] = FE_UV.reshape((self.fe_mesh.GetNumberOfNodes(), 2))

            absolute_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
            self.fe_mesh['abs_error'] = absolute_error.reshape((self.fe_mesh.GetNumberOfNodes(), 2))


            plot_mesh_vec_data(1,[FOL_UV[0::2],FOL_UV[1::2],absolute_error[0::2],absolute_error[1::2]],
                            ["U","V","abs_error_U","abs_error_V"],
                            fig_title="implicit FOL solution and error",
                            file_name=os.path.join(self.test_directory,f"FOL-UV-dist_test_{eval_id}.png"))
            plot_mesh_vec_data(1,[self.K_matrix[eval_id,:],FE_UV[0::2],FE_UV[1::2]],
                            ["K","U","V"],
                            fig_title="conductivity and FEM solution",
                            file_name=os.path.join(self.test_directory,f"FEM-KUV-dist_test_{eval_id}.png"))

        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)
        else:
            self.fe_mesh.Finalize(export_dir=self.test_directory)
if __name__ == '__main__':
    unittest.main()