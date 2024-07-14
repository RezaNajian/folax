import pytest
import unittest
import os
import numpy as np
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.mechanical_2D_fe_quad import MechanicalLoss2D
from fol.solvers.fe_solver import FiniteElementSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *

class TestMechanical2D(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        test_name = 'test_mechanical_2D'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)
        self.model_info = create_2D_square_model_info_mechanical(L=1,N=11,Ux_left=0.0,Ux_right=0.1,Uy_left=0.0,Uy_right=0.1)
        self.fe_model = FiniteElementModel("FE_model",self.model_info)
        self.mechanical_loss = MechanicalLoss2D("mechanical_loss_2d",self.fe_model,{"young_modulus":1,"poisson_ratio":0.3,"num_gp":2})
        self.fe_solver = FiniteElementSolver("fe_solver",self.mechanical_loss)
        fourier_control_settings = {"x_freqs":np.array([1,2,3]),"y_freqs":np.array([1,2,3]),"z_freqs":np.array([0]),"beta":2}
        self.fourier_control = FourierControl("fourier_control",fourier_control_settings,self.fe_model)
        self.fol = FiniteElementOperatorLearning("fol_mechanical_loss_2d",self.fourier_control,[self.mechanical_loss],[],
                                                "swish",working_directory=self.test_directory)
        
        self.coeffs_matrix,self.K_matrix = create_random_fourier_samples(self.fourier_control,0)

    def test_compute(self):
        self.fol.Initialize()
        self.fol.Train(loss_functions_weights=[1],X_train=self.coeffs_matrix[-1,:].reshape(-1,1).T,batch_size=1,num_epochs=200,
                       learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",relative_error=1e-6)
        UV_FOL = np.array(self.fol.Predict(self.coeffs_matrix[-1,:].reshape(-1,1).T))
        UV_FEM = np.array(self.fe_solver.SingleSolve(self.K_matrix[-1,:],np.zeros(UV_FOL.shape)))
        l2_error = 100 * np.linalg.norm(UV_FOL-UV_FEM,ord=2)/ np.linalg.norm(UV_FEM,ord=2)
        self.assertLessEqual(l2_error, 1)

        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)
        else:
            plot_mesh_vec_data(1,[self.K_matrix[-1,:],UV_FOL[0::2],UV_FOL[1::2]],["K","U","V"],file_name=os.path.join(self.test_directory,"FOL-KUV-dist.png"))
            plot_mesh_vec_data(1,[self.K_matrix[-1,:],UV_FEM[0::2],UV_FEM[1::2]],["K","U","V"],file_name=os.path.join(self.test_directory,"FEM-KUV-dist.png"))

if __name__ == '__main__':
    unittest.main()