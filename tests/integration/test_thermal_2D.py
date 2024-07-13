import pytest
import unittest
import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.thermal_2D_fe_quad import ThermalLoss2D
from fol.solvers.fe_solver import FiniteElementSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *

class TestThermal2D(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        test_name = 'test_thermal_2D'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)
        self.model_info = create_2D_square_model_info_thermal(L=1,N=11,T_left=1.0,T_right=0.1)
        self.fe_model = FiniteElementModel("FE_model",self.model_info)
        self.thermal_loss = ThermalLoss2D("thermal_loss",self.fe_model,{"num_gp":2})
        self.fe_solver = FiniteElementSolver("fe_solver",self.thermal_loss)
        fourier_control_settings = {"x_freqs":np.array([1,2,3]),"y_freqs":np.array([1,2,3]),"z_freqs":np.array([0]),"beta":2}
        self.fourier_control = FourierControl("fourier_control",fourier_control_settings,self.fe_model)
        self.fol = FiniteElementOperatorLearning("first_fol",self.fourier_control,[self.thermal_loss],[],
                                                "swish",working_directory=self.test_directory)
        
        self.coeffs_matrix,self.K_matrix = create_random_fourier_samples(self.fourier_control,0)

    def test_compute(self):
        self.fol.Initialize()
        self.fol.Train(loss_functions_weights=[1],X_train=self.coeffs_matrix[-1,:].reshape(-1,1).T,batch_size=1,num_epochs=200,
                       learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",relative_error=1e-6)
        T_FOL = np.array(self.fol.Predict(self.coeffs_matrix[-1,:].reshape(-1,1).T))
        T_FEM = np.array(self.fe_solver.SingleSolve(self.K_matrix[-1,:],np.zeros(self.K_matrix.shape)))
        l2_error = 100 * np.linalg.norm(T_FOL-T_FEM,ord=2)/ np.linalg.norm(T_FEM,ord=2)
        self.assertLessEqual(l2_error, 1)

        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)
        else:
            pointwise_err = 100 * abs(T_FEM-T_FOL)/abs(T_FEM)
            plot_mesh_vec_data(1,[self.K_matrix[-1,:],T_FOL,T_FEM,pointwise_err],["K","T_FOL","T_FEM","rel err %"],file_name=os.path.join(self.test_directory,"FOL-KT-dist.png"))

if __name__ == '__main__':
    unittest.main()
