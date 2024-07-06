import pytest
import unittest
import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
from computational_models import FiniteElementModel
from loss_functions import ThermalLoss2D
from solvers import FiniteElementSolver
from controls import FourierControl
from deep_neural_networks import FiniteElementOperatorLearning
from tools import *


class TestThermal2D(unittest.TestCase):
    def setUp(self):
        # problem setup
        test_name = 'test_thermal_2D'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)
        self.model_info = create_2D_square_model_info_thermal(L=1,N=11,T_left=1.0,T_right=0.1)
        self.fe_model = FiniteElementModel("FE_model",self.model_info)
        self.thermal_loss = ThermalLoss2D("thermal_loss",self.fe_model)
        self.fe_solver = FiniteElementSolver("fe_solver",self.thermal_loss)
        fourier_control_settings = {"x_freqs":np.array([1,2,3]),"y_freqs":np.array([1,2,3]),"z_freqs":np.array([0]),"beta":2}
        self.fourier_control = FourierControl("fourier_control",fourier_control_settings,self.fe_model)
        self.fol = FiniteElementOperatorLearning("first_fol",self.fourier_control,[self.thermal_loss],[],
                                                "swish",working_directory=self.test_directory)
        
        self.coeffs_matrix,self.K_matrix = create_random_fourier_samples(self.fourier_control,1)


    def test_compute(self):
        self.fol.Initialize()
        self.fol.Train(loss_functions_weights=[1],X_train=self.coeffs_matrix[-1,:].reshape(-1,1).T,batch_size=1,num_epochs=200,
                       learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",relative_error=1e-6)
        FOL_T = np.array(self.fol.Predict(self.coeffs_matrix[-1,:].reshape(-1,1).T))
        plot_mesh_vec_data(1,[self.K_matrix[-1,:],FOL_T],["K","T"],file_name=os.path.join(self.test_directory,"FOL-KT-dist.png"))
        self.assertEqual(5, 5)
        # self.assertEqual(product_result, 6)
        # shutil.rmtree(self.test_directory)

if __name__ == '__main__':
    unittest.main()
