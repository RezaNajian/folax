import pytest
import unittest
import os
import numpy as np
from fol.IO.mesh_io import MeshIO
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.thermal_3D_fe_tetra import ThermalLoss3DTetra
from fol.solvers.nonlinear_solver import NonLinearSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *

class TestMechanical3D(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        test_name = 'test_thermal_3D_nonlin'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)
        point_bc_settings = {"T":{"left":1,"right":0.1}}
        self.io = MeshIO("box_io",os.path.join(os.path.dirname(os.path.abspath(__file__)),"meshes"),"box_3D_coarse.med",point_bc_settings)
        self.model_info = self.io.Import()
        self.fe_model = FiniteElementModel("FE_model",self.model_info)
        self.mechanical_loss = ThermalLoss3DTetra("thermal",self.fe_model,{"beta":0,"c":4})
        self.fe_solver = NonLinearSolver("fe_solver",self.mechanical_loss,max_num_itr=3)
        fourier_control_settings = {"x_freqs":np.array([0]),"y_freqs":np.array([0]),"z_freqs":np.array([0]),"beta":2}
        self.fourier_control = FourierControl("fourier_control",fourier_control_settings,self.fe_model)
        self.fol = FiniteElementOperatorLearning("fol_thermal_nonlin",self.fourier_control,[self.mechanical_loss],[],
                                                "swish",working_directory=self.test_directory)
        
        self.coeffs_matrix,self.K_matrix = create_random_fourier_samples(self.fourier_control,0)

    def test_compute(self):
        self.fol.Initialize()
        self.fol.Train(loss_functions_weights=[1],X_train=self.coeffs_matrix[-1,:].reshape(-1,1).T,batch_size=1,num_epochs=2000,
                       learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",relative_error=1e-6)
        T_FOL = np.array(self.fol.Predict(self.coeffs_matrix[-1,:].reshape(-1,1).T))
        T_FEM = np.array(self.fe_solver.SingleSolve(self.K_matrix[-1,:],np.zeros(T_FOL.shape)))
        l2_error = 100 * np.linalg.norm(T_FOL-T_FEM,ord=2)/ np.linalg.norm(T_FEM,ord=2)
        self.assertLessEqual(l2_error, 1)

        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)
        else:
            solution_file = os.path.join(self.test_directory, "results.vtu")
            self.io.mesh_io.point_data['K'] = np.array(self.K_matrix[-1,:])
            self.io.mesh_io.point_data['T_FOL'] = np.array(T_FOL)
            self.io.mesh_io.point_data['T_FEM'] = np.array(T_FEM)
            self.io.mesh_io.write(solution_file)

if __name__ == '__main__':
    unittest.main()