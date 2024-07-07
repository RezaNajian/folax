import pytest
import unittest
import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
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







# import sys
# import os
# # Add the parent directory to sys.path
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)
# import numpy as np
# from computational_models import FiniteElementModel
# from loss_functions import MechanicalLoss3D
# from solvers import FiniteElementSolver
# from controls import FourierControl
# from deep_neural_networks import FiniteElementOperatorLearning
# from tools import *
# import pickle

# # problem setup
# model_settings = {"Lx":1,"Ly":1,"Lz":1,
#                   "Nx":15,"Ny":15,"Nz":15,
#                   "Ux_left":0.0,"Ux_right":"",
#                   "Uy_left":0.0,"Uy_right":0.05,
#                   "Uz_left":0.0,"Uz_right":-0.05}

# working_directory_name = 'test_mechanical_3D'
# case_dir = os.path.join('.', working_directory_name)
# clean_dir = True
# if clean_dir:
#     create_clean_directory(working_directory_name)
# sys.stdout = Logger(os.path.join(case_dir,"test_mechanical_3D.log"))
# model_info,model_io = create_3D_box_model_info_mechanical(model_settings,case_dir)

# x_freqs = np.array([2,4,6])
# y_freqs = np.array([2,4,6])
# z_freqs = np.array([2,4,6])

# fe_model = FiniteElementModel("first_FE_model",model_info)
# first_mechanical_loss_3d = MechanicalLoss3D("first_mechanical_loss_3d",fe_model)
# first_fe_solver = FiniteElementSolver("first_fe_solver",first_mechanical_loss_3d)
# fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":5,"min":1e-2,"max":1}
# fourier_control = FourierControl("first_fourier_control",fourier_control_settings,fe_model)

# # create some random coefficients & K for training
# create_random_K = True
# if create_random_K:
#     number_of_random_samples = 20
#     coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
#     export_dict = model_settings.copy()
#     export_dict["coeffs_matrix"] = coeffs_matrix
#     export_dict["K_matrix"] = K_matrix
#     export_dict["x_freqs"] = x_freqs
#     export_dict["y_freqs"] = y_freqs
#     export_dict["z_freqs"] = z_freqs
#     with open(os.path.join(case_dir,'fourier_control_dict.pkl'), 'wb') as f:
#         pickle.dump(export_dict,f)
# else:
#     with open(os.path.join(case_dir,'fourier_control_dict.pkl'), 'rb') as f:
#         loaded_dict = pickle.load(f)
    
#     coeffs_matrix = loaded_dict["coeffs_matrix"]
#     K_matrix = loaded_dict["K_matrix"]

# # now save K matrix 
# for i in range(K_matrix.shape[0]):
#     solution_file = os.path.join(case_dir, f"K_{i}.vtu")
#     model_io.point_data['K'] = np.array(K_matrix[i,:])
#     model_io.write(solution_file)

# eval_id = 10
# solve_FOL = True
# solve_FE = True
# if solve_FOL:
#     # now we need to create, initialize and train fol
#     fol = FiniteElementOperatorLearning("first_fol",fourier_control,[first_mechanical_loss_3d],[50,50],
#                                         "swish",load_NN_params=False,NN_params_file_name="test_mechanical_3D_params.npy",
#                                         working_directory=working_directory_name)
#     fol.Initialize()
#     fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix,batch_size=5,num_epochs=1000,
#             learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
#             relative_error=1e-10)

#     FOL_UVW = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T))
#     solution_file = os.path.join(case_dir, f"K_{eval_id}_FOL_results.vtu")
#     model_io.point_data['K'] = np.array(K_matrix[eval_id,:])
#     model_io.point_data['U_FOL'] = FOL_UVW.reshape((fe_model.GetNumberOfNodes(), 3))
#     model_io.write(solution_file)

# if solve_FE:
#     FE_UVW = np.array(first_fe_solver.SingleSolve(K_matrix[eval_id],np.zeros(3*fe_model.GetNumberOfNodes())))                     
#     solution_file = os.path.join(case_dir, f"K_{eval_id}_FE_results.vtu")
#     model_io.point_data['K'] = np.array(K_matrix[eval_id,:])
#     model_io.point_data['U_FE'] = FE_UVW.reshape((fe_model.GetNumberOfNodes(), 3))
#     model_io.write(solution_file)