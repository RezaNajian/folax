import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
from computational_models import FiniteElementModel
from loss_functions import MechanicalLoss3D
from solvers import FiniteElementSolver
from controls import FourierControl
from deep_neural_networks import FiniteElementOperatorLearning
from tools import *
import pickle

# problem setup
model_settings = {"Lx":1,"Ly":1,"Lz":1,
                  "Nx":10,"Ny":10,"Nz":10,
                  "Ux_left":0.0,"Ux_right":"",
                  "Uy_left":0.0,"Uy_right":0.05,
                  "Uz_left":0.0,"Uz_right":-0.05}

working_directory_name = 'test_mechanical_3D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,"test_mechanical_3D.log"))
model_info,model_io = create_3D_box_model_info_mechanical(model_settings,case_dir)

x_freqs = np.array([2,4,6])
y_freqs = np.array([2,4,6])
z_freqs = np.array([2,4,6])

fe_model = FiniteElementModel("first_FE_model",model_info)
first_mechanical_loss_3d = MechanicalLoss3D("first_mechanical_loss_3d",fe_model)
first_fe_solver = FiniteElementSolver("first_fe_solver",first_mechanical_loss_3d)
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":5,"min":1e-2,"max":1}
fourier_control = FourierControl("first_fourier_control",fourier_control_settings,fe_model)

# create some random coefficients & K for training
create_random_K = True
if create_random_K:
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control)
    export_dict = model_settings.copy()
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["K_matrix"] = K_matrix
    export_dict["x_freqs"] = x_freqs
    export_dict["y_freqs"] = y_freqs
    export_dict["z_freqs"] = z_freqs
    # with open('fourier_control_dict.pkl', 'wb') as f:
    #     pickle.dump(export_dict,f)
else:
    with open('fourier_control_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]
    K_matrix = loaded_dict["K_matrix"]

# rand_K = np.random.normal(size=fe_model.GetNumberOfNodes())
# rand_UVW = np.random.normal(size=3*fe_model.GetNumberOfNodes())
# residuals,stiffness = first_mechanical_loss_3d.ComputeResidualsAndStiffness(rand_K,rand_UVW)

# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",fourier_control,[first_mechanical_loss_3d],[50,50],
                                    "swish",load_NN_params=False,NN_params_file_name="test_mechanical_3D_params.npy",
                                    working_directory=working_directory_name)
fol.Initialize()
fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix,batch_size=5,num_epochs=1000,
          learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
          relative_error=1e-10,plot_save_rate=10)

FOL_UVW_matrix = np.array(fol.Predict(coeffs_matrix))
FE_UVW_matrix = np.array(first_fe_solver.BatchSolve(K_matrix,np.zeros((K_matrix.shape[0],3*fe_model.GetNumberOfNodes()))))

# # FOL_UVW_matrix = FOL_UVW_matrix.reshape(-1,1).T

for i in range(K_matrix.shape[0]):
    solution_file = os.path.join(case_dir, f"case_{i}.vtu")
    model_io.point_data['K'] = np.array(K_matrix[i,:])
    model_io.point_data['U_FOL'] = np.array(FOL_UVW_matrix[i]).reshape((fe_model.GetNumberOfNodes(), 3))
    model_io.point_data['U_FE'] = np.array(FE_UVW_matrix[i]).reshape((fe_model.GetNumberOfNodes(), 3))
    model_io.write(solution_file)
