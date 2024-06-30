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
                  "Uy_left":0.0,"Uy_right":-0.05,
                  "Uz_left":0.0,"Uz_right":-0.05}

# fourier freqs
x_freqs = np.array([2,4,6])
y_freqs = np.array([2,4,6])
z_freqs = np.array([2,4,6])

# directory & save handling
working_directory_name = f'mechanical_3D_Nx_{model_settings["Nx"]}_Ny_{model_settings["Ny"]}_Nz_{model_settings["Nz"]}'
case_dir = os.path.join('.', working_directory_name)
clean_dir = True
if clean_dir:
    create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# creation of the model
model_info,model_io = create_3D_box_model_info_mechanical(model_settings,case_dir)

# creation of the objects
fe_model = FiniteElementModel("FE_model",model_info)
mechanical_loss_3d = MechanicalLoss3D("mechanical_loss_3d",fe_model)
fe_solver = FiniteElementSolver("fe_solver",mechanical_loss_3d)
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":20,"min":1e-2,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_model)

# create some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 200
    coeffs_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = model_settings.copy()
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = x_freqs
    export_dict["y_freqs"] = y_freqs
    export_dict["z_freqs"] = z_freqs
    with open(f'fourier_control_dict_Nx_{model_settings["Nx"]}_Ny_{model_settings["Ny"]}_Nz_{model_settings["Nz"]}.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'fourier_control_dict_Nx_{model_settings["Nx"]}_Ny_{model_settings["Ny"]}_Nz_{model_settings["Nz"]}.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]

K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

# now save K matrix 
for i in range(K_matrix.shape[0]):
    solution_file = os.path.join(case_dir, f"K_{i}.vtu")
    model_io.point_data['K'] = np.array(K_matrix[i,:])
    model_io.write(solution_file)


# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",fourier_control,[mechanical_loss_3d],[50,50],
                                    "swish",load_NN_params=False,working_directory=working_directory_name)
fol.Initialize()
fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix,batch_size=5,num_epochs=100,
            learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
            relative_error=1e-10,NN_params_save_file_name="NN_params_"+working_directory_name)

