import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
from computational_models import FiniteElementModel
from loss_functions import MechanicalLoss3DTetra
from solvers import FiniteElementSolver
from controls import FourierControl
from deep_neural_networks import FiniteElementOperatorLearning
from tools import *
import pickle

# problem setup
model_settings = {"Lx":1,"Ly":1,"Lz":1,
                  "Ux_left":0.0,"Ux_right":"",
                  "Uy_left":0.0,"Uy_right":-0.05,
                  "Uz_left":0.0,"Uz_right":-0.05}

# fourier freqs
x_freqs = np.array([2,4,6])
y_freqs = np.array([2,4,6])
z_freqs = np.array([2,4,6])

# directory & save handling
working_directory_name = "box_3D_tetra"
case_dir = os.path.join('.', working_directory_name)
clean_dir = False
if clean_dir:
    create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

model_info,model_io = import_box_model_info_mechanical("Mesh_075e-2.med",'.',model_settings)

# creation of the objects
fe_model = FiniteElementModel("FE_model",model_info)
mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",fe_model)

fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_model)

# create some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 200
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = model_settings.copy()
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = x_freqs
    export_dict["y_freqs"] = y_freqs
    export_dict["z_freqs"] = z_freqs
    with open(f'fourier_control_dict.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'fourier_control_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]

K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

# now save K matrix 
for i in range(K_matrix.shape[0]):
    solution_file = os.path.join(case_dir, f"K_{i}.vtu")
    model_io.point_data['K'] = np.array(K_matrix[i,:])
    model_io.write(solution_file)

eval_id = 69

# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",fourier_control,[mechanical_loss_3d],[1],
                                    "tanh",load_NN_params=False,working_directory=working_directory_name)
fol.Initialize()

print(fol.output_size)

hjh

fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[eval_id].reshape(-1,1).T,batch_size=1,num_epochs=1000,
            learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
            relative_error=1e-10,NN_params_save_file_name="NN_params_"+working_directory_name)

solution_file = os.path.join(case_dir, f"K_{eval_id}_comp.vtu")
FOL_UVW = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T))
model_io.point_data['U_FOL'] = FOL_UVW.reshape((fe_model.GetNumberOfNodes(), 3))

first_fe_solver = FiniteElementSolver("first_fe_solver",mechanical_loss_3d)
FE_UVW = np.array(first_fe_solver.SingleSolve(K_matrix[eval_id],np.zeros(3*fe_model.GetNumberOfNodes())))  
model_io.point_data['U_FE'] = FE_UVW.reshape((fe_model.GetNumberOfNodes(), 3))

model_io.write(solution_file)