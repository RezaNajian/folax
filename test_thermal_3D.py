import numpy as np
from computational_models import FiniteElementModel
from loss_functions import ThermalLoss3D
from solvers import FiniteElementSolver
from controls import FourierControl
from deep_neural_networks import FiniteElementOperatorLearning
from usefull_functions import *

# problem setup
Lx=10
Ly=1
Lz=1
Nx=20
Ny=10
Nz=10 
T_left=1
T_right=0.1
x_freqs = np.array([1,2,3])
y_freqs = np.array([1])
z_freqs = np.array([1])
case_dir = os.path.join('.', 'test_thermal_3D')
model_info,model_io = create_3D_box_model_info_thermal(Nx,Ny,Nz,Lx,Ly,Lz,T_left,T_right,case_dir)

fe_model = FiniteElementModel("first_FE_model",model_info)

first_3D_thermal_loss = ThermalLoss3D("first_3D_thermal_loss",fe_model)
first_fe_solver = FiniteElementSolver("first_fe_solver",first_3D_thermal_loss)
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":2}
fourier_control = FourierControl("first_fourier_control",fourier_control_settings,fe_model)

# create some random coefficients & K for training
coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control)

# now we need to create, initialize and train fol
first_3D_fol = FiniteElementOperatorLearning("first_3D_fol",fourier_control,[first_3D_thermal_loss],[5,10],
                                             "swish",load_NN_params=False,NN_params_file_name="test.npy")
first_3D_fol.Initialize()

first_3D_fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix,batch_size=2,num_epochs=100,
                   learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
                   relative_error=1e-7,save_NN_params=True,NN_params_save_file_name="test.npy")

first_3D_fol.ReTrain(loss_functions_weights=[1],X_train=coeffs_matrix,batch_size=2,num_epochs=100,
                     convergence_criterion="total_loss",relative_error=1e-7,save_NN_params=True,
                     NN_params_save_file_name="test.npy")

FOL_T_matrix = np.array(first_3D_fol.Predict(coeffs_matrix))
FE_T_matrix = np.array(first_fe_solver.BatchSolve(K_matrix,np.zeros(K_matrix.shape)))

for i in range(K_matrix.shape[0]):
    solution_file = os.path.join(case_dir, f"case_{i}.vtu")
    model_io.point_data['K'] = np.array(K_matrix[i,:])
    model_io.point_data['T_FOL'] = np.array(FOL_T_matrix[i,:])
    model_io.point_data['T_FE'] = np.array(FE_T_matrix[i,:])
    model_io.write(solution_file)