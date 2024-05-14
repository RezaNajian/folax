import numpy as np
from computational_models import FiniteElementModel
from loss_functions import ThermalLoss3D
from solvers import FiniteElementSolver
from controls import FourierControl
from deep_neural_networks import FiniteElementOperatorLearning
from usefull_functions import *

# problem setup
Lx=10
Ly=0.1
Lz=0.1
Nx=10
Ny=10
Nz=10 
T_left=1
T_right=0.1
x_freqs = np.array([1,2,3])
y_freqs = np.array([1,2,3])
z_freqs = np.array([1,2,3])
model_info = create_3D_square_model_info_thermal(Nx,Ny,Nz,Lx,Ly,Lz,T_left,T_right)

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

# plot_data_input(FOL_T_matrix,10,'FOL T distributions')
# plot_data_input(FE_T_matrix,10,'FE T distributions')

# # compute error
# relative_error = np.zeros((0,FOL_T_matrix.shape[1]))
# for i in range(FOL_T_matrix.shape[0]):
#     FEM_T = FE_T_matrix[i,:].reshape(-1)
#     FOL_T = FOL_T_matrix[i,:].reshape(-1)
#     err = np.zeros((FOL_T.size))
#     for dof_index,dof in enumerate(["T"]):
#         non_dirichlet_indices = first_thermal_loss.number_dofs_per_node*first_thermal_loss.fe_model.GetDofsDict()[dof]["non_dirichlet_nodes_ids"] + dof_index
#         non_dirichlet_FOL_values = FOL_T[non_dirichlet_indices]
#         non_dirichlet_FE_values = FEM_T[non_dirichlet_indices]
#         err[non_dirichlet_indices] = 100 * abs(non_dirichlet_FOL_values-non_dirichlet_FE_values)/abs(non_dirichlet_FE_values)
#     relative_error = np.vstack((relative_error,err))

# test_index=1
# plot_mesh_vec_data(1,[K_matrix[test_index,:],FOL_T_matrix[test_index,:]],["K","T"],file_name="FOL-KT-dist.png")
# plot_mesh_vec_data(1,[K_matrix[test_index,:],FE_T_matrix[test_index,:]],["K","T"],file_name="FE-KT-dist.png")
# plot_mesh_vec_data(1,[K_matrix[test_index,:],relative_error[test_index,:]],["K","err(T) % "],file_name="KT-diff-dist.png")
