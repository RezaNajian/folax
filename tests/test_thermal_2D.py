import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
from computational_models import FiniteElementModel
from loss_functions import ThermalLoss2D
from solvers import FiniteElementSolver
from controls import FourierControl
from deep_neural_networks import FiniteElementOperatorLearning
from tools import *

# problem setup
L=1
N=51 
T_left=1
T_right=0.1
x_freqs = np.array([1,2,3])
y_freqs = np.array([1,2,3])
z_freqs = np.array([0])
case_dir = os.path.join('.', 'test_thermal_2D')
create_clean_directory('test_thermal_2D')
sys.stdout = Logger(os.path.join(case_dir,"test_thermal_2D.log"))


model_info = create_2D_square_model_info_thermal(L,N,T_left,T_right)
fe_model = FiniteElementModel("first_FE_model",model_info)
first_thermal_loss = ThermalLoss2D("first_thermal_loss",fe_model)
first_fe_solver = FiniteElementSolver("first_fe_solver",first_thermal_loss)
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":2}
fourier_control = FourierControl("first_fourier_control",fourier_control_settings,fe_model)

# create some random coefficients & K for training
coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control)

# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",fourier_control,[first_thermal_loss],[5,10],
                                    "swish",load_NN_params=False,NN_params_file_name="test_thermal_2D_params.npy",
                                    working_directory='test_thermal_2D')
fol.Initialize()
fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[-1].reshape(-1,1).T,batch_size=1,num_epochs=10000,
          learning_rate=0.001,plot_rate=100000,optimizer="adam",convergence_criterion="total_loss",
          relative_error=1e-100,save_NN_params=True,NN_params_save_file_name="test_thermal_2D_params.npy")

fol.ReTrain(loss_functions_weights=[1],X_train=coeffs_matrix,batch_size=20,num_epochs=100,
            convergence_criterion="total_loss",relative_error=1e-7,save_NN_params=True,
            NN_params_save_file_name="test_thermal_2D_params.npy")

FOL_T_matrix = np.array(fol.Predict(coeffs_matrix))
FE_T_matrix = np.array(first_fe_solver.BatchSolve(K_matrix,np.zeros(K_matrix.shape)))

# plot_data_input(FOL_T_matrix,10,'FOL T distributions')
# plot_data_input(FE_T_matrix,10,'FE T distributions')

# compute error
relative_error = np.zeros((0,FOL_T_matrix.shape[1]))
for i in range(FOL_T_matrix.shape[0]):
    FEM_T = FE_T_matrix[i,:].reshape(-1)
    FOL_T = FOL_T_matrix[i,:].reshape(-1)
    err = np.zeros((FOL_T.size))
    for dof_index,dof in enumerate(["T"]):
        non_dirichlet_indices = first_thermal_loss.number_dofs_per_node*first_thermal_loss.fe_model.GetDofsDict()[dof]["non_dirichlet_nodes_ids"] + dof_index
        non_dirichlet_FOL_values = FOL_T[non_dirichlet_indices]
        non_dirichlet_FE_values = FEM_T[non_dirichlet_indices]
        err[non_dirichlet_indices] = 100 * abs(non_dirichlet_FOL_values-non_dirichlet_FE_values)/abs(non_dirichlet_FE_values)
    relative_error = np.vstack((relative_error,err))

test_index=1
plot_mesh_vec_data(1,[K_matrix[test_index,:],FOL_T_matrix[test_index,:]],["K","T"],file_name=os.path.join(case_dir,"FOL-KT-dist.png"))
plot_mesh_vec_data(1,[K_matrix[test_index,:],FE_T_matrix[test_index,:]],["K","T"],file_name=os.path.join(case_dir,"FE-KT-dist.png"))
plot_mesh_vec_data(1,[K_matrix[test_index,:],relative_error[test_index,:]],["K","err(T) % "],file_name=os.path.join(case_dir,"KT-diff-dist.png"))
