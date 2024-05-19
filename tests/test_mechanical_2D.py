import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
from computational_models import FiniteElementModel
from loss_functions import MechanicalLoss2D
from solvers import FiniteElementSolver
from controls import FourierControl
from deep_neural_networks import FiniteElementOperatorLearning
from tools import *
# problem setup
L=1
N=21 
x_freqs = np.array([1,2,3])
y_freqs = np.array([1,2,3])
z_freqs = np.array([0])
case_dir = os.path.join('.', 'test_mechanical_2D')
create_clean_directory('test_mechanical_2D')
sys.stdout = Logger(os.path.join(case_dir,"test_mechanical_2D.log"))
model_info = create_2D_square_model_info_mechanical(L,N,Ux_left=0.0,Ux_right=0.1,Uy_left=0.0,Uy_right=0.1)

fe_model = FiniteElementModel("first_FE_model",model_info)
first_mechanical_loss = MechanicalLoss2D("first_mechanical_loss",fe_model)
first_fe_solver = FiniteElementSolver("first_fe_solver",first_mechanical_loss)
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":2}
fourier_control = FourierControl("first_fourier_control",fourier_control_settings,fe_model)

# create some random coefficients & K for training
coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control)

# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",fourier_control,[first_mechanical_loss],[5,10],
                                    "swish",load_NN_params=False,NN_params_file_name="test_mechanical_2D_params.npy",
                                    working_directory='test_mechanical_2D')
fol.Initialize()
fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix,batch_size=20,num_epochs=1000,
          learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
          relative_error=1e-7,save_NN_params=False,NN_params_save_file_name="test_mechanical_2D_params.npy")

fol.ReTrain(loss_functions_weights=[1],X_train=coeffs_matrix,batch_size=20,num_epochs=100,
            convergence_criterion="total_loss",relative_error=1e-7,save_NN_params=False,
            NN_params_save_file_name="test_mechanical_2D_params.npy")

FOL_UV_matrix = np.array(fol.Predict(coeffs_matrix))
FE_UV_matrix = np.array(first_fe_solver.BatchSolve(K_matrix,np.zeros(FOL_UV_matrix.shape)))

# compute error
relative_error = np.zeros((0,FOL_UV_matrix.shape[1]))
for i in range(FOL_UV_matrix.shape[0]):
    FEM_UV = FE_UV_matrix[i,:].reshape(-1)
    FOL_UV = FOL_UV_matrix[i,:].reshape(-1)
    err = np.zeros((FOL_UV.size))
    for dof_index,dof in enumerate(["Ux","Uy"]):
        non_dirichlet_indices = first_mechanical_loss.number_dofs_per_node*first_mechanical_loss.fe_model.GetDofsDict()[dof]["non_dirichlet_nodes_ids"] + dof_index
        non_dirichlet_FOL_values = FOL_UV[non_dirichlet_indices]
        non_dirichlet_FE_values = FEM_UV[non_dirichlet_indices]
        err[non_dirichlet_indices] = 100 * abs(non_dirichlet_FOL_values-non_dirichlet_FE_values)/abs(non_dirichlet_FE_values)
    relative_error = np.vstack((relative_error,err))

test_index=1
plot_mesh_vec_data(1,[K_matrix[test_index,:],FOL_UV_matrix[test_index,0::2],FOL_UV_matrix[test_index,1::2]],["K","U","V"],file_name=os.path.join(case_dir,"FOL-KUV-dist.png"))
plot_mesh_vec_data(1,[K_matrix[test_index,:],FE_UV_matrix[test_index,0::2],FE_UV_matrix[test_index,1::2]],["K","U","V"],file_name=os.path.join(case_dir,"FE-KUV-dist.png"))
plot_mesh_vec_data(1,[K_matrix[test_index,:],relative_error[test_index,0::2],relative_error[test_index,1::2]],["K","err(U) % ","err(V) % "],file_name=os.path.join(case_dir,"KUV-diff-dist.png"))