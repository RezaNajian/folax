import numpy as np
from computational_models import FiniteElementModel
from loss_functions import MechanicalLoss3D
from solvers import FiniteElementSolver
from controls import FourierControl
from deep_neural_networks import FiniteElementOperatorLearning
from usefull_functions import *
# problem setup
model_settings = {"Lx":10,"Ly":1,"Lz":1,
                  "Nx":10,"Ny":2,"Nz":2,
                  "Ux_left":0.0,"Ux_right":"",
                  "Uy_left":0.0,"Uy_right":"",
                  "Uz_left":0.0,"Uz_right":0.05}
 
case_dir = os.path.join('.', 'test_mechanical_3D')
model_info,model_io = create_3D_box_model_info_mechanical(model_settings,case_dir)

x_freqs = np.array([3])
y_freqs = np.array([1])
z_freqs = np.array([1])

fe_model = FiniteElementModel("first_FE_model",model_info)
first_mechanical_loss_3d = MechanicalLoss3D("first_mechanical_loss_3d",fe_model)
first_fe_solver = FiniteElementSolver("first_fe_solver",first_mechanical_loss_3d)
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":2}
fourier_control = FourierControl("first_fourier_control",fourier_control_settings,fe_model)

# create some random coefficients & K for training
coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control)

# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",fourier_control,[first_mechanical_loss_3d],[5,10],
                                    "swish",load_NN_params=False,NN_params_file_name="test.npy")
fol.Initialize()
fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix,batch_size=5,num_epochs=10000,
          learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
          relative_error=1e-10,save_NN_params=False,NN_params_save_file_name="test.npy")

FOL_UVW_matrix = np.array(fol.Predict(coeffs_matrix))
FE_UVW_matrix = np.array(first_fe_solver.BatchSolve(K_matrix,np.zeros(FOL_UVW_matrix.shape)))

for i in range(K_matrix.shape[0]):
    solution_file = os.path.join(case_dir, f"case_{i}.vtu")
    model_io.point_data['K'] = np.array(K_matrix[i,:])
    model_io.point_data['U_FOL'] = np.array(FOL_UVW_matrix[i]).reshape((fe_model.GetNumberOfNodes(), 3))
    model_io.point_data['U_FE'] = np.array(FE_UVW_matrix[i]).reshape((fe_model.GetNumberOfNodes(), 3))
    model_io.write(solution_file)
