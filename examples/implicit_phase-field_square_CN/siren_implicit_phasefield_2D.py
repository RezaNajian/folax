import sys
import os
import optax
import numpy as np
from flax import nnx
import jax
from fol.loss_functions.phasefield_2D_fe_quad_CN import AllenCahnLoss2DQuad
from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.implicit_transient_parametric_operator_learning_super_res import ImplicitParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver_phasefield import FiniteElementNonLinearResidualBasedSolverPhasefield
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import MLP
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import time 
from datetime import datetime

# directory & save handling
working_directory_name = 'siren_implicit_AllenCahn_2D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":64,
                "T_left":1.0,"T_right":-1.0}
num_steps = 20
# creation of the model
mesh_res_rate = 1
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh_pred = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"]*mesh_res_rate)
# create fe-based loss function
bc_dict = {"T":{}}#"left":model_settings["T_left"],"right":model_settings["T_right"]
Dirichlet_BCs = False

material_dict = {"rho":1.0,"cp":1.0,"dt":0.0002,"epsilon":0.1}
dt_res_rate = 1
material_dict_pred = {"rho":material_dict["rho"],"cp":material_dict["cp"],"dt":material_dict["dt"]/dt_res_rate,"epsilon":material_dict["epsilon"]}
phasefield_loss_2d = AllenCahnLoss2DQuad("phasefield_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)
phasefield_loss_2d_pred = AllenCahnLoss2DQuad("phasefield_loss_2d_pred",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict_pred},
                                                                            fe_mesh=fe_mesh_pred)
no_control = NoControl("No_Control",fe_mesh)

fe_mesh.Initialize()
fe_mesh_pred.Initialize()
phasefield_loss_2d.Initialize()
phasefield_loss_2d_pred.Initialize()
no_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = True
if create_random_coefficients:
    # number_of_random_samples = 200
    # coeffs_matrix,K_matrix = create_random_fourier_samples(_control,number_of_random_samples)
    # export_dict = model_settings.copy()
    # export_dict["coeffs_matrix"] = coeffs_matrix
    # export_dict["x_freqs"] = fourier_control.x_freqs
    # export_dict["y_freqs"] = fourier_control.y_freqs
    # export_dict["z_freqs"] = fourier_control.z_freqs
    # with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'wb') as f:
    #     pickle.dump(export_dict,f)
    # def generate_random_pattern(rows, cols, mean=0, std=1):
    #     """Generate a random pattern in a 2-D domain using Gaussian random process."""
    #     # Generate a 2-D array with random values from a Gaussian distribution
    #     random_pattern = np.random.normal(loc=mean, scale=std, size=(rows, cols))
    #     random_pattern = (random_pattern-np.min(random_pattern))/(np.max(random_pattern)-np.min(random_pattern))
    #     return random_pattern
    # coeffs_matrix = generate_random_pattern(model_settings["N"], model_settings["N"], mean=0, std=1)
    def generate_random_smooth_pattern(L,N,epsilon):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(epsilon, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        X1, X2 = np.meshgrid(x, y)
        X = np.vstack([X1.ravel(), X2.ravel()]).T
        u = gp.sample_y(X, n_samples=1, random_state=0).ravel()
        scaled_u = 2.0*(u - np.min(u)) / (np.max(u) - np.min(u)) -1.0
        return scaled_u.reshape(1,-1)
    
    def generate_double_bubble(L,N,epsilon):
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        u = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                func1 = np.tanh((0.2-np.sqrt((x[i]-0.6*L)**2 + (y[j]-0.5*L)**2))/epsilon)
                func2 = np.tanh((0.2-np.sqrt((x[i]-0.4*L)**2 + (y[j]-0.5*L)**2))/epsilon)
                if func1>func2:
                    u[i,j] = func1
                else:
                    u[i,j] = func2
        return u.reshape(1,-1)
    coeffs_matrix = generate_double_bubble(model_settings["L"],model_settings["N"],material_dict["epsilon"])
    coeffs_matrix_fine = coeffs_matrix#generate_random_smooth_pattern(model_settings["L"],model_settings["N"]*mesh_res_rate,material_dict["epsilon"])

else:
    pass
    # with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f) 
    # coeffs_matrix = loaded_dict["coeffs_matrix"]
    # coeffs_matrix = jnp.load("gaussian_kernel_50000_N21.npy")


# K_matrix = []
T_matrix = no_control.ComputeBatchControlledVariables(coeffs_matrix)

export_Ks = False
# if export_Ks:
#     for i in range(K_matrix.shape[0]):
#         fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
#     fe_mesh.Finalize(export_dir=case_dir)
#     exit()

# specify id of the K of interest
eval_id = 0

# design siren NN for learning
hidden_layers = [100,100]
siren_NN = MLP(input_size=4,
                    output_size=1,
                    hidden_layers=hidden_layers,
                    activation_settings={"type":"sin",
                                         "prediction_gain":30,
                                         "initialization_gain":1.0},
                    skip_connections_settings={"active":False,"frequency":1})

lr = 1e-4
# create fol optax-based optimizer
chained_transform = optax.chain(optax.normalize_by_update_norm(),
                                optax.adam(lr))

# create fol
start_time = time.time()
fol = ImplicitParametricOperatorLearning(name="dis_fol",control=no_control,
                                        loss_function=phasefield_loss_2d,
                                        loss_function_pred=phasefield_loss_2d_pred,
                                        flax_neural_network=siren_NN,
                                        optax_optimizer=chained_transform,
                                        checkpoint_settings={"restore_state":False,
                                        "state_directory":case_dir+"/flax_state"},
                                        working_directory=case_dir)

fol.Initialize()

t_init = 0.0
t_current = t_init
FOL_T_temp = coeffs_matrix.flatten()
FOL_T = np.zeros((fe_mesh_pred.GetNumberOfNodes(),num_steps))
# For the first time step
fol.Train(train_set=(jnp.concatenate((jnp.array([t_current]),FOL_T_temp)).reshape(-1,1).T,),batch_size=100,
            convergence_settings={"num_epochs":1000,"relative_error":1e-5},
            plot_settings={"plot_save_rate":1000},
            save_settings={"save_nn_model":True})
FOL_T_temp_fine = np.array(fol.Predict_fine(jnp.array([t_current]))).reshape(-1)
FOL_T_temp = np.array(fol.Predict(jnp.array([t_current]))).reshape(-1)
FOL_T[:,0] = FOL_T_temp_fine
# For the subsequent time steps the checkpoint function should be activated
for i in range(num_steps-1):
    t_current += material_dict["dt"]
    chained_transform = optax.chain(optax.normalize_by_update_norm(),
                                optax.adam(lr))
    fol = ImplicitParametricOperatorLearning(name="dis_fol",control=no_control,
                                            loss_function=phasefield_loss_2d,
                                            loss_function_pred=phasefield_loss_2d_pred,
                                            flax_neural_network=siren_NN,
                                            optax_optimizer=chained_transform,
                                            checkpoint_settings={"restore_state":True,
                                            "state_directory":case_dir+"/flax_state"},
                                            working_directory=case_dir)
    fol.Initialize()
    fol.Train(train_set=(jnp.concatenate((jnp.array([t_current]),FOL_T_temp)).reshape(-1,1).T,),batch_size=100,
                convergence_settings={"num_epochs":1000,"relative_error":1e-5},
                plot_settings={"plot_save_rate":1000},
                save_settings={"save_nn_model":True})
    FOL_T_temp_fine = np.array(fol.Predict_fine(jnp.array([t_current]))).reshape(-1)
    FOL_T[:,i+1] = FOL_T_temp_fine
    FOL_T_temp = np.array(fol.Predict(jnp.array([t_current]))).reshape(-1)

end_time = time.time()
execution_time = end_time - start_time
now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"{current_time} - Info : iFOL part - finished in {execution_time:.4f} seconds")
fe_mesh['T_FOL'] = FOL_T
# solve FE here
start_time = time.time()
fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-7,"atol":1e-7,
                                            "maxiter":1000,"pre-conditioner":"none","Dirichlet_BCs":Dirichlet_BCs},
                "nonlinear_solver_settings":{"rel_tol":1e-7,"abs_tol":1e-7,
                                            "maxiter":20,"load_incr":1}}
nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolverPhasefield("nonlinear_fe_solver",phasefield_loss_2d_pred,fe_setting)
nonlinear_fe_solver.Initialize()
num_steps_FE = num_steps*dt_res_rate
FE_T = np.zeros((fe_mesh_pred.GetNumberOfNodes(),num_steps_FE))
FE_T_temp = coeffs_matrix_fine.flatten()
for i in range(num_steps_FE):
    FE_T_temp = np.array(nonlinear_fe_solver.Solve(FE_T_temp,FE_T_temp))  #np.zeros(fe_mesh.GetNumberOfNodes())
    FE_T[:,i] = FE_T_temp   
end_time = time.time()
execution_time = end_time - start_time
now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"{current_time} - Info : FEM part - finished in {execution_time:.4f} seconds")
 
fe_mesh['T_FE'] = FE_T

absolute_error = np.abs(FOL_T- FE_T[:,(dt_res_rate-1)::dt_res_rate])
fe_mesh['abs_error'] = absolute_error#.reshape((fe_mesh.GetNumberOfNodes(), 1))
time_list = [int(num_steps/5) - 1,int(num_steps/2) - 1,num_steps - 1]
time_list_FE = [int(num_steps_FE/5)- 1,int(num_steps_FE/2)- 1,num_steps_FE-1] 
plot_mesh_vec_data_phasefield(1,[coeffs_matrix_fine[eval_id],FOL_T[:,time_list[0]],FOL_T[:,time_list[1]],FOL_T[:,time_list[2]]],#,absolute_error
                   ["","","",""],
                   fig_title="Initial condition and implicit FOL solution",cmap = "jet",
                   file_name=os.path.join(case_dir,"FOL-T-dist.png"))
plot_mesh_vec_data_phasefield(1,[coeffs_matrix_fine[eval_id],FE_T[:,time_list_FE[0]],FE_T[:,time_list_FE[1]],FE_T[:,time_list_FE[2]]],
                   ["","","",""],
                   fig_title="Initial condition and FEM solution",cmap = "jet",
                   file_name=os.path.join(case_dir,"FEM-T-dist.png"))
plot_mesh_vec_data(1,[coeffs_matrix_fine[eval_id],absolute_error[:,time_list[0]],absolute_error[:,time_list[1]],absolute_error[:,time_list[2]]],
                   ["","","",""],
                   fig_title="Initial condition and iFOL error against FEM",cmap = "jet",
                   file_name=os.path.join(case_dir,"FOL-T-Error-dist.png"))

fe_mesh.Finalize(export_dir=case_dir)
