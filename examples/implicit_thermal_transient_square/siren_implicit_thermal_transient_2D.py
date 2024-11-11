import sys
import os
import optax
import numpy as np
from flax import nnx
import jax
from fol.loss_functions.thermal_transient_2D_fe_quad import ThermalTransientLoss2DQuad
from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.implicit_transient_parametric_operator_learning_super_res import ImplicitParametricOperatorLearning
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from siren_nn import Siren
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# directory & save handling
working_directory_name = 'siren_implicit_thermal_2D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":50,
                "T_left":1.0,"T_right":0.0}

# creation of the model
mesh_res_rate = 5
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh_pred = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"]*mesh_res_rate) 

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]}}#

material_dict = {"rho":1.0,"cp":1.0,"dt":0.05}
thermal_loss_2d = ThermalTransientLoss2DQuad("thermal_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)
thermal_loss_2d_pred = ThermalTransientLoss2DQuad("thermal_loss_2d_pred",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh_pred)


no_control = NoControl("No_Control",fe_mesh)

fe_mesh.Initialize()
fe_mesh_pred.Initialize()
thermal_loss_2d.Initialize()
thermal_loss_2d_pred.Initialize()
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
    def generate_random_smooth_pattern(L,N):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        X1, X2 = np.meshgrid(x, y)
        X = np.vstack([X1.ravel(), X2.ravel()]).T
        y_sample = gp.sample_y(X, n_samples=1, random_state=0).ravel()
        scaled_y_sample = (y_sample - np.min(y_sample)) / (np.max(y_sample) - np.min(y_sample))
        return scaled_y_sample
    # coeffs_matrix = generate_random_smooth_pattern(model_settings["L"],model_settings["N"]).reshape(1,-1)
    coeffs_matrix = np.full((1,model_settings["N"]**2),0.0)

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
siren_NN = Siren(4,1,[100,100])

# create fol optax-based optimizer
chained_transform = optax.chain(optax.normalize_by_update_norm(),
                                optax.adam(1e-4))

# create fol
fol = ImplicitParametricOperatorLearning(name="dis_fol",control=no_control,
                                        loss_function=thermal_loss_2d,
                                        loss_function_pred=thermal_loss_2d_pred,
                                        flax_neural_network=siren_NN,
                                        optax_optimizer=chained_transform,
                                        checkpoint_settings={"restore_state":False,
                                        "state_directory":case_dir+"/flax_state"},
                                        working_directory=case_dir)

fol.Initialize()

t_init = 0.0
t_current = t_init
num_steps = 10
FOL_T_temp = T_matrix
FOL_T = np.zeros((fe_mesh_pred.GetNumberOfNodes(),num_steps))
# For the first time step
fol.Train(train_set=(jnp.concatenate((jnp.array([t_current]),FOL_T_temp)).reshape(-1,1).T,),batch_size=100,
            convergence_settings={"num_epochs":2000,"relative_error":1e-8},
            plot_settings={"plot_save_rate":1000},
            save_settings={"save_nn_model":True})
FOL_T_temp_fine = np.array(fol.Predict_fine(jnp.array([t_current]))).reshape(-1)
FOL_T_temp = np.array(fol.Predict(jnp.array([t_current]))).reshape(-1)
FOL_T[:,0] = FOL_T_temp_fine
# For the subsequent time steps the checkpoint function should be activated
for i in range(num_steps-1):
    t_current += material_dict["dt"]
    fol = ImplicitParametricOperatorLearning(name="dis_fol",control=no_control,
                                            loss_function=thermal_loss_2d,
                                            loss_function_pred=thermal_loss_2d_pred,
                                            flax_neural_network=siren_NN,
                                            optax_optimizer=chained_transform,
                                            checkpoint_settings={"restore_state":True,
                                            "state_directory":case_dir+"/flax_state"},
                                            working_directory=case_dir)
    fol.Initialize()
    fol.Train(train_set=(jnp.concatenate((jnp.array([t_current]),FOL_T_temp)).reshape(-1,1).T,),batch_size=100,
                convergence_settings={"num_epochs":2000,"relative_error":1e-8},
                plot_settings={"plot_save_rate":1000},
                save_settings={"save_nn_model":True})
    FOL_T_temp_fine = np.array(fol.Predict_fine(jnp.array([t_current]))).reshape(-1)
    FOL_T[:,i+1] = FOL_T_temp_fine
    FOL_T_temp = np.array(fol.Predict(jnp.array([t_current]))).reshape(-1)

fe_mesh['T_FOL'] = FOL_T#.reshape((fe_mesh.GetNumberOfNodes(), 1))
# solve FE here
fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                            "maxiter":1000,"pre-conditioner":"ilu"},
                "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                            "maxiter":10,"load_incr":5}}
linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",thermal_loss_2d_pred,fe_setting)
linear_fe_solver.Initialize()
FE_T = np.zeros((fe_mesh_pred.GetNumberOfNodes(),num_steps))
# FE_T_temp = T_matrix#[eval_id]
FE_T_temp = np.full((fe_mesh_pred.GetNumberOfNodes()),0.0)
for i in range(num_steps):
    FE_T_temp = np.array(linear_fe_solver.Solve(FE_T_temp,FE_T_temp))  #np.zeros(fe_mesh.GetNumberOfNodes())
    FE_T[:,i] = FE_T_temp    
fe_mesh['T_FE'] = FE_T#.reshape((fe_mesh.GetNumberOfNodes(), 1))

absolute_error = np.abs(FOL_T- FE_T)
fe_mesh['abs_error'] = absolute_error#.reshape((fe_mesh.GetNumberOfNodes(), 1))

plot_mesh_vec_data(1,[coeffs_matrix[eval_id],FOL_T[:,0], FOL_T[:,2], FOL_T[:,-1]],#,absolute_error
                   ["T_init","T_1","T_3","T_fin"],
                   fig_title="Initial condition and implicit FOL solution",cmap = "jet",
                   file_name=os.path.join(case_dir,"FOL-T-dist.png"))
plot_mesh_vec_data(1,[coeffs_matrix[eval_id],FE_T[:,0], FE_T[:,2], FE_T[:,-1]],
                   ["T_init","T_1","T_3","T_fin"],
                   fig_title="Initial condition and FEM solution",cmap = "jet",
                   file_name=os.path.join(case_dir,"FEM-T-dist.png"))
plot_mesh_vec_data(1,[coeffs_matrix[eval_id],absolute_error[:,0], absolute_error[:,2], absolute_error[:,-1]],
                   ["T_init","Error_1","Error_3","Error_fin"],
                   fig_title="Initial condition and iFOL error against FEM",cmap = "jet",
                   file_name=os.path.join(case_dir,"FOL-T-Error-dist.png"))

fe_mesh.Finalize(export_dir=case_dir)
