import sys
import os
import optax
import numpy as np
from fol.loss_functions.thermal_transient_hetero_3D_fe_hex import ThermalTransientLossHetero3DHex
from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning_field_hetero import MetaImplicitParametricOperatorLearning
from fol.solvers.fe_linear_residual_based_solver_hetero import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle

# directory & save handling
working_directory_name = 'meta_learning_thermal_transient_box_hetero_N16_3'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":16,
                "T_left":1.0,"T_right":0.0}

# creation of the model
fe_mesh = create_3D_box_mesh(Nx=model_settings["N"]-1,
                             Ny=model_settings["N"]-1,
                             Nz=model_settings["N"]-1,
                             Lx=model_settings["L"],
                             Ly=model_settings["L"],
                             Lz=model_settings["L"],
                             case_dir=case_dir)

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]}}

material_dict = {"rho":1.0,"cp":1.0,"dt":0.001}
thermal_loss_3d = ThermalTransientLossHetero3DHex("thermal_transient_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)
no_control = NoControl("No_control",fe_mesh)

fe_mesh.Initialize()
thermal_loss_3d.Initialize()
no_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    # def generate_random_smooth_patterns_3D(L, N, num_samples=2100):
    #     # Define the kernel for the Gaussian Process
    #     kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
    #     gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)

    #     # Create the 3D grid
    #     x = np.linspace(0, L, N)
    #     y = np.linspace(0, L, N)
    #     z = np.linspace(0, L, N)
    #     X1, X2, X3 = np.meshgrid(x, y, z, indexing='ij')  # Use 'ij' indexing for 3D
    #     X = np.vstack([X1.ravel(), X2.ravel(), X3.ravel()]).T

    #     # Generate multiple samples
    #     y_samples = gp.sample_y(X, n_samples=num_samples, random_state=0)

    #     # Normalize each sample
    #     scaled_y_samples = np.array([
    #         ((y_sample - np.min(y_sample)) / (np.max(y_sample) - np.min(y_sample)))
    #         for y_sample in y_samples.T
    #     ])

    #     return scaled_y_samples
    # coeffs_matrix = generate_random_smooth_patterns_3D(model_settings["L"],model_settings["N"])
    pass
else:
    coeffs_matrix = np.load("../training_data/3d_fft_N16_num5000.npy").reshape(5000,model_settings["N"]**3)

T_matrix = no_control.ComputeBatchControlledVariables(coeffs_matrix)
hetero_matrix = generate_composite_material(fe_mesh.GetNodesCoordinates()).reshape(1,-1)
K_matrix = no_control.ComputeBatchControlledVariables(hetero_matrix)
# export_Ks = False
# if export_Ks:
#     for i in range(K_matrix.shape[0]):
#         fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
#     fe_mesh.Finalize(export_dir=case_dir)
#     exit()

# design synthesizer & modulator NN for hypernetwork
characteristic_length = model_settings["N"]
characteristic_length = 128
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=3,
                     output_size=1,
                     hidden_layers=[characteristic_length] * 6,
                     activation_settings={"type":"sin",
                                          "prediction_gain":60,
                                          "initialization_gain":1.0},
                     skip_connections_settings={"active":False,"frequency":1})

latent_size =  2* characteristic_length
modulator_nn = MLP(name="modulator_nn",
                   input_size=latent_size,
                   use_bias=False) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 10000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=no_control,
                                            loss_function=thermal_loss_3d,
                                            hetero_info=K_matrix,
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3,
                                            checkpoint_settings={"restore_state":False,
                                            "state_directory":case_dir+"/flax_state"},
                                            working_directory=case_dir)
fol.Initialize()


train_start_id = 0
train_end_id = 4096
test_start_id = 1*train_end_id
test_end_id = int(1.2*train_end_id)
# here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
        #   test_set=(coeffs_matrix[test_start_id:test_end_id,:],),
#         #   test_settings={"test_frequency":10},
fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
            batch_size=32,
            convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
            plot_settings={"plot_save_rate":100},
            save_settings={"save_nn_model":True,
                         "best_model_checkpointing":True,
                         "best_model_checkpointing_frequency":100})

# load teh best model
fol.RestoreCheckPoint(fol.checkpoint_settings)

# relative_L2_error = 0.0
num_steps = 100
eval_start_id = 4050
eval_end_id = 4051
FOL_T = np.zeros((fe_mesh.GetNumberOfNodes(),num_steps))
for test in range(eval_start_id,eval_end_id):
    eval_id = test
    FOL_T_temp  = coeffs_matrix[eval_id,:]
    for i in range(num_steps):
        FOL_T_temp = np.array(fol.Predict(FOL_T_temp.reshape(-1,1).T)).reshape(-1)
        FOL_T[:,i] = FOL_T_temp 

    fe_mesh['T_FOL'] = FOL_T#.reshape((fe_mesh.GetNumberOfNodes(), 1))

    # solve FE here
    fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu"},
                    "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                "maxiter":10,"load_incr":5}}
    linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",thermal_loss_3d,fe_setting)
    linear_fe_solver.Initialize()
    FE_T = np.zeros((fe_mesh.GetNumberOfNodes(),num_steps))
    FE_T_temp = coeffs_matrix[eval_id,:]
    for i in range(num_steps):
        FE_T_temp = np.array(linear_fe_solver.Solve(FE_T_temp,K_matrix,FE_T_temp))  #np.zeros(fe_mesh.GetNumberOfNodes())
        FE_T[:,i] = FE_T_temp    
    fe_mesh['T_FE'] = FE_T#.reshape((fe_mesh.GetNumberOfNodes(), 1))

    absolute_error = np.abs(FOL_T- FE_T)
    fe_mesh['abs_error'] = absolute_error#.reshape((fe_mesh.GetNumberOfNodes(), 1))
    
fe_mesh['Heterogeneity'] = K_matrix.reshape(-1,1)
fe_mesh.Finalize(export_dir=case_dir)
