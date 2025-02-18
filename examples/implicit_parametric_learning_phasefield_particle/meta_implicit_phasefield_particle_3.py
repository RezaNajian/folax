import sys
import os
import optax
import numpy as np
from fol.loss_functions.phasefield import PhaseFieldLoss2DTri
from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver_phasefield import FiniteElementNonLinearResidualBasedSolverPhasefield
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from jax import config
config.update("jax_default_matmul_precision", "float32")
# directory & save handling
working_directory_name = 'meta_learning_phasefield_particle_3'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
# model_settings = {"L":1,"N":51,
#                 "T_left":1.0,"T_right":0.0}

# creation of the model
mesh_res_rate = 1
fe_mesh = Mesh("fol_io","Li_battery_particle.med",'../meshes/')
fe_mesh_pred = Mesh("fol_io","Li_battery_particle.med",'../meshes/')
# create fe-based loss function
bc_dict = {"T":{}}#"left":model_settings["T_left"],"right":model_settings["T_right"]
Dirichlet_BCs = False
material_dict = {"rho":1.0,"cp":1.0,"dt":0.001,"epsilon":0.2}
phasefield_loss_2d = PhaseFieldLoss2DTri("phasefield_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)
phasefield_loss_2d_pred = PhaseFieldLoss2DTri("phasefield_loss_2d_pred",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "material_dict":material_dict},
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
    def generate_random_smooth_patterns_from_mesh(coords, num_samples=10000,smoothness_levels=[0.15, 0.2, 0.3, 0.4, 0.5]):
        """
        Generate mixed random smooth patterns using a Gaussian Process with varying smoothness levels.

        Parameters:
            mesh (meshio.Mesh): A meshio object containing the coordinate information.
            num_samples (int): Total number of samples to generate (divided among smoothness levels).
            smoothness_levels (list): List of length scales for different smoothness levels.

        Returns:
            np.ndarray: A shuffled array of normalized samples from all smoothness levels.
        """
        # Extract coordinate points from the mesh
        X = coords[:, :2]  # Ensure only the first two coordinates are used if it's 3D

        all_samples = []

        for length_scale in smoothness_levels:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)

            # Generate an equal number of samples per smoothness level
            num_per_level = num_samples // len(smoothness_levels)
            y_samples = gp.sample_y(X, n_samples=num_per_level, random_state=0)

            # Normalize each sample
            scaled_y_samples = np.array([
                2 * (y_sample - np.min(y_sample)) / (np.max(y_sample) - np.min(y_sample)) - 1
                for y_sample in y_samples.T
            ])

            all_samples.append(scaled_y_samples)

        # Concatenate all samples from different smoothness levels
        mixed_samples = np.vstack(all_samples)

        # Shuffle the samples randomly
        np.random.shuffle(mixed_samples)

        return mixed_samples

    coeffs_matrix = generate_random_smooth_patterns_from_mesh(fe_mesh.GetNodesCoordinates())
else:
    coeffs_matrix = np.load("../training_data/gaussian_N51_num12000.npy")

T_matrix = no_control.ComputeBatchControlledVariables(coeffs_matrix)

# export_Ks = False
# if export_Ks:
#     for i in range(K_matrix.shape[0]):
#         fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
#     fe_mesh.Finalize(export_dir=case_dir)
#     exit()

# design synthesizer & modulator NN for hypernetwork
# characteristic_length = model_settings["N"]
characteristic_length = 256
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=2,
                     output_size=1,
                     hidden_layers=[characteristic_length] * 6,
                     activation_settings={"type":"sin",
                                          "prediction_gain":30,
                                          "initialization_gain":1.0},
                     skip_connections_settings={"active":False,"frequency":1})

latent_size = characteristic_length
modulator_nn = MLP(name="modulator_nn",
                   input_size=latent_size,
                   use_bias=False) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 10000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-6, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=no_control,
                                            loss_function=phasefield_loss_2d_pred,
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3,
                                            checkpoint_settings={"restore_state":False,
                                            "state_directory":case_dir+"/flax_state"},
                                            working_directory=case_dir)
fol.Initialize()


train_start_id = 0
train_end_id = 8000
test_start_id = 1*train_end_id
test_end_id = int(1.2*train_end_id)
# # here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
#         #   test_set=(coeffs_matrix[test_start_id:test_end_id,:],),
#         #   test_settings={"test_frequency":10},
fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
            batch_size=100,
            convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
            plot_settings={"plot_save_rate":100},
            save_settings={"save_nn_model":True,
                         "best_model_checkpointing":True,
                         "best_model_checkpointing_frequency":100})

# # load teh best model
fol.RestoreCheckPoint(fol.checkpoint_settings)
# relative_L2_error = 0.0
# for test in range(test_start_id,test_end_id):
#     eval_id = test
#     FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
#     fe_mesh['T_FOL'] = FOL_T.reshape((fe_mesh.GetNumberOfNodes(), 1))

#     # solve FE here
#     fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl","tol":1e-6,"atol":1e-6,
#                                                 "maxiter":1000,"pre-conditioner":"ilu"},
#                     "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
#                                                 "maxiter":10,"load_incr":5}}
#     linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",phasefield_loss_2d,fe_setting)
#     linear_fe_solver.Initialize()
#     FE_T = np.array(linear_fe_solver.Solve(coeffs_matrix[eval_id],coeffs_matrix[eval_id]))  
#     fe_mesh['T_FE'] = FE_T.reshape((fe_mesh.GetNumberOfNodes(), 1))

#     absolute_error = abs(FOL_T.reshape(-1,1)- FE_T.reshape(-1,1))
#     fe_mesh['abs_error'] = absolute_error#.reshape((fe_mesh.GetNumberOfNodes(), 1))


#     plot_mesh_vec_data(1,[coeffs_matrix[eval_id],FOL_T,FE_T,absolute_error],
#                     ["T_init","FOL_T","FE_T","abs_error"],
#                     fig_title="implicit FOL solution and error",cmap = "jet",
#                     file_name=os.path.join(case_dir,f"FOL-T-dist_test_{eval_id}.png"))
#     relative_L2_error += np.linalg.norm(absolute_error)/np.linalg.norm(FE_T)

# print("Average relative L2 error:", relative_L2_error/test_end_id)

# relative_L2_error = 0.0
num_steps = 50
eval_start_id = 0
eval_end_id = 1
FOL_T = np.zeros((fe_mesh_pred.GetNumberOfNodes(),num_steps))
for test in range(eval_start_id,eval_end_id):
    eval_id = test
    FOL_T_temp  = coeffs_matrix[eval_id,:]
    for i in range(num_steps):
        FOL_T_temp = np.array(fol.Predict(FOL_T_temp.reshape(-1,1).T)).reshape(-1)
        FOL_T[:,i] = FOL_T_temp 

    fe_mesh['T_FOL'] = FOL_T#.reshape((fe_mesh.GetNumberOfNodes(), 1))

    # solve FE here
    fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu","Dirichlet_BCs":Dirichlet_BCs},
                    "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                "maxiter":10,"load_incr":5}}
    nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolverPhasefield("nonlinear_fe_solver",phasefield_loss_2d_pred,fe_setting)
    nonlinear_fe_solver.Initialize()
    FE_T = np.zeros((fe_mesh_pred.GetNumberOfNodes(),num_steps))
    FE_T_temp = coeffs_matrix[eval_id,:]
    for i in range(num_steps):
        FE_T_temp = np.array(nonlinear_fe_solver.Solve(FE_T_temp,FE_T_temp))  #np.zeros(fe_mesh.GetNumberOfNodes())
        FE_T[:,i] = FE_T_temp    
    fe_mesh['T_FE'] = FE_T#.reshape((fe_mesh.GetNumberOfNodes(), 1))

    absolute_error = np.abs(FOL_T- FE_T)
    fe_mesh['abs_error'] = absolute_error#.reshape((fe_mesh.GetNumberOfNodes(), 1))


    # plot_mesh_vec_data(1,[coeffs_matrix[eval_id],FOL_T,FE_T,absolute_error],
    #                 ["T_init","FOL_T","FE_T","abs_error"],
    #                 fig_title="implicit FOL solution and error",cmap = "jet",
    #                 file_name=os.path.join(case_dir,f"FOL-T-dist_test_{eval_id}.png"))
    # time_list = [int(num_steps/5) - 1,int(num_steps/2) - 1,num_steps - 1]
    # plot_mesh_vec_data_phasefield(1,[coeffs_matrix[eval_id],FOL_T[:,time_list[0]],FOL_T[:,time_list[1]],FOL_T[:,time_list[2]]],
    #             ["","","",""],
    #             fig_title="",cmap = "jet",
    #             file_name=os.path.join(case_dir,f"FOL-dist_test_{eval_id}_zssr.png"))
    
    # plot_mesh_vec_data_phasefield(1,[coeffs_matrix[eval_id],FE_T[:,time_list[0]],FE_T[:,time_list[1]],FE_T[:,time_list[2]]],
    #             ["","","",""],
    #             fig_title="",cmap = "jet",
    #             file_name=os.path.join(case_dir,f"FE-dist_test_{eval_id}_zssr.png"))
    # plot_mesh_vec_data(1,[coeffs_matrix[eval_id],absolute_error[:,time_list[0]],absolute_error[:,time_list[1]],absolute_error[:,time_list[2]]],
    #             ["","","",""],
    #             fig_title="",cmap = "jet",
    #             file_name=os.path.join(case_dir,f"FOL-FE-error-dist_test_{eval_id}_zssr.png"))
fe_mesh.Finalize(export_dir=case_dir)
