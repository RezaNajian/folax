import sys
import os
import optax
import numpy as np
from fol.loss_functions.thermal_transient_2D_fe_quad import ThermalTransientLoss2DQuad
from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# directory & save handling
working_directory_name = 'meta_learning_thermal_transient_2D_one_modulator_per_synthesizer_layer_FiLM3'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":21,
                "T_left":1.0,"T_right":0.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]}}

material_dict = {"rho":1.0,"cp":1.0,"dt":0.05}
thermal_loss_2d = ThermalTransientLoss2DQuad("thermal_transient_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)
no_control = NoControl("No_control",fe_mesh)

fe_mesh.Initialize()
thermal_loss_2d.Initialize()
no_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = True
if create_random_coefficients:
    def generate_random_smooth_patterns(L, N, num_samples=1000):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)

        # Create the grid
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        X1, X2 = np.meshgrid(x, y)
        X = np.vstack([X1.ravel(), X2.ravel()]).T

        # Generate multiple samples
        y_samples = gp.sample_y(X, n_samples=num_samples, random_state=0)

        # Normalize each sample
        scaled_y_samples = np.array([((y_sample - np.min(y_sample)) / (np.max(y_sample) - np.min(y_sample))) 
                                     for y_sample in y_samples.T])

        return scaled_y_samples
    coeffs_matrix = generate_random_smooth_patterns(model_settings["L"],model_settings["N"])
else:
    pass
    # with open(f'no_control_dict.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)
    # coeffs_matrix = loaded_dict["coeffs_matrix"]

# ATTENTION: we need to normalize the features
# coeffs_matrix_min = np.min(coeffs_matrix)
# coeffs_matrix_max = np.max(coeffs_matrix)
# no_control.scale_min = coeffs_matrix_min
# no_control.scale_max = coeffs_matrix_max
# coeffs_matrix = (coeffs_matrix-coeffs_matrix_min)/(coeffs_matrix_max-coeffs_matrix_min)
# print(coeffs_matrix.shape)

T_matrix = no_control.ComputeBatchControlledVariables(coeffs_matrix)

# export_Ks = False
# if export_Ks:
#     for i in range(K_matrix.shape[0]):
#         fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
#     fe_mesh.Finalize(export_dir=case_dir)
#     exit()

# design synthesizer & modulator NN for hypernetwork
num_nodes = model_settings["N"]
hidden_layers = [model_settings["N"]]*6
# hidden_layers = [250,250]
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=3,
                     output_size=1,
                     hidden_layers=hidden_layers,
                     activation_settings={"type":"sin",
                                          "prediction_gain":30,
                                          "initialization_gain":1.0},
                     skip_connections_settings={"active":False,"frequency":1})

modulator_nn = MLP(name="modulator_nn",
                   input_size=101)# Can play around with this latent size (maybe 64-128)
                   #hidden_layers=hidden_layers,# Remove this hidden layer for FiLM
                   #activation_settings={"type":"relu"},
                   #skip_connections_settings={"active":False,"frequency":1}) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
main_loop_transform = optax.chain(optax.adam(1e-5))
latent_loop_transform = optax.chain(optax.adam(1e-4))

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=no_control,
                                                loss_function=thermal_loss_2d,
                                                flax_neural_network=hyper_network,
                                                latent_loop_optax_optimizer=latent_loop_transform,
                                                main_loop_optax_optimizer=main_loop_transform,
                                                checkpoint_settings={"restore_state":False,
                                                "state_directory":case_dir+"/flax_state"},
                                                working_directory=case_dir)
fol.Initialize()


train_start_id = 0
train_end_id = 50

# here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),batch_size=1,
            convergence_settings={"num_epochs":10000,"relative_error":1e-100,
                                  "absolute_error":1e-100},
            plot_settings={"plot_save_rate":100},
            save_settings={"save_nn_model":True})

eval_start_id = 0
eval_end_id = 60
for test in range(eval_start_id,eval_end_id):
    eval_id = test
    FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
    fe_mesh['T_FOL'] = FOL_T.reshape((fe_mesh.GetNumberOfNodes(), 1))

    # solve FE here
    fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu"},
                    "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                "maxiter":10,"load_incr":5}}
    linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",thermal_loss_2d,fe_setting)
    linear_fe_solver.Initialize()
    FE_T = np.array(linear_fe_solver.Solve(coeffs_matrix[eval_id],coeffs_matrix[eval_id]))  
    fe_mesh['T_FE'] = FE_T.reshape((fe_mesh.GetNumberOfNodes(), 1))

    absolute_error = abs(FOL_T.reshape(-1,1)- FE_T.reshape(-1,1))
    fe_mesh['abs_error'] = absolute_error#.reshape((fe_mesh.GetNumberOfNodes(), 1))


    plot_mesh_vec_data(1,[coeffs_matrix[eval_id],FOL_T,FE_T,absolute_error],
                    ["T_init","FOL_T","FE_T","abs_error"],
                    fig_title="implicit FOL solution and error",cmap = "jet",
                    file_name=os.path.join(case_dir,f"FOL-T-dist_test_{eval_id}.png"))
    
fe_mesh.Finalize(export_dir=case_dir)
