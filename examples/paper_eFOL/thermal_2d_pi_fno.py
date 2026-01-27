import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
import numpy as np
import optax
from flax import nnx
import jax
from fol.loss_functions.thermal import ThermalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.deep_neural_networks.fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearning
from fol.deep_neural_networks.fourier_neural_operator_networks import FourierNeuralOperator2D
from fol.deep_neural_networks.ported_fourier_neural_operator_networks.fno import FNO
from fol.controls.fourier_control import FourierControl
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import MLP
from fol.tools.decoration_functions import *
from flax.nnx import bridge
from utilities import *
import pickle

# jax.config.update('jax_default_matmul_precision','high')
# jax.config.update('jax_enable_x64', True)

# directory & save handling
working_directory_name = 'nn_output_thermal_pi_fno_2D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":52,
                "left":1.0,"right":0.1}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["left"],"right":model_settings["right"]}}

thermal_loss_2d = ThermalLoss2DQuad("thermal_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,"loss_function_exponent":2,
                                                                        "beta":2,"c":4},
                                                                        fe_mesh=fe_mesh)

# create Fourier parametrization/control
x_freqs = np.array([3,5,7])
y_freqs = np.array([2,4,7])
z_freqs = np.array([0])
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":5,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

fe_mesh.Initialize()
thermal_loss_2d.Initialize()
fourier_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 10000
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = {}
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = x_freqs
    export_dict["y_freqs"] = y_freqs
    export_dict["z_freqs"] = z_freqs
    with open(f'fourier_control_dict_efol_paper.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(os.path.join(os.path.dirname(__file__),f'fourier_control_dict_efol_paper.pkl'), 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]
    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)


# now save K matrix 
export_Ks = False
if export_Ks:
    for i in range(K_matrix.shape[0]):
        fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
    fe_mesh.Finalize(export_dir=case_dir)


# ----------------------------------------------------------------------------
# 3) Build the FNO (Fourier Neural Operator) in JAX/Flax 
# ----------------------------------------------------------------------------
fno_model = FNO(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    n_modes=(12,12),
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    rngs=nnx.Rngs(0)
)

# Count trainable parameters 
params = nnx.state(fno_model, nnx.Param)
total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"FNO trainable parameters:{total_params}")

# Sanity-check forward pass with a small batch:
# FNO expects shape (batch, Nx, Ny, channels)
# ISince K_matrix is stored as flattened vectors, we reshape to (B,N,N,1)
init_out = fno_model(K_matrix[0:8].reshape(8,model_settings["N"],model_settings["N"],1))


num_epochs = 500
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-6, transition_steps=num_epochs)
lr = 1e-5
optimizer = optax.chain(optax.adam(lr))

# create fol
pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(name="dd_fno_pr_learning",
                                                                control=fourier_control,
                                                                loss_function=thermal_loss_2d,
                                                                flax_neural_network=fno_model,
                                                                optax_optimizer=optimizer)

pi_fno_pr_learning.Initialize()

train_start_id = 0
train_end_id = 1
test_start_id = 999
test_end_id = 1000
#here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
pi_fno_pr_learning.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
                        test_set=(coeffs_matrix[test_start_id:test_end_id,:],),
                        test_frequency=100,
                        batch_size=100,
                        convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                        plot_settings={"plot_save_rate":100},
                        train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
                        working_directory=case_dir)

# load teh best model
pi_fno_pr_learning.RestoreState(restore_state_directory=case_dir+"/flax_final_state")

for test in [1,2,3]:
        eval_id = test
        FOL_T = np.array(pi_fno_pr_learning.Predict(coeffs_matrix[eval_id].reshape(-1,1).T)).reshape(-1)
        fe_mesh[f'T_FOL_{eval_id}'] = FOL_T.reshape((fe_mesh.GetNumberOfNodes(), 1))
        fe_mesh[f'K_{eval_id}'] = K_matrix[eval_id].reshape((fe_mesh.GetNumberOfNodes(), 1))

        # solve FE here
        fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                    "nonlinear_solver_settings":{"rel_tol":1e-6,"abs_tol":1e-6,
                                                    "maxiter":20,"load_incr":4}}
        nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",thermal_loss_2d,fe_setting)
        nonlin_fe_solver.Initialize()
        FE_T = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id],np.zeros(fe_mesh.GetNumberOfNodes())))  
        fe_mesh[f'T_FE_{eval_id}'] = FE_T.reshape((fe_mesh.GetNumberOfNodes(), 1))

        absolute_error = abs(FOL_T.reshape(-1,1)-FE_T.reshape(-1,1))
        relative_error = 100 * absolute_error/abs(FE_T.reshape(-1,1))
        fe_mesh[f'relative_error_{eval_id}'] = relative_error.reshape((fe_mesh.GetNumberOfNodes(), 1))
        fe_mesh[f'absolute_error_{eval_id}'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 1))

        plot_thermal_paper(vectors_list=[K_matrix[eval_id],FOL_T,FE_T], file_name=case_dir+f"/thermal_2d_sample_{eval_id}")
