import sys
import os
import optax
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from mechanical2D_usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import MLP
from fol.deep_neural_networks.fourier_neural_operator_networks import FourierNeuralOperator2D
from fol.tools.decoration_functions import *
from flax.nnx import bridge
import pickle
from flax import nnx
import jax

jax.config.update('jax_default_matmul_precision','high')
jax.config.update('jax_enable_x64', True)

# directory & save handling
working_directory_name = 'pi_fno_mechanical_2D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":41,
                    "Ux_left":0.0,"Ux_right":0.5,
                    "Uy_left":0.0,"Uy_right":0.5}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()

# create fe-based loss function
bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
        "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}

fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                            "beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)
fourier_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 10
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = model_settings.copy()
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = fourier_control.x_freqs
    export_dict["y_freqs"] = fourier_control.y_freqs
    export_dict["z_freqs"] = fourier_control.z_freqs
    with open(f'fourier_control_dict.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'fourier_control_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]

K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

material_dict = {"young_modulus":1,"poisson_ratio":0.3}
loss_settings={"dirichlet_bc_dict":bc_dict,
               "material_dict":material_dict,
               "loss_function_exponent":1.0}
mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad("mechanical_loss_2d",loss_settings=loss_settings,
                                                                            fe_mesh=fe_mesh)
mechanical_loss_2d.Initialize()

fno_model = bridge.ToNNX(FourierNeuralOperator2D(modes1=12,
                                             modes2=12,
                                             width=32,
                                             depth=4,
                                             channels_last_proj=128,
                                             padding=45,
                                             out_channels=2,
                                             output_scale=0.1),rngs=nnx.Rngs(0)).lazy_init(K_matrix[0:1].reshape(1,model_settings["N"],model_settings["N"],1)) 

num_epochs = 1000
optimizer = optax.chain(optax.adam(1e-4))

# create fol
pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(name="pi_fno_pr_learning",
                                                                        control=fourier_control,
                                                                        loss_function=mechanical_loss_2d,
                                                                        flax_neural_network=fno_model,
                                                                        optax_optimizer=optimizer)

pi_fno_pr_learning.Initialize()

train_start_id = 0
train_end_id = 1
train_set_pr = coeffs_matrix[train_start_id:train_end_id,:]     # for parametric training
#here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
pi_fno_pr_learning.Train(train_set=(train_set_pr,),
                        batch_size=1,
                        convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                        plot_settings={"plot_save_rate":100},
                        train_checkpoint_settings={"least_loss_checkpointing":False,"frequency":10},
                        working_directory=case_dir)


# load teh best model
# pi_fno_pr_learning.RestoreState(restore_state_directory=case_dir+"/flax_final_state")

fe_setting = {"linear_solver_settings":{"solver":"JAX-direct","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu"},
                    "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                                "maxiter":20,"load_incr":40}}

for test in np.arange(train_start_id,train_end_id):
    eval_id = test
    FOL_UV = np.array(pi_fno_pr_learning.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
    fe_mesh[f'U_FOL_{eval_id}'] = FOL_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))
    fe_mesh[f'K_{eval_id}'] = K_matrix[eval_id].reshape((fe_mesh.GetNumberOfNodes(), 1))

    ## solve FE here
    linear_fe_solver = FiniteElementNonLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_2d,fe_setting)
    linear_fe_solver.Initialize()
    FE_UV = np.array(linear_fe_solver.Solve(K_matrix[eval_id].flatten(),jnp.zeros(2*fe_mesh.GetNumberOfNodes())))
    fe_mesh[f'U_FE_{eval_id}'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

    absolute_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
    fe_mesh['abs_error'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 2))

    # plot U
    vectors_list = [K_matrix[eval_id].flatten(),FOL_UV[::2],FE_UV[::2]]
    plot_mesh_res(vectors_list, file_name=case_dir+f'/plot_U_{eval_id}.png',dir="U")

    # plot V
    vectors_list = [K_matrix[eval_id].flatten(),FOL_UV[1::2],FE_UV[1::2]]
    plot_mesh_res(vectors_list, file_name=case_dir+f'/plot_V_{eval_id}.png',dir="V")


fe_mesh.Finalize(export_dir=case_dir, export_format='vtk')

