import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import jax
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.loss_functions.mechanical_saint_venant import SaintVenantMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.identity_control import IdentityControl
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearning
from fol.deep_neural_networks.ported_fourier_neural_operator_networks.fno import FNO
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
import optax
from flax import nnx
from mechanical2d_utilities import *
from fol.tools.decoration_functions import *
import requests
import zipfile


# directory & save handling
working_directory_name = "2d_hyperelastic_fno"   # should be the same dir that contains network parameters
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":42,
                "Ux_left":0.0,"Ux_right":0.5,
                "Uy_left":0.0,"Uy_right":0.5}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()

# create fe-based loss function
bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
            "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}

material_dict = {"young_modulus":1,"poisson_ratio":0.3}

mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                    "num_gp":2,
                                                                                    "material_dict":material_dict},
                                                    fe_mesh=fe_mesh)

mechanical_loss_2d.Initialize()

# define a control class which reconstrcut the input space from a reduced space
# identity control maps X: -> X
identity_control = IdentityControl('identity_control', control_settings={},num_vars=fe_mesh.GetNumberOfNodes())
identity_control.Initialize()


# define fourier control to create synthethic microstructures
fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                            "beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)
fourier_control.Initialize()

# load fourier coefficients and compute K
with open(os.path.join('.',f'fourier_control_dict.pkl'), 'rb') as f:
    loaded_dict = pickle.load(f)
coeffs_matrix = loaded_dict["coeffs_matrix"]

K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

# K_matrix_tests = np.loadtxt("ifol_fourier_test_samples_K_matrix_res_81.txt")
# print(K_matrix_tests.shape)

# ----------------------------------------------------------------------------
# Build the FNO (Fourier Neural Operator) in JAX/Flax 
# ----------------------------------------------------------------------------
fno_dict = {"in_channel": 1, "out_channel":2,
            "hidden_channels":32,"n_modes":(8,8),"n_layers":4,
            "lifting_channel_ratio":2, "projection_channel_ratio":2}

fno_model = FNO(
    in_channels=fno_dict["in_channel"],
    out_channels=fno_dict["out_channel"],
    hidden_channels=fno_dict["hidden_channels"],
    n_modes=fno_dict["n_modes"],
    n_layers=fno_dict["n_layers"],
    lifting_channel_ratio=fno_dict["lifting_channel_ratio"],
    projection_channel_ratio=fno_dict["projection_channel_ratio"],
    scale_factor=0.1,
    rngs=nnx.Rngs(0)
)

# Count trainable parameters 
params = nnx.state(fno_model, nnx.Param)
total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"FNO trainable parameters:{total_params}")

# Sanity-check forward pass with a small batch:
# FNO expects shape (batch, Nx, Ny, channels)
# ISince K_matrix is stored as flattened vectors, we reshape to (B,N,N,1)
init_out = fno_model(K_matrix[0:8,:].reshape(8,model_settings["N"],model_settings["N"],fno_dict["in_channel"]))


num_epochs = 2000
lr = 1e-5
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=lr, transition_steps=num_epochs)
optimizer = optax.chain(optax.adam(learning_rate_scheduler))

# create fol
pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(name="dd_fno_pr_learning",
                                                                        control=identity_control,
                                                                        loss_function=mechanical_loss_2d,
                                                                        flax_neural_network=fno_model,
                                                                        optax_optimizer=optimizer)

pi_fno_pr_learning.Initialize()



# split the data to train and test sets
otf_id = 0
train_set_otf = K_matrix[otf_id,:].reshape(-1,1).T     # for On The Fly training

train_start_id = 0
train_end_id = 160
train_set_pr = K_matrix[train_start_id:train_end_id,:]     # for parametric training

test_start_id = 180
test_end_id = 200
test_set_pr = K_matrix[test_start_id:test_end_id,:]

# OTF or Parametric 
parametric_learning = True
if parametric_learning:
    train_set = train_set_pr
    test_set = test_set_pr
    tests = range(test_start_id,test_end_id)
else:
    train_set = train_set_otf   
    test_set = train_set
    tests = [otf_id]
# here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
train_settings_dict = {"batch_size": 8,
                        "num_epoch":num_epochs,
                        "parametric_learning": parametric_learning,
                        "OTF_id": otf_id,
                        "train_start_id": train_start_id,
                        "train_end_id": train_end_id,
                        "test_start_id": test_start_id,
                        "test_end_id": test_end_id}


train_from_scratch = True
if train_from_scratch:
    pi_fno_pr_learning.Train(train_set=(train_set,),
                        test_set=(test_set,),
                        batch_size=train_settings_dict["batch_size"],
                        restore_nnx_state_settings={'restore':False, "state_directory":case_dir+"/flax_train_state"},
                        convergence_settings={"num_epochs":train_settings_dict["num_epoch"],"relative_error":1e-100,"absolute_error":1e-100},
                        plot_settings={"save_frequency":10},
                        train_checkpoint_settings={"least_loss_checkpointing":False,"frequency":10},
                        test_checkpoint_settings={"least_loss_checkpointing":False,"frequency":10},
                        data_model_sharding_settings ={"sharding":False,"num_data_devices":4,"num_nnx_model_devices":1},
                        working_directory=case_dir)
else:
    # load the best model
    pi_fno_pr_learning.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

test_ids = [0, 4, 5, 7, 11, 12, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
rho = []
cosine = []

use_warmstart = True
# for eval_id in test_ids[:10]:
for eval_id in range(10):
    # predict the result from fno
    fno_uvw = np.array(pi_fno_pr_learning.Predict(K_matrix[eval_id].reshape(-1,1).T)).reshape(-1)
    fe_mesh[f'FNO_U_{eval_id}'] = fno_uvw.reshape((fe_mesh.GetNumberOfNodes(), 2))
    fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))
    # iFOL_stress = compute_stress_neohooke_quad(loss_function=mechanical_loss_2d, disp_field_vec=jnp.array(ifol_uvw), K_matrix=jnp.array(K_matrix[eval_id,:]))

    # solve FE here to compare the result
    fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                "nonlinear_solver_settings":{"rel_tol":1e-7,"abs_tol":5e-8,
                                                "maxiter":4,"load_incr":56, 
                                                # --- Line search ---
                                                "line_search_type": "none",  # "none", "residual", "energy"
                                                "ls_beta": 0.5,
                                                "ls_c": 1e-4,
                                                "ls_maxiter": 12,}}
    nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_2d,fe_setting,history_plot_settings={"plot":True})
    nonlin_fe_solver.Initialize()
    if not use_warmstart:
        FE_UVW = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id,:],np.zeros(2*fe_mesh.GetNumberOfNodes())))  
        # for load_step in nonlin_fe_solver.load_history.keys():
        #     for iter in nonlin_fe_solver.load_history[load_step].keys():
        #         fe_mesh[f"Newton_step_{load_step}_{iter}"] = nonlin_fe_solver.load_history[load_step][iter].reshape((fe_mesh.GetNumberOfNodes(), 2))

        fe_mesh[f'FE_U_{eval_id}'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))

        abs_err = abs(FE_UVW.reshape(-1,1) - fno_uvw.reshape(-1,1))
        # fe_mesh[f"abs_U_error_{eval_id}"] = abs_err.reshape((fe_mesh.GetNumberOfNodes(), 2))

        # FE_stress = compute_stress_neohooke_quad(loss_function=mechanical_loss_2d, disp_field_vec=jnp.array(FE_UVW), K_matrix=jnp.array(K_matrix[eval_id,:]))
        # stress_error = abs(iFOL_stress.reshape(-1) - FE_stress.reshape(-1))
        # fe_mesh[f"FE_FirstPiola_{eval_id}"] = FE_stress.reshape((fe_mesh.GetNumberOfNodes(), 3))
        # fe_mesh[f"iFOL_FirstPiola_{eval_id}"] = iFOL_stress.reshape((fe_mesh.GetNumberOfNodes(), 3))
        # fe_mesh[f"error_FirstPiola_{eval_id}"] = stress_error.reshape((fe_mesh.GetNumberOfNodes(), 3))


    # solve the Newton-Raphson initialized by ifol in one load increment
    fol_info("solve fe hybrid in one load step")
    nin_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                                "maxiter":10,"load_incr":1}}
    nin_nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nin_nonlin_fe_solver",mechanical_loss_2d,nin_setting)
    nin_nonlin_fe_solver.Initialize()
    if use_warmstart:
        try:    
            NiN_UVW = np.array(nin_nonlin_fe_solver.Solve(K_matrix[eval_id,:],fno_uvw.reshape(2*fe_mesh.GetNumberOfNodes())))  
            # rho.append((nin_nonlin_fe_solver.rho,eval_id))
            # cosine.append((nin_nonlin_fe_solver.cosine,eval_id))
            # print(f"rho values : {nin_nonlin_fe_solver.rho}")
        except Exception as e:
            fol_info(f"Error occured {type(e).__name__}: e")
            NiN_UVW = np.zeros(2*fe_mesh.GetNumberOfNodes())
            # rho.append((nin_nonlin_fe_solver.rho,eval_id))
            # cosine.append((nin_nonlin_fe_solver.cosine,eval_id))
            # print(f"rho values : {nin_nonlin_fe_solver.rho}")

        fe_mesh[f'NiN_U_{eval_id}'] = NiN_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))

    # plot the result
    # plot_iFOL_HFE(topology_field=K_matrix[eval_id,:], ifol_sol_field=ifol_uvw.reshape(2*fe_mesh.GetNumberOfNodes()), hfe_sol_field=NiN_UVW,
    #             err_sol_field=abs_err, file_name=os.path.join(case_dir,'plots')+f"/ifol_fe-nin_error_{eval_id}",
    #             fig_titles=['Elasticity Morph.','iFOL','FE-NIN','iFOL-FE Abs Diff.'])

# export the result in a .vtk file
fe_mesh.Finalize(export_dir=case_dir)
