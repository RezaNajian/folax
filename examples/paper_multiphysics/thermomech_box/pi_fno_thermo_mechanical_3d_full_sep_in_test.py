import sys
import os
import optax
import numpy as np
from fol.loss_functions.thermo_mechanical_nonlinear import ThermoMechanicalLoss3DHexa
from fol.controls.fourier_control import FourierControl
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.controls.identity_control import IdentityControl
from thermo_mechanical_useful_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import MLP

from fourier_neural_operator_networks import FourierNeuralOperator3DFullSeparate
from fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearning

from fol.tools.decoration_functions import *
from flax.nnx import bridge
import pickle
from flax import nnx
import jax

jax.config.update('jax_default_matmul_precision','high')
# jax.config.update('jax_enable_x64', True)

# directory & save handling
main_name = 'pi_fno_thermomech_3D_full_sep'
working_directory_name = main_name + '_in_test'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

model_settings = {"L":1,"N":22,
                  "Ux_left":0.0,"Ux_right":0.10,
                  "Uy_left":0.0,"Uy_right":0.10,
                  "Uz_left":0.0,"Uz_right":0.10,
                  "T_left":0.5,"T_right":0.0}
# creation of the model
mesh_res_rate = 1
fe_mesh = create_3D_box_mesh_structured(Nx=model_settings["N"],Ny=model_settings["N"],Nz=model_settings["N"],
                             Lx=model_settings["L"],Ly=model_settings["L"],Lz=model_settings["L"])
# # create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]},
           "Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
           "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]},
           "Uz":{"left":model_settings["Uz_left"],"right":model_settings["Uz_right"]}}#
Dirichlet_BCs = False
initial_temp = np.full((1,fe_mesh.GetNumberOfNodes()),1e-4)
material_dict = {"young_modulus":1.0,"poisson_ratio":0.3,"T0":initial_temp.flatten()}
thermomech_loss_3d = ThermoMechanicalLoss3DHexa("thermomechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "material_dict":material_dict, "alpha":1.5,
                                                                            "beta":2.0, 
                                                                            "c":2.0},
                                                                            fe_mesh=fe_mesh)
no_control = IdentityControl("No_Control",fe_mesh)
fe_mesh.Initialize()
thermomech_loss_3d.Initialize()
no_control.Initialize()
# create fe-based loss function

initial_temp = np.full((1,fe_mesh.GetNumberOfNodes()),0.0)
material_dict = {"young_modulus":1.0,"poisson_ratio":0.3,"T0":initial_temp.flatten()}
# freq_sets = [
#     (np.array([2, 4, 6]),    np.array([2, 4, 6]))
# ]
freq_sets = [      
    (np.array([2, 4, 6]),    np.array([2, 4, 6])),
    (np.array([1, 2, 3]),    np.array([1, 2, 3])),
    (np.array([3, 4, 5]),    np.array([3, 4, 5])),
    (np.array([4, 6, 8]),    np.array([4, 6, 8]))
]

N_samples_per_set = 1500

freq_sets = [      
    (np.array([2, 4, 6]),    np.array([2, 4, 6]),   np.array([2, 4, 6])),
    (np.array([1, 2, 3]),    np.array([1, 2, 3]),   np.array([1, 2, 3])),
    (np.array([3, 4, 5]),    np.array([3, 4, 5]),   np.array([3, 4, 5])),
    (np.array([4, 6, 8]),    np.array([4, 6, 8]),   np.array([4, 6, 8]))
]

all_K_list = []
create_random_coefficients = False
if create_random_coefficients:
    for idx, (x_freqs, y_freqs, z_freqs) in enumerate(freq_sets):
        control_settings = {
            "x_freqs": x_freqs,
            "y_freqs": y_freqs,
            "z_freqs": z_freqs,
            "beta": 10,
            "min": 0.1,
            "max": 1.0
        }

        fourier_control = FourierControl("K", control_settings, fe_mesh)
        fourier_control.Initialize()

        coeffs_matrix, K_matrix = create_random_fourier_samples(fourier_control, N_samples_per_set)
        np.save(f"coeffs_matrix_{idx}.npy", coeffs_matrix)
        # np.save(os.path.join(case_dir, f"K_matrix_{idx}.npy"), K_matrix)
        all_K_list.append(K_matrix)
        # all_labels.append(np.full(N_samples_per_set, idx))  
    # coeffs_matrix_all = np.vstack(all_coeffs_list)
    K_matrix = np.vstack(all_K_list)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(K_matrix)
    E_matrix = K_matrix

else:
    for idx, (x_freqs, y_freqs, z_freqs) in enumerate(freq_sets):
        control_settings = {
            "x_freqs": x_freqs,
            "y_freqs": y_freqs,
            "z_freqs": z_freqs,
            "beta": 10,
            "min": 0.1,
            "max": 1.0
        }
        coeffs_matrix = np.load(f"coeffs_matrix_{idx}.npy")
        fourier_control = FourierControl("K", control_settings, fe_mesh)
        fourier_control.Initialize()
        K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)
        all_K_list.append(K_matrix)
    K_matrix = np.vstack(all_K_list)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(K_matrix)
    E_matrix = K_matrix


loss_settings={"dirichlet_bc_dict":bc_dict,
               "material_dict":material_dict,
               "loss_function_exponent":1.0}
thermomech_loss_3d = ThermoMechanicalLoss3DHexa("thermothermomech_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "material_dict":material_dict, "alpha":1.5,
                                                                            "beta":2.0, 
                                                                            "c":2.0},
                                                                            fe_mesh=fe_mesh)
thermomech_loss_3d.Initialize()

def merge_state(dst: nnx.State, src: nnx.State):
    for k, v in src.items():
        if isinstance(v, nnx.State):
            merge_state(dst[k], v)
        else:
            dst[k] = v

fno_model = bridge.ToNNX(FourierNeuralOperator3DFullSeparate(modes1=10,
                                                modes2=10,
                                                modes3=10,
                                                width=8,
                                                depth=4,
                                                channels_last_proj=128,
                                                out_channels=1,
                                                padding=4,
                                                output_scale=0.001),rngs=nnx.Rngs(0)).lazy_init(K_matrix[0:1].reshape(1,model_settings["N"],model_settings["N"],model_settings["N"],1)) 

# replace RNG key by a dummy to allow checkpoint restoration later
graph_def, state = nnx.split(fno_model)
rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
merge_state(state, rngs_key)
fno_model = nnx.merge(graph_def, state)

# get total number of fno params
params = nnx.state(fno_model, nnx.Param)
total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"total number of fno network param:{total_params}")

num_epochs = 1000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-2, end_value=1e-3, transition_steps=num_epochs)
optimizer = optax.chain(optax.adam(learning_rate_scheduler))

# create fol
pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(name="pi_fno_pr_learning",
                                                                        control=no_control,
                                                                        loss_function=thermomech_loss_3d,
                                                                        flax_neural_network=fno_model,
                                                                        optax_optimizer=optimizer)

pi_fno_pr_learning.Initialize()

train_start_id = 0
train_end_id = 5000
test_start_id = 5000
test_end_id = 6000
# train_start_id = 0
# train_end_id = 1
# test_start_id = 0
# test_end_id = 1
#here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
# pi_fno_pr_learning.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
#                         test_set=(coeffs_matrix[test_start_id:test_end_id,:],),
#                         test_frequency=100,
#                         batch_size=100,
#                         convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
#                         plot_settings={"plot_save_rate":100},
#                         train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
#                         working_directory=case_dir)
# pi_fno_pr_learning.Train(train_set=(K_matrix[train_start_id:train_end_id,:],),
#           batch_size=50,
#           convergence_settings={"num_epochs":num_epochs,
#                                 "relative_error":1e-100,
#                                 "absolute_error":1e-100},
#           train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
#           working_directory=case_dir,
#           plot_settings={"plot_list":["total_loss","phy1_loss","phy2_loss"],
#                          "plot_frequency":1,"save_frequency":100,
#                          "save_directory":".","multiphysics":True})
# load teh best model
pi_fno_pr_learning.RestoreState(restore_state_directory=main_name+"/flax_final_state")


# ---- config ----
# start_id = 0
# num_samples = 50
test_start_id = 5000
test_end_id = 5020
# initialize FE solver once
fe_setting = {
    "linear_solver_settings":{
        "solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
        "maxiter":1000,"pre-conditioner":"ilu"},
    "nonlinear_solver_settings":{
        "rel_tol":1e-7,"abs_tol":1e-7,"maxiter":10,"load_incr":5}
}
nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolver(
    "nonlinear_fe_solver", thermomech_loss_3d, fe_setting
)
nonlinear_fe_solver.Initialize()
K_matrix_all = K_matrix
for i in range(test_start_id, test_end_id):    
    K_matrix_temp = K_matrix_all[i:i+1]
    FOL_TUVW = np.array(pi_fno_pr_learning.Predict(K_matrix_temp)) 
    FOL_TUVW = FOL_TUVW.reshape((fe_mesh.GetNumberOfNodes(), 4))
    fe_mesh['K_matrix'] = K_matrix.reshape(-1,1)
    fe_mesh['sol_FOL'] = FOL_TUVW

    nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlinear_fe_solver",thermomech_loss_3d,fe_setting)
    nonlinear_fe_solver.Initialize()

    FE_TUVW = np.array(nonlinear_fe_solver.Solve(K_matrix_temp.flatten(),np.zeros((fe_mesh.GetNumberOfNodes()*4))))  #
    FE_TUVW = reshape_T_U_to_nodewise3D(FE_TUVW, fe_mesh.GetNumberOfNodes())
    fe_mesh['sol_FE'] = FE_TUVW
    fe_mesh['sol_Disp'] = FE_TUVW[:,1:]
    absolute_error = np.abs(FOL_TUVW- FE_TUVW)
    fe_mesh['abs_error'] = absolute_error

    # # Stress and heat flux
    FOL_UVW = FOL_TUVW[:,1:]
    FE_UVW = FE_TUVW[:,1:]
    FE_T = FE_TUVW[:,0]
    FOL_T = FOL_TUVW[:,0]

    FOLstress_at_nodes = GetStressVector3D(thermomech_loss_3d,fe_mesh,  K_matrix_temp.flatten(),
                            FOL_UVW.flatten(),FOL_T.flatten(),initial_temp.flatten())
    FE_stress_at_nodes = GetStressVector3D(thermomech_loss_3d,fe_mesh,  K_matrix_temp.flatten(),
                                           FE_UVW.flatten(),FE_T.flatten(),initial_temp.flatten())
    absolute_error = np.abs(FOLstress_at_nodes - FE_stress_at_nodes)
    fe_mesh['FOL_stress'] = FOLstress_at_nodes
    fe_mesh['FE_stress'] = FE_stress_at_nodes   
    fe_mesh['abs_error_stress'] = absolute_error

    FOL_heat_flux_at_nodes = GetHeatFluxVector3D(thermomech_loss_3d,fe_mesh,K_matrix_temp.flatten(),FOL_T.flatten())
    FE_heat_flux_at_nodes = GetHeatFluxVector3D(thermomech_loss_3d,fe_mesh,K_matrix_temp.flatten(),FE_T.flatten())
    heat_flux_absolute_error = np.abs(FOL_heat_flux_at_nodes - FE_heat_flux_at_nodes)
    fe_mesh['FOL_heat_flux'] = FOL_heat_flux_at_nodes
    fe_mesh['FE_heat_flux'] = FE_heat_flux_at_nodes
    fe_mesh['abs_error_heat_flux'] = heat_flux_absolute_error

    fe_mesh.mesh_io.write(os.path.join(case_dir, f"poly_test_case_{i}.vtu"),file_format="vtu")
