import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
# import jax
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_default_matmul_precision","highest")
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.identity_control import IdentityControl
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
import optax
from flax import nnx
from mechanical2d_utilities import *
from fol.tools.decoration_functions import *
import timeit, statistics

### Script's goal:
####### the following script is to 
####### check the inference time for iFOL and solving time for FE and HFE

# directory & save handling
working_directory_name = "mechanical_2d_base_from_ifol_meta"
case_dir = os.path.join('.', working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":21,
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

identity_control = IdentityControl('identity_control', control_settings={}, fe_mesh=fe_mesh)
identity_control.Initialize()


# load data dir
data_directory_name = f"{working_directory_name}_data"
data_dir = os.path.join('.', data_directory_name)

phase_contrast = 0.1

# load data from a pkl file
# load the lowest resolutions
with open(os.path.join(data_dir,f"U_base_res_21_bc_0.5_phase_contrast_{phase_contrast}.pkl"), 'rb') as f:
            U_dict_base = pickle.load(f)

# load the middle resolutions
with open(os.path.join(data_dir,f"U_base_res_41_bc_0.5_phase_contrast_{phase_contrast}.pkl"), 'rb') as f:
            U_dict_zssr1 = pickle.load(f)

# load the highest resolutions
with open(os.path.join(data_dir,f"U_base_res_81_bc_0.5_phase_contrast_{phase_contrast}.pkl"), 'rb') as f:
            U_dict_zssr2 = pickle.load(f)

# load a pretrained ifol
ifol_settings_dict = {
    "characteristic_length": 64,
    "synthesizer_depth": 4,
    "activation_settings":{"type":"sin",
                            "prediction_gain":30,
                            "initialization_gain":1.0},
    "skip_connections_settings": {"active":False,"frequency":1},
    "latent_size":  8*64,
    "modulator_bias": False,
    "main_loop_transform": 1e-5,
    "latent_step_optimizer": 1e-4,
    "ifol_nn_latent_step_size": 1e-2
}

# design synthesizer & modulator NN for hypernetwork
# characteristic_length = model_settings["N"]
characteristic_length = ifol_settings_dict["characteristic_length"]
synthesizer_nn = MLP(name="synthesizer_nn",
                    input_size=3,
                    output_size=2,
                    hidden_layers=[characteristic_length] * ifol_settings_dict["synthesizer_depth"],
                    activation_settings=ifol_settings_dict["activation_settings"],
                    skip_connections_settings=ifol_settings_dict["skip_connections_settings"])

latent_size = ifol_settings_dict["latent_size"]
modulator_nn = MLP(name="modulator_nn",
                input_size=latent_size,
                use_bias=ifol_settings_dict["modulator_bias"]) 

hyper_network = HyperNetwork(name="hyper_nn",
                            modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                            coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
#learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.adam(ifol_settings_dict["main_loop_transform"]))
latent_step_optimizer = optax.chain(optax.adam(ifol_settings_dict["latent_step_optimizer"]))

# create fol
ifol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_fol",control=identity_control,
                                                        loss_function=mechanical_loss_2d,
                                                        flax_neural_network=hyper_network,
                                                        main_loop_optax_optimizer=main_loop_transform,
                                                        latent_step_optax_optimizer=latent_step_optimizer,
                                                        latent_step_size=ifol_settings_dict["ifol_nn_latent_step_size"],
                                                        num_latent_iterations=3)
ifol.Initialize()


# load the best model
ifol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

# extract corresponding indices:
eval_id_list_zssr2 = []
for eval_id in range(400):
        if U_dict_zssr2.get(f"U_FE_81_{eval_id}") is not None:
                eval_id_list_zssr2.append(eval_id)


K_matrix_base_list = []
K_matrix_zssr1_list = []
K_matrix_zssr2_list = []
for eval_id in eval_id_list_zssr2:
    K_matrix_base_list.append(U_dict_base[f"K_matrix_21_{eval_id}"])
    K_matrix_zssr1_list.append(U_dict_zssr1[f"K_matrix_41_{eval_id}"])
    K_matrix_zssr2_list.append(U_dict_zssr2[f"K_matrix_81_{eval_id}"])

K_matrix_base = np.array(K_matrix_base_list)
K_matrix_zssr1 = np.array(K_matrix_zssr1_list)
K_matrix_zssr2 = np.array(K_matrix_zssr2_list)

################################# for iFOL inference #################################
def ifol_time(K_matrix, n_repeat=5,n_number=5):
    jit_time = timeit.repeat(lambda: ifol.Predict(K_matrix.reshape(-1,1).T), 
                            repeat=1, number=1)

    # Benchmark
    n_repeat = n_repeat   # How many times to repeat the timing
    n_number = n_number    # How many times to run the function in each repeat

    times = timeit.repeat(lambda: ifol.Predict(K_matrix.reshape(-1,1).T), 
                        repeat=n_repeat, number=n_number)
    normalized_times = np.array(times) / n_number

    print(f"jit time: {jit_time[0]:.6f} sec")
    print(f"Per-run mean time: {statistics.mean(normalized_times):.6f} sec")
    print(f"Per-run std dev  : {statistics.stdev(normalized_times):.6f} sec")

    return statistics.mean(normalized_times)

################################# for HFE #################################
hfe_setting = {"linear_solver_settings":{"solver":"JAX-direct","tol":1e-10,"atol":1e-10},
                                "nonlinear_solver_settings":{"rel_tol":1e-6,"abs_tol":1e-6,
                                                                "maxiter":20,"load_incr":1}}
hybrid_nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("hybrid_nonlin_fe_solver",mechanical_loss_2d,hfe_setting)
hybrid_nonlin_fe_solver.Initialize()


def hfem_time(K_matrix, ifol_uv, n_repeat=5, n_number=5):
    # ifol_uv = iFOL_UV[idx,:]
    jit_time = timeit.repeat(lambda: hybrid_nonlin_fe_solver.Solve(K_matrix,ifol_uv), 
                            repeat=1, number=1)

    # Benchmark
    n_repeat = n_repeat   # How many times to repeat the timing
    n_number = n_number    # How many times to run the function in each repeat

    times = timeit.repeat(lambda: hybrid_nonlin_fe_solver.Solve(K_matrix,ifol_uv), 
                        repeat=n_repeat, number=n_number)
    normalized_times = np.array(times) / n_number

    print(f"jit time: {jit_time[0]:.6f} sec")
    print(f"Per-run mean time: {statistics.mean(normalized_times):.6f} sec")
    print(f"Per-run std dev  : {statistics.stdev(normalized_times):.6f} sec")

    return statistics.mean(normalized_times)

################################# to save the results for different samples #################################
ifol_time_dict = {}
hfem_time_dict = {}
n_repeat = 2
n_number = 3
ifol_time_calc_list = []
hfem_time_calc_list = []
number_of_sample = 20

if model_settings["N"] == 21:
    K_matrix = K_matrix_base[:number_of_sample,:]
if model_settings["N"] == 41:
    K_matrix = K_matrix_zssr1[:number_of_sample,:]
if model_settings["N"] == 81:
    K_matrix = K_matrix_zssr2[:number_of_sample,:]

iFOL_UV = np.array(ifol.Predict(K_matrix))

res = int(K_matrix[0,:].size**0.5)
ifol_time_dict[f"for resolution: {res}"] = {}
hfem_time_dict[f"for resolution: {res}"] = {}

ifol_time_dict[f"for resolution: {res}"]["number of experiments"]= n_repeat
ifol_time_dict[f"for resolution: {res}"]["number of function calls"]= n_number

hfem_time_dict[f"for resolution: {res}"]["number of experiments"]= n_repeat
hfem_time_dict[f"for resolution: {res}"]["number of function calls"]= n_number

sample_counter = 0
for idx in range(number_of_sample):
    sample_counter += 1
    ifol_time_calc = ifol_time(K_matrix=K_matrix[idx,:],n_repeat=n_repeat,n_number=n_number)
    ifol_time_calc_list.append(ifol_time_calc)
    
    hfem_time_calc = hfem_time(K_matrix=K_matrix[idx,:],ifol_uv=iFOL_UV[idx,:],n_repeat=n_repeat,n_number=n_number)
    hfem_time_calc_list.append(hfem_time_calc)

ifol_mean_time_for_all_samples = statistics.mean(ifol_time_calc_list)
ifol_std_for_all_samples = statistics.stdev(ifol_time_calc_list)

hfem_mean_time_for_all_samples = statistics.mean(hfem_time_calc_list)
hfem_std_for_all_samples = statistics.stdev(hfem_time_calc_list)

ifol_time_dict[f"for resolution: {res}"]["Per-run mean time"]= ifol_mean_time_for_all_samples
ifol_time_dict[f"for resolution: {res}"]["Per-run std dev"]= ifol_std_for_all_samples
ifol_time_dict[f"for resolution: {res}"]["number of samples"]= sample_counter

hfem_time_dict[f"for resolution: {res}"]["Per-run mean time"]= hfem_mean_time_for_all_samples
hfem_time_dict[f"for resolution: {res}"]["Per-run std dev"]= hfem_std_for_all_samples
hfem_time_dict[f"for resolution: {res}"]["number of samples"]= sample_counter


output_directory_name = f"{working_directory_name}_cost_evaluation"
output_dir = os.path.join('.',output_directory_name)
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir,f'cost_eval_res_{res}')
data_to_dump = {"iFOL Cost Evaluation":ifol_time_dict,
                "HFEM Cost Evaluation": hfem_time_dict}
with open(output_filename, 'w') as f:
    json.dump(data_to_dump, f, indent=4)

