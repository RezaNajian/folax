import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
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
# jax.config.update('jax_enable_x64', True)

# directory & save handling
working_directory_name = 'pi_fno_mechanical_2D'
case_dir = os.path.join('.', working_directory_name)
# create_clean_directory(working_directory_name)
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
    with open(f'fourier_control_dict_N_21.pkl', 'rb') as f:
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

def merge_state(dst: nnx.State, src: nnx.State):
    for k, v in src.items():
        if isinstance(v, nnx.State):
            merge_state(dst[k], v)
        else:
            dst[k] = v

fno_model = bridge.ToNNX(FourierNeuralOperator2D(modes1=12,
                                                modes2=12,
                                                width=32,
                                                depth=4,
                                                channels_last_proj=128,
                                                out_channels=2,
                                                output_scale=0.001),rngs=nnx.Rngs(0)).lazy_init(K_matrix[0:1].reshape(1,model_settings["N"],model_settings["N"],1)) 

# replace RNG key by a dummy to allow checkpoint restoration later
graph_def, state = nnx.split(fno_model)
rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
merge_state(state, rngs_key)
fno_model = nnx.merge(graph_def, state)

# get total number of fno params
params = nnx.state(fno_model, nnx.Param)
total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"total number of fno network param:{total_params}")

num_epochs = 10000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-5, transition_steps=num_epochs)
optimizer = optax.chain(optax.adam(1e-6))

# create fol
pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(name="pi_fno_pr_learning",
                                                                        control=fourier_control,
                                                                        loss_function=mechanical_loss_2d,
                                                                        flax_neural_network=fno_model,
                                                                        optax_optimizer=optimizer)

pi_fno_pr_learning.Initialize()

train_start_id = 0
train_end_id = 8000
test_start_id = 8000
test_end_id = 10000
#here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
# pi_fno_pr_learning.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
#                         test_set=(coeffs_matrix[test_start_id:test_end_id,:],),
#                         test_frequency=100,
#                         batch_size=350,
#                         convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
#                         plot_settings={"plot_save_rate":100},
#                         train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
#                         working_directory=case_dir)

# load teh best model
pi_fno_pr_learning.RestoreState(restore_state_directory=case_dir+"/flax_final_state")

with open(f'train_data_dict_n8050_N_42.pkl', 'rb') as f:
    train_data_dict = pickle.load(f)

FNO_UVs = np.array(pi_fno_pr_learning.Predict(train_data_dict["input_matrix"]))
import matplotlib.pyplot
plt.imshow(FNO_UVs[0,::2].reshape(42,42))
# plt.savefig("ifol2.png")
plt.show()
plt.imshow(FNO_UVs[49,::2].reshape(42,42))
# plt.savefig("ifol2.png")
plt.show()

plt.imshow(train_data_dict["output_matrix"][0,::2].reshape(42,42))
# plt.savefig("ifol2.png")
plt.show()
plt.imshow(train_data_dict["output_matrix"][49,::2].reshape(42,42))
# plt.savefig("ifol2.png")
plt.show()


############# mean absolute errors for test data ############
# mean absolute errors for test data
idx = np.ix_(range(50))
idx_ifol = np.ix_(range(50))
FEM_UV,FNO_UV = train_data_dict["output_matrix"][idx],FNO_UVs[idx_ifol]
absolute_error_test = np.abs(FNO_UV- FEM_UV)
test_mae_err_for_samples = np.sum(absolute_error_test,axis=1) / absolute_error_test.shape[-1]
test_mae_err_total = np.mean(test_mae_err_for_samples)
print("mean absolute error for test set: ",test_mae_err_total)

# max absolute errors
test_max_err_for_samples = np.max(absolute_error_test,axis=1)
test_max_err_total = np.mean(test_max_err_for_samples)
print("max absolute error for test set: ",test_max_err_total)

