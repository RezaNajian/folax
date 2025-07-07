import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
import optax
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.loss_functions.regression_loss import RegressionLoss
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.deep_o_net_parametric_operator_learning import DataDrivenDeepONetParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from mechanical2D_usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import MLP
from fol.deep_neural_networks.deep_o_nets import DeepONet
from fol.tools.decoration_functions import *
import pickle
import jax

jax.config.update('jax_default_matmul_precision','high')
jax.config.update('jax_enable_x64', True)

# directory & save handling
working_directory_name = 'dd_deep_onet_mechanical_2D'
case_dir = os.path.join('.', working_directory_name)
# create_clean_directory(working_directory_name)
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
    with open(f'fourier_control_dict_N_21.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]

K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

material_dict = {"young_modulus":1,"poisson_ratio":0.3}
loss_settings={"dirichlet_bc_dict":bc_dict,
            "material_dict":material_dict}
mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad("mechanical_loss_2d",loss_settings=loss_settings,                                                                      fe_mesh=fe_mesh)
mechanical_loss_2d.Initialize()

# create regression loss
reg_loss = RegressionLoss("reg_loss",loss_settings={"nodal_unknows":["Ux","Uy"]},fe_mesh=fe_mesh)
reg_loss.Initialize()

characteristic_length = 64
num_dofs = len(mechanical_loss_2d.GetDOFs())
activation_function = "relu"
# design branch & trunk NN for vanilla deep_onet
branch_nn = MLP(name="branch_nn",
                input_size=fourier_control.GetNumberOfControlledVariables(),
                hidden_layers=[characteristic_length] * 4,
                output_size=num_dofs*characteristic_length,
                activation_settings={"type":activation_function})

trunk_nn = MLP(name="trunk_nn",
                input_size=3,
                hidden_layers=[characteristic_length] * 4,
                output_size=num_dofs*characteristic_length,
                activation_settings={"type":activation_function})

deep_onet = DeepONet("main_deeponet",
                     branch_nn=branch_nn,
                     trunk_nn=trunk_nn,
                     output_dimension=num_dofs,
                     activation_function_name=activation_function,
                     use_bias=True,
                     output_scale_factor=1.0)

num_epochs = 10000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-6, transition_steps=num_epochs)
optimizer = optax.chain(optax.adam(1e-4))

# create fol
deeponet_learning = DataDrivenDeepONetParametricOperatorLearning(name="deeponet_learning",
                                                                    control=fourier_control,
                                                                    loss_function=reg_loss,
                                                                    flax_neural_network=deep_onet,
                                                                    optax_optimizer=optimizer)

deeponet_learning.Initialize()


train_start_id = 0
train_end_id = 10
solve_FE = False
if solve_FE:
    input_matrix = np.empty((0,10))
    output_matrix = np.empty((0,2*fe_mesh.GetNumberOfNodes()))
    for eval_id in np.arange(train_start_id,train_end_id):
        fe_setting = {"linear_solver_settings":{"solver":"JAX-direct","tol":1e-6,"atol":1e-6,
                                                        "maxiter":1000,"pre-conditioner":"ilu"},
                            "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                                        "maxiter":20,"load_incr":40}}
        linear_fe_solver = FiniteElementNonLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_2d,fe_setting)
        linear_fe_solver.Initialize()
        FE_UV = np.array(linear_fe_solver.Solve(K_matrix[eval_id],jnp.zeros(2*fe_mesh.GetNumberOfNodes())))
        input_matrix = np.vstack((input_matrix,coeffs_matrix[eval_id,:]))
        output_matrix = np.vstack((output_matrix,FE_UV))

    train_data_dict = {"input_matrix":input_matrix,
                        "output_matrix":output_matrix}

    with open(f'train_data_dict.pkl', 'wb') as f:
        pickle.dump(train_data_dict,f)
else:
    with open(f'train_data_dict_10K.pkl', 'rb') as f:
        train_data_dict = pickle.load(f)

# deeponet_learning.Train(train_set=(train_data_dict["input_matrix"],train_data_dict["output_matrix"]),
#                         batch_size=1,
#                         convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
#                         plot_settings={"plot_save_rate":100},
#                         train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
#                         working_directory=case_dir)

# load the best model
deeponet_learning.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

# deeponet sols
# DON_UVs = np.array(deeponet_learning.Predict(train_data_dict["input_matrix"]))

# k matrix
# K_matrix = fourier_control.ComputeBatchControlledVariables(train_data_dict["input_matrix"])

# for idx in range(K_matrix.shape[0]):
#     K,FEM_UV,DON_UV = K_matrix[idx],train_data_dict["output_matrix"][idx],DON_UVs[idx]
#     # plot U
#     vectors_list = [K,DON_UV[::2],FEM_UV[::2]]
#     plot_mesh_res(vectors_list, file_name=case_dir+f'/plot_U_{idx}.png',dir="U")
#     # plot V
#     vectors_list = [K,DON_UV[1::2],FEM_UV[1::2]]
#     plot_mesh_res(vectors_list, file_name=case_dir+f'/plot_V_{idx}.png',dir="V")




with open(f'train_data_dict_10K.pkl', 'rb') as f:
    train_data_dict = pickle.load(f)

DON_UVs = np.array(deeponet_learning.Predict(train_data_dict["input_matrix"][8000:8050,:]))
import matplotlib.pyplot
plt.imshow(DON_UVs[0,::2].reshape(41,41))
# plt.savefig("ifol2.png")
plt.show()
plt.imshow(DON_UVs[49,::2].reshape(41,41))
# plt.savefig("ifol2.png")
plt.show()

plt.imshow(train_data_dict["output_matrix"][8000,::2].reshape(41,41))
# plt.savefig("ifol2.png")
plt.show()
plt.imshow(train_data_dict["output_matrix"][8049,::2].reshape(41,41))
# plt.savefig("ifol2.png")
plt.show()


############# mean absolute errors for test data ############
# mean absolute errors for test data
idx = np.ix_(range(8000,8050))
idx_ifol = np.ix_(range(50))
FEM_UV,DON_UV = train_data_dict["output_matrix"][idx],DON_UVs[idx_ifol]
absolute_error_test = np.abs(DON_UV- FEM_UV)
test_mae_err_for_samples = np.sum(absolute_error_test,axis=1) / absolute_error_test.shape[-1]
test_mae_err_total = np.mean(test_mae_err_for_samples)
print("mean absolute error for test set: ",test_mae_err_total)

# max absolute errors
test_max_err_for_samples = np.max(absolute_error_test,axis=1)
test_max_err_total = np.mean(test_max_err_for_samples)
print("max absolute error for test set: ",test_max_err_total)


