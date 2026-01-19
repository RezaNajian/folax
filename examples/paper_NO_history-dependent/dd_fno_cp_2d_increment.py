import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import optax
import jax
import numpy as np
from flax import nnx
from flax.nnx import bridge
from fol.loss_functions.regression_loss import RegressionLoss
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.ported_fourier_neural_operator_networks.fno import FNO
from fol.deep_neural_networks.fourier_parametric_operator_learning import DataDrivenFourierParametricOperatorLearning
from fol.deep_neural_networks.fourier_neural_operator_networks import FourierNeuralOperator2D
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DTetra
import pickle
from mechanical3d_utilities import *


# directory & save handling
working_directory_name = 'nn_output_fno_cp_2d'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":128,
                    "Ux_left":0.0,"Ux_right":0.,
                    "Uy_left":0.0,"Uy_right":0.}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()

gt_dict_load_train_set = {}
with open(os.path.join(os.path.dirname(__file__),f'nn_output_gt_cp/data_mixed_{model_settings["N"]}x{model_settings["N"]}_simulation_num_from_0_to_199.pkl'), 'rb') as f:
    gt_dict_load_train_set = pickle.load(f)

gt_dict_load_val_set = {}
with open(os.path.join(os.path.dirname(__file__),f'nn_output_gt_cp/data_mixed_{model_settings["N"]}x{model_settings["N"]}_simulation_num_from_2000_to_2099.pkl'), 'rb') as f:
    gt_dict_load_val_set = pickle.load(f)

gt_dict_load_test_set = {}
with open(os.path.join(os.path.dirname(__file__),f'nn_output_gt_cp/data_mixed_{model_settings["N"]}x{model_settings["N"]}_simulation_num_from_2100_to_2199.pkl'), 'rb') as f:
    gt_dict_load_test_set = pickle.load(f)

increments = []
for k in gt_dict_load_train_set.keys():
    increments.append(k.split('_')[-1])

K_current_train_set = gt_dict_load_train_set["increment_75"]
K_next_train_set = gt_dict_load_train_set["increment_150"]
K_current_val_set = gt_dict_load_val_set["increment_75"]
K_next_val_set = gt_dict_load_val_set["increment_150"]
K_current_test_set = gt_dict_load_test_set["increment_75"]
K_next_test_set = gt_dict_load_test_set["increment_150"]
for increment_current, increment_next in zip(increments[1:], increments[2:]):
    print(f"increment current: {increment_current}, increment next: {increment_next}")
    K_current_train_set = np.concatenate([K_current_train_set,gt_dict_load_train_set[f"increment_{increment_current}"]])
    K_next_train_set = np.concatenate([K_next_train_set,gt_dict_load_train_set[f"increment_{increment_next}"]])
    K_current_val_set = np.concatenate([K_current_val_set,gt_dict_load_val_set[f"increment_{increment_current}"]])
    K_next_val_set = np.concatenate([K_next_val_set,gt_dict_load_val_set[f"increment_{increment_next}"]])
    K_current_test_set = np.concatenate([K_current_test_set,gt_dict_load_test_set[f"increment_{increment_current}"]])
    K_next_test_set = np.concatenate([K_next_test_set,gt_dict_load_test_set[f"increment_{increment_next}"]])

K_matrix_current_step_train_set = K_current_train_set
K_matrix_next_step_train_set = K_next_train_set
K_matrix_current_step_val_set = K_current_val_set
K_matrix_next_step_val_set = K_next_val_set
K_matrix_current_step_test_set = K_current_test_set
K_matrix_next_step_test_set = K_next_test_set


# plot some data to check
# for id in [0,20,40,60,80,86,106,126,146,166,186]:   # for train set
for id in [0,5]:   # for test set
    random_sample_id = id

    fig, ax = plt.subplots(1, 5, figsize=(24, 8))
    N = model_settings["N"]

    # First sample
    im00 = ax[0].imshow(K_matrix_current_step_train_set[random_sample_id, ::5].reshape(N, N), cmap='viridis')
    ax[0].set_title(f"Current Step Von-Mises Stress (Sample {random_sample_id})")
    fig.colorbar(im00, ax=ax[0], fraction=0.046, pad=0.04)

    im01 = ax[1].imshow(K_matrix_current_step_train_set[random_sample_id, 1::5].reshape(N, N), cmap='coolwarm')
    ax[1].set_title(f"Current Step $alpha_{0}$ (Sample {random_sample_id})")
    fig.colorbar(im01, ax=ax[1], fraction=0.046, pad=0.04)

    # Second sample
    im02 = ax[2].imshow(K_matrix_current_step_train_set[random_sample_id, 2::5].reshape(N, N), cmap='coolwarm')
    ax[2].set_title(f"Current Step $alpha_{1}$ (Sample {random_sample_id})")
    fig.colorbar(im02, ax=ax[2], fraction=0.046, pad=0.04)

    im03 = ax[3].imshow(K_matrix_current_step_train_set[random_sample_id, 3::5].reshape(N, N), cmap='coolwarm')
    ax[3].set_title(f"Current Step $alpha_{2}$ (Sample {random_sample_id})")
    fig.colorbar(im03, ax=ax[3], fraction=0.046, pad=0.04)

    im04 = ax[4].imshow(K_matrix_current_step_train_set[random_sample_id, 4::5].reshape(N, N), cmap='coolwarm')
    ax[4].set_title(f"Current Step $alpha_{3}$ (Sample {random_sample_id})")
    fig.colorbar(im04, ax=ax[4], fraction=0.046, pad=0.04)

    # Layout & save
    for a in ax.flat:
        a.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, f'von_mises_res_{model_settings["N"]}_{id}.png'), dpi=300)
    plt.close()

train_stress_scale_factor = max(abs(np.max(K_matrix_current_step_train_set[:,::5])),abs(np.max(K_matrix_next_step_train_set[:,::5])))
K_matrix_current_step_train_set[:,::5] = K_matrix_current_step_train_set[:,::5] / train_stress_scale_factor
K_matrix_next_step_train_set[:,::5] = K_matrix_next_step_train_set[:,::5] / train_stress_scale_factor

val_stress_scale_factor = max(abs(np.max(K_matrix_current_step_val_set[:,::5])),abs(np.max(K_matrix_next_step_val_set[:,::5])))
K_matrix_current_step_val_set[:,::5] = K_matrix_current_step_val_set[:,::5] / val_stress_scale_factor
K_matrix_next_step_val_set[:,::5] = K_matrix_next_step_val_set[:,::5] / val_stress_scale_factor

test_stress_scale_factor = max(abs(np.max(K_matrix_current_step_test_set[:,::5])),abs(np.max(K_matrix_next_step_test_set[:,::5])))
K_matrix_current_step_test_set[:,::5] = K_matrix_current_step_test_set[:,::5] / test_stress_scale_factor
K_matrix_next_step_test_set[:,::5] = K_matrix_next_step_test_set[:,::5] / test_stress_scale_factor

# plot some data to check
# for id in [0,20,40,60,80,86,106,126,146,166,186]:   # for train set
for id in [0,5]:   # for test set
    random_sample_id = id

    fig, ax = plt.subplots(1, 5, figsize=(24, 8))
    N = model_settings["N"]

    # First sample
    im00 = ax[0].imshow(train_stress_scale_factor*(K_matrix_current_step_train_set[random_sample_id, ::5].reshape(N, N)), cmap='viridis')
    ax[0].set_title(f"Current Step Von-Mises Stress (Sample {random_sample_id})")
    fig.colorbar(im00, ax=ax[0], fraction=0.046, pad=0.04)

    im01 = ax[1].imshow(K_matrix_current_step_train_set[random_sample_id, 1::5].reshape(N, N), cmap='coolwarm')
    ax[1].set_title(f"Current Step $alpha_{0}$ (Sample {random_sample_id})")
    fig.colorbar(im01, ax=ax[1], fraction=0.046, pad=0.04)

    # Second sample
    im02 = ax[2].imshow(K_matrix_current_step_train_set[random_sample_id, 2::5].reshape(N, N), cmap='coolwarm')
    ax[2].set_title(f"Current Step $alpha_{1}$ (Sample {random_sample_id})")
    fig.colorbar(im02, ax=ax[2], fraction=0.046, pad=0.04)

    im03 = ax[3].imshow(K_matrix_current_step_train_set[random_sample_id, 3::5].reshape(N, N), cmap='coolwarm')
    ax[3].set_title(f"Current Step $alpha_{2}$ (Sample {random_sample_id})")
    fig.colorbar(im03, ax=ax[3], fraction=0.046, pad=0.04)

    im04 = ax[4].imshow(K_matrix_current_step_train_set[random_sample_id, 4::5].reshape(N, N), cmap='coolwarm')
    ax[4].set_title(f"Current Step $alpha_{3}$ (Sample {random_sample_id})")
    fig.colorbar(im04, ax=ax[4], fraction=0.046, pad=0.04)

    # Layout & save
    for a in ax.flat:
        a.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, f'von_mises_normalize_res_{model_settings["N"]}_{id}.png'), dpi=300)
    plt.close()

train_on_von_mises = False
if train_on_von_mises:
    data_current_step_train_set = K_matrix_current_step_train_set[:,::5]
    data_current_step_val_set = K_matrix_current_step_val_set[:,::5]
    data_current_step_test_set = K_matrix_current_step_test_set[:,::5]

    data_next_step_train_set = K_matrix_next_step_train_set[:,::5]
    data_next_step_val_set = K_matrix_next_step_val_set[:,::5]
    data_next_step_test_set = K_matrix_next_step_test_set[:,::5]

    data_increment_delta_train_set = K_matrix_next_step_train_set[:,::5] - K_matrix_current_step_train_set[:,::5]
    data_increment_delta_val_set = K_matrix_next_step_val_set[:,::5] - K_matrix_current_step_val_set[:,::5]
    data_increment_delta_test_set = K_matrix_next_step_test_set[:,::5] - K_matrix_current_step_test_set[:,::5]
    nodal_unknowns = ["von_mises"]

else:
    data_current_step_train_set = K_matrix_current_step_train_set
    data_current_step_val_set = K_matrix_current_step_val_set
    data_current_step_test_set = K_matrix_current_step_test_set

    data_next_step_train_set = K_matrix_next_step_train_set
    data_next_step_val_set = K_matrix_next_step_val_set
    data_next_step_test_set = K_matrix_next_step_test_set

    data_increment_delta_train_set = K_matrix_next_step_train_set - K_matrix_current_step_train_set
    data_increment_delta_val_set = K_matrix_next_step_val_set - K_matrix_current_step_val_set
    data_increment_delta_test_set = K_matrix_next_step_test_set - K_matrix_current_step_test_set
    nodal_unknowns = ["von_mises","Ori_0","Ori_1","Ori_2","Ori_3"]


# create identity control
identity_control = IdentityControl("ident_control",num_vars=K_matrix_current_step_train_set.shape[1],control_settings={})

# create regression loss
reg_loss = RegressionLoss("reg_loss",loss_settings={"nodal_unknows":nodal_unknowns},fe_mesh=fe_mesh)

# initialize all 
reg_loss.Initialize()
identity_control.Initialize()


# ----------------------------------------------------------------------------
# 3) Build the FNO (Fourier Neural Operator) in JAX/Flax 
# ----------------------------------------------------------------------------
fno_model = FNO(
    in_channels=len(nodal_unknowns),
    out_channels=len(nodal_unknowns),
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
init_out = fno_model(data_current_step_train_set[0:8].reshape(8,model_settings["N"],model_settings["N"],len(nodal_unknowns)))


num_epochs = 5
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-6, transition_steps=num_epochs)
lr = 1e-5
optimizer = optax.chain(optax.adam(lr))

# create fol
dd_fno_pr_learning = DataDrivenFourierParametricOperatorLearning(name="dd_fno_pr_learning",
                                                                control=identity_control,
                                                                loss_function=reg_loss,
                                                                flax_neural_network=fno_model,
                                                                optax_optimizer=optimizer)

dd_fno_pr_learning.Initialize()

train_start_id = 0
train_end_id = 80
print(f"train end id: {train_end_id}")
test_start_id = 0
test_end_id = 10
batch_size = 8
dd_fno_pr_learning.Train(train_set=(data_current_step_train_set[train_start_id:train_end_id,:],data_increment_delta_train_set[train_start_id:train_end_id,:]),
                         test_set=(data_current_step_val_set[test_start_id:test_end_id,:],data_increment_delta_val_set[test_start_id:test_end_id,:]),
                        batch_size=batch_size,
                        convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                        plot_settings={"plot_save_rate":100},
                        train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
                        working_directory=case_dir)

# load the best model
dd_fno_pr_learning.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

# deeponet sols
# FNO_UVs = np.array(dd_fno_pr_learning.Predict(data_current_step[train_start_id:train_end_id]))

type_index = {'von_mises':0, 'orientation_1':1, 'orientation_2':2,'orientation_3':3,'orientation_4':4,}
i = type_index["von_mises"]

time_interval = gt_dict_load_train_set["increment_75"].shape[0]  # is equal to the number of simulations
print(f"plot for train set--time interval is: {time_interval}")

for case_id in [0, 1, 4]:

    ground_truth, iFOL_pred, error = [], [], []
    ifol_current_step = data_current_step_train_set[case_id, :]
    for eval_id in range(10):
        
        iFOL_increment_to_the_next_step = np.array(dd_fno_pr_learning.Predict(ifol_current_step.reshape(-1, 1).T)).reshape(-1)

        if train_on_von_mises:
            iFOL_von_mises_increment_to_the_next_step = iFOL_increment_to_the_next_step
            iFOL_von_mises_current_step = ifol_current_step
        else:
            iFOL_von_mises_increment_to_the_next_step = iFOL_increment_to_the_next_step[i::5]
            iFOL_von_mises_current_step = ifol_current_step[i::5]

        ifol_next_step = ifol_current_step + iFOL_increment_to_the_next_step
        ifol_current_step = ifol_next_step
        if i==0:
            scale_factor = train_stress_scale_factor
        else:
            scale_factor = 1
        iFOL_von_mises_next_step = scale_factor * (iFOL_von_mises_current_step + iFOL_von_mises_increment_to_the_next_step)
        von_mises_gt_next = scale_factor * data_next_step_train_set[case_id + time_interval * eval_id, i::5]
        abs_err = np.abs(iFOL_von_mises_next_step.reshape(-1) - von_mises_gt_next.reshape(-1))

        fe_mesh[f'Von_Mises_iFOL_{case_id}_{eval_id}'] = iFOL_von_mises_next_step.reshape((fe_mesh.GetNumberOfNodes(), 1))
        fe_mesh[f'Von_Mises_FFT_{case_id}_{eval_id}'] = von_mises_gt_next.reshape((fe_mesh.GetNumberOfNodes(), 1))
        fe_mesh[f'abs_err_{case_id}_{eval_id}'] = abs_err.reshape((fe_mesh.GetNumberOfNodes(), 1))

        ground_truth.append(von_mises_gt_next.reshape((fe_mesh.GetNumberOfNodes(), 1)))
        iFOL_pred.append(iFOL_von_mises_next_step.reshape((fe_mesh.GetNumberOfNodes(), 1)))
        error.append(abs_err.reshape((fe_mesh.GetNumberOfNodes(), 1)))
    
    ground_truth = np.array(ground_truth)
    iFOL_pred = np.array(iFOL_pred)
    error = np.array(error)

    output_name = list(type_index.keys())[list(type_index.values()).index(i)]
    von_mises_plot(iFOL_pred,ground_truth,error,N,case_id,case_dir,f'{output_name}_train_set')
    plot_value_1d(pred_field_list=[iFOL_pred[1,:],iFOL_pred[4,:],iFOL_pred[-1,:]],
                  gt_field_list=[ground_truth[1,:],ground_truth[4,:],ground_truth[-1,:]],
                  y_eval_value_grid=64,time=['2','5','10'],case_dir=case_dir,filename=f"{output_name}_1d_train_case_{case_id}")


time_interval = gt_dict_load_test_set["increment_75"].shape[0]  # is equal to the number of simulations
print(f"plot for train set--time interval is: {time_interval}")


for case_id in [0, 1, 4]:

    ground_truth, iFOL_pred, error = [], [], []
    ifol_current_step = data_current_step_test_set[case_id, :]
    for eval_id in range(10):
        
        iFOL_increment_to_the_next_step = np.array(dd_fno_pr_learning.Predict(ifol_current_step.reshape(-1, 1).T)).reshape(-1)

        if train_on_von_mises:
            iFOL_von_mises_increment_to_the_next_step = iFOL_increment_to_the_next_step
            iFOL_von_mises_current_step = ifol_current_step
        else:
            iFOL_von_mises_increment_to_the_next_step = iFOL_increment_to_the_next_step[i::5]
            iFOL_von_mises_current_step = ifol_current_step[i::5]

        ifol_next_step = ifol_current_step + iFOL_increment_to_the_next_step
        ifol_current_step = ifol_next_step
        if i==0:
            scale_factor = test_stress_scale_factor
        else:
            scale_factor = 1
        iFOL_von_mises_next_step = scale_factor * (iFOL_von_mises_current_step + iFOL_von_mises_increment_to_the_next_step)
        von_mises_gt_next = scale_factor * data_next_step_test_set[case_id + time_interval * eval_id, i::5]
        abs_err = np.abs(iFOL_von_mises_next_step.reshape(-1) - von_mises_gt_next.reshape(-1))

        fe_mesh[f'Von_Mises_iFOL_{case_id}_{eval_id}'] = iFOL_von_mises_next_step.reshape((fe_mesh.GetNumberOfNodes(), 1))
        fe_mesh[f'Von_Mises_FFT_{case_id}_{eval_id}'] = von_mises_gt_next.reshape((fe_mesh.GetNumberOfNodes(), 1))
        fe_mesh[f'abs_err_{case_id}_{eval_id}'] = abs_err.reshape((fe_mesh.GetNumberOfNodes(), 1))

        ground_truth.append(von_mises_gt_next.reshape((fe_mesh.GetNumberOfNodes(), 1)))
        iFOL_pred.append(iFOL_von_mises_next_step.reshape((fe_mesh.GetNumberOfNodes(), 1)))
        error.append(abs_err.reshape((fe_mesh.GetNumberOfNodes(), 1)))
    
    ground_truth = np.array(ground_truth)
    iFOL_pred = np.array(iFOL_pred)
    error = np.array(error)

    von_mises_plot(iFOL_pred,ground_truth,error,N,case_id,case_dir,f'{output_name}_test_set')
    plot_value_1d(pred_field_list=[iFOL_pred[1,:],iFOL_pred[4,:],iFOL_pred[-1,:]],
                  gt_field_list=[ground_truth[1,:],ground_truth[4,:],ground_truth[-1,:]],
                  y_eval_value_grid=64,time=['2','5','10'],case_dir=case_dir,filename=f"{output_name}_1d_test_case_{case_id}")


fe_mesh.Finalize(export_dir=case_dir)

print(f"{'#'*20} network information {'#'*20}")
print(f"Number of modes in the first dimension: {fno_model._n_modes[0]}")
print(f"Number of modes in the second dimension: {fno_model._n_modes[1]}")
print(f"Number of channels to which the input is lifted: {fno_model.hidden_channels}")
print(f"Number of Fourier stages: {fno_model.n_layers}")
print(f"Number of channels in the hidden layer of the last: {fno_model.projection_channels}")
print(f"Number of output channels: {fno_model.out_channels}")
print(f"Actication function: {str(fno_model.non_linearity)}")
print(f"Number of epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")
print(f"Trainig sample's id: {train_start_id} -> {train_end_id}")
print(f"Number of learnable parameters: {total_params}")
