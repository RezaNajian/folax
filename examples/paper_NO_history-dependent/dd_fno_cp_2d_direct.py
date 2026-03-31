import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import optax
import jax
import numpy as np
from flax import nnx
from fol.loss_functions.regression_loss import RegressionLoss
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.ported_fourier_neural_operator_networks.fno import FNO
from fol.deep_neural_networks.fourier_parametric_operator_learning import DataDrivenFourierParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
import pickle
from mechanical3d_utilities import *


# directory & save handling
working_directory_name = 'nn_output_fno_cp_2d_direct_fullfield_frequency'
case_dir = os.path.join('.', working_directory_name)
# create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# loading directory
loading_directory_name = 'nn_output_gt_cp'
load_case_dir = os.path.join(os.path.dirname(__file__), loading_directory_name)
path = os.path.join(load_case_dir, 'Cu_textured_dataset.hdf5')
path_incr_24 = os.path.join(load_case_dir, 'Cu_simulations_quaternion_orientation_5.hdf5')

# problem setup
model_settings = {"L":1,"N":128,
                    "Ux_left":0.0,"Ux_right":0.,
                    "Uy_left":0.0,"Uy_right":0.}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()

# load data and create train set
N = model_settings["N"]
increments=list(range(75,975,75))  
extrapolate = 2         # Last two increments are meant to be extrapolated
time_steps = len(increments)
w = 1
excluded_idx=[264, 286, 484, 982, 1480, 1572, 1686, 1732, 1840, 2246, 2481]
train_sim_ids, val_sim_ids, test_sim_ids = (500,501),(2249,2254),(2484,2499)

# which_data = ["orientations"]
which_data=["von_Mises_stress", "gamma_slip_(-1-11)", "gamma_slip_(-11-1)", "gamma_slip_(1-1-1)", "gamma_slip_(111)"]

X_train, Y_train, scales = create_time_channel_set(sim_ids=train_sim_ids, increments=increments[:-extrapolate], 
                                                    max_increment=increments[-1],min_increment=increments[0], which_data=which_data, 
                                                    N=N, excluded_idx=excluded_idx,path=path, dtype=np.float32,normalize=True)

X_val, Y_val, _ = create_time_channel_set(sim_ids=val_sim_ids, increments=increments[:-extrapolate], 
                                           max_increment=increments[-1],min_increment=increments[0], which_data=which_data, 
                                           N=N, excluded_idx=excluded_idx,path=path, dtype=np.float32,normalize=True)
X_test, Y_test, _ = create_time_channel_set(sim_ids=test_sim_ids, increments=increments, 
                                            max_increment=increments[-1],min_increment=increments[0], which_data=which_data, 
                                            N=N, excluded_idx=excluded_idx,path=path, dtype=np.float32,normalize=True)

Cin = int(X_train[0,...].size / (N*N))
Cout = int(Y_train[0,...].size / (N*N))
print(f"Cin: {Cin}")
print(f"Cout: {Cout}")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")


delta = False
if delta:
    Y_train_delta = Y_train.reshape(X_train.shape[0],N,N,Cout) - X_train.reshape(X_train.shape[0],N,N,Cin)[:,:,:,:Cout]
    Y_val_delta = Y_val.reshape(X_val.shape[0],N,N,Cout) - X_val.reshape(X_val.shape[0],N,N,Cin)[:,:,:,:Cout]
    Y_test_delta = Y_test.reshape(X_test.shape[0],N,N,Cout) - X_test.reshape(X_test.shape[0],N,N,Cin)[:,:,:,:Cout]

    Y_train_delta = Y_train_delta.reshape(X_train.shape[0],-1)
    Y_val_delta = Y_val_delta.reshape(X_val.shape[0],-1)
    Y_test_delta = Y_test_delta.reshape(X_test.shape[0],-1)

# plot some data to check
# plot_to_check(X_train,N,window_lenght=1, channel_length=Cin, indices=[0,1,2,3], case_dir=case_dir,plot_name='train_X')
# plot_to_check(Y_train,N,window_lenght=1, channel_length=Cout, indices=[0,1,2,3], case_dir=case_dir,plot_name='train_Y')

print(f"X_train shape: {X_train.shape}\nY_train shape: {Y_train.shape}\nY_train_full_field shape: {Y_train.shape}")
print(f"X_val shape: {X_val.shape}\nY_val shape: {Y_val.shape}\n")
print(f"X_test shape: {X_test.shape}\nY_test shape: {Y_test.shape}\nY_test_full_field shape: {Y_test.shape}")

output_nodal_unknowns = which_data
input_nodal_unknowns = which_data + ['dt']
        

print(f"for X inputs")
for c in range(Cin):
    print(f"for input channel {c}")
    print(f'maximum of data current step train: {np.max(X_train[...,c])}')
    print(f'maximum of data current step val: {np.max(X_val[...,c])}')
    print(f'maximum of data current step test: {np.max(X_test[...,c])}')

print(f"for Y inputs")
for c in range(Cout):
    print(f"for input channel {c}")
    print(f'maximum of data increment step train: {np.max(Y_train[...,c])}')
    print(f'maximum of data increment step val: {np.max(Y_val[...,c])}')
    print(f'maximum of data increment step test: {np.max(Y_test[...,c])}')

# create identity control
identity_control = IdentityControl("ident_control",num_vars=X_train.shape[1],control_settings={})

# create regression loss
reg_loss = RegressionLoss("reg_loss",loss_settings={"nodal_unknows":output_nodal_unknowns},fe_mesh=fe_mesh)

# initialize all 
reg_loss.Initialize()
identity_control.Initialize()


# ----------------------------------------------------------------------------
# Build the FNO (Fourier Neural Operator) in JAX/Flax 
# ----------------------------------------------------------------------------
fno_dict = {"in_channel": Cin, "out_channel":Cout,
            "hidden_channels":64,"n_modes":(12,12),"n_layers":4,
            "lifting_channel_ratio":4, "projection_channel_ratio":4,
            "constant_param_embedding":{"num_frequencies":4,"embedding_type":"nerf",     # "nerf" or "transformer" 
                                "modulation":"frequency"}, "full_field": (not delta)}                              # "frequency" or "amplitude"}

fno_model = FNO(
    in_channels=fno_dict["in_channel"],
    out_channels=fno_dict["out_channel"],
    hidden_channels=fno_dict["hidden_channels"],
    n_modes=fno_dict["n_modes"],
    n_layers=fno_dict["n_layers"],
    lifting_channel_ratio=fno_dict["lifting_channel_ratio"],
    projection_channel_ratio=fno_dict["projection_channel_ratio"],
    rngs=nnx.Rngs(0),
    constant_parameter_index = [5],
    constant_param_embedding = fno_dict["constant_param_embedding"]
)

# Count trainable parameters 
params = nnx.state(fno_model, nnx.Param)
total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"FNO trainable parameters:{total_params}")

# Sanity-check forward pass with a small batch:
# FNO expects shape (batch, Nx, Ny, channels)
# ISince K_matrix is stored as flattened vectors, we reshape to (B,N,N,1)
init_out = fno_model(X_train[0:8].reshape(8,model_settings["N"],model_settings["N"],len(input_nodal_unknowns)))


num_epochs = 2000
lr = 1e-5
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=lr, transition_steps=num_epochs)
optimizer = optax.chain(optax.adam(learning_rate_scheduler))

# create fol
dd_fno_pr_learning = DataDrivenFourierParametricOperatorLearning(name="dd_fno_pr_learning",
                                                                control=identity_control,
                                                                loss_function=reg_loss,
                                                                flax_neural_network=fno_model,
                                                                optax_optimizer=optimizer)

dd_fno_pr_learning.Initialize()

train_start_id = 0
train_end_id = -1
print(f"train end id: {train_end_id}")
val_start_id = 0
val_end_id = -1
test_start_id = 0
test_end_id = 10
batch_size = 8
if delta:
    train_set = (X_train[train_start_id:train_end_id,:],Y_train_delta[train_start_id:train_end_id,:])
    test_set = (X_val[val_start_id:val_end_id,:],Y_val_delta[val_start_id:val_end_id,:])
else:
    train_set = (X_train[train_start_id:train_end_id,:],Y_train[train_start_id:train_end_id,:])
    test_set = (X_val[val_start_id:val_end_id,:],Y_val[val_start_id:val_end_id,:])
dd_fno_pr_learning.Train(train_set=train_set,
                        test_set=test_set,
                        batch_size=batch_size,
                        restore_nnx_state_settings={'restore':False, "state_directory":case_dir+"/flax_train_state"},
                        convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                        plot_settings={"save_frequency":10},
                        train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
                        test_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
                        data_model_sharding_settings ={"sharding":False,"num_data_devices":4,"num_nnx_model_devices":1},
                        working_directory=case_dir)

# load the best model
dd_fno_pr_learning.RestoreState(restore_state_directory=os.path.join(os.path.dirname(__file__),case_dir,"flax_train_state"))

# ----------------------------------------------------------------------------
# Plot Prediction vs Ground Truth values 
# ----------------------------------------------------------------------------

type_index = {'von_mises':0, 'orientation_1':1, 'orientation_2':2,'orientation_3':3,'orientation_4':4,}
i = type_index["von_mises"]

for case_id in [0, 120, 280, 420, 890, 1345, 1367, 1992]:   # less than train_/test_sim_ids

    gt_sol = gt_loader(which_data=which_data,sim_id=case_id,increments=increments,w=w,path=path,excluded_idx=excluded_idx,N=N)

    x0=gt_sol[0,:].reshape(N,N,Cout)
    # x0_norm = x0 / np.max(x0,axis=(0,1))
    x0_norm = x0 / scales["physical"]
    fno_sol = rollout_direct(model=dd_fno_pr_learning, x0=x0_norm[None,...], steps=(time_steps - w), time_scale = (time_steps - w - extrapolate), N=N, C=Cout, w=w, delta=delta).squeeze().reshape((time_steps - w),-1)
    print(f"fno_sol shape: {fno_sol.shape}")
    print(f"gt_sol shape: {gt_sol.shape}")
    assert fno_sol.shape == ((time_steps - w),N*N*Cout)
    assert gt_sol.shape == ((time_steps - w),N*N*Cout)

    if i==0:
        scale_factor = scales["physical"][0]
    else:
        scale_factor = 1
    model_vm_pred = scale_factor * fno_sol[:,::Cout].reshape(fno_sol.shape[0],-1)
    gt_vm = gt_sol[:,::Cout]
    abs_err = np.abs(model_vm_pred.reshape(fno_sol.shape[0],-1) - gt_vm)

    for t in range(time_steps - w):
        fe_mesh[f'Von_Mises_FNO_train_{case_id}_t_0{t}'] = model_vm_pred[t,:].reshape((fe_mesh.GetNumberOfNodes(), 1))
        fe_mesh[f'Von_Mises_FFT_train_{case_id}_0{t}'] = gt_vm[t,:].reshape((fe_mesh.GetNumberOfNodes(), 1))
        fe_mesh[f'abs_err_train_{case_id}_0{t}'] = abs_err[t,:].reshape((fe_mesh.GetNumberOfNodes(), 1))
    fe_mesh.Finalize(export_dir=case_dir)

    output_name = list(type_index.keys())[list(type_index.values()).index(i)]
    von_mises_plot(model_vm_pred,gt_vm,abs_err,(time_steps - w),N,case_id,case_dir,f'{output_name}_train_set',start_id=w)
    radial_power_spectrum_plot(model_vm_pred.reshape(fno_sol.shape[0],N,N),gt_vm.reshape(fno_sol.shape[0],N,N),
                               (time_steps - w),N,case_id,case_dir,f'{output_name}_RPS_train_set',start_id=w)
    plot_value_1d(pred_field_list=[model_vm_pred[1,:],model_vm_pred[4,:],model_vm_pred[-1,:]],
                  gt_field_list=[gt_vm[1,:],gt_vm[4,:],gt_vm[-1,:]],
                  y_eval_value_grid=64,time=['2','5','10'],case_dir=case_dir,filename=f"{output_name}_1d_train_case_{case_id}",start_id=w)


for case_id in [2258, 2298, 2302, 2345, 2379, 2421, 2462, 2492]:   # less than train_/test_sim_ids

    gt_sol = gt_loader(which_data=which_data,sim_id=case_id,increments=increments,w=w,path=path,excluded_idx=excluded_idx,N=N)  # (steps,N*N*C)

    x0=gt_sol[0,:].reshape(N,N,Cout)
    # x0_norm = x0 / np.max(x0,axis=(0,1))
    x0_norm = x0 / scales["physical"]
    fno_sol = rollout_direct(model=dd_fno_pr_learning, x0=x0_norm, steps=(time_steps - w), time_scale = (time_steps - w - extrapolate), N=N, C=Cout, w=w, delta=delta).squeeze().reshape((time_steps - w),-1)  # (steps,N*N*C)
    print(f"fno_sol shape: {fno_sol.shape}")
    print(f"gt_sol shape: {gt_sol.shape}")
    assert fno_sol.shape == ((time_steps - w),N*N*Cout)
    assert gt_sol.shape == ((time_steps - w),N*N*Cout)

    if i==0:
        scale_factor = scales["physical"][0]
    else:
        scale_factor = 1
    model_vm_pred = scale_factor * fno_sol[:,::Cout].reshape(fno_sol.shape[0],-1)
    gt_vm = gt_sol[:,::Cout]
    abs_err = np.abs(model_vm_pred.reshape(fno_sol.shape[0],-1) - gt_vm)

    for t in range(time_steps - w):
        fe_mesh[f'Von_Mises_FNO_test_{case_id}_t_0{t}'] = model_vm_pred[t,:].reshape((fe_mesh.GetNumberOfNodes(), 1))
        fe_mesh[f'Von_Mises_FFT_test_{case_id}_0{t}'] = gt_vm[t,:].reshape((fe_mesh.GetNumberOfNodes(), 1))
        fe_mesh[f'abs_err_test_{case_id}_0{t}'] = abs_err[t,:].reshape((fe_mesh.GetNumberOfNodes(), 1))
    fe_mesh.Finalize(export_dir=case_dir)

    output_name = list(type_index.keys())[list(type_index.values()).index(i)]
    von_mises_plot(model_vm_pred,gt_vm,abs_err,(time_steps - w),N,case_id,case_dir,f'{output_name}_test_set',start_id=w)
    radial_power_spectrum_plot(model_vm_pred.reshape(fno_sol.shape[0],N,N),gt_vm.reshape(fno_sol.shape[0],N,N),
                               (time_steps - w),N,case_id,case_dir,f'{output_name}_RPS_test_set',start_id=w)
    plot_value_1d(pred_field_list=[model_vm_pred[1,:],model_vm_pred[4,:],model_vm_pred[-1,:]],
                  gt_field_list=[gt_vm[1,:],gt_vm[4,:],gt_vm[-1,:]],
                  y_eval_value_grid=64,time=['2','5','10'],case_dir=case_dir,filename=f"{output_name}_1d_test_case_{case_id}",start_id=w)


fe_mesh.Finalize(export_dir=case_dir)

# # -------------------------------------------------
# # Report FNO Hyper-parameters to Document
# # -------------------------------------------------

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

print(f"FNO dict param:\n{fno_dict}")
# # -------------------------------------------------
# # Calculation of error in time
# # -------------------------------------------------


print(f"steps: {time_steps - w} , N: {N}, C: {Cout}, w: {w}")
#   shape: (steps,B,N*N*C)
excluded_idx = excluded_idx or []
valid_sim_ids = [s for s in range(*test_sim_ids) if s not in excluded_idx]
B = len(valid_sim_ids)
gt_sol_batch = np.zeros(((time_steps - w),B,N*N*Cout))
fno_sol_batch = np.zeros(((time_steps - w),B,N*N*Cout))
for ind,sim_id in enumerate(valid_sim_ids):    
    gt_sol = gt_loader(which_data=which_data,sim_id=sim_id,increments=increments,w=w,path=path,excluded_idx=excluded_idx,N=N)
    # x0=gt_sol[0,...]
    x0=gt_sol[0,...].reshape(N,N,Cout)
    norm_scales = scales["physical"]
    x0_norm = x0 / norm_scales
    fno_sol = rollout_direct(model=dd_fno_pr_learning, x0=x0_norm, steps=(time_steps - w), time_scale = (time_steps - w - extrapolate), N=N, C=Cout, w=w, delta=delta).squeeze()
    fno_sol_rescaled = (fno_sol * norm_scales)
    gt_sol_batch[:,ind,:] = gt_sol.reshape((time_steps - w),-1)
    fno_sol_batch[:,ind,:] = fno_sol_rescaled.reshape((time_steps - w),-1)

# compute error for each time steps
abs_error_for_step = []
# error for von_Mises_stress --> ::Cout
for step in range(time_steps - w):
    abs_error_for_step.append(np.abs(fno_sol_batch[step,:].reshape(B,N,N,Cout) - gt_sol_batch[step,:].reshape(B,N,N,Cout)))        # shape= (B, N*N*C)
error_steps_array = np.array(abs_error_for_step)

error_data_dict = {}
pred_data_dict = {}
test_data_dict = {}
for step in range(time_steps - w):
    error_data_dict[f"step_{step}"] = error_steps_array[step,:,:]
    pred_data_dict[f"step_{step}"] = fno_sol_batch[step,:,:]
    test_data_dict[f"step_{step}"] = gt_sol_batch[step,:,:]

print(error_data_dict.keys())

error_time_plot_pointwise(error_data_dict,which_channel=0,w=w,filename='boxplot_error_in_time_pointwise.png',case_dir=case_dir)
error_time_plot_mae(error_data_dict,which_physics='von_mises',w=w,filename='boxplot_error_in_time_mae.png',case_dir=case_dir)


with open(case_dir+f'/error_data_dict.pkl', 'wb') as f:
    pickle.dump(error_data_dict,f)
with open(case_dir+f'/pred_data_dict.pkl', 'wb') as f:
    pickle.dump(pred_data_dict,f)
with open(case_dir+f'/test_data_dict.pkl', 'wb') as f:
    pickle.dump(test_data_dict,f)

