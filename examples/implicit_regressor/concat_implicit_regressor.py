import sys
import os
import optax
import numpy as np
from fol.loss_functions.regression_loss import RegressionLoss
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.implicit_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import MLP
import pickle

# directory & save handling
working_directory_name = 'concat_implicit_regressor'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":51}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create regression loss function
reg_loss = RegressionLoss("reg_loss",loss_settings={"nodal_unknows":["K"]},fe_mesh=fe_mesh)

fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                            "beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

fe_mesh.Initialize()
reg_loss.Initialize()
fourier_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 200
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = model_settings.copy()
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = fourier_control.x_freqs
    export_dict["y_freqs"] = fourier_control.y_freqs
    export_dict["z_freqs"] = fourier_control.z_freqs
    with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'fourier_control_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]

# ATTENTION: we need to normalize the features
coeffs_matrix_min = np.min(coeffs_matrix)
coeffs_matrix_max = np.max(coeffs_matrix)
fourier_control.scale_min = coeffs_matrix_min
fourier_control.scale_max = coeffs_matrix_max
coeffs_matrix = (coeffs_matrix-coeffs_matrix_min)/(coeffs_matrix_max-coeffs_matrix_min)


K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

export_Ks = False
if export_Ks:
    for i in range(K_matrix.shape[0]):
        fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
    fe_mesh.Finalize(export_dir=case_dir)
    exit()


# design siren NN for learning
concate_network = MLP(name="concat_network",
                      input_size=13,
                      output_size=1,
                      hidden_layers=[200,200,200],
                      activation_settings={"type":"sin","prediction_gain":30,"initialization_gain":3},
                      skip_connections_settings={"active":False,"frequency":1})

# create fol optax-based optimizer

learning_rate_scheduler = optax.linear_schedule(init_value=1e-3, end_value=1e-5, transition_steps=500)
chained_transform = optax.chain(optax.adam(learning_rate_scheduler))

# create fol
fol = ImplicitParametricOperatorLearning(name="implicit_ol",control=fourier_control,
                                        loss_function=reg_loss,
                                        flax_neural_network=concate_network,
                                        optax_optimizer=chained_transform,
                                        checkpoint_settings={"restore_state":False,
                                        "state_directory":case_dir+"/flax_state"},
                                        working_directory=case_dir)

fol.Initialize()

train_start_id = 1
train_end_id = 2

# here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),batch_size=1,
            convergence_settings={"num_epochs":500,"relative_error":1e-100,
                                  "absolute_error":1e-100},
            plot_settings={"plot_save_rate":10000},
            save_settings={"save_nn_model":True})

for test in range(1):
    eval_id = train_start_id
    fe_mesh[f'Pred_K_{eval_id}'] = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
    fe_mesh[f'GT_K_{eval_id}'] = np.array(K_matrix[eval_id,:])
    fe_mesh[f'abs_error_{eval_id}'] = abs(fe_mesh[f'Pred_K_{eval_id}']-fe_mesh[f'GT_K_{eval_id}'])

    plot_mesh_vec_data(1,[fe_mesh[f'Pred_K_{eval_id}'],fe_mesh[f'GT_K_{eval_id}'],fe_mesh[f'abs_error_{eval_id}']],
                        ["Pred_K","GT_K","abs_error"],
                        fig_title="",
                        file_name=os.path.join(case_dir,f"test_{eval_id}.png"))

fe_mesh.Finalize(export_dir=case_dir)
