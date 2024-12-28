import sys
import os
import optax
import numpy as np
from fol.loss_functions.regression_loss import RegressionLoss
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle

# directory & save handling
working_directory_name = 'meta_alpha_meta_implicit_regressor'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":11}

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

K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

export_Ks = False
if export_Ks:
    for i in range(K_matrix.shape[0]):
        fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
    fe_mesh.Finalize(export_dir=case_dir)
    exit()


# design siren NN for learning
characteristic_length = model_settings["N"]
characteristic_length = 64
synthesizer_nn = MLP(name="regressor_synthesizer",
                    input_size=3,
                    output_size=1,
                    hidden_layers=[characteristic_length] * 6,
                    activation_settings={"type":"sin",
                                         "prediction_gain":30,
                                         "initialization_gain":1.0})

latent_size = 10
modulator_nn = MLP(name="modulator_nn",
                   input_size=latent_size,
                   use_bias=False) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 2000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))
latent_step_optimizer = optax.chain(optax.normalize_by_update_norm(),optax.adam(1e-6))

# create fol
fol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=fourier_control,
                                                loss_function=reg_loss,
                                                flax_neural_network=hyper_network,
                                                main_loop_optax_optimizer=main_loop_transform,
                                                latent_step_optax_optimizer=latent_step_optimizer,
                                                latent_step_size=1e-2,
                                                checkpoint_settings={"restore_state":False,
                                                "state_directory":case_dir+"/flax_state",
                                                "meta_state_directory":case_dir+"/meta_state"},
                                                working_directory=case_dir)

fol.Initialize()

train_start_id = 0
train_end_id = 20
test_start_id = 3 * train_end_id
test_end_id = 4 * train_end_id

# here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
          test_set=(coeffs_matrix[test_start_id:test_end_id,:],),
           test_settings={"test_frequency":10},batch_size=1,
          convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
          plot_settings={"plot_save_rate":100},
          save_settings={"save_nn_model":True,
                         "best_model_checkpointing":True,
                         "best_model_checkpointing_frequency":10})


# load teh best model
fol.RestoreCheckPoint(fol.checkpoint_settings)

for eval_id in list(np.arange(train_start_id,test_end_id)):
    fe_mesh[f'Pred_K_{eval_id}'] = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
    fe_mesh[f'GT_K_{eval_id}'] = np.array(K_matrix[eval_id,:])
    fe_mesh[f'abs_error_{eval_id}'] = abs(fe_mesh[f'Pred_K_{eval_id}']-fe_mesh[f'GT_K_{eval_id}'])

    plot_mesh_vec_data(1,[fe_mesh[f'Pred_K_{eval_id}'],fe_mesh[f'GT_K_{eval_id}'],fe_mesh[f'abs_error_{eval_id}']],
                        ["Pred_K","GT_K","abs_error"],
                        fig_title="",
                        file_name=os.path.join(case_dir,f"test_{eval_id}.png"))

fe_mesh.Finalize(export_dir=case_dir)
