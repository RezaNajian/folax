import sys
import os
import optax
import numpy as np

from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.loss_functions.transient_thermal import TransientThermalLoss2DQuad
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# directory & save handling
working_directory_name = 'meta_implicit_pl_transient_nonlinear_thermal_2D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":51,
                "T_left":1.0,"T_right":0.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]}}

k0 = np.load("k0.npy").flatten()
material_dict = {"rho":1.0,"cp":1.0,"k0":k0,"beta":1.0,"c":4.0}
time_integration_dict = {"method":"implicit-euler","time_step":0.005}
transient_thermal_loss_2d = TransientThermalLoss2DQuad("thermal_transient_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict,
                                                                            "time_integration_dict":time_integration_dict},
                                                                            fe_mesh=fe_mesh)
transient_thermal_loss_2d.Initialize()

identity_control = IdentityControl("ident_control",num_vars=fe_mesh.GetNumberOfNodes())
identity_control.Initialize()

# design synthesizer & modulator NN for hypernetwork
characteristic_length = 64
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=3,
                     output_size=1,
                     hidden_layers=[characteristic_length] * 6,
                     activation_settings={"type":"sin",
                                          "prediction_gain":30,
                                          "initialization_gain":1.0},
                     skip_connections_settings={"active":False,"frequency":1})

latent_size = characteristic_length
modulator_nn = MLP(name="modulator_nn",
                   input_size=latent_size,
                   use_bias=False) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 2000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.adam(1e-5))
latent_step_optimizer = optax.chain(optax.adam(1e-4))
# create fol
ifol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_ol",
                                                        control=identity_control,
                                                        loss_function=transient_thermal_loss_2d,
                                                        flax_neural_network=hyper_network,
                                                        main_loop_optax_optimizer=main_loop_transform,
                                                        latent_step_optax_optimizer=latent_step_optimizer,
                                                        latent_step_size=0.01)

ifol.Initialize()


# train_start_id = 0
# train_end_id = 6000

# fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
#             batch_size=120,
#             convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
#             plot_settings={"plot_save_rate":100})


exit()

fol.RestoreState(os.path.join(".","flax_final_state"))
num_steps = 50

# FOL inference
eval_id = 2
FOL_T = np.array(fol.Predict_all(coeffs_matrix[eval_id,:].reshape(1,-1),num_steps))  
FOL_T = np.squeeze(FOL_T,axis=1).T    
fe_mesh['T_FOL'] = FOL_T    
# solve FE here
fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                            "maxiter":1000,"pre-conditioner":"ilu"},
                "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                            "maxiter":10,"load_incr":1}}
nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolverHetero("nonlinear_fe_solver",thermal_loss_2d,fe_setting)
nonlinear_fe_solver.Initialize()
FE_T = np.zeros((fe_mesh.GetNumberOfNodes(),num_steps))
FE_T_temp = coeffs_matrix[eval_id,:]
for i in range(num_steps):
    FE_T_temp = np.array(nonlinear_fe_solver.Solve(FE_T_temp,K_matrix,FE_T_temp))  #np.zeros(fe_mesh.GetNumberOfNodes())
    FE_T[:,i] = FE_T_temp    
absolute_error = np.abs(FOL_T- FE_T)
fe_mesh['abs_error'] = absolute_error
    
fe_mesh['Heterogeneity'] = K_matrix.reshape(-1,1)
fe_mesh['T_init'] = coeffs_matrix[eval_id,:].reshape(-1,1)
fe_mesh['T_FOL'] = FOL_T
fe_mesh['T_FE'] = FE_T

# time_list = [int(num_steps/5) - 1,int(num_steps/2) - 1,num_steps - 1]
time_list = [0,1,4,9,19,24,49]


np.save(os.path.join(case_dir,"test2_FOL_T.npy"),FOL_T)
np.save(os.path.join(case_dir,"test2_FE_T.npy"),FE_T)

plot_mesh_vec_data_thermal_row(1,[coeffs_matrix[eval_id,:]],
                   [""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test2_initial_condition.png"))

plot_mesh_quad(fe_mesh.GetNodesCoordinates()[:,:-1],
               fe_mesh.GetElementsNodes("quad"),
               background=hetero_info[::-1], 
               filename=os.path.join(case_dir,"FE_mesh_hetero_info.png"))

plot_mesh_vec_data_thermal_row(1,[FOL_T[:,time_list[0]],FOL_T[:,time_list[1]],
                                  FOL_T[:,time_list[2]],FOL_T[:,time_list[3]],
                                  FOL_T[:,time_list[4]],FOL_T[:,time_list[5]],
                                  FOL_T[:,time_list[6]]],
                   ["","","","","","",""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test2_FOL_summary.png"))
plot_mesh_vec_data_thermal_row(1,[FE_T[:,time_list[0]],FE_T[:,time_list[1]],
                                  FE_T[:,time_list[2]],FE_T[:,time_list[3]],
                                  FE_T[:,time_list[4]],FE_T[:,time_list[5]],
                                  FE_T[:,time_list[6]]],
                   ["","","","","","",""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test2_FE_summary.png"))

plot_mesh_vec_data_thermal_row(1,[absolute_error[:,time_list[0]],absolute_error[:,time_list[1]],
                                  absolute_error[:,time_list[2]],absolute_error[:,time_list[3]],
                                  absolute_error[:,time_list[4]],absolute_error[:,time_list[5]],
                                  absolute_error[:,time_list[6]]],
                   ["","","","","","",""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test2_Error_summary.png"))

fe_mesh.Finalize(export_dir=case_dir)
