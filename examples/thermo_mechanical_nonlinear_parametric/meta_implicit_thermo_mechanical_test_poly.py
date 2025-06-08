import sys
import os
import optax
import numpy as np
from fol.loss_functions.thermo_mechanical_nonlinear import ThermoMechanicalLoss2DQuad
from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.controls.fourier_control import FourierControl
from fol.controls.voronoi_control2D import VoronoiControl2D
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning_field_hetero_multiphysics import MetaImplicitParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from thermo_mechanical_useful_functions import *
from jax import config
import jax 
jax.config.update("jax_enable_x64", False)
config.update("jax_default_matmul_precision", "highest")
# directory & save handling
working_directory_name = 'meta_learning_thermo_mechanical_poly'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))
# problem setup 
model_settings = {"L":1,"N":51,
                  "Ux_left":0.0,"Ux_right":0.00,
                  "Uy_top":0.0,"Uy_bottom":0.00,
                  "T_left":1.0,"T_right":0.0}

# creation of the model
mesh_res_rate = 1
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
# # create fe-based loss function
# bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]},
#            "Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
#            "Uy":{"top":model_settings["Uy_top"],"bottom":model_settings["Uy_bottom"]}}#
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]},
           "Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
           "Uy":{"top":model_settings["Uy_top"],"bottom":model_settings["Uy_bottom"]}}#
Dirichlet_BCs = False
material_dict = {"young_modulus":1.0,"poisson_ratio":0.3}
thermomech_loss_2d = ThermoMechanicalLoss2DQuad("thermomechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "material_dict":material_dict, "alpha":1.5,
                                                                            "beta":2.0, 
                                                                            "c":2.0},
                                                                            fe_mesh=fe_mesh)
no_control = NoControl("No_Control",fe_mesh)
fe_mesh.Initialize()
thermomech_loss_2d.Initialize()
no_control.Initialize()

voronoi_control_settings = {"number_of_seeds":5,"E_values":(0.3,1)}
voronoi_control = VoronoiControl2D("first_voronoi_control",voronoi_control_settings,fe_mesh)
voronoi_control.Initialize()
number_of_random_samples = 2
test_K_coeff, K_matrix = create_random_voronoi_samples(voronoi_control,number_of_random_samples)
test_E_coeff = test_K_coeff
E_matrix = K_matrix

initial_temp = np.zeros((1,fe_mesh.GetNumberOfNodes()))

characteristic_length = 256
output_size = 3
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=3,
                     output_size=output_size,
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
num_epochs = 3000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
# main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))#
main_loop_transform = optax.chain(
    optax.normalize_by_update_norm(),
    optax.adam(learning_rate_scheduler)
)

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=no_control,
                                            loss_function=thermomech_loss_2d,
                                            fixed_feature = initial_temp.flatten(),
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3)
fol.Initialize()

train_start_id = 0
train_end_id = 4000
test_start_id = 4000
test_end_id = 5000

# fol.Train(train_set=(K_matrix[train_start_id:train_end_id,:],E_matrix[train_start_id:train_end_id,:]),
#           test_set=(K_matrix[test_start_id:test_end_id,:],E_matrix[test_start_id:test_end_id,:]),
#           batch_size=100,
#           convergence_settings={"num_epochs":num_epochs,
#                                 "relative_error":1e-100,
#                                 "absolute_error":1e-100},
#           working_directory=case_dir)

fol.RestoreState(restore_state_directory="flax_final_state")

test_start_id = 0
test_end_id = 1

FOL_TUV = np.array(fol.Predict((K_matrix[test_start_id:test_end_id,:],E_matrix[test_start_id:test_end_id,:])))  
FOL_TUV = FOL_TUV.reshape((fe_mesh.GetNumberOfNodes(), 3))
fe_mesh['sol_FOL'] = FOL_TUV

# solve FE here
fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-10,"atol":1e-10,
                                            "maxiter":1000,"pre-conditioner":"ilu","Dirichlet_BCs":Dirichlet_BCs},
                "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                            "maxiter":10,"load_incr":1}}
nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlinear_fe_solver",thermomech_loss_2d,fe_setting)
nonlinear_fe_solver.Initialize()

FE_TUV = np.array(nonlinear_fe_solver.SolveMulti(K_matrix[test_start_id:test_end_id,:].flatten(),E_matrix[test_start_id:test_end_id,:].flatten()
                                         ,np.zeros((fe_mesh.GetNumberOfNodes()*output_size)),initial_temp.flatten()))  #
FE_TUV = reshape_T_U_to_nodewise(FE_TUV, fe_mesh.GetNumberOfNodes())
fe_mesh['sol_FE'] = FE_TUV
absolute_error = np.abs(FOL_TUV- FE_TUV)
fe_mesh['abs_error'] = absolute_error


plot_mesh_vec_data_thermal_clear(1,[K_matrix[test_start_id:test_end_id,:].flatten(),FOL_TUV[:,0],FOL_TUV[:,1],FOL_TUV[:,2]],
                   ["Heterogeneity","Temperature","X-displacement","Y-displacement"],
                   fig_title="Initial condition and implicit FOL solution",cmap = "rainbow",
                   file_name=os.path.join(case_dir,"sol_FOL.png"))

plot_mesh_vec_data_thermal_clear(1,[K_matrix[test_start_id:test_end_id,:].flatten(),FE_TUV[:,0],FE_TUV[:,1],FE_TUV[:,2]],
                   ["Heterogeneity","Temperature","X-displacement","Y-displacement"],
                   fig_title="Initial condition and FE solution",cmap = "rainbow",
                   file_name=os.path.join(case_dir,"sol_FE.png"))

plot_mesh_vec_data_thermal_clear(1,[K_matrix[test_start_id:test_end_id,:].flatten(),absolute_error[:,0],absolute_error[:,1],absolute_error[:,2]],
                   ["Heterogeneity","Temperature","X-displacement","Y-displacement"],
                   fig_title="Absolute error",cmap = "jet",
                   file_name=os.path.join(case_dir,"sol_AbsError.png"))


# Stress and heat flux
FOL_UV = FOL_TUV[:,1:]
FE_UV = FE_TUV[:,1:]
FE_T = FE_TUV[:,0]
FOL_T = FOL_TUV[:,0]

FOLstress_at_nodes = GetStressVector2D(thermomech_loss_2d,fe_mesh, E_matrix[test_start_id:test_end_id,:].flatten(),
                                       FOL_UV.flatten(),FOL_T.flatten(),initial_temp.flatten())
FE_stress_at_nodes = GetStressVector2D(thermomech_loss_2d,fe_mesh, E_matrix[test_start_id:test_end_id,:].flatten(),
                                       FE_UV.flatten(),FE_T.flatten(),initial_temp.flatten())
absolute_error = np.abs(FOLstress_at_nodes - FE_stress_at_nodes)
fe_mesh['FOL_stress'] = FOLstress_at_nodes
fe_mesh['FE_stress'] = FE_stress_at_nodes   
fe_mesh['abs_error_stress'] = absolute_error

plot_mesh_vec_data_thermal_clear(1,[E_matrix[test_start_id:test_end_id,:].flatten(),FOLstress_at_nodes[:,0],FOLstress_at_nodes[:,1],FOLstress_at_nodes[:,2]],
                   ["Heterogeneity","XX-stress","YY-stress","XY-stress"],
                   fig_title="FOL stress",cmap = "jet",
                   file_name=os.path.join(case_dir,"stress_FOL.png"))

plot_mesh_vec_data_thermal_clear(1,[E_matrix[test_start_id:test_end_id,:].flatten(),FE_stress_at_nodes[:,0],FE_stress_at_nodes[:,1],FE_stress_at_nodes[:,2]],
                   ["Heterogeneity","XX-stress","YY-stress","XY-stress"],
                   fig_title="FE stress",cmap = "jet",
                   file_name=os.path.join(case_dir,"stress_FE.png"))
plot_mesh_vec_data_thermal_clear(1,[E_matrix[test_start_id:test_end_id,:].flatten(),absolute_error[:,0],absolute_error[:,1],absolute_error[:,2]],
                   ["Heterogeneity","XX-stress","YY-stress","XY-stress"],
                   fig_title="Absolute error",cmap = "jet",
                   file_name=os.path.join(case_dir,"stress_AbsError.png"))

FOL_heat_flux_at_nodes = GetHeatFluxVector2D(thermomech_loss_2d,fe_mesh,K_matrix[test_start_id].flatten(),FOL_T.flatten())
FE_heat_flux_at_nodes = GetHeatFluxVector2D(thermomech_loss_2d,fe_mesh,K_matrix[test_start_id].flatten(),FE_T.flatten())
heat_flux_absolute_error = np.abs(FOL_heat_flux_at_nodes - FE_heat_flux_at_nodes)
FOL_heat_flux_magnitude = np.linalg.norm(FOL_heat_flux_at_nodes, axis=1)
FE_heat_flux_magnitude = np.linalg.norm(FE_heat_flux_at_nodes, axis=1)
heat_flux_magnitude_absolute_error = np.abs(FOL_heat_flux_magnitude - FE_heat_flux_magnitude)
fe_mesh['FOL_heat_flux'] = FOL_heat_flux_at_nodes
fe_mesh['FE_heat_flux'] = FE_heat_flux_at_nodes
fe_mesh['abs_error_heat_flux'] = heat_flux_absolute_error
plot_mesh_vec_data_thermal_clear(1,[K_matrix[test_start_id].flatten(),FOL_heat_flux_at_nodes[:,0],FOL_heat_flux_at_nodes[:,1],FOL_heat_flux_magnitude],
                   ["Heterogeneity","X-heat flux","Y-heat flux","Heat flux magnitude"],
                   fig_title="FOL heat flux",cmap = "inferno",
                   file_name=os.path.join(case_dir,"heat_flux_FOL.png"))
plot_mesh_vec_data_thermal_clear(1,[K_matrix[test_start_id].flatten(),FE_heat_flux_at_nodes[:,0],FE_heat_flux_at_nodes[:,1],FE_heat_flux_magnitude],
                   ["Heterogeneity","X-heat flux","Y-heat flux","Heat flux magnitude"],
                   fig_title="FE heat flux",cmap = "inferno",
                   file_name=os.path.join(case_dir,"heat_flux_FE.png"))
plot_mesh_vec_data_thermal_clear(1,[K_matrix[test_start_id].flatten(),heat_flux_absolute_error[:,0],heat_flux_absolute_error[:,1],heat_flux_magnitude_absolute_error],
                   ["Heterogeneity","X-heat flux","Y-heat flux","Heat flux magnitude"],
                   fig_title="Absolute error",cmap = "inferno",
                   file_name=os.path.join(case_dir,"heat_flux_AbsError.png"))
fe_mesh.Finalize(export_dir=case_dir)
