import sys
import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import optax
import numpy as np
from fol.loss_functions.thermo_mechanical_nonlinear import ThermoMechanicalLoss3DHexa
from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
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
working_directory_name = 'meta_learning_thermo_mechanical'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))
# problem setup 
model_settings = {"L":1,"N":21,
                  "Ux_left":0.0,"Ux_right":0.00,
                  "Uy_top":0.0,"Uy_bottom":0.00,
                  "Uz_front":0.0,"Uz_back":0.00,
                  "T_left":1.0,"T_right":0.0}

# creation of the model
mesh_res_rate = 1
# fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh = create_3D_box_mesh(Nx=model_settings["N"],Ny=model_settings["N"],Nz=model_settings["N"],
                             Lx=model_settings["L"],Ly=model_settings["L"],Lz=model_settings["L"],case_dir=case_dir)

# # create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]},
           "Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
           "Uy":{"top":model_settings["Uy_top"],"bottom":model_settings["Uy_bottom"]},
           "Uz":{"front":model_settings["Uz_front"],"back":model_settings["Uz_back"]}}#
Dirichlet_BCs = False
material_dict = {"young_modulus":1.0,"poisson_ratio":0.3,"T0":jnp.full((fe_mesh.GetNumberOfNodes(),),1e-4),}
thermomech_loss_3d = ThermoMechanicalLoss3DHexa("thermomechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "material_dict":material_dict, "alpha":1.5,
                                                                            "beta":2.0, 
                                                                            "c":2.0},
                                                                            fe_mesh=fe_mesh)
no_control = NoControl("No_Control",fe_mesh)

x_freqs = np.array([1,2,3])
y_freqs = np.array([1,2,3])
z_freqs = np.array([1,2,3])
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":5,"min":1e-1,"max":1}
fourier_control = FourierControl("K",fourier_control_settings,fe_mesh)

fe_mesh.Initialize()
thermomech_loss_3d.Initialize()
no_control.Initialize()
fourier_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = True
if create_random_coefficients:
    number_of_random_samples = 1
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = {}
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = x_freqs
    export_dict["y_freqs"] = y_freqs
    export_dict["z_freqs"] = z_freqs
    with open(f'fourier_control_dict.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'fourier_control_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

# test_E_coeff = generate_morph_pattern_smooth(fe_mesh.GetNodesCoordinates()).reshape(1,-1)
# test_K_coeff = generate_morph_pattern_smooth(fe_mesh.GetNodesCoordinates()).reshape(1,-1)
# test_E_coeff = np.full((1,fe_mesh.GetNumberOfNodes()),1.0)
# test_K_coeff = np.full((1,fe_mesh.GetNumberOfNodes()),1.0)
test_E_coeff = K_matrix.reshape(1,-1)
test_K_coeff = K_matrix.reshape(1,-1)
initial_temp = np.full((1,fe_mesh.GetNumberOfNodes()),1e-4)

E_matrix = test_E_coeff.flatten()#fourier_control.ComputeBatchControlledVariables(coeffs_matrix)
K_matrix = test_K_coeff.flatten()#fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

characteristic_length = 256
output_size = 4
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
num_epochs = 200
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
# main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))#
main_loop_transform = optax.chain(
    optax.normalize_by_update_norm(),
    optax.adam(learning_rate_scheduler)
)

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=no_control,
                                            loss_function=thermomech_loss_3d,
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3)
fol.Initialize()

train_start_id = 0
train_end_id = 1

fol.Train(train_set=(test_K_coeff,test_E_coeff),
          batch_size=1,
          convergence_settings={"num_epochs":num_epochs,
                                "relative_error":1e-100,
                                "absolute_error":1e-100},
          working_directory=case_dir)

# num_steps = 1
FOL_TUV = np.array(fol.Predict((test_K_coeff)))  
FOL_TUV = FOL_TUV.reshape((fe_mesh.GetNumberOfNodes(), 4))
fe_mesh['sol_FOL'] = FOL_TUV

# solve FE here
fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-10,"atol":1e-10,
                                            "maxiter":1000,"pre-conditioner":"ilu","Dirichlet_BCs":Dirichlet_BCs},
                "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                            "maxiter":10,"load_incr":1}}
nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlinear_fe_solver",thermomech_loss_3d,fe_setting)
nonlinear_fe_solver.Initialize()

FE_TUV = np.array(nonlinear_fe_solver.SolveMulti(test_K_coeff.flatten(),np.zeros((fe_mesh.GetNumberOfNodes()*output_size))))  #
FE_TUV = reshape_T_U_to_nodewise3D(FE_TUV, fe_mesh.GetNumberOfNodes())
fe_mesh['sol_FE'] = FE_TUV
absolute_error = np.abs(FOL_TUV- FE_TUV)
fe_mesh['abs_error'] = absolute_error
fe_mesh['Heterogeneity'] = test_K_coeff.T

# FOL_TUV = FOL_TUV.reshape((model_settings["N"],model_settings["N"], 3))
# FE_TUV = FE_TUV.reshape((model_settings["N"],model_settings["N"], 3))
# absolute_error = absolute_error.reshape((model_settings["N"],model_settings["N"], 3))
# test_K_coeff = test_K_coeff.reshape((model_settings["N"],model_settings["N"]))
# FOL_TUV = FOL_TUV[::-1,:,:]
# FE_TUV = FE_TUV[::-1,:,:]
# absolute_error = absolute_error[::-1,:,:]
# test_K_coeff = test_K_coeff[::-1,:]
# FOL_TUV = FOL_TUV.reshape((fe_mesh.GetNumberOfNodes(), 3))
# FE_TUV = FE_TUV.reshape((fe_mesh.GetNumberOfNodes(), 3))
# absolute_error = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 3))
# test_K_coeff = test_K_coeff.reshape((1,fe_mesh.GetNumberOfNodes()))

# plot_mesh_vec_data_thermal_clear(1,[test_K_coeff[0],FOL_TUV[:,0],FOL_TUV[:,1],FOL_TUV[:,2]],
#                    ["Heterogeneity","Temperature","X-displacement","Y-displacement"],
#                    fig_title="Initial condition and implicit FOL solution",cmap = "rainbow",
#                    file_name=os.path.join(case_dir,"sol_FOL.png"))

# plot_mesh_vec_data_thermal_clear(1,[test_K_coeff[0],FE_TUV[:,0],FE_TUV[:,1],FE_TUV[:,2]],
#                    ["Heterogeneity","Temperature","X-displacement","Y-displacement"],
#                    fig_title="Initial condition and FE solution",cmap = "rainbow",
#                    file_name=os.path.join(case_dir,"sol_FE.png"))

# plot_mesh_vec_data_thermal_clear(1,[test_K_coeff[0],absolute_error[:,0],absolute_error[:,1],absolute_error[:,2]],
#                    ["Heterogeneity","Temperature","X-displacement","Y-displacement"],
#                    fig_title="Absolute error",cmap = "jet",
#                    file_name=os.path.join(case_dir,"sol_AbsError.png"))


# # Stress and heat flux
# FOL_UV = FOL_TUV[:,1:]
# FE_UV = FE_TUV[:,1:]
# FE_T = FE_TUV[:,0]
# FOL_T = FOL_TUV[:,0]

# FOLstress_at_nodes = GetStressVector2D(thermomech_loss_3d,fe_mesh, test_E_coeff.flatten(),
#                                        FOL_UV.flatten(),FOL_T.flatten(),initial_temp.flatten())
# FE_stress_at_nodes = GetStressVector2D(thermomech_loss_3d,fe_mesh, test_E_coeff.flatten(),
#                                        FE_UV.flatten(),FE_T.flatten(),initial_temp.flatten())
# absolute_error = np.abs(FOLstress_at_nodes - FE_stress_at_nodes)
# fe_mesh['FOL_stress'] = FOLstress_at_nodes
# fe_mesh['FE_stress'] = FE_stress_at_nodes   
# fe_mesh['abs_error_stress'] = absolute_error

# plot_mesh_vec_data_thermal_clear(1,[test_E_coeff.flatten(),FOLstress_at_nodes[:,0],FOLstress_at_nodes[:,1],FOLstress_at_nodes[:,2]],
#                    ["Heterogeneity","XX-stress","YY-stress","XY-stress"],
#                    fig_title="FOL stress",cmap = "jet",
#                    file_name=os.path.join(case_dir,"stress_FOL.png"))

# plot_mesh_vec_data_thermal_clear(1,[test_E_coeff.flatten(),FE_stress_at_nodes[:,0],FE_stress_at_nodes[:,1],FE_stress_at_nodes[:,2]],
#                    ["Heterogeneity","XX-stress","YY-stress","XY-stress"],
#                    fig_title="FE stress",cmap = "jet",
#                    file_name=os.path.join(case_dir,"stress_FE.png"))
# plot_mesh_vec_data_thermal_clear(1,[test_E_coeff.flatten(),absolute_error[:,0],absolute_error[:,1],absolute_error[:,2]],
#                    ["Heterogeneity","XX-stress","YY-stress","XY-stress"],
#                    fig_title="Absolute error",cmap = "jet",
#                    file_name=os.path.join(case_dir,"stress_AbsError.png"))

# FOL_heat_flux_at_nodes = GetHeatFluxVector2D(thermomech_loss_3d,fe_mesh,test_K_coeff.flatten(),FOL_T.flatten())
# FE_heat_flux_at_nodes = GetHeatFluxVector2D(thermomech_loss_3d,fe_mesh,test_K_coeff.flatten(),FE_T.flatten())
# heat_flux_absolute_error = np.abs(FOL_heat_flux_at_nodes - FE_heat_flux_at_nodes)
# fe_mesh['FOL_heat_flux'] = FOL_heat_flux_at_nodes
# fe_mesh['FE_heat_flux'] = FE_heat_flux_at_nodes
# fe_mesh['abs_error_heat_flux'] = heat_flux_absolute_error
# plot_mesh_vec_data_thermal_clear(1,[test_K_coeff.flatten(),FOL_heat_flux_at_nodes[:,0],FOL_heat_flux_at_nodes[:,1]],
#                    ["Heterogeneity","X-heat flux","Y-heat flux"],
#                    fig_title="FOL heat flux",cmap = "inferno",
#                    file_name=os.path.join(case_dir,"heat_flux_FOL.png"))
# plot_mesh_vec_data_thermal_clear(1,[test_K_coeff.flatten(),FE_heat_flux_at_nodes[:,0],FE_heat_flux_at_nodes[:,1]],
#                    ["Heterogeneity","X-heat flux","Y-heat flux"],
#                    fig_title="FE heat flux",cmap = "inferno",
#                    file_name=os.path.join(case_dir,"heat_flux_FE.png"))
# plot_mesh_vec_data_thermal_clear(1,[test_K_coeff.flatten(),heat_flux_absolute_error[:,0],heat_flux_absolute_error[:,1]],
#                    ["Heterogeneity","X-heat flux","Y-heat flux"],
#                    fig_title="Absolute error",cmap = "inferno",
#                    file_name=os.path.join(case_dir,"heat_flux_AbsError.png"))
fe_mesh.Finalize(export_dir=case_dir)
