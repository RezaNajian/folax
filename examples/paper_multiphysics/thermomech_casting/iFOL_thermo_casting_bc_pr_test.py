import sys
import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import optax
import numpy as np
from fol.loss_functions.thermo_mechanical_nonlinear import ThermoMechanicalLoss3DTetra
# from fol.loss_functions.thermal import ThermalLoss3DTetra
from fol.mesh_input_output.mesh import Mesh
from fol.controls.identity_control import IdentityControl
from fol.controls.fourier_control import FourierControl
from fol.controls.dirichlet_control import DirichletControl
from meta_implicit_parametric_operator_learning_no_ad import MetaImplicitParametricOperatorLearning
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
name_main = 'iFOL_thermo_mechanical'
working_directory_name = name_main+'_test'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))
# problem setup 
model_settings = {"L":1,"N":20,
                  "Ux_left":0.0,"Ux_right":0.0,
                  "Uy_top":0.0,"Uy_bottom":0.0,
                  "Uz_front":0.0,"Uz_back":0.0,
                  "T_left":1.0,"T_right":0.0}

# creation of the model
mesh_res_rate = 1

fe_mesh = Mesh("fol_io","casting_base.med")
fe_mesh.Initialize()
# # create fe-based loss function

bc_dict = {"T":{"Inlet_top":1.0,"F_bottom":0.001,"Surroundings":0.001},#
           "Ux":{"Inlet_top":0.0,"Inlet_bottom":0.0,"F_bottom":0.0},#"F_right1":model_settings["Ux_right"],"F_right1":model_settings["Ux_right"]
           "Uy":{"Inlet_top":0.0,"Inlet_bottom":0.0,"F_bottom":0.0},
           "Uz":{"Inlet_top":0.0,"F_bottom":0.0}}#,"F_rear":model_settings["Uz_back"]

Dirichlet_BCs = False
material_dict = {"young_modulus":1.0,"poisson_ratio":0.3,"T0":jnp.full((fe_mesh.GetNumberOfNodes(),),1e-4),}
thermomech_loss_3d = ThermoMechanicalLoss3DTetra("thermomechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "material_dict":material_dict, "alpha":1.5,
                                                                            "beta":2.0, 
                                                                            "c":2.0,
                                                                            "K_matrix":np.ones((fe_mesh.GetNumberOfNodes(),)),
                                                                            "parametric_boundary_learning":True},
                                                                            fe_mesh=fe_mesh)
no_control = IdentityControl("No_Control",fe_mesh)

thermomech_loss_3d.Initialize()
no_control.Initialize()
displ_control_settings = {"learning_boundary": {"T":{"Inlet_top"}}}

displ_control = DirichletControl("displ_control",displ_control_settings,fe_mesh,thermomech_loss_3d)
displ_control.Initialize()


create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 500
    bc_matrix,bc_nodal_value_matrix = create_uniform_dist_bc_samples(displ_control,
                                                                    numberof_sample=number_of_random_samples,
                                                                    low=0.001, high=1.0)
    export_dict = {}
    export_dict["bc_matrix"] = bc_matrix
    export_dict["point_bc_settings"] = bc_dict
    export_dict["displ_control_settings"] = displ_control_settings
    with open(f'bc_control_dict.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'bc_control_dict.pkl', 'rb') as f:
        loaded_control_dict = pickle.load(f)
    
    bc_matrix = loaded_control_dict["bc_matrix"]

K_matrix = np.full((1,fe_mesh.GetNumberOfNodes()),1.0)
test_E_coeff = K_matrix[0].reshape(1,-1)
test_K_coeff = K_matrix[0].reshape(1,-1)
initial_temp = np.full((1,fe_mesh.GetNumberOfNodes()),1e-4)
E_matrix = test_E_coeff.flatten()#fourier_control.ComputeBatchControlledVariables(coeffs_matrix)
K_matrix = test_K_coeff.flatten()#fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

characteristic_length = 128
output_size = 4
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=3,
                     output_size=output_size,
                     hidden_layers=[characteristic_length] * 6,
                     activation_settings={"type":"leaky_relu",
                                          "prediction_gain":30,
                                          "initialization_gain":1.0},
                    skip_connections_settings={"active":True,"frequency":1})
latent_size = characteristic_length
modulator_nn = MLP(name="modulator_nn",
                   input_size=1,
                   hidden_layers=[characteristic_length]*4,
                    activation_settings={"type":"leaky_relu",
                                          "prediction_gain":30,
                                          "initialization_gain":1.0},
                   skip_connections_settings={"active":False,"frequency":1}) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 2000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
# main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))#
main_loop_transform = optax.chain(
    optax.normalize_by_update_norm(),
    optax.adam(learning_rate_scheduler)
)

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=displ_control,
                                            loss_function=thermomech_loss_3d,
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3)
fol.Initialize()

train_start_id = 0
train_end_id = 50
wanted_bc = np.array([[0.3]])  # T, Ux, Uy, Uz
bc_matrix = np.vstack((bc_matrix,wanted_bc))
print(bc_matrix.shape)
print(bc_matrix[-1,:])
# fol.Train(train_set=(bc_matrix[train_start_id:train_end_id,:].reshape(-1,1),),
#           batch_size=1,
#           convergence_settings={"num_epochs":num_epochs,
#                                 "relative_error":1e-100,
#                                 "absolute_error":1e-100},
#           working_directory=case_dir,
#           plot_settings={"plot_list":["total_loss","phy1_loss","phy2_loss"],
#                "plot_frequency":1,"save_frequency":100,
#                "save_directory":".","multiphysics":True})
fol.RestoreState(restore_state_directory=name_main+"/flax_final_state")

test_start_id = 51
test_end_id = 61

for i in np.arange(test_start_id,test_end_id,1):
    FOL_TUVW = np.array(fol.Predict((bc_matrix[i:i+1,:].reshape(-1,1))))  
    FOL_TUVW = FOL_TUVW.reshape((fe_mesh.GetNumberOfNodes(), 4))
    fe_mesh['sol_FOL'] = FOL_TUVW

    # solve FE here
    # Initialize the fe loss function with the desired boundary conditions
    bc_dict = {"T":{"Inlet_top":bc_matrix[i,:],"F_bottom":0.001,"Surroundings":0.001},#
               "Ux":{"Inlet_top":0.0,"Inlet_bottom":0.0,"F_bottom":0.0},#"F_right1":model_settings["Ux_right"],"F_right1":model_settings["Ux_right"]
               "Uy":{"Inlet_top":0.0,"Inlet_bottom":0.0,"F_bottom":0.0},
               "Uz":{"Inlet_top":0.0,"F_bottom":0.0}}#,"F_rear":model_settings["Uz_back"]

    Dirichlet_BCs = False
    material_dict = {"young_modulus":1.0,"poisson_ratio":0.3,"T0":jnp.full((fe_mesh.GetNumberOfNodes(),),1e-4),}
    thermomech_loss_3d = ThermoMechanicalLoss3DTetra("thermomechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "material_dict":material_dict, "alpha":1.5,
                                                                                "beta":2.0, 
                                                                                "c":2.0},
                                                                                fe_mesh=fe_mesh)

    thermomech_loss_3d.Initialize()

    fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu","Dirichlet_BCs":Dirichlet_BCs},
                    "nonlinear_solver_settings":{"rel_tol":1e-4,"abs_tol":1e-4,
                                                "maxiter":10,"load_incr":1}}
    nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlinear_fe_solver",thermomech_loss_3d,fe_setting)
    nonlinear_fe_solver.Initialize()

    FE_TUVW = np.array(nonlinear_fe_solver.Solve(test_K_coeff.flatten(),np.zeros((fe_mesh.GetNumberOfNodes()*4))))  #
    # FE_TUVW = FE_TUVW.reshape((fe_mesh.GetNumberOfNodes(), output_size))
    FE_TUVW = reshape_T_U_to_nodewise3D(FE_TUVW, fe_mesh.GetNumberOfNodes())
    fe_mesh['sol_FE'] = FE_TUVW
    fe_mesh['sol_Disp'] = FE_TUVW[:,1:]
    absolute_error = np.abs(FOL_TUVW- FE_TUVW)
    fe_mesh['abs_error'] = absolute_error
    fe_mesh['relative_error'] = absolute_error / (np.abs(FE_TUVW)+1e-10)
    fe_mesh['Heterogeneity'] = test_K_coeff.reshape((fe_mesh.GetNumberOfNodes(),1))

    # # Stress and heat flux
    FOL_UVW = FOL_TUVW[:,1:]
    FE_UVW = FE_TUVW[:,1:]
    FE_T = FE_TUVW[:,0]
    FOL_T = FOL_TUVW[:,0]

    FOLstress_at_nodes = GetStressVector3D(thermomech_loss_3d,fe_mesh, test_E_coeff.flatten(),
                            FOL_UVW.flatten(),FOL_T.flatten(),initial_temp.flatten())
    FE_stress_at_nodes = GetStressVector3D(thermomech_loss_3d,fe_mesh, test_E_coeff.flatten(),
                                           FE_UVW.flatten(),FE_T.flatten(),initial_temp.flatten())
    absolute_error = np.abs(FOLstress_at_nodes - FE_stress_at_nodes)
    fe_mesh['FOL_stress'] = FOLstress_at_nodes
    fe_mesh['FE_stress'] = FE_stress_at_nodes   
    fe_mesh['abs_error_stress'] = absolute_error
    fe_mesh['relative_error_stress'] = absolute_error / (np.abs(FE_stress_at_nodes)+1e-10)

    FOL_heat_flux_at_nodes = GetHeatFluxVector3D(thermomech_loss_3d,fe_mesh,test_K_coeff.flatten(),FOL_T.flatten())
    FE_heat_flux_at_nodes = GetHeatFluxVector3D(thermomech_loss_3d,fe_mesh,test_K_coeff.flatten(),FE_T.flatten())
    heat_flux_absolute_error = np.abs(FOL_heat_flux_at_nodes - FE_heat_flux_at_nodes)
    fe_mesh['FOL_heat_flux'] = FOL_heat_flux_at_nodes
    fe_mesh['FE_heat_flux'] = FE_heat_flux_at_nodes
    fe_mesh['abs_error_heat_flux'] = heat_flux_absolute_error
    fe_mesh['relative_error_heat_flux'] = heat_flux_absolute_error / (np.abs(FE_heat_flux_at_nodes)+1e-10)

    fe_mesh.mesh_io.write(os.path.join(case_dir, f"casting_case_{i}.vtk"),file_format="vtk")

