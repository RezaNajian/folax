import sys
import os
import optax
import numpy as np

from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning_field_hetero_bd import MetaImplicitParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver_hetero import FiniteElementNonLinearResidualBasedSolverHetero
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
from fol.loss_functions.thermal_transient_nonlinear_hetero import ThermalTransientLossNonlinear2DQuad
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import jax
jax.config.update("jax_default_matmul_precision", "float32")
# directory & save handling
working_directory_name = 'siren_implicit_thermal_nonlinear_2D_square'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":51,
                "T_left":1.0,"T_right":0.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]}}

material_dict = {"rho":1.0,"cp":1.0,"dt":0.005,"alpha_k":1.5}
thermal_loss_2d = ThermalTransientLossNonlinear2DQuad("thermal_transient_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)
no_control = NoControl("No_control",fe_mesh)

fe_mesh.Initialize()
thermal_loss_2d.Initialize()
no_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = True
if create_random_coefficients:
    def generate_random_smooth_patterns(L, N, num_samples=9000, smoothness_levels=[0.025, 0.05, 0.1, 0.2, 0.3, 0.4]):
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        X1, X2 = np.meshgrid(x, y)
        X = np.vstack([X1.ravel(), X2.ravel()]).T
        all_samples = []
        for length_scale in smoothness_levels:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
            num_per_level = num_samples // len(smoothness_levels)
            y_samples = gp.sample_y(X, n_samples=num_per_level, random_state=0)
            scaled_y_samples = np.array([(y_sample - np.min(y_sample)) / (np.max(y_sample) - np.min(y_sample))
                                         for y_sample in y_samples.T])
            all_samples.append(scaled_y_samples)

        mixed_samples = np.vstack(all_samples)
        np.random.shuffle(mixed_samples)

        return mixed_samples
    
    def generate_morph_pattern(N):
        hetero_morph = np.full((N * N), 1.0)
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)
        shapes = [
            {"type": "circle", "center": (0.6, 0.9), "radius": 0.25},
            {"type": "ellipse", "center": (0.5, 0.4), "radii": (0.35, 0.2), "rotation": np.pi / 6},
            {"type": "circle", "center": (0.5, 0.0), "radius": 0.2}#,
            # {"type": "rectangle", "center": (0.25, 0.25), "size": (0.2, 0.1), "rotation": 0}
        ]
        # Apply conditions for each shape
        for shape in shapes:
            if shape["type"] == "circle":
                mask = (X - shape["center"][0])**2 + ((1-Y) - shape["center"][1])**2 < shape["radius"]**2
            elif shape["type"] == "ellipse":
                a, b = shape["radii"]
                cx, cy = shape["center"]
                X_rot = (X - cx) * np.cos(shape["rotation"]) - ((1-Y) - cy) * np.sin(shape["rotation"])
                Y_rot = (X - cx) * np.sin(shape["rotation"]) + ((1-Y) - cy) * np.cos(shape["rotation"])
                mask = (X_rot**2 / a**2) + (Y_rot**2 / b**2) < 1
            elif shape["type"] == "rectangle":
                w, h = shape["size"]
                cx, cy = shape["center"]
                mask = (np.abs(X - cx) < w / 2) & (np.abs((1-Y) - cy) < h / 2)

            hetero_morph[mask.ravel()] = 0.1

        return hetero_morph
    
    coeffs_matrix = generate_random_smooth_patterns(model_settings["L"],model_settings["N"])
    hetero_info = generate_morph_pattern(model_settings["N"]).reshape(1,-1) #np.full((1,fe_mesh.GetNumberOfNodes()),1.0)#
    np.save(os.path.join(case_dir,"coeffs_matrix.npy"), coeffs_matrix)
    np.save(os.path.join(case_dir,"hetero_info.npy"), hetero_info)
else:
    pass

T_matrix = no_control.ComputeBatchControlledVariables(coeffs_matrix)
K_matrix = no_control.ComputeBatchControlledVariables(hetero_info)

# design synthesizer & modulator NN for hypernetwork
characteristic_length = model_settings["N"]
characteristic_length = 256
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
num_epochs = 10000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=no_control,
                                            loss_function=thermal_loss_2d,
                                            hetero_info=K_matrix.flatten(),
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3)
fol.Initialize()


train_start_id = 0
train_end_id = 6000

fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
            batch_size=120,
            convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
            plot_settings={"plot_save_rate":100},working_directory=case_dir)

num_steps = 50
# FOL inference
eval_id = 8500
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
# FE calculation
for i in range(num_steps):
    FE_T_temp = np.array(nonlinear_fe_solver.Solve(FE_T_temp,K_matrix,FE_T_temp))  #np.zeros(fe_mesh.GetNumberOfNodes())
    FE_T[:,i] = FE_T_temp    
fe_mesh['T_FE'] = FE_T
absolute_error = np.abs(FOL_T- FE_T)
fe_mesh['abs_error'] = absolute_error
    
fe_mesh['Heterogeneity'] = K_matrix.reshape(-1,1)
fe_mesh['T_init'] = coeffs_matrix[eval_id,:].reshape(-1,1)

time_list = [int(num_steps/5) - 1,int(num_steps/2) - 1,num_steps - 1]
time_list2 = [0,1,int(num_steps/10) - 1,int(num_steps/5) - 1]

plot_mesh_quad(fe_mesh.GetNodesCoordinates()[:,:-1],fe_mesh.GetElementsNodes("quad"),background=hetero_info.reshape(model_settings["N"],model_settings["N"]), filename=os.path.join(case_dir,"FE_mesh_hetero_info.png"))

plot_mesh_vec_data_thermal_clear(1,[coeffs_matrix[eval_id],FOL_T[:,time_list[0]],FOL_T[:,time_list[1]],FOL_T[:,time_list[2]]],
                   ["","","",""],
                   fig_title="Initial condition and implicit FOL solution",cmap = "jet",
                   file_name=os.path.join(case_dir,"FOL-T-dist.png"))

plot_mesh_vec_data_thermal_clear(1,[coeffs_matrix[eval_id],FE_T[:,time_list[0]],FE_T[:,time_list[1]],FE_T[:,time_list[2]]],
                   ["","","",""],
                   fig_title="Initial condition and FEM solution",cmap = "jet",
                   file_name=os.path.join(case_dir,"FEM-T-dist.png"))
plot_mesh_vec_data_thermal_clear(1,[coeffs_matrix[eval_id],absolute_error[:,time_list[0]],absolute_error[:,time_list[1]],absolute_error[:,time_list[2]]],
                   ["","","",""],
                   fig_title="Initial condition and iFOL error against FEM",cmap = "jet",
                   file_name=os.path.join(case_dir,"FOL-T-Error-dist.png"))

plot_mesh_vec_data_thermal_clear(1,[FOL_T[:,time_list2[0]],FOL_T[:,time_list2[1]],FOL_T[:,time_list2[2]],FOL_T[:,time_list2[3]]],
                   ["","","",""],
                   fig_title="Initial condition and implicit FOL solution",cmap = "jet",
                   file_name=os.path.join(case_dir,"FOL-T-dist_initial_steps.png"))

plot_mesh_vec_data_thermal_clear(1,[FE_T[:,time_list2[0]],FE_T[:,time_list2[1]],FE_T[:,time_list2[2]],FE_T[:,time_list2[3]]],
                   ["","","",""],
                   fig_title="Initial condition and FEM solution",cmap = "jet",
                   file_name=os.path.join(case_dir,"FEM-T-dist_initial_steps.png"))
plot_mesh_vec_data_thermal_clear(1,[absolute_error[:,time_list2[0]],absolute_error[:,time_list2[1]],absolute_error[:,time_list2[2]],absolute_error[:,time_list2[3]]],
                   ["","","",""],
                   fig_title="Initial condition and iFOL error against FEM",cmap = "jet",
                   file_name=os.path.join(case_dir,"FOL-T-Error-dist_initial_steps.png"))

fe_mesh.Finalize(export_dir=case_dir)
