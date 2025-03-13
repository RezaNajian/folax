import sys
import os
import optax
import numpy as np
from fol.loss_functions.phasefield import PhaseFieldLoss2DTri
from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning_pf import MetaImplicitParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver_phasefield import FiniteElementNonLinearResidualBasedSolverPhasefield
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from jax import config

config.update("jax_default_matmul_precision", "float32")
# directory & save handling
working_directory_name = 'meta_learning_phasefield_particle_bumps'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# creation of the model
mesh_res_rate = 1
fe_mesh = Mesh("fol_io","Li_battery_particle_bumps.med")
# create fe-based loss function
bc_dict = {"T":{}}#"left":model_settings["T_left"],"right":model_settings["T_right"]
Dirichlet_BCs = False
material_dict = {"rho":1.0,"cp":1.0,"dt":0.001,"epsilon":0.2}
phasefield_loss_2d = PhaseFieldLoss2DTri("phasefield_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)
no_control = NoControl("No_Control",fe_mesh)

fe_mesh.Initialize()
phasefield_loss_2d.Initialize()
no_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = True
if create_random_coefficients:
    def generate_random_smooth_patterns_from_mesh(coords, num_samples=10000,smoothness_levels=[0.15, 0.2, 0.3, 0.4, 0.5]):
        X = coords[:, :2]  
        all_samples = []

        for length_scale in smoothness_levels:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
            num_per_level = num_samples // len(smoothness_levels)
            y_samples = gp.sample_y(X, n_samples=num_per_level, random_state=0)
            scaled_y_samples = np.array([
                2 * (y_sample - np.min(y_sample)) / (np.max(y_sample) - np.min(y_sample)) - 1
                for y_sample in y_samples.T
            ])
            all_samples.append(scaled_y_samples)
        mixed_samples = np.vstack(all_samples)
        np.random.shuffle(mixed_samples)

        return mixed_samples

    coeffs_matrix = generate_random_smooth_patterns_from_mesh(fe_mesh.GetNodesCoordinates())
    np.save(os.path.join(case_dir,f"particle_pf_2d_gaussian_N{fe_mesh.GetNodesCoordinates().shape[0]}_num10000.npy"), coeffs_matrix)
else:
    pass
    # coeffsmatrix_ = np.load("training_data/particle_pf_2d_gaussian_N2323_num10000.npy")
    # np.save(os.path.join(case_dir,f"particle_pf_2d_gaussian_N{fe_mesh.GetNodesCoordinates().shape[0]}_num10000.npy"), coeffs_matrix)

T_matrix = no_control.ComputeBatchControlledVariables(coeffs_matrix)

characteristic_length = 256
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=2,
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
num_epochs = 1
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=no_control,
                                            loss_function=phasefield_loss_2d,
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3)
fol.Initialize()


train_start_id = 0
train_end_id = 8000

fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
          batch_size=100,
          convergence_settings={"num_epochs":num_epochs,
                                "relative_error":1e-100,
                                "absolute_error":1e-100},
          working_directory=case_dir)

num_steps = 10
FOL_T = np.zeros((fe_mesh.GetNumberOfNodes(),num_steps))
eval_id = 8500
FOL_T_temp  = coeffs_matrix[eval_id,:]
for i in range(num_steps):
    FOL_T_temp = np.array(fol.Predict(FOL_T_temp.reshape(-1,1).T)).reshape(-1)
    FOL_T[:,i] = FOL_T_temp 
fe_mesh['T_FOL'] = FOL_T
# solve FE here
fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                            "maxiter":1000,"pre-conditioner":"ilu","Dirichlet_BCs":Dirichlet_BCs},
                "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                            "maxiter":10,"load_incr":5}}
nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolverPhasefield("nonlinear_fe_solver",phasefield_loss_2d,fe_setting)
nonlinear_fe_solver.Initialize()
FE_T = np.zeros((fe_mesh.GetNumberOfNodes(),num_steps))
FE_T_temp = coeffs_matrix[eval_id,:]
for i in range(num_steps):
    FE_T_temp = np.array(nonlinear_fe_solver.Solve(FE_T_temp,FE_T_temp))  #np.zeros(fe_mesh.GetNumberOfNodes())
    FE_T[:,i] = FE_T_temp    
fe_mesh['T_FE'] = FE_T
absolute_error = np.abs(FOL_T- FE_T)
fe_mesh['abs_error'] = absolute_error
time_list = [int(num_steps/5) - 1,int(num_steps/2) - 1,num_steps - 1]
time_list2 = [0,1,int(num_steps/10) - 1,int(num_steps/5) - 1]

plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [coeffs_matrix[eval_id],FOL_T[:,time_list[0]],FOL_T[:,time_list[1]],FOL_T[:,-1]],
                  filename=os.path.join(case_dir,"FOL_predictions_all.png"),value_range=(-1,1))
plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [coeffs_matrix[eval_id],FE_T[:,time_list[0]],FE_T[:,time_list[1]],FE_T[:,-1]],
                  filename=os.path.join(case_dir,"FE_solutions_all.png"),value_range=(-1,1))
plot_triangulated_error(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [coeffs_matrix[eval_id],absolute_error[:,time_list[0]],absolute_error[:,time_list[1]],absolute_error[:,-1]],
                  filename=os.path.join(case_dir,"FOL-FE_errors_all.png"))
plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [FOL_T[:,time_list2[0]],FOL_T[:,time_list2[1]],FOL_T[:,time_list2[2]],FOL_T[:,time_list2[3]]],
                  filename=os.path.join(case_dir,"FOL_predictions_initialsteps.png"),value_range=(-1,1))
plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [FE_T[:,time_list2[0]],FE_T[:,time_list2[1]],FE_T[:,time_list2[2]],FE_T[:,time_list2[3]]],
                  filename=os.path.join(case_dir,"FE_solutions_initialsteps.png"),value_range=(-1,1))
plot_triangulated_error(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [absolute_error[:,time_list2[0]],absolute_error[:,time_list2[1]],absolute_error[:,time_list2[2]],absolute_error[:,time_list2[3]]],
                  filename=os.path.join(case_dir,"FOL-FE_errors_initialsteps.png"))
plot_mesh_tri(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                            filename=os.path.join(case_dir,'FE_mesh_particle.png'))
fe_mesh.Finalize(export_dir=case_dir)
