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
from scipy.spatial.distance import cdist
from jax import config
config.update("jax_default_matmul_precision", "float32")
# directory & save handling
working_directory_name = 'meta_learning_phasefield_particle_bumps_2'
case_dir = os.path.join('.', working_directory_name)
# create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# creation of the model
mesh_res_rate = 1
name = "original"
fe_mesh = Mesh("fol_io","Li_battery_particle_bumps.med")
# create fe-based loss function
bc_dict = {"T":{}}#"left":1.0,"right":-1.0
Dirichlet_BCs = False
material_dict = {"rho":1.0,"cp":1.0,"dt":0.001,"epsilon":0.2}
phasefield_loss_2d = PhaseFieldLoss2DTri("phasefield_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)

no_control = NoControl("No_Control",fe_mesh)

fe_mesh.Initialize()
phasefield_loss_2d.Initialize()
no_control.Initialize()

def generate_fixed_gaussian_basis_field(coords, num_samples=1, num_basis=25, length_scale=0.1, random_seed=1):
    """
    Generate multiple random smooth patterns using fixed Gaussian/RBF basis functions.

    Parameters:
        coords (np.ndarray): Array of shape (N, 2) representing mesh node coordinates.
        num_samples (int): Number of random field samples to generate.
        num_basis (int): Number of fixed RBF basis functions.
        length_scale (float): Controls the smoothness of the function.
        random_seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Generated fields of shape (num_samples, N)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Normalize mesh coordinates to [0,1]
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    norm_coords = (coords - min_coords) / (max_coords - min_coords)

    # Define fixed basis function centers
    basis_centers = np.random.rand(num_basis, 2)
    # Compute distances
    distances = cdist(norm_coords, basis_centers)
    basis_values = np.exp(- (distances ** 2) / (2 * length_scale ** 2))

    # Generate multiple sets of coefficients
    coefficients = np.random.randn(num_samples, num_basis)
    # Compute all fields
    fields = coefficients @ basis_values.T  # shape: (num_samples, N)

    # Normalize each field to [-1, 1]
    min_vals = fields.min(axis=1, keepdims=True)
    max_vals = fields.max(axis=1, keepdims=True)
    fields = 2 * (fields - min_vals) / (max_vals - min_vals) - 1

    return fields

coeffs_matrix = generate_fixed_gaussian_basis_field(fe_mesh.GetNodesCoordinates(), num_samples=3200, length_scale=0.15)

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
num_epochs = 10000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-6, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=no_control,
                                            loss_function=phasefield_loss_2d,
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3)
fol.Initialize()


# train_start_id = 0
# train_end_id = 8000
# coeffs_matrix = np.load("particle_pf_2d_gaussian_N2624_num10000.npy")
# fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
#           batch_size=100,
#           convergence_settings={"num_epochs":num_epochs,
#                                 "relative_error":1e-100,
#                                 "absolute_error":1e-100},
#           working_directory=case_dir)

fol.RestoreState(os.path.join(".","flax_final_state"))

num_steps = 50
eval_id = 0
FOL_T = np.array(fol.Predict_all(coeffs_matrix[eval_id].reshape(1,-1),num_steps))
FOL_T = np.squeeze(FOL_T,axis=1).T
fe_mesh['T_FOL'] = FOL_T

# solve FE here
FE_T = np.zeros((fe_mesh.GetNumberOfNodes(),num_steps))
FE_T_temp = coeffs_matrix[eval_id,:]
fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-8,"atol":1e-8,
                                            "maxiter":1000,"pre-conditioner":"ilu","Dirichlet_BCs":Dirichlet_BCs},
                "nonlinear_solver_settings":{"rel_tol":1e-6,"abs_tol":1e-6,
                                            "maxiter":10,"load_incr":1}}
nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolverPhasefield("nonlinear_fe_solver",phasefield_loss_2d,fe_setting)
nonlinear_fe_solver.Initialize()
for i in range(num_steps):
    FE_T_temp = np.array(nonlinear_fe_solver.Solve(FE_T_temp,FE_T_temp))  #np.zeros(fe_mesh.GetNumberOfNodes())
    FE_T[:,i] = FE_T_temp 
fe_mesh['T_FE'] = FE_T

absolute_error = np.abs(FOL_T-FE_T)

time_list = [0,1,4,9]

np.save(os.path.join(case_dir,f"test_{name}_FOL_T.npy"),FOL_T)
np.save(os.path.join(case_dir,f"test_{name}_FE_T.npy"),FE_T)

plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [coeffs_matrix[eval_id]],
                  filename=os.path.join(case_dir,f"test3_{name}_init_phi.png"),value_range=(-1,1),row=True)

plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [FOL_T[:,time_list[0]],FOL_T[:,time_list[1]],
                   FOL_T[:,time_list[2]],FOL_T[:,time_list[3]]],
                  filename=os.path.join(case_dir,f"test3_{name}_FOL_summary.png"),value_range=(-1,1),row=True)
plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [FE_T[:,time_list[0]],FE_T[:,time_list[1]],
                   FE_T[:,time_list[2]],FE_T[:,time_list[3]]],
                  filename=os.path.join(case_dir,f"test3_{name}_FE_summary.png"),value_range=(-1,1),row=True)
plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [absolute_error[:,time_list[0]],absolute_error[:,time_list[1]],
                   absolute_error[:,time_list[2]],absolute_error[:,time_list[3]]],
                  filename=os.path.join(case_dir,f"test3_{name}_Error_summary.png"),value_range=None,row=True)
plot_mesh_tri(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                            filename=os.path.join(case_dir,f'test_{name}_FE_mesh_particle.png'))

fe_mesh.Finalize(export_dir=case_dir)
