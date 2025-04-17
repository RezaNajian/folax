import sys
import os
import optax
import numpy as np
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.loss_functions.phase_field import AllenCahnLoss2DTri
from fol.controls.identity_control import IdentityControl
from scipy.spatial.distance import cdist
import pickle

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

# directory & save handling
working_directory_name = 'transient_allen_cahn_2D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))


fe_mesh = Mesh("fol_io","Li_battery_particle_bumps.med",'../../meshes/')

bc_dict = {"Phi":{}}
material_dict = {"rho":1.0,"cp":1.0,"dt":0.001,"epsilon":0.2}
phasefield_loss_2d = AllenCahnLoss2DTri("phasefield_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)
fe_mesh.Initialize()
phasefield_loss_2d.Initialize()

# create identity control
identity_control = IdentityControl("ident_control",num_vars=phasefield_loss_2d.GetNumberOfUnknowns())
identity_control.Initialize()

# generate some randome spatial fields
generate_new_samples=False
if generate_new_samples:
    sample_matrix = generate_fixed_gaussian_basis_field(fe_mesh.GetNodesCoordinates(), num_samples=200, length_scale=0.15)
    with open(f'sample_matrix.pkl', 'wb') as f:
        pickle.dump({"sample_matrix":sample_matrix},f)
else:
    with open(f'sample_matrix.pkl', 'rb') as f:
        sample_matrix = pickle.load(f)["sample_matrix"]

characteristic_length = 64
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
num_epochs = 10
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-6, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(1e-5))

# create ifol
ifol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=identity_control,
                                            loss_function=phasefield_loss_2d,
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3)
ifol.Initialize()


train_start_id = 0
train_end_id = 10
ifol.Train(train_set=(sample_matrix[train_start_id:train_end_id,:],),
          batch_size=1,
          convergence_settings={"num_epochs":num_epochs,
                                "relative_error":1e-100,
                                "absolute_error":1e-100},
          working_directory=case_dir)


# load the best model
ifol.RestoreState(restore_state_directory=case_dir+"/flax_final_state")

# FE-based time loop
fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                            "maxiter":10,"load_incr":1}}
nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",phasefield_loss_2d,fe_setting)
nonlin_fe_solver.Initialize()
eval_id = 0 
phi_fe = sample_matrix[eval_id].flatten()
fe_mesh[f'phi_fe_{0}'] = phi_fe
phi_ifol = sample_matrix[eval_id].flatten()
fe_mesh[f'phi_ifol_{0}'] = phi_ifol
for i in range(1,10):
    phi_fe = np.array(nonlin_fe_solver.Solve(phi_fe,np.zeros(fe_mesh.GetNumberOfNodes())))
    fe_mesh[f'phi_fe_{i}'] = phi_fe
    phi_ifol = np.array(ifol.Predict(phi_ifol.reshape(-1,1).T)).reshape(-1)
    fe_mesh[f'phi_ifol_{i}'] = phi_ifol

fe_mesh.Finalize(export_dir=case_dir)
