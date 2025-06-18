# ──────────────────────────────────────────────────────────────────────────────
#  Imports
# ──────────────────────────────────────────────────────────────────────────────
import sys, os, pickle, json, pathlib
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")

from fol.tools.usefull_functions import (
    create_random_fourier_samples,
    create_random_voronoi_samples,
)
from fol.controls.fourier_control import FourierControl
from fol.controls.voronoi_control3D import VoronoiControl3D
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.nns import MLP, HyperNetwork
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import (
    MetaAlphaMetaImplicitParametricOperatorLearning,
)
from fol.loss_functions.mechanical import MechanicalLoss3DTetra
from fol.mesh_input_output.mesh import Mesh
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.logging_functions import Logger
from fol.tools.decoration_functions import *
import optax

# ──────────────────────────────────────────────────────────────────────────────
#  User options
# ──────────────────────────────────────────────────────────────────────────────
control_type = "fourier"   # "voronoi" or "fourier"
phase_mode = "multi"       # "dual" or "multi"
num_phases = 2             # Only used if phase_mode == "multi"
num_samples = 1
create_new_samples = True

# ──────────────────────────────────────────────────────────────────────────────
#  Run directory & logging
# ──────────────────────────────────────────────────────────────────────────────
working_directory_name = "mechanical_low_fourier"
case_dir = os.path.join(".", working_directory_name)
os.makedirs(case_dir, exist_ok=True)
os.makedirs(f"{case_dir}/meshes", exist_ok=True)
os.makedirs(f"{case_dir}/trained_samples", exist_ok=True)
os.makedirs(f"{case_dir}/tested_samples", exist_ok=True)
sys.stdout = Logger(os.path.join(case_dir, f"{working_directory_name}.log"))

# ──────────────────────────────────────────────────────────────────────────────
#  Mesh + loss
# ──────────────────────────────────────────────────────────────────────────────

fe_mesh = Mesh(
    "box_3D",               # the mesh name/key
    "box_3D.med",           # the filename
    "meshes"                # relative path to the folder
)
fe_mesh.Initialize()

bc_dict = {
    "Ux": {"left": 0.0},
    "Uy": {"left": 0.0, "right": -0.05},
    "Uz": {"left": 0.0, "right": -0.05},
}

material_dict = {"young_modulus": 1.0, "poisson_ratio": 0.30}
mechanical_loss_3d = MechanicalLoss3DTetra(
    "mech_loss",
    loss_settings={
        "dirichlet_bc_dict": bc_dict,
        "material_dict": material_dict,
        "body_force": jnp.zeros((3, 1)),
    },
    fe_mesh=fe_mesh,
)
mechanical_loss_3d.Initialize()

# ──────────────────────────────────────────────────────────────────────────────
#  Phase values
# ──────────────────────────────────────────────────────────────────────────────
if phase_mode == "dual":
    E_values = [0.1, 1.0]
else:
    E_values = list(np.round(np.linspace(0.1, 1.0, num_phases), 6))

# ──────────────────────────────────────────────────────────────────────────────
#  Generator setup
# ──────────────────────────────────────────────────────────────────────────────
if control_type == "voronoi":
    a_seeds = 10
    generators = {
        "A": VoronoiControl3D("voronoi_A", {"number_of_seeds": a_seeds, "E_values": E_values}, fe_mesh),
    }
    for ctrl in generators.values():
        ctrl.Initialize()

elif control_type == "fourier":
    fourier_settings = {
        "x_freqs": np.array([2, 4, 6]),
        "y_freqs": np.array([2, 4, 6]),
        "z_freqs": np.array([2, 4, 6]),
        "beta": 5,
        "min": 0.1,
        "max": 1.0,
    }
    generators = {
        "A": FourierControl("fourier_A", fourier_settings, fe_mesh),
    }
    for ctrl in generators.values():
        ctrl.Initialize()
else:
    raise ValueError("Invalid control_type. Must be 'voronoi' or 'fourier'.")

# ──────────────────────────────────────────────────────────────────────────────
#  Generate / load samples (SINGLE sample only)
# ──────────────────────────────────────────────────────────────────────────────
if create_new_samples:
    ctrl = generators["A"]

    if control_type == "voronoi":
        _, K_matrix = create_random_voronoi_samples(ctrl, num_samples, 3)
    elif control_type == "fourier":
        _, K_matrix = create_random_fourier_samples(ctrl, num_samples)

    key = jax.random.PRNGKey(42)
    K_matrix = K_matrix[jax.random.permutation(key, K_matrix.shape[0])]
    K_matrix = np.clip(K_matrix, 1e-2, 1.0)

    with open(f"{case_dir}/K_matrix.pkl", "wb") as f:
        pickle.dump({"K_matrix": K_matrix}, f)
else:
    with open(f"{case_dir}/K_matrix.pkl", "rb") as f:
        K_matrix = np.clip(pickle.load(f)["K_matrix"][:num_samples], 1e-2, 1.0)

print("K_matrix shape:", K_matrix.shape)

# ──────────────────────────────────────────────────────────────────────────────
#  Identity control (used for training)
# ──────────────────────────────────────────────────────────────────────────────
identity_control = IdentityControl("identity_control", num_vars=fe_mesh.GetNumberOfNodes())
identity_control.Initialize()

# ──────────────────────────────────────────────────────────────────────────────
#  Network architecture
# ──────────────────────────────────────────────────────────────────────────────
char_len = 64
latent_fac = 8
latent_size = char_len * latent_fac  # 512

synthesizer = MLP(
    "synth",
    input_size=3,
    output_size=3,
    hidden_layers=[char_len] * 2,
    activation_settings={"type": "sin", "prediction_gain": 30, "initialization_gain": 1.0},
)

modulator = MLP("mod", input_size=latent_size, use_bias=False)

hyper_net = HyperNetwork(
    "hyper",
    modulator,
    synthesizer,
    coupling_settings={"modulator_to_synthesizer_coupling_mode": "one_modulator_per_synthesizer_layer"},
)

# ──────────────────────────────────────────────────────────────────────────────
#  Meta-learner object
# ──────────────────────────────────────────────────────────────────────────────
num_epochs = 150
main_opt = optax.adam(1e-5)
latent_opt = optax.adam(1e-4)

fol = MetaAlphaMetaImplicitParametricOperatorLearning(
    name="meta_implicit_ol",
    control=identity_control,
    loss_function=mechanical_loss_3d,
    flax_neural_network=hyper_net,
    main_loop_optax_optimizer=main_opt,
    latent_step_optax_optimizer=latent_opt,
    latent_step_size=1e-3,
)
fol.Initialize()

# ──────────────────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────────────────
train_ids = slice(0, 1)
test_ids = slice(0, 1)

fol.Train(
    train_set=(K_matrix[train_ids],),
    test_set=(K_matrix[test_ids],),
    test_frequency=10,
    batch_size=1,
    convergence_settings={"num_epochs": num_epochs, "relative_error": 1e-100, "absolute_error": 1e-100},
    plot_settings={
        "save": True,
        "save_frequency": 1,
        "plot_list": ["total_loss"],
        "save_directory": case_dir,
    },
    train_checkpoint_settings={
        "least_loss_checkpointing": True,
        "frequency": 10,
        "state_directory": f"{case_dir}/flax_train",
    },
    test_checkpoint_settings={
        "least_loss_checkpointing": True,
        "frequency": 10,
        "state_directory": f"{case_dir}/flax_test",
    },
    restore_nnx_state_settings={"restore": False, "state_directory": f"{case_dir}/flax_final"},
    working_directory=case_dir,
)

fol.RestoreState(f"{case_dir}/flax_train")

# ──────────────────────────────────────────────────────────────────────────────
#  Save K field and prediction
# ──────────────────────────────────────────────────────────────────────────────
for i in range(*train_ids.indices(num_samples)):
    fe_mesh[f"K_{i}"] = K_matrix[i]
fe_mesh.Finalize(export_dir=f"{case_dir}/trained_samples")

for i in range(*test_ids.indices(num_samples)):
    fe_mesh[f"K_{i}"] = K_matrix[i]

    FOL_UVW = np.array(fol.Predict(K_matrix[i : i + 1])).reshape(-1)
    fe_mesh[f"U_FOL_{i}"] = FOL_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    fe_setting = {
        "linear_solver_settings": {
            "solver": "JAX-bicgstab",
            "tol": 1e-6,
            "atol": 1e-6,
            "maxiter": 1000,
            "pre-conditioner": "ilu",
        },
        "nonlinear_solver_settings": {"rel_tol": 1e-5, "abs_tol": 1e-5, "maxiter": 10, "load_incr": 5},
    }
    linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver", mechanical_loss_3d, fe_setting)
    linear_fe_solver.Initialize()

    FE_UVW = np.array(linear_fe_solver.Solve(K_matrix[i], np.zeros(3 * fe_mesh.GetNumberOfNodes())))
    fe_mesh[f"U_FE_{i}"] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    abs_error = np.abs(FOL_UVW - FE_UVW)
    fe_mesh[f"abs_error_{i}"] = abs_error.reshape((fe_mesh.GetNumberOfNodes(), 3))

    _, r_FEM = mechanical_loss_3d.ComputeJacobianMatrixAndResidualVector(K_matrix[i], FE_UVW)
    _, r_FOL = mechanical_loss_3d.ComputeJacobianMatrixAndResidualVector(K_matrix[i], FOL_UVW)
    fe_mesh[f"res_FE_{i}"] = r_FEM.reshape((fe_mesh.GetNumberOfNodes(), 3))
    fe_mesh[f"res_FOL_{i}"] = r_FOL.reshape((fe_mesh.GetNumberOfNodes(), 3))

    FOL_energy = mechanical_loss_3d.ComputeSingleLoss(K_matrix[i], FOL_UVW.flatten()[mechanical_loss_3d.non_dirichlet_indices])[0]
    FE_energy = mechanical_loss_3d.ComputeSingleLoss(K_matrix[i], FE_UVW.flatten()[mechanical_loss_3d.non_dirichlet_indices])[0]

    fol_info(
        f"Test sample {i}: FOL_loss = {np.sqrt(max(FOL_energy, 0.0)):.6e}, "
        f"FE_loss = {np.sqrt(max(FE_energy, 0.0)):.6e}"
    )

fe_mesh.Finalize(export_dir=f"{case_dir}/tested_samples")
# ──────────────────────────────────────────────────────────────────────────────
#  Plotting the best test sample
# ──────────────────────────────────────────────────────────────────────────────
from fol.inference.plotter import Plotter3D

# Path to the VTK you just exported
vtk_path = os.path.join(case_dir, "tested_samples", "box_3D.vtk")

# (Optional) place plots in a dedicated subfolder
plot_dir = os.path.join(case_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Instantiate and render
plotter = Plotter3D(vtk_path)
plotter.render_all_panels()

print(f"3D panels and combined figure saved under: {plot_dir}")
