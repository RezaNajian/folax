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
from fol.solvers.fe_linear_residual_based_solver import (
    FiniteElementLinearResidualBasedSolver,
)
from fol.tools.logging_functions import Logger
from fol.tools.decoration_functions import *
import optax

# ──────────────────────────────────────────────────────────────────────────────
#  Run directory & logging
# ──────────────────────────────────────────────────────────────────────────────
working_directory_name = "check"
case_dir = os.path.join('.', working_directory_name)
os.makedirs(case_dir, exist_ok=True)
os.makedirs(f"{case_dir}/meshes", exist_ok=True)
os.makedirs(f"{case_dir}/trained_samples", exist_ok=True)
os.makedirs(f"{case_dir}/tested_samples", exist_ok=True)

sys.stdout = Logger(os.path.join(case_dir, f"{working_directory_name}.log"))

# ──────────────────────────────────────────────────────────────────────────────
#  Mesh + loss
# ──────────────────────────────────────────────────────────────────────────────
fe_mesh = Mesh("box_3D", "Box_3D_Tetra_20.med", "meshes")
fe_mesh.Initialize()

bc_dict = {
    "Ux": {"left": 0.0,},
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
#  USER OPTIONS – mixed generators (4 Voronoi + 5 Fourier)
# ──────────────────────────────────────────────────────────────────────────────
control_type = "mixed"  # mix of voronoi + fourier

phase_mode = "multi"
num_phases = 100

if phase_mode == "dual":
    E_values = [0.01, 1.0]
else:
    E_values = list(np.round(np.linspace(0.01, 1.0, num_phases), 6))

# Voronoi seed counts
voronoi_seed_counts = {
    "V_A": 10,
    "V_B": 20,
    "V_C": 30,
    "V_D": 40,
}

# Fourier settings (EXPANDED)
fourier_settings = {
    "F_A": {
        "x_freqs": np.array([2, 4, 6]),
        "y_freqs": np.array([2, 4, 6]),
        "z_freqs": np.array([2, 4, 6]),
        "beta": 20,
        "min": 0.01,
        "max": 1.0,
    },
    "F_B": {
        "x_freqs": np.array([4, 8, 12]),
        "y_freqs": np.array([4, 8, 12]),
        "z_freqs": np.array([4, 8, 12]),
        "beta": 20,
        "min": 0.01,
        "max": 1.0,
    },
    "F_C": {
        "x_freqs": np.array([1, 3, 5]),
        "y_freqs": np.array([1, 3, 5]),
        "z_freqs": np.array([1, 3, 5]),
        "beta": 15,
        "min": 0.01,
        "max": 1.0,
    },
    "F_D": {
        "x_freqs": np.array([2, 4, 6]),
        "y_freqs": np.array([2, 4, 6]),
        "z_freqs": np.array([2, 4, 6]),
        "beta": 20,
        "min": 0.01,
        "max": 1.0,
    },
    "F_E": {
        "x_freqs": np.array([0, 2, 3]),
        "y_freqs": np.array([0, 2, 3]),
        "z_freqs": np.array([0, 2, 3]),
        "beta": 10,
        "min": 0.01,
        "max": 1.0,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
#  Instantiate controls (EXPANDED)
# ──────────────────────────────────────────────────────────────────────────────
if control_type == "mixed":
    generators = {}
    for tag, nseed in voronoi_seed_counts.items():
        generators[tag] = VoronoiControl3D(tag, {"number_of_seeds": nseed, "E_values": E_values}, fe_mesh)
    for tag, sett in fourier_settings.items():
        generators[tag] = FourierControl(tag, sett, fe_mesh)
    for ctrl in generators.values():
        ctrl.Initialize()
else:
    raise ValueError("control_type must be 'mixed' for this sampling scheme")

_is_fourier = {tag: tag.startswith("F_") for tag in generators}

# ──────────────────────────────────────────────────────────────────────────────
#  Generate / load samples (UPDATED proportions)
# ──────────────────────────────────────────────────────────────────────────────
num_samples = 1
create_new_samples = True

if create_new_samples:
    prop = {
        "V_A": 0.12,
        "V_B": 0.12,
        "V_C": 0.12,
        "V_D": 0.12,
        "F_A": 0.10,
        "F_B": 0.10,
        "F_C": 0.10,
        "F_D": 0.10,
        "F_E": 0.12,
    }
    counts = {k: int(v * num_samples) for k, v in prop.items()}
    remainder = num_samples - sum(counts.values())
    counts["V_A"] += remainder

    batches = []
    for tag, ctrl in generators.items():
        n = counts[tag]
        if n == 0:
            continue
        if _is_fourier[tag]:
            _, K = create_random_fourier_samples(ctrl, n)
        else:
            _, K = create_random_voronoi_samples(ctrl, n, dim=3)
        batches.append(K)

    K_matrix = np.vstack(batches)
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
#  Identity control
# ──────────────────────────────────────────────────────────────────────────────
identity_control = IdentityControl("identity_control", num_vars=fe_mesh.GetNumberOfNodes())
identity_control.Initialize()

# ──────────────────────────────────────────────────────────────────────────────
#  Network architecture (UPDATED)
# ──────────────────────────────────────────────────────────────────────────────
char_len = 64            # ↓ 64 neurons / layer
latent_fac = 12           # ↓ 8 latent blocks
latent_size = char_len * latent_fac  # 512

synthesizer = MLP(
    "synth",
    input_size=3,
    output_size=3,
    hidden_layers=[char_len] * 8,  # 8‑layer MLP
    activation_settings={
        "type": "sin",
        "prediction_gain": 30,
        "initialization_gain": 1.0,
    },
)

modulator = MLP("mod", input_size=latent_size, use_bias=False)

hyper_net = HyperNetwork(
    "hyper",
    modulator,
    synthesizer,
    coupling_settings={
        "modulator_to_synthesizer_coupling_mode": "one_modulator_per_synthesizer_layer",
    },
)

# ──────────────────────────────────────────────────────────────────────────────
#  Meta‑learning object (UPDATED optimisers)
# ──────────────────────────────────────────────────────────────────────────────
num_epochs = 5000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_opt = optax.chain(
    optax.normalize_by_update_norm(),
    optax.adam(learning_rate_scheduler))      
latent_opt = optax.chain(
    optax.normalize_by_update_norm(),
    optax.adamw(1e-5))
    
    
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
test_ids = slice(0,1)

fol.Train(
    train_set=(K_matrix[train_ids],),
    test_set=(K_matrix[test_ids],),
    test_frequency=10,
    batch_size=20,
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
#  Post‑processing: save training K‑fields
# ──────────────────────────────────────────────────────────────────────────────
for i in range(*train_ids.indices(num_samples)):
    fe_mesh[f"K_{i}"] = K_matrix[i]
fe_mesh.Finalize(export_dir=f"{case_dir}/trained_samples")

# ──────────────────────────────────────────────────────────────────────────────
#  Evaluate on test samples and export results
# ──────────────────────────────────────────────────────────────────────────────
for i in range(*test_ids.indices(num_samples)):
    fe_mesh[f"K_{i}"] = K_matrix[i]

    # FOL prediction
    FOL_UVW = np.array(fol.Predict(K_matrix[i : i + 1])).reshape(-1)
    fe_mesh[f"U_FOL_{i}"] = FOL_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    # FEM reference
    fe_setting = {
        "linear_solver_settings": {
            "solver": "JAX-bicgstab",
            "tol": 1e-6,
            "atol": 1e-6,
            "maxiter": 1000,
            "pre-conditioner": "ilu",
        },
        "nonlinear_solver_settings": {"rel_tol": 1e-5, "abs_tol": 1e-5, "maxiter": 1000, "load_incr": 5},
    }
    linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver", mechanical_loss_3d, fe_setting)
    linear_fe_solver.Initialize()

    FE_UVW = np.array(
        linear_fe_solver.Solve(K_matrix[i], np.zeros(3 * fe_mesh.GetNumberOfNodes()))
    )
    fe_mesh[f"U_FE_{i}"] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    # Absolute error
    abs_error = np.abs(FOL_UVW - FE_UVW)
    fe_mesh[f"abs_error_{i}"] = abs_error.reshape((fe_mesh.GetNumberOfNodes(), 3))

    # Residuals
    _, r_FEM = mechanical_loss_3d.ComputeJacobianMatrixAndResidualVector(K_matrix[i], FE_UVW)
    _, r_FOL = mechanical_loss_3d.ComputeJacobianMatrixAndResidualVector(K_matrix[i], FOL_UVW)
    fe_mesh[f"res_FE_{i}"] = r_FEM.reshape((fe_mesh.GetNumberOfNodes(), 3))
    fe_mesh[f"res_FOL_{i}"] = r_FOL.reshape((fe_mesh.GetNumberOfNodes(), 3))

    # Log energy-based losses for info
    FOL_energy = mechanical_loss_3d.ComputeSingleLoss(
        K_matrix[i], FOL_UVW.flatten()[mechanical_loss_3d.non_dirichlet_indices]
    )[0]
    FE_energy = mechanical_loss_3d.ComputeSingleLoss(
        K_matrix[i], FE_UVW.flatten()[mechanical_loss_3d.non_dirichlet_indices]
    )[0]

    fol_info(
        f"Test sample {i}: FOL_loss = {np.sqrt(max(FOL_energy, 0.0)):.6e}, "
        f"FE_loss = {np.sqrt(max(FE_energy, 0.0)):.6e}"
    )

# save all accumulated test fields
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

