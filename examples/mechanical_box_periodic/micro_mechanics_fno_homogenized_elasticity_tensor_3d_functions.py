"""
Functions for the 3-D periodic micro-mechanics FNO experiment with homogenized elasticity tensor.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path


def configure_jax_platform_from_argv(argv):
    platform = "gpu"
    for index, argument in enumerate(argv):
        if argument == "--jax-platform" and index + 1 < len(argv):
            platform = argv[index + 1]
            break
        if argument.startswith("--jax-platform="):
            platform = argument.split("=", 1)[1]
            break
    if platform == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = "all"
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            os.environ.pop("CUDA_VISIBLE_DEVICES")
    elif platform == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif platform != "auto":
        raise ValueError("--jax-platform must be one of: gpu, cpu, auto")


configure_jax_platform_from_argv(sys.argv[1:])

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

if "JAX_PLATFORMS" in os.environ:
    jax.config.update("jax_platforms", os.environ["JAX_PLATFORMS"])
if "JAX_PLATFORM_NAME" in os.environ:
    jax.config.update("jax_platform_name", os.environ["JAX_PLATFORM_NAME"])
if "JAX_CUDA_VISIBLE_DEVICES" in os.environ:
    jax.config.update("jax_cuda_visible_devices", os.environ["JAX_CUDA_VISIBLE_DEVICES"])
jax.config.update("jax_default_matmul_precision", "highest")

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use low-, baseline-, and high-frequency material families.  Each tuple is
# applied to x, y, and z, keeping the coefficient-vector size identical across
# families while broadening the microstructures seen during training.
FOURIER_FREQUENCY_SETS = (
    (1, 2, 3),
    (2, 4, 6),
    (4, 6, 8),
)

from fol.controls.fourier_control import FourierControl
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.ported_fourier_neural_operator_networks.fno import FNO
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearningMulti
from micro_mechanics_fe_homogenized_elasticity_tensor_3d import (
    VOIGT_LABELS,
    build_control_input,
    create_mechanical_loss,
    create_volume_average_stress_function,
    macro_displacement,
    voigt_strain_to_macro_deformation_grad,
)
from periodic_voronoi_control3D import VoronoiControl3D
from usefull_functions import (
    create_3D_box_mesh_structured,
    plot_displacement_components_pyvista,
    plot_displacement_contours_pyvista,
)


def parse_args(
    default_material_distribution="fourier",
    default_case_dir=None,
    default_model_dir=None,
    default_stage="all",
    default_allow_distribution_shift=False,
    default_n=22,
    default_num_train=3000,
    default_num_test=20,
    stage_choices=("all", "data", "train", "compare", "evaluate"),
    description=None,
):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument(
        "--stage",
        choices=stage_choices,
        default=default_stage,
        help="'evaluate' generates missing test data and compares without training.",
    )
    parser.add_argument(
        "--N", type=int, default=default_n, help="Nodes per coordinate direction."
    )
    parser.add_argument("--L", type=float, default=1.0, help="RVE side length.")
    parser.add_argument("--num-train", type=int, default=default_num_train)
    parser.add_argument("--num-test", type=int, default=default_num_test)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help="Prediction batch size; keep small for high-resolution 3-D inference.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strain-amplitude", type=float, default=0.1)
    parser.add_argument(
        "--material-distribution",
        choices=("fourier", "polycrystal"),
        default=default_material_distribution,
        help="Material-field family used for training and held-out samples.",
    )
    parser.add_argument(
        "--num-grains",
        type=int,
        default=16,
        help="Number of periodic 3-D Voronoi grains per polycrystal sample.",
    )
    parser.add_argument(
        "--grain-values",
        type=float,
        nargs=2,
        default=(0.1, 1.0),
        metavar=("MIN", "MAX"),
        help=(
            "Continuous uniform range for independently sampled Voronoi-grain "
            "material multipliers."
        ),
    )
    parser.add_argument(
        "--coeffs-file",
        type=Path,
        default=THIS_DIR / "3d_coeffs_matrix_0.npy",
        help=(
            "Fourier coefficient matrix. If it does not exist, a reproducible "
            "matrix is generated from --seed and saved at this path."
        ),
    )
    parser.add_argument(
        "--solver", choices=("JAX-direct", "JAX-bicgstab"), default="JAX-bicgstab"
    )
    parser.add_argument("--solver-tol", type=float, default=1e-6)
    parser.add_argument("--solver-maxiter", type=int, default=5000)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--jax-platform", choices=("gpu", "cpu", "auto"), default="gpu")
    if default_case_dir is None:
        default_case_dir = THIS_DIR / "periodic_bcs_FNO_homogenized_3D"
    parser.add_argument("--case-dir", type=Path, default=default_case_dir)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help="Checkpoint directory containing <load-case>.pkl files. Defaults to CASE_DIR/models.",
    )
    parser.add_argument(
        "--allow-distribution-shift",
        action="store_true",
        default=default_allow_distribution_shift,
        help="Allow checkpoints trained on another material-field family.",
    )
    parser.add_argument("--regenerate-data", action="store_true")
    parser.add_argument(
        "--repair-invalid-fem",
        action="store_true",
        help=(
            "Re-solve only saved FEM sample/load-case targets containing NaN or "
            "Inf values, leaving every valid target unchanged."
        ),
    )
    parser.add_argument(
        "--repair-solver",
        choices=("JAX-direct", "JAX-bicgstab"),
        default="JAX-direct",
        help="Linear solver used only for invalid FEM target repairs.",
    )
    parser.add_argument("--no-vtk", action="store_true")
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        default=True,
        help="Disable FE/FNO displacement contour images during comparison.",
    )
    parser.add_argument(
        "--visualization-sample-id",
        type=int,
        default=0,
        help="Held-out sample index to visualize (default: 0).",
    )
    parser.add_argument(
        "--visualization-load-case",
        choices=("all",) + VOIGT_LABELS,
        default="xx",
        help="Load case to visualize, or 'all' for all six cases (default: xx).",
    )
    parser.add_argument(
        "--warp-factor",
        type=float,
        default=1.0,
        help="Displacement scale used to warp the rendered mesh.",
    )
    parser.add_argument(
        "--camera-zoom",
        type=float,
        default=0.85,
        help="PyVista camera zoom used for displacement images.",
    )
    return parser.parse_args()


def validate_args(args):
    if (
        args.num_train < 0
        or args.num_test < 1
        or args.batch_size < 1
        or args.inference_batch_size < 1
    ):
        raise ValueError(
            "num-train must be nonnegative; num-test, batch-size, and "
            "inference-batch-size must be positive."
        )
    if args.stage in ("all", "train") and args.num_train < 1:
        raise ValueError("num-train must be positive when training models.")
    if args.benchmark_repeats < 1:
        raise ValueError("benchmark-repeats must be positive.")
    if args.num_grains < 1:
        raise ValueError("num-grains must be positive.")
    args.grain_values = tuple(args.grain_values)
    if not np.all(np.isfinite(args.grain_values)):
        raise ValueError("grain-values must be finite.")
    if args.grain_values[0] > args.grain_values[1]:
        raise ValueError("grain-values MIN must not exceed MAX.")


def create_problem(args):
    mesh = create_3D_box_mesh_structured(
        Nx=args.N, Ny=args.N, Nz=args.N, Lx=args.L, Ly=args.L, Lz=args.L
    )
    mesh.Initialize()
    loss = create_mechanical_loss(mesh, {"young_modulus": 1.0, "poisson_ratio": 0.3})
    settings = {
        "linear_solver_settings": {
            "solver": args.solver,
            "tol": args.solver_tol,
            "atol": args.solver_tol,
            "maxiter": args.solver_maxiter,
            "pre-conditioner": "ilu",
        },
        "nonlinear_solver_settings": {
            "rel_tol": 1e-5,
            "abs_tol": 1e-5,
            "maxiter": 10,
            "load_incr": 5,
        },
    }
    solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver", loss, settings)
    solver.Initialize()
    stress_function = create_volume_average_stress_function(mesh, loss)
    return mesh, loss, solver, stress_function


def load_material_fields(args, mesh):
    requested = args.num_train + args.num_test
    if args.material_distribution == "polycrystal":
        field_value_range = tuple(args.grain_values)
        control = VoronoiControl3D(
            "periodic_polycrystal",
            {
                "number_of_seeds": args.num_grains,
                "E_values": field_value_range,
                "Lx": args.L,
                "Ly": args.L,
                "Lz": args.L,
            },
            mesh,
        )
        control.Initialize()
        rng = np.random.default_rng(args.seed)
        x_coordinates = rng.uniform(0.0, args.L, size=(requested, args.num_grains))
        y_coordinates = rng.uniform(0.0, args.L, size=(requested, args.num_grains))
        z_coordinates = rng.uniform(0.0, args.L, size=(requested, args.num_grains))
        grain_values = rng.uniform(
            field_value_range[0],
            field_value_range[1],
            size=(requested, args.num_grains),
        )
        coefficients = np.concatenate(
            (x_coordinates, y_coordinates, z_coordinates, grain_values), axis=1
        ).astype(np.float32)
        fields = np.asarray(
            control.ComputeBatchControlledVariables(jnp.asarray(coefficients)),
            dtype=np.float32,
        )
        np.save(args.case_dir / "polycrystal_coefficients.npy", coefficients)
        return fields

    number_of_sets = len(FOURIER_FREQUENCY_SETS)

    def balanced_counts(total):
        quotient, remainder = divmod(total, number_of_sets)
        return [quotient + (set_id < remainder) for set_id in range(number_of_sets)]

    train_counts = balanced_counts(args.num_train)
    test_counts = balanced_counts(args.num_test)
    maximum_requested = max(
        train_count + test_count
        for train_count, test_count in zip(train_counts, test_counts)
    )

    frequency_count = len(FOURIER_FREQUENCY_SETS[0])
    if any(
        len(frequencies) != frequency_count
        for frequencies in FOURIER_FREQUENCY_SETS
    ):
        raise ValueError("All Fourier frequency sets must contain the same number of modes.")
    number_of_coefficients = frequency_count**3 + 1

    if args.coeffs_file.exists():
        coefficients = np.load(args.coeffs_file, allow_pickle=False)
    else:
        rng = np.random.default_rng(args.seed)
        coefficients = rng.standard_normal(
            size=(maximum_requested, number_of_coefficients)
        ).astype(np.float32)
        args.coeffs_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.coeffs_file, coefficients)
        print(
            f"Generated {len(coefficients)} Fourier coefficient samples "
            f"from seed {args.seed} and saved them to {args.coeffs_file}."
        )

    if coefficients.ndim != 2 or coefficients.shape[1] != number_of_coefficients:
        raise ValueError(
            f"{args.coeffs_file} must have shape (samples, {number_of_coefficients}), "
            f"but has shape {coefficients.shape}."
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{args.coeffs_file} contains NaN or Inf coefficients.")
    if maximum_requested > len(coefficients):
        raise ValueError(
            f"Each Fourier frequency set needs up to {maximum_requested} coefficient "
            f"samples, but {args.coeffs_file} has {len(coefficients)}."
        )

    training_fields = []
    test_fields = []
    for set_id, (frequencies, train_count, test_count) in enumerate(
        zip(FOURIER_FREQUENCY_SETS, train_counts, test_counts)
    ):
        frequency_array = np.asarray(frequencies)
        control = FourierControl(
            f"K_frequency_set_{set_id}",
            {
                "x_freqs": frequency_array,
                "y_freqs": frequency_array,
                "z_freqs": frequency_array,
                "beta": 10,
                "min": 0.1,
                "max": 1.0,
            },
            mesh,
        )
        control.Initialize()
        count = train_count + test_count
        order = np.random.default_rng(args.seed + set_id).permutation(len(coefficients))
        set_fields = np.asarray(
            control.ComputeBatchControlledVariables(coefficients[order[:count]]),
            dtype=np.float32,
        )
        training_fields.append(set_fields[:train_count])
        test_fields.append(set_fields[train_count:])

    # Keep the dataset contract used by training/evaluation: all training
    # samples first, followed by all held-out samples.  Shuffle within each
    # partition so batches do not contain long single-frequency runs.
    training_fields = np.concatenate(training_fields, axis=0)
    test_fields = np.concatenate(test_fields, axis=0)
    rng = np.random.default_rng(args.seed)
    training_fields = training_fields[rng.permutation(len(training_fields))]
    test_fields = test_fields[rng.permutation(len(test_fields))]
    print(
        "Fourier frequency sets (x=y=z): "
        f"{FOURIER_FREQUENCY_SETS}; training samples per set={train_counts}, "
        f"held-out samples per set={test_counts}."
    )
    return np.concatenate((training_fields, test_fields), axis=0)


def periodic_grid_metadata(loss, number_of_nodes):
    input_representatives = np.asarray(loss.input_representative_dofs, dtype=np.int32)
    output_representatives = np.asarray(loss.representative_dofs, dtype=np.int32)
    unique_nodes = len(input_representatives)
    grid_size = int(round(unique_nodes ** (1.0 / 3.0)))
    expected_output_size = 3 * unique_nodes
    if grid_size**3 != unique_nodes or len(output_representatives) != expected_output_size:
        raise RuntimeError(
            "The periodic representative DOFs do not form the structured (N-1)^3 grid required by FNO: "
            f"input={unique_nodes}, output={len(output_representatives)}."
        )
    if np.max(output_representatives) >= 3 * number_of_nodes:
        raise RuntimeError("Invalid representative DOF index in periodic projection.")
    return input_representatives, output_representatives, grid_size


def dataset_path(args):
    return args.case_dir / "physics_informed_dataset.npz"


def generate_fem_dataset(args, mesh, loss, solver):
    material_fields = load_material_fields(args, mesh)
    input_reps, output_reps, grid_size = periodic_grid_metadata(
        loss, mesh.GetNumberOfNodes()
    )
    targets = np.empty(
        (args.num_test, len(VOIGT_LABELS), len(output_reps)), dtype=np.float32
    )
    solve_seconds = np.empty((args.num_test, len(VOIGT_LABELS)), dtype=np.float64)
    zeros = np.zeros(3 * mesh.GetNumberOfNodes())

    test_material_fields = material_fields[args.num_train :]
    print(
        f"Training uses FE loss directly; generating FEM references only for "
        f"{args.num_test} held-out samples."
    )
    for sample_id, k_field in enumerate(test_material_fields):
        print(f"FEM reference: sample {sample_id + 1}/{args.num_test}")
        for case_id, label in enumerate(VOIGT_LABELS):
            strain = np.zeros(6)
            strain[case_id] = args.strain_amplitude
            deformation_gradient = voigt_strain_to_macro_deformation_grad(strain)
            control_input = build_control_input(k_field, deformation_gradient)
            start = time.perf_counter()
            full_fluctuation = np.asarray(solver.SolveReduced(control_input, zeros))
            solve_seconds[sample_id, case_id] = time.perf_counter() - start
            targets[sample_id, case_id] = full_fluctuation[output_reps]
            print(
                f"  {label}: {solve_seconds[sample_id, case_id]:.3f} s, "
                f"||u~||={np.linalg.norm(targets[sample_id, case_id]):.6e}"
            )

    np.savez_compressed(
        dataset_path(args),
        material_fields=material_fields,
        reduced_test_targets=targets,
        fem_solve_seconds=solve_seconds,
        input_representatives=input_reps,
        output_representatives=output_reps,
        grid_size=grid_size,
        N=args.N,
        L=args.L,
        strain_amplitude=args.strain_amplitude,
        num_train=args.num_train,
        num_test=args.num_test,
        seed=args.seed,
        material_distribution=args.material_distribution,
        fourier_frequency_sets=np.asarray(FOURIER_FREQUENCY_SETS, dtype=np.int32),
        fourier_train_counts=np.asarray(
            [
                args.num_train // len(FOURIER_FREQUENCY_SETS)
                + (set_id < args.num_train % len(FOURIER_FREQUENCY_SETS))
                for set_id in range(len(FOURIER_FREQUENCY_SETS))
            ],
            dtype=np.int32,
        ),
        fourier_test_counts=np.asarray(
            [
                args.num_test // len(FOURIER_FREQUENCY_SETS)
                + (set_id < args.num_test % len(FOURIER_FREQUENCY_SETS))
                for set_id in range(len(FOURIER_FREQUENCY_SETS))
            ],
            dtype=np.int32,
        ),
        polycrystal_dimension=3 if args.material_distribution == "polycrystal" else 0,
        grain_value_sampling=(
            "continuous_uniform"
            if args.material_distribution == "polycrystal"
            else "not_applicable"
        ),
        num_grains=args.num_grains,
        grain_values=np.asarray(args.grain_values, dtype=np.float32),
        voigt_labels=np.asarray(VOIGT_LABELS),
    )
    print(f"Saved material fields and held-out FEM references to {dataset_path(args)}")


def load_dataset(args):
    path = dataset_path(args)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run with --stage data or --stage all first.")
    data = np.load(path)
    checks = {
        "N": args.N,
        "num_train": args.num_train,
        "num_test": args.num_test,
        "seed": args.seed,
    }
    for key, expected in checks.items():
        actual = int(data[key])
        if actual != expected:
            raise ValueError(f"Dataset {key}={actual}, but command line requests {expected}.")
    saved_length = float(data["L"])
    if not np.isclose(saved_length, args.L):
        raise ValueError(f"Dataset L={saved_length}, but the command line requests {args.L}.")
    saved_strain_amplitude = float(data["strain_amplitude"])
    if not np.isclose(saved_strain_amplitude, args.strain_amplitude):
        raise ValueError(
            f"Dataset strain_amplitude={saved_strain_amplitude}, but the command line "
            f"requests {args.strain_amplitude}."
        )
    saved_distribution = (
        str(data["material_distribution"])
        if "material_distribution" in data.files
        else "fourier"
    )
    if saved_distribution != args.material_distribution:
        raise ValueError(
            f"Dataset material distribution is {saved_distribution!r}, but the command "
            f"line requests {args.material_distribution!r}."
        )
    if args.material_distribution == "fourier":
        if "fourier_frequency_sets" not in data.files:
            raise ValueError("Dataset predates multi-frequency Fourier training.")
        saved_frequency_sets = np.asarray(data["fourier_frequency_sets"])
        requested_frequency_sets = np.asarray(FOURIER_FREQUENCY_SETS)
        if not np.array_equal(saved_frequency_sets, requested_frequency_sets):
            raise ValueError(
                "Dataset Fourier frequency sets differ from the configured sets."
            )
    if args.material_distribution == "polycrystal" and "num_grains" in data.files:
        saved_dimension = (
            int(data["polycrystal_dimension"])
            if "polycrystal_dimension" in data.files
            else 2
        )
        if saved_dimension != 3:
            raise ValueError(
                f"Dataset polycrystal_dimension={saved_dimension}, but 3-D is required."
            )
        saved_sampling = (
            str(data["grain_value_sampling"])
            if "grain_value_sampling" in data.files
            else "discrete"
        )
        if saved_sampling != "continuous_uniform":
            raise ValueError(
                f"Dataset grain_value_sampling={saved_sampling!r}, but continuous_uniform "
                "is required."
            )
        if int(data["num_grains"]) != args.num_grains:
            raise ValueError(
                f"Dataset num_grains={int(data['num_grains'])}, but the command line "
                f"requests {args.num_grains}."
            )
        saved_grain_values = np.asarray(data["grain_values"])
        requested_grain_values = np.asarray(args.grain_values)
        if saved_grain_values.shape != requested_grain_values.shape or not np.allclose(
            saved_grain_values, requested_grain_values
        ):
            raise ValueError("Dataset grain values differ from the command line.")
    return data


def ensure_fem_dataset(args, mesh, loss, solver):
    """Generate missing or incompatible FEM reference data, otherwise reuse it."""
    path = dataset_path(args)
    if args.regenerate_data or not path.exists():
        generate_fem_dataset(args, mesh, loss, solver)
        return load_dataset(args)

    try:
        data = load_dataset(args)
    except ValueError as error:
        print(f"Existing dataset is incompatible: {error}")
        print(f"Regenerating {path} with the requested settings.")
        generate_fem_dataset(args, mesh, loss, solver)
        return load_dataset(args)

    print(f"Reusing {path} (pass --regenerate-data to replace it).")
    return data


def model_directory(args, label):
    root = args.model_dir if args.model_dir is not None else args.case_dir / "models"
    return root / f"{label}.pkl"


def model_metadata_path(args):
    if args.model_dir is None:
        return args.case_dir / "model_metadata.json"
    return args.model_dir.parent / "model_metadata.json"


def create_fno(args, grid_size, case_id):
    modes = tuple(min(args.modes, grid_size) for _ in range(3))
    return FNO(
        in_channels=10,
        out_channels=3,
        hidden_channels=args.width,
        n_modes=modes,
        n_layers=args.layers,
        rngs=nnx.Rngs(args.seed + case_id),
    )


@nnx.jit
def predict(model, inputs):
    return model(inputs)


def save_model(model, checkpoint_path):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(pickle.dumps(nnx.state(model)))


def restore_model(model, checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint {checkpoint_path} is missing.")
    restored = pickle.loads(checkpoint_path.read_bytes())
    nnx.update(model, restored)
    return model


class PickleCheckpointPhysicsInformedFNO(
    PhysicsInformedFourierParametricOperatorLearningMulti
):
    """Use Folax training/checkpoint scheduling with synchronous NNX pickles."""

    def SaveCheckPoint(self, check_point_type, checkpoint_state_dir):
        checkpoint_path = Path(f"{checkpoint_state_dir}.pkl")
        save_model(self.flax_neural_network, checkpoint_path)
        print(f"Saved {check_point_type} NNX state to {checkpoint_path}")

    def RestoreState(self, restore_state_directory):
        checkpoint_path = Path(f"{restore_state_directory}.pkl")
        restore_model(self.flax_neural_network, checkpoint_path)
        print(f"Restored NNX state from {checkpoint_path}")


def prepare_test_targets(data):
    grid_size = int(data["grid_size"])
    targets = np.asarray(data["reduced_test_targets"]).reshape(
        int(data["num_test"]), len(VOIGT_LABELS), grid_size, grid_size, grid_size, 3
    )
    return targets.astype(np.float32)


def build_case_control_inputs(material_fields, case_id, strain_amplitude):
    strain = np.zeros(6)
    strain[case_id] = strain_amplitude
    deformation_gradient = np.asarray(voigt_strain_to_macro_deformation_grad(strain))
    batch_size, number_of_nodes = material_fields.shape
    controls = np.empty((batch_size, number_of_nodes, 10), dtype=np.float32)
    controls[:, :, 0] = material_fields
    controls[:, :, 1:4] = deformation_gradient[0]
    controls[:, :, 4:7] = deformation_gradient[1]
    controls[:, :, 7:10] = deformation_gradient[2]
    return controls


def create_operator_learner(args, fe_loss, model, label):
    """Construct the standard Folax physics-informed FNO wrapper."""
    identity_control = IdentityControl(
        f"identity_control_{label}",
        control_settings={},
        num_vars=fe_loss.fe_mesh.GetNumberOfNodes(),
    )
    operator_learner = PickleCheckpointPhysicsInformedFNO(
        name=f"pi_fno_{label}",
        control=identity_control,
        loss_function=fe_loss,
        flax_neural_network=model,
        optax_optimizer=optax.adam(args.learning_rate),
    )
    operator_learner.Initialize()
    return operator_learner


def train_models(args, data, fe_loss):
    material_fields = np.asarray(data["material_fields"], dtype=np.float32)
    train_material_fields = material_fields[: args.num_train]
    grid_size = int(data["grid_size"])
    model_metadata = {
        "model_config": {
            "width": args.width,
            "modes": args.modes,
            "layers": args.layers,
            "grid_size": grid_size,
            "architecture": "ported_fno",
            "in_channels": 10,
            "out_channels": 3,
            "training_objective": "periodic_finite_element_loss",
        },
        "training_driver": "PhysicsInformedFourierParametricOperatorLearningMulti",
        "checkpoint_pattern": "models/<load_case>.pkl",
        "material_distribution": args.material_distribution,
        "num_train": args.num_train,
        "fourier_frequency_sets": (
            [list(frequencies) for frequencies in FOURIER_FREQUENCY_SETS]
            if args.material_distribution == "fourier"
            else None
        ),
        "num_grains": args.num_grains,
        "grain_values": list(args.grain_values),
        "polycrystal_dimension": (
            3 if args.material_distribution == "polycrystal" else None
        ),
        "grain_value_sampling": (
            "continuous_uniform"
            if args.material_distribution == "polycrystal"
            else None
        ),
    }

    for case_id, label in enumerate(VOIGT_LABELS):
        case_controls = build_case_control_inputs(
            train_material_fields, case_id, args.strain_amplitude
        )
        model = create_fno(args, grid_size, case_id)
        operator_learner = create_operator_learner(args, fe_loss, model, label)
        model_root = args.model_dir if args.model_dir is not None else args.case_dir / "models"
        training_directory = model_root / label
        training_directory.mkdir(parents=True, exist_ok=True)

        print(f"Training physics-informed FNO for {label} with periodic FE loss")
        start = time.perf_counter()
        operator_learner.Train(
            train_set=(jnp.asarray(case_controls),),
            batch_size=args.batch_size,
            convergence_settings={
                "num_epochs": args.epochs,
                "relative_error": 1e-100,
                "absolute_error": 1e-100,
            },
            plot_settings={
                "plot_list": ["total_loss"],
                "save_frequency": max(1, args.epochs // 10),
            },
            save_nnx_state_settings={
                "save_final_state": True,
                "interval_state_checkpointing": True,
                "interval_state_checkpointing_frequency": 100,
            },
            working_directory=str(training_directory),
        )
        elapsed = time.perf_counter() - start
        save_model(model, model_directory(args, label))
        print(f"  saved {label} model ({elapsed:.2f} s)")

    with model_metadata_path(args).open("w", encoding="utf-8") as stream:
        json.dump(model_metadata, stream, indent=2)


def expand_reduced_field(loss, reduced_field):
    full = np.asarray(
        loss.ExpandReducedDeltaDofVector(jnp.asarray(reduced_field.reshape(-1)))
    ).copy()
    full[np.asarray(loss.dirichlet_indices, dtype=np.int32)] = np.asarray(loss.dirichlet_values)
    return full.reshape(-1, 3)


def relative_l2(prediction, reference, epsilon=1e-12):
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), epsilon))


def timed_fno_prediction(model, inputs, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = np.asarray(predict(model, jnp.asarray(inputs)))
        samples.append((time.perf_counter() - start) / len(inputs))
    return result, float(np.median(samples))


def compare_models(args, data, mesh, loss, solver, stress_function):
    with model_metadata_path(args).open(encoding="utf-8") as stream:
        model_metadata = json.load(stream)
    expected_model_config = {
        "width": args.width,
        "modes": args.modes,
        "layers": args.layers,
        "grid_size": int(data["grid_size"]),
        "architecture": "ported_fno",
        "in_channels": 10,
        "out_channels": 3,
        "training_objective": "periodic_finite_element_loss",
    }
    if model_metadata.get("model_config") != expected_model_config:
        raise ValueError(
            "Checkpoint architecture differs from the command line: "
            f"saved={model_metadata.get('model_config')}, requested={expected_model_config}."
        )
    saved_distribution = model_metadata.get("material_distribution", "fourier")
    args.checkpoint_material_distribution = saved_distribution
    if saved_distribution != args.material_distribution:
        message = (
            f"Checkpoint material distribution is {saved_distribution!r}, while the "
            f"evaluation distribution is {args.material_distribution!r}."
        )
        if not args.allow_distribution_shift:
            raise ValueError(message + " Pass --allow-distribution-shift to continue.")
        print(f"WARNING: {message} Evaluating out of distribution.")
    if args.material_distribution == "fourier" and saved_distribution == "fourier":
        expected_frequency_sets = [
            list(frequencies) for frequencies in FOURIER_FREQUENCY_SETS
        ]
        if (
            model_metadata.get("num_train") != args.num_train
            or model_metadata.get("fourier_frequency_sets")
            != expected_frequency_sets
        ):
            raise ValueError(
                "Checkpoint training count or Fourier frequency sets differ from "
                "the requested multi-frequency configuration."
            )
    if args.material_distribution == "polycrystal" and saved_distribution == "polycrystal":
        saved_grain_values = np.asarray(model_metadata.get("grain_values", ()))
        requested_grain_values = np.asarray(args.grain_values)
        if (
            model_metadata.get("polycrystal_dimension", 2) != 3
            or model_metadata.get("grain_value_sampling", "discrete")
            != "continuous_uniform"
            or model_metadata.get("num_grains") != args.num_grains
            or (
                saved_grain_values.shape != requested_grain_values.shape
                or not np.allclose(saved_grain_values, requested_grain_values)
            )
        ):
            raise ValueError(
                "Checkpoint polycrystal settings differ from the command line."
            )
    targets = prepare_test_targets(data)
    test_slice = slice(args.num_train, args.num_train + args.num_test)
    test_targets = targets
    test_materials = np.asarray(data["material_fields"])[test_slice]
    input_representatives = np.asarray(data["input_representatives"], dtype=np.int32)
    grid_size = int(data["grid_size"])
    predictions = np.empty_like(test_targets)
    fno_seconds = np.empty(len(VOIGT_LABELS))

    for case_id, label in enumerate(VOIGT_LABELS):
        case_controls = build_case_control_inputs(
            test_materials, case_id, args.strain_amplitude
        )
        test_inputs = case_controls[:, input_representatives, :].reshape(
            args.num_test, grid_size, grid_size, grid_size, 10
        )
        model = restore_model(
            create_fno(args, grid_size, case_id), model_directory(args, label)
        )
        normalized_prediction, fno_seconds[case_id] = timed_fno_prediction(
            model, test_inputs, args.benchmark_repeats
        )
        predictions[:, case_id] = normalized_prediction

    displacement_relative_l2 = np.zeros((args.num_test, len(VOIGT_LABELS)))
    displacement_mae = np.zeros_like(displacement_relative_l2)
    stress_relative_l2 = np.zeros_like(displacement_relative_l2)
    stress_mae = np.zeros_like(displacement_relative_l2)
    fem_tensors = np.zeros((args.num_test, 6, 6))
    fno_tensors = np.zeros_like(fem_tensors)

    for sample_id, k_field in enumerate(test_materials):
        for case_id, label in enumerate(VOIGT_LABELS):
            fem_reduced = test_targets[sample_id, case_id]
            fno_reduced = predictions[sample_id, case_id]
            displacement_relative_l2[sample_id, case_id] = relative_l2(
                fno_reduced, fem_reduced
            )
            displacement_mae[sample_id, case_id] = float(np.mean(np.abs(fno_reduced - fem_reduced)))
            strain = np.zeros(6)
            strain[case_id] = args.strain_amplitude
            deformation_gradient = voigt_strain_to_macro_deformation_grad(strain)
            macro_u = np.asarray(macro_displacement(mesh, deformation_gradient))
            fem_total = expand_reduced_field(loss, fem_reduced) + macro_u
            fno_total = expand_reduced_field(loss, fno_reduced) + macro_u
            fem_stress, _ = stress_function(k_field, fem_total)
            fno_stress, _ = stress_function(k_field, fno_total)
            fem_stress = np.asarray(fem_stress)
            fno_stress = np.asarray(fno_stress)
            fem_tensors[sample_id, :, case_id] = fem_stress / args.strain_amplitude
            fno_tensors[sample_id, :, case_id] = fno_stress / args.strain_amplitude
            stress_relative_l2[sample_id, case_id] = relative_l2(fno_stress, fem_stress)
            stress_mae[sample_id, case_id] = float(
                np.mean(np.abs(fno_stress - fem_stress))
            )

    tensor_relative_error = np.asarray(
        [relative_l2(fno_tensors[i], fem_tensors[i]) for i in range(args.num_test)]
    )
    fem_seconds = benchmark_fem(args, mesh, solver, test_materials[0])
    rows = []
    for case_id, label in enumerate(VOIGT_LABELS):
        rows.append(
            {
                "load_case": label,
                "mean_displacement_relative_l2": np.mean(displacement_relative_l2[:, case_id]),
                "std_displacement_relative_l2": np.std(displacement_relative_l2[:, case_id]),
                "mean_displacement_mae": np.mean(displacement_mae[:, case_id]),
                "std_displacement_mae": np.std(displacement_mae[:, case_id]),
                "mean_stress_relative_l2": np.mean(stress_relative_l2[:, case_id]),
                "std_stress_relative_l2": np.std(stress_relative_l2[:, case_id]),
                "mean_stress_mae": np.mean(stress_mae[:, case_id]),
                "std_stress_mae": np.std(stress_mae[:, case_id]),
                "fem_seconds_per_sample": fem_seconds[case_id],
                "fno_seconds_per_sample": fno_seconds[case_id],
                "online_speedup": fem_seconds[case_id] / max(fno_seconds[case_id], 1e-12),
            }
        )
    write_comparison(args, rows, tensor_relative_error)
    np.savez(
        args.case_dir / "fno_fem_comparison.npz",
        displacement_relative_l2=displacement_relative_l2,
        displacement_mae=displacement_mae,
        stress_relative_l2=stress_relative_l2,
        stress_mae=stress_mae,
        fem_homogenized_tensors=fem_tensors,
        fno_homogenized_tensors=fno_tensors,
        tensor_relative_error=tensor_relative_error,
        fem_seconds_per_sample=fem_seconds,
        fno_seconds_per_sample=fno_seconds,
        checkpoint_material_distribution=saved_distribution,
        evaluation_material_distribution=args.material_distribution,
        voigt_labels=np.asarray(VOIGT_LABELS),
    )
    plot_comparison(args, rows, tensor_relative_error)
    if args.visualize:
        visualize_displacement_comparison(
            args, mesh, loss, test_targets, predictions
        )
    if not args.no_vtk:
        export_comparison_vtk(
            args, mesh, test_materials[0], test_targets[0], predictions[0], loss
        )


def benchmark_fem(args, mesh, solver, k_field):
    zeros = np.zeros(3 * mesh.GetNumberOfNodes())
    seconds = np.zeros(len(VOIGT_LABELS))
    for case_id in range(len(VOIGT_LABELS)):
        strain = np.zeros(6)
        strain[case_id] = args.strain_amplitude
        control = build_control_input(k_field, voigt_strain_to_macro_deformation_grad(strain))
        repeats = []
        for _ in range(args.benchmark_repeats):
            start = time.perf_counter()
            _ = np.asarray(solver.SolveReduced(control, zeros))
            repeats.append(time.perf_counter() - start)
        seconds[case_id] = np.median(repeats)
    return seconds


def write_comparison(args, rows, tensor_relative_error):
    path = args.case_dir / "performance_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    error_statistic_keys = (
        "mean_displacement_relative_l2",
        "std_displacement_relative_l2",
        "mean_displacement_mae",
        "std_displacement_mae",
        "mean_stress_relative_l2",
        "std_stress_relative_l2",
        "mean_stress_mae",
        "std_stress_mae",
    )
    summary = {
        "checkpoint_material_distribution": getattr(
            args, "checkpoint_material_distribution", args.material_distribution
        ),
        "evaluation_material_distribution": args.material_distribution,
        "model_directory": str(
            args.model_dir if args.model_dir is not None else args.case_dir / "models"
        ),
        "mean_homogenized_tensor_relative_frobenius_error": float(np.mean(tensor_relative_error)),
        "std_homogenized_tensor_relative_frobenius_error": float(np.std(tensor_relative_error)),
        "median_homogenized_tensor_relative_frobenius_error": float(np.median(tensor_relative_error)),
        "load_case_error_statistics": {
            row["load_case"]: {
                key: float(row[key]) for key in error_statistic_keys
            }
            for row in rows
        },
        "mean_online_speedup_over_six_cases": float(
            sum(row["fem_seconds_per_sample"] for row in rows)
            / max(sum(row["fno_seconds_per_sample"] for row in rows), 1e-12)
        ),
    }
    with (args.case_dir / "comparison_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    evaluation_set_name = getattr(args, "evaluation_set_name", "held-out set")
    print(f"\nFNO versus FEM ({evaluation_set_name})")
    for row in rows:
        print(
            f"{row['load_case']}: "
            f"u rel-L2={row['mean_displacement_relative_l2']:.4e} "
            f"+/- {row['std_displacement_relative_l2']:.4e}, "
            f"stress rel-L2={row['mean_stress_relative_l2']:.4e} "
            f"+/- {row['std_stress_relative_l2']:.4e}, "
            f"speedup={row['online_speedup']:.2f}x"
        )
    print(json.dumps(summary, indent=2))


def plot_comparison(args, rows, tensor_relative_error):
    labels = [row["load_case"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(
        labels,
        [row["mean_displacement_relative_l2"] for row in rows],
        yerr=[row["std_displacement_relative_l2"] for row in rows],
        capsize=4,
    )
    axes[0].set_title("Displacement relative L2")
    axes[1].bar(
        labels,
        [row["mean_stress_relative_l2"] for row in rows],
        yerr=[row["std_stress_relative_l2"] for row in rows],
        capsize=4,
    )
    axes[1].set_title("Average-stress relative L2")
    axes[2].bar(labels, [row["online_speedup"] for row in rows])
    axes[2].set_title("Online speedup (FEM/FNO)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Tensor relative Frobenius error: "
        f"{np.mean(tensor_relative_error):.3e} +/- {np.std(tensor_relative_error):.3e}"
    )
    figure.tight_layout()
    figure.savefig(args.case_dir / "fno_fem_comparison.png", dpi=200)
    plt.close(figure)


def visualize_displacement_comparison(args, mesh, loss, fem_targets, fno_predictions):
    """Render FE/FNO total and fluctuation displacement fields for test samples."""
    sample_id = args.visualization_sample_id
    number_of_samples = fem_targets.shape[0]
    if sample_id < 0 or sample_id >= number_of_samples:
        raise ValueError(
            f"visualization-sample-id {sample_id} is outside "
            f"[0, {number_of_samples - 1}]."
        )

    if args.visualization_load_case == "all":
        case_ids = range(len(VOIGT_LABELS))
    else:
        case_ids = (VOIGT_LABELS.index(args.visualization_load_case),)

    for case_id in case_ids:
        label = VOIGT_LABELS[case_id]
        strain = np.zeros(6)
        strain[case_id] = args.strain_amplitude
        deformation_gradient = voigt_strain_to_macro_deformation_grad(strain)
        macro_u = np.asarray(macro_displacement(mesh, deformation_gradient))
        fem_fluctuation = expand_reduced_field(
            loss, fem_targets[sample_id, case_id]
        )
        fno_fluctuation = expand_reduced_field(
            loss, fno_predictions[sample_id, case_id]
        )
        total_fields = {
            "Finite element": fem_fluctuation + macro_u,
            "FNO prediction": fno_fluctuation + macro_u,
        }
        fluctuation_fields = {
            "Finite element fluctuation": fem_fluctuation,
            "FNO fluctuation": fno_fluctuation,
        }
        output_directory = (
            args.case_dir
            / "visualizations"
            / f"sample_{sample_id:04d}"
            / label
        )
        output_directory.mkdir(parents=True, exist_ok=True)

        plot_displacement_contours_pyvista(
            fe_mesh=mesh,
            displacement_fields=total_fields,
            output_path=output_directory / "displacement_total_contours.png",
            warp_factor=args.warp_factor,
            camera_zoom=args.camera_zoom,
        )
        plot_displacement_components_pyvista(
            fe_mesh=mesh,
            displacement_fields=total_fields,
            output_directory=output_directory,
            filename_prefix="displacement_total",
            warp_factor=args.warp_factor,
            camera_zoom=args.camera_zoom,
        )
        plot_displacement_contours_pyvista(
            fe_mesh=mesh,
            displacement_fields=fluctuation_fields,
            output_path=output_directory / "displacement_fluctuation_contours.png",
            warp_factor=args.warp_factor,
            camera_zoom=args.camera_zoom,
        )
        plot_displacement_components_pyvista(
            fe_mesh=mesh,
            displacement_fields=fluctuation_fields,
            output_directory=output_directory,
            filename_prefix="displacement_fluctuation",
            warp_factor=args.warp_factor,
            camera_zoom=args.camera_zoom,
        )


def export_comparison_vtk(args, mesh, k_field, fem_targets, fno_predictions, loss):
    vtk_directory = args.case_dir / "vtk_comparison"
    vtk_directory.mkdir(parents=True, exist_ok=True)
    mesh["K_matrix"] = k_field
    for case_id, label in enumerate(VOIGT_LABELS):
        fem = expand_reduced_field(loss, fem_targets[case_id])
        fno = expand_reduced_field(loss, fno_predictions[case_id])
        mesh[f"U_FE_fluctuation_{label}"] = fem
        mesh[f"U_FNO_fluctuation_{label}"] = fno
        mesh[f"U_FNO_abs_error_{label}"] = np.abs(fno - fem)
    mesh.Finalize(export_dir=vtk_directory)
