#!/usr/bin/env python3
# ============================================================
# PHASE_0_PARAMETRIC_3D_NEW.py  (REVISED)
#
# Key revisions vs your version:
#   (A) Fix “too fast convergence” from minibatch early-stop:
#       - Added fullbatch_for_convergence flag (default True)
#       - When enabled, training batch_size is forced to len(train_list)
#         so residual_rms_batch_mean is computed on the whole train set,
#         not on a single easy sample (batch_size=1 issue).
#
#   (B) Add per-sample “training history” presence:
#       - Added copy_training_artifacts_to_samples flag (default True)
#       - Copies training_history_parametric.png + residual CSV/PNGs
#         into every PHASE_0_PARAMETRIC_sample_* folder.
#
# Everything else is kept in the same structure you wrote.
# ============================================================

import sys
import os

# ---- GPU memory behavior (must be set BEFORE importing jax) ----
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import optax
from flax import nnx
from flax.nnx import bridge
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pickle
import csv
import time
import glob

from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DHexa
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.identity_control import IdentityControl
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fourier_parametric_operator_learning import (
    PhysicsInformedFourierParametricOperatorLearning3D,
)
from fol.deep_neural_networks.fourier_neural_operator_networks import FourierNeuralOperator3D
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger

from newton_residual_tracker import custom_newton_solve
from training_residual_tracker import TrainingResidualTracker

# =========================
RESIDUAL_TARGET = 1e-5
MASTER_STATS_NAME = "stats_phase0_parametric.csv"
# =========================



# ---- shared utilities (extracted to common/) --------------------------
from common.script_utils import (
    safe_rename,
    _parse_ids,
    _sanitize_label,
    _clean_mkdir,
    _find_first_existing,
    _read_training_residual_csv,
    _summarize_training,
    _append_row_csv,
    _write_single_row_csv,
    _ensure_vtu_in_dir,
    _copy_training_artifacts,
    _patch_rng_keys,
)
from common.plot_3d import (
    _midz_slice_scalar,
    _save_slice_png,
    _save_2x2_panel,
)
# -----------------------------------------------------------------------
def main(
    fol_num_epochs: int = 2000,
    solve_FE: bool = True,
    clean_dir: bool = False,
    use_fno_warmstart: bool = False,

    # PKL / sample generation controls
    create_new_samples: bool = True,
    num_samples: int = 200,
    random_seed: int = 0,
    pkl_path: str = "fourier_control_dict3D.pkl",

    # train/test selection (INCLUSIVE ranges)
    train_ids: str = "0-79",
    test_ids: str = "81-99",

    default_label: str = "fourier_control3D",
    out_root: str = "PHASE_0_PARAMETRIC_3D_Fourier",

    batch_size: int = 5,
    test_frequency: int = 100,
    save_frequency: int = 100,

    # ---------- NEW FLAGS ----------
    fullbatch_for_convergence: bool = True,
    copy_training_artifacts_to_samples: bool = True,
):
    out_root = os.path.abspath(os.path.expanduser(out_root))
    os.makedirs(out_root, exist_ok=True)

    master_stats_csv = os.path.join(out_root, MASTER_STATS_NAME)

    STAT_FIELDS = [
        "phase",
        "sample_id",
        "label",
        "pkl_path",
        "case_dir",
        "create_new_samples",
        "num_samples",
        "train_ids",
        "test_ids",
        "train_count",
        "test_count",
        "fol_num_epochs_requested",
        "epochs_completed",
        "last_epoch",
        "train_residual_final",
        "train_residual_min",
        "epoch_at_min",
        "epoch_first_below_target",
        "residual_target",
        "final_total_loss",
        "train_time_s",
        "batch_size",
        "solve_FE",
        "use_fno_warmstart",
        "newton_total_iters",
        "newton_final_residual",
        "fe_time_s",
        "ux_rms", "ux_max",
        "uy_rms", "uy_max",
        "uz_rms", "uz_max",
        "uv_rms", "uv_max",
    ]

    # ============================================================
    # 1) problem setup (3D)
    # ============================================================
    model_settings = {
        "L": 1.0,
        "N": 20,  # Nx=Ny=Nz
        "Ux_left": 0.0,
        "Ux_right": 0.1,
        "Uy_left": 0.0,
        "Uy_right": 0.1,
        "Uz_left": 0.0,
        "Uz_right": 0.0,
    }

    L = float(model_settings["L"])
    N = int(model_settings["N"])
    Nx = Ny = Nz = N

    fe_mesh = create_3D_box_mesh_structured(Nx=Nx, Ny=Ny, Nz=Nz, Lx=L, Ly=L, Lz=L)

    bc_dict = {
        "Ux": {"left": model_settings["Ux_left"], "right": model_settings["Ux_right"]},
        "Uy": {"left": model_settings["Uy_left"], "right": model_settings["Uy_right"]},
        "Uz": {"left": model_settings["Uz_left"], "right": model_settings["Uz_right"]},
    }

    material_dict = {"young_modulus": 1.0, "poisson_ratio": 0.3}
    mechanical_loss_3d = NeoHookeMechanicalLoss3DHexa(
        "mechanical_loss_3d",
        loss_settings={"dirichlet_bc_dict": bc_dict, "material_dict": material_dict},
        fe_mesh=fe_mesh,
    )

    fe_mesh.Initialize()
    mechanical_loss_3d.Initialize()

    n_nodes = fe_mesh.GetNumberOfNodes()
    if n_nodes != Nx * Ny * Nz:
        raise ValueError(f"Mesh nodes mismatch: fe_mesh has {n_nodes} nodes but Nx*Ny*Nz={Nx*Ny*Nz}")

    # ============================================================
    # 2) load PKL OR generate PKL
    # ============================================================
    pkl_path = os.path.abspath(os.path.expanduser(pkl_path))

    if (not create_new_samples) and (not os.path.exists(pkl_path)):
        raise FileNotFoundError(
            f"[PHASE_0_PARAMETRIC_3D] pkl not found: {pkl_path}\n"
            f"Set create_new_samples=True to generate it."
        )

    if create_new_samples:
        np.random.seed(int(random_seed))

        fourier_control_settings = {
            "x_freqs": np.array([2, 4, 6]),
            "y_freqs": np.array([2, 4, 6]),
            "z_freqs": np.array([2, 4, 6]),
            "beta": 20.0,
            "min": 1e-1,
            "max": 1.0,
        }
        fourier_control = FourierControl("fourier_control", fourier_control_settings, fe_mesh)
        fourier_control.Initialize()

        coeffs_matrix, K_matrix = create_random_fourier_samples(fourier_control, int(num_samples))
        coeffs_matrix = np.asarray(coeffs_matrix, dtype=np.float32)
        K_matrix = np.asarray(K_matrix, dtype=np.float32)

        export_dict = {
            "coeffs_matrix": coeffs_matrix,
            "K_matrix": K_matrix,
            "x_freqs": np.asarray(fourier_control_settings["x_freqs"]),
            "y_freqs": np.asarray(fourier_control_settings["y_freqs"]),
            "z_freqs": np.asarray(fourier_control_settings["z_freqs"]),
            "beta": float(fourier_control_settings["beta"]),
            "min": float(fourier_control_settings["min"]),
            "max": float(fourier_control_settings["max"]),
        }

        os.makedirs(os.path.dirname(pkl_path) or ".", exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(export_dict, f)

        print(f"[PHASE_0_PARAMETRIC_3D] Wrote NEW pkl: {pkl_path}")
        print(f"[PHASE_0_PARAMETRIC_3D] coeffs_matrix shape={coeffs_matrix.shape}, K_matrix shape={K_matrix.shape}")

        d = export_dict
    else:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

    # ============================================================
    # 3) K_matrix from pkl OR coeffs_matrix→FourierControl
    # ============================================================
    if "K_matrix" in d:
        K_matrix = np.asarray(d["K_matrix"], dtype=np.float32)
        if num_samples is not None:
            K_matrix = K_matrix[: int(num_samples)]
        print(f"[PHASE_0_PARAMETRIC_3D] Found K_matrix in pkl: {K_matrix.shape}")

    elif "coeffs_matrix" in d:
        coeffs_matrix = np.asarray(d["coeffs_matrix"], dtype=np.float32)
        if num_samples is not None:
            coeffs_matrix = coeffs_matrix[: int(num_samples)]
        print(f"[PHASE_0_PARAMETRIC_3D] Found coeffs_matrix in pkl: {coeffs_matrix.shape}")

        x_freqs = np.asarray(d.get("x_freqs", np.array([2, 4, 6])))
        y_freqs = np.asarray(d.get("y_freqs", np.array([2, 4, 6])))
        z_freqs = np.asarray(d.get("z_freqs", np.array([2, 4, 6])))
        beta = float(d.get("beta", 20.0))
        kmin = float(d.get("min", 1e-1))
        kmax = float(d.get("max", 1.0))

        fourier_control_settings = {
            "x_freqs": x_freqs,
            "y_freqs": y_freqs,
            "z_freqs": z_freqs,
            "beta": beta,
            "min": kmin,
            "max": kmax,
        }
        fourier_control = FourierControl("fourier_control", fourier_control_settings, fe_mesh)
        fourier_control.Initialize()

        K_matrix_jax = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)
        K_matrix = np.asarray(K_matrix_jax, dtype=np.float32)
        print(f"[PHASE_0_PARAMETRIC_3D] Computed K_matrix from coeffs_matrix: {K_matrix.shape}")

    else:
        raise KeyError("[PHASE_0_PARAMETRIC_3D] pkl must contain either 'K_matrix' or 'coeffs_matrix'")

    if K_matrix.ndim != 2:
        raise ValueError(f"[PHASE_0_PARAMETRIC_3D] K_matrix must be 2D, got {K_matrix.shape}")

    num_samples_eff, k_nodes = K_matrix.shape
    if k_nodes != n_nodes:
        raise ValueError(
            f"[PHASE_0_PARAMETRIC_3D] K_matrix has {k_nodes} nodes, expected {n_nodes}. "
            f"Your PKL was generated for a different (Nx,Ny,Nz)."
        )

    labels_vec = None
    for key in ["labels", "sample_labels", "source_labels", "types"]:
        if key in d:
            labels_vec = list(d[key])
            if len(labels_vec) != num_samples_eff:
                labels_vec = None
            break

    train_list = _parse_ids(train_ids, num_samples_eff)
    test_list = _parse_ids(test_ids, num_samples_eff)

    label = _sanitize_label(default_label)

    print(f"[PHASE_0_PARAMETRIC_3D] out_root={out_root}")
    print(f"[PHASE_0_PARAMETRIC_3D] train_ids={train_ids} -> {len(train_list)} samples")
    print(f"[PHASE_0_PARAMETRIC_3D] test_ids={test_ids}  -> {len(test_list)} samples")

    # ============================================================
    # 4) training folder (single)
    # ============================================================
    train_case_dir = os.path.join(out_root, f"PHASE_0_PARAMETRIC_train_{label}")
    _clean_mkdir(train_case_dir)
    sys.stdout = Logger(os.path.join(train_case_dir, f"PHASE_0_PARAMETRIC_train_{label}.log"))

    X_train = K_matrix[train_list, :].astype(np.float32)
    X_test  = K_matrix[test_list, :].astype(np.float32)

    # ---------- NEW: enforce full-batch to make residual-based early-stop meaningful ----------
    requested_batch_size = int(batch_size)
    train_count = len(train_list)
    if fullbatch_for_convergence:
        batch_size_used = train_count
        print(
            f"[PHASE_0_PARAMETRIC_3D] fullbatch_for_convergence=True -> "
            f"forcing batch_size from {requested_batch_size} to {batch_size_used} "
            f"(so residual_rms_batch_mean is over ALL train samples, not a minibatch)."
        )
    else:
        batch_size_used = max(1, min(requested_batch_size, train_count))
        if batch_size_used < train_count:
            print(
                f"[PHASE_0_PARAMETRIC_3D] WARN: batch_size={batch_size_used} < train_count={train_count}. "
                f"Early-stop on residual_rms_batch_mean may trigger on an easy minibatch."
            )

    # ============================================================
    # 5) build model
    # ============================================================
    fno_model = bridge.ToNNX(
        FourierNeuralOperator3D(
            modes1=6,
            modes2=6,
            modes3=6,
            width=8,
            depth=4,
            channels_last_proj=32,
            out_channels=3,
            output_scale=0.1,
        ),
        rngs=nnx.Rngs(0),
    ).lazy_init(X_train[0].reshape(1, Nx, Ny, Nz, 1))

    fno_model = _patch_rng_keys(fno_model)

    params = nnx.state(fno_model, nnx.Param)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print(f"[PHASE_0_PARAMETRIC_3D] total number of FNO network parameters: {total_params}")

    adam_optimizer = optax.chain(optax.adam(1e-3))
    identity_control = IdentityControl("identity_control", {"num_vars": n_nodes}, fe_mesh)

    pi_fno_param = PhysicsInformedFourierParametricOperatorLearning3D(
        name="pi_fno_parametric_identity_3d",
        control=identity_control,
        loss_function=mechanical_loss_3d,
        flax_neural_network=fno_model,
        optax_optimizer=adam_optimizer,
    )
    pi_fno_param.Initialize()

    # ============================================================
    # 6) PARAMETRIC TRAIN
    # ============================================================
    train_tag = f"parametric_identity_{label}"
    train_res_tracker = TrainingResidualTracker(out_dir=train_case_dir, tag=train_tag)

    print("[PHASE_0_PARAMETRIC_3D] Starting parametric training...")
    t_train0 = time.time()

    pi_fno_param.Train(
        train_set=(jnp.asarray(X_train),),
        test_set=(jnp.asarray(X_test),),
        test_frequency=int(test_frequency),
        batch_size=int(batch_size_used),
        convergence_settings={
            "num_epochs": int(fol_num_epochs),
            "convergence_criterion": "residual_rms_batch_mean",
            "relative_error": 1e-100,
            "absolute_error": float(RESIDUAL_TARGET),
        },
        plot_settings={
            "plot_list": ["total_loss", "residual_rms_batch_mean"],
            "plot_frequency": 1,
            "save_frequency": int(save_frequency),
            "save_directory": train_case_dir,
            "test_frequency": int(test_frequency),
        },
        train_checkpoint_settings={
            "least_loss_checkpointing": True,
            "frequency": int(save_frequency),
            "state_directory": os.path.join(train_case_dir, "flax_train_state"),
        },
        working_directory=train_case_dir,
        training_residual_tracker=train_res_tracker,
    )

    train_time_s = time.time() - t_train0
    train_res_tracker.finalize()

    safe_rename(
        os.path.join(train_case_dir, "training_history.png"),
        os.path.join(train_case_dir, "training_history_parametric.png"),
    )

    pi_fno_param.RestoreState(restore_state_directory=os.path.join(train_case_dir, "flax_train_state"))
    print("[PHASE_0_PARAMETRIC_3D] Restored best parametric checkpoint from flax_train_state.")

    train_csv = os.path.join(train_case_dir, f"{train_tag}_residual_rms.csv")
    if not os.path.exists(train_csv):
        train_csv = _find_first_existing([os.path.join(train_case_dir, "*residual_rms*.csv")])
    train_stats = _summarize_training(train_csv, residual_target=RESIDUAL_TARGET)

    # ============================================================
    # 7) EVALUATE test_list -> per-sample folders
    # ============================================================
    for sample_id in test_list:
        sample_label = _sanitize_label(labels_vec[sample_id]) if labels_vec is not None else label
        case_dir = os.path.join(out_root, f"PHASE_0_PARAMETRIC_sample_{sample_id}_{sample_label}")
        _clean_mkdir(case_dir)
        sys.stdout = Logger(os.path.join(case_dir, f"PHASE_0_PARAMETRIC_sample_{sample_id}_{sample_label}.log"))

        # ---------- NEW: copy training artifacts into each sample folder ----------
        if copy_training_artifacts_to_samples:
            _copy_training_artifacts(train_case_dir=train_case_dir, case_dir=case_dir, train_tag=train_tag)

        print(f"[PHASE_0_PARAMETRIC_3D] sample_id={sample_id}, label={sample_label}")
        print(f"[PHASE_0_PARAMETRIC_3D] pkl_path={pkl_path}")
        print(f"[PHASE_0_PARAMETRIC_3D] output_dir={case_dir}")

        K_vec = K_matrix[sample_id, :].astype(np.float32)

        _save_slice_png(
            K_vec, Nx, Ny, Nz,
            os.path.join(case_dir, f"heterogeneity_sample_{sample_id}.png"),
            title=f"Heterogeneity mid-z (sid={sample_id})",
        )

        # predict
        FNO_UVW = np.array(pi_fno_param.Predict(K_vec.reshape(1, -1))).reshape(-1)

        # export fields (fresh mesh per sample)
        export_mesh = create_3D_box_mesh_structured(Nx=Nx, Ny=Ny, Nz=Nz, Lx=L, Ly=L, Lz=L)
        export_mesh.Initialize()
        export_mesh[f"U_FNO_{sample_id}_param"] = FNO_UVW.reshape((n_nodes, 3))
        export_mesh[f"K_field_{sample_id}_param"] = K_vec.reshape((n_nodes,))

        newton_total_iters = ""
        newton_final_residual = ""
        fe_time_s = ""

        ux_rms = np.nan; ux_max = np.nan
        uy_rms = np.nan; uy_max = np.nan
        uz_rms = np.nan; uz_max = np.nan
        uv_rms = np.nan; uv_max = np.nan

        if solve_FE:
            fe_setting = {
                "linear_solver_settings": {
                    "solver": "JAX-bicgstab",
                    "tol": 1e-6,
                    "atol": 1e-6,
                    "maxiter": 1000,
                    "pre-conditioner": "ilu",
                },
                "nonlinear_solver_settings": {
                    "rel_tol": 1e-5,
                    "abs_tol": 1e-5,
                    "maxiter": 8,
                    "load_incr": 21,
                },
            }

            nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver(
                f"nonlin_fe_solver_param_{sample_id}",
                mechanical_loss_3d,
                fe_setting,
            )
            nonlin_fe_solver.Initialize()

            ndofs = 3 * n_nodes
            initial_dofs = FNO_UVW.copy() if use_fno_warmstart else np.zeros(ndofs)

            t_fe0 = time.time()
            FE_UVW_jax, residuals, total_iters = custom_newton_solve(
                fe_solver=nonlin_fe_solver,
                control_vars=K_vec,
                initial_dofs=initial_dofs,
                case_dir=case_dir,
                sample_tag=f"sample_{sample_id}_{sample_label}_param",
            )
            fe_time_s = time.time() - t_fe0

            FE_UVW = np.array(FE_UVW_jax).reshape(-1)
            export_mesh[f"U_FE_{sample_id}_param"] = FE_UVW.reshape((n_nodes, 3))

            newton_total_iters = int(total_iters)
            newton_final_residual = float(residuals[-1]) if (residuals is not None and len(residuals) > 0) else ""

            iters = np.arange(1, len(residuals) + 1)
            plt.figure(figsize=(6, 4))
            plt.semilogy(iters, residuals, marker="o")
            plt.xlabel("Global Newton iteration")
            plt.ylabel(r"RMS residual $\|r\|_{\mathrm{rms}}$")
            plt.title(f"Newton convergence – sample {sample_id} (PARAM-3D)")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, f"newton_residuals_sample_{sample_id}_param.png"))
            plt.close()

            abs_err_uvw = np.abs(FNO_UVW - FE_UVW)
            abs_err_ux = abs_err_uvw[0::3]
            abs_err_uy = abs_err_uvw[1::3]
            abs_err_uz = abs_err_uvw[2::3]

            export_mesh[f"abs_error_{sample_id}_param"] = abs_err_uvw.reshape((n_nodes, 3))
            export_mesh[f"abs_error_ux_{sample_id}_param"] = abs_err_ux.reshape((n_nodes,))
            export_mesh[f"abs_error_uy_{sample_id}_param"] = abs_err_uy.reshape((n_nodes,))
            export_mesh[f"abs_error_uz_{sample_id}_param"] = abs_err_uz.reshape((n_nodes,))

            ux_rms = float(np.sqrt(np.mean(abs_err_ux**2))); ux_max = float(np.max(abs_err_ux))
            uy_rms = float(np.sqrt(np.mean(abs_err_uy**2))); uy_max = float(np.max(abs_err_uy))
            uz_rms = float(np.sqrt(np.mean(abs_err_uz**2))); uz_max = float(np.max(abs_err_uz))

            ev = np.sqrt(
                (FNO_UVW[0::3] - FE_UVW[0::3])**2 +
                (FNO_UVW[1::3] - FE_UVW[1::3])**2 +
                (FNO_UVW[2::3] - FE_UVW[2::3])**2
            )
            uv_rms = float(np.sqrt(np.mean(ev**2)))
            uv_max = float(np.max(ev))

            _save_2x2_panel(
                K_vec=K_vec,
                fno_ux=FNO_UVW[0::3],
                fe_ux=FE_UVW[0::3],
                abs_err_ux=abs_err_ux,
                Nx=Nx, Ny=Ny, Nz=Nz,
                out_path=os.path.join(case_dir, f"plot_results_sample_{sample_id}_param.png"),
            )
        else:
            Ksl = _midz_slice_scalar(K_vec, Nx, Ny, Nz).T
            Usl = _midz_slice_scalar(FNO_UVW[0::3], Nx, Ny, Nz).T
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            im0 = axs[0].imshow(Ksl, origin="lower"); axs[0].set_title("Heterogeneity"); plt.colorbar(im0, ax=axs[0], fraction=0.046)
            im1 = axs[1].imshow(Usl, origin="lower"); axs[1].set_title("FNO_Ux");      plt.colorbar(im1, ax=axs[1], fraction=0.046)
            for a in axs:
                a.set_xticks([]); a.set_yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, f"plot_results_sample_{sample_id}_param.png"), dpi=200)
            plt.close()

        export_mesh.Finalize(export_dir=case_dir)
        _ensure_vtu_in_dir(case_dir)

        row = {
            "phase": "PHASE_0_PARAMETRIC_3D",
            "sample_id": int(sample_id),
            "label": sample_label,
            "pkl_path": pkl_path,
            "case_dir": case_dir,
            "create_new_samples": bool(create_new_samples),
            "num_samples": int(num_samples_eff),
            "train_ids": train_ids,
            "test_ids": test_ids,
            "train_count": len(train_list),
            "test_count": len(test_list),
            "fol_num_epochs_requested": int(fol_num_epochs),
            "epochs_completed": train_stats["epochs_completed"],
            "last_epoch": train_stats["last_epoch"],
            "train_residual_final": train_stats["final_residual"],
            "train_residual_min": train_stats["min_residual"],
            "epoch_at_min": train_stats["epoch_at_min"],
            "epoch_first_below_target": train_stats["epoch_first_below_target"],
            "residual_target": float(RESIDUAL_TARGET),
            "final_total_loss": train_stats["final_total_loss"],
            "train_time_s": float(train_time_s),
            "batch_size": int(batch_size_used),  # <-- store USED batch size
            "solve_FE": bool(solve_FE),
            "use_fno_warmstart": bool(use_fno_warmstart),
            "newton_total_iters": newton_total_iters,
            "newton_final_residual": newton_final_residual,
            "fe_time_s": fe_time_s,
            "ux_rms": ux_rms, "ux_max": ux_max,
            "uy_rms": uy_rms, "uy_max": uy_max,
            "uz_rms": uz_rms, "uz_max": uz_max,
            "uv_rms": uv_rms, "uv_max": uv_max,
        }

        per_sample_csv = os.path.join(case_dir, f"stats_param_sample_{sample_id}.csv")
        _write_single_row_csv(per_sample_csv, STAT_FIELDS, row)
        _append_row_csv(master_stats_csv, STAT_FIELDS, row)

        print(f"[PHASE_0_PARAMETRIC_3D] Wrote per-sample stats: {per_sample_csv}")
        print(f"[PHASE_0_PARAMETRIC_3D] Appended master stats: {master_stats_csv}")

        if clean_dir:
            shutil.rmtree(case_dir)
            print(f"[PHASE_0_PARAMETRIC_3D] Cleaned directory {case_dir}.")


if __name__ == "__main__":
    fol_num_epochs = 5000
    solve_FE = True
    clean_dir = False
    use_fno_warmstart = False

    create_new_samples = True
    num_samples = 200
    random_seed = 0

    batch_size = 1
    test_frequency = 100
    save_frequency = 100

    # NEW defaults (you can override from CLI if you want)
    fullbatch_for_convergence = True
    copy_training_artifacts_to_samples = True

    script_dir = Path(__file__).resolve().parent
    pkl_path = str(script_dir / "fourier_control_dict3D.pkl")

    train_ids = "0-80"
    test_ids = "20-99"
    default_label = "fourier_control3D"
    out_root = str(script_dir / "PHASE_0_PARAMETRIC_3D_Fourier_non_warm")

    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("fol_num_epochs="):
            fol_num_epochs = int(arg.split("=")[1])
        elif arg.startswith("solve_FE="):
            solve_FE = arg.split("=")[1].lower() == "true"
        elif arg.startswith("clean_dir="):
            clean_dir = arg.split("=")[1].lower() == "true"
        elif arg.startswith("use_fno_warmstart="):
            use_fno_warmstart = arg.split("=")[1].lower() == "true"
        elif arg.startswith("create_new_samples="):
            create_new_samples = arg.split("=")[1].lower() == "true"
        elif arg.startswith("num_samples="):
            num_samples = int(arg.split("=")[1])
        elif arg.startswith("random_seed="):
            random_seed = int(arg.split("=")[1])
        elif arg.startswith("batch_size="):
            batch_size = int(arg.split("=")[1])
        elif arg.startswith("test_frequency="):
            test_frequency = int(arg.split("=")[1])
        elif arg.startswith("save_frequency="):
            save_frequency = int(arg.split("=")[1])
        elif arg.startswith("pkl_path="):
            pkl_path = arg.split("=", 1)[1]
        elif arg.startswith("train_ids="):
            train_ids = arg.split("=", 1)[1]
        elif arg.startswith("test_ids="):
            test_ids = arg.split("=", 1)[1]
        elif arg.startswith("default_label="):
            default_label = arg.split("=")[1]
        elif arg.startswith("out_root="):
            out_root = arg.split("=", 1)[1]
        elif arg.startswith("fullbatch_for_convergence="):
            fullbatch_for_convergence = arg.split("=")[1].lower() == "true"
        elif arg.startswith("copy_training_artifacts_to_samples="):
            copy_training_artifacts_to_samples = arg.split("=")[1].lower() == "true"
        else:
            print(
                "Usage:\n"
                "  python PHASE_0_PARAMETRIC_3D_NEW.py "
                "create_new_samples=False num_samples=200 random_seed=0 "
                "fol_num_epochs=2000 batch_size=5 "
                "train_ids=0-2 test_ids=3-4 "
                "solve_FE=True use_fno_warmstart=False "
                "pkl_path=/path/to/fourier_control_dict3D.pkl "
                "out_root=/path/to/out "
                "fullbatch_for_convergence=True "
                "copy_training_artifacts_to_samples=True\n"
                "\n"
                "Notes:\n"
                "  - train_ids/test_ids ranges are INCLUSIVE.\n"
                "  - If pkl is missing, set create_new_samples=True.\n"
            )
            sys.exit(1)

    main(
        fol_num_epochs=fol_num_epochs,
        solve_FE=solve_FE,
        clean_dir=clean_dir,
        use_fno_warmstart=use_fno_warmstart,
        create_new_samples=create_new_samples,
        num_samples=num_samples,
        random_seed=random_seed,
        pkl_path=pkl_path,
        train_ids=train_ids,
        test_ids=test_ids,
        default_label=default_label,
        out_root=out_root,
        batch_size=batch_size,
        test_frequency=test_frequency,
        save_frequency=save_frequency,
        fullbatch_for_convergence=fullbatch_for_convergence,
        copy_training_artifacts_to_samples=copy_training_artifacts_to_samples,
    )