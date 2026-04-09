"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: August, 2025
 License: FOL/LICENSE
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import optax
from flax import nnx
from flax.nnx import bridge
import jax
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pickle
import csv
import time
import glob

from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.identity_control import IdentityControl
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fourier_parametric_operator_learning import (
    PhysicsInformedFourierParametricOperatorLearning,
)
from fol.deep_neural_networks.fourier_neural_operator_networks import FourierNeuralOperator2D
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger

from fol.tools.newton_residual_tracker import custom_newton_solve
from training_residual_tracker import TrainingResidualTracker


# =========================
RESIDUAL_TARGET = 1e-4
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
)
# -----------------------------------------------------------------------
def main(
    fol_num_epochs: int = 2000,
    solve_FE: bool = True,
    clean_dir: bool = False,
    use_fno_warmstart: bool = False,
    skip_training: bool = False,
    pkl_path: str = "fourier_control_dict.pkl",
    train_ids: str = "0-150",
    test_ids: str = "150-200",
    default_label: str = "fourier_control",
    out_root: str = "PHASE_0_PARAMETRIC_Fourier",
    batch_size: int = 16,
    test_frequency: int = 100,
    save_frequency: int = 100,
):
    out_root = os.path.abspath(os.path.expanduser(out_root))
    os.makedirs(out_root, exist_ok=True)

    if skip_training and not solve_FE:
        raise ValueError("skip_training=True requires solve_FE=True")

    effective_use_fno_warmstart = bool(use_fno_warmstart and not skip_training)
    if skip_training and use_fno_warmstart:
        print("[PHASE_0_PARAMETRIC] skip_training=True -> ignoring use_fno_warmstart and using zero initial DOFs.")

    master_stats_csv = os.path.join(out_root, MASTER_STATS_NAME)

    STAT_FIELDS = [
        "phase",
        "sample_id",
        "label",
        "pkl_path",
        "case_dir",
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
        "skip_training",
        "use_fno_warmstart",
        "newton_total_iters",
        "newton_final_residual",
        "fe_time_s",
        "ux_rms", "ux_max",
        "uy_rms", "uy_max",
        "uv_rms", "uv_max",
    ]

    # ============================================================
    # 1) load PKL
    # ============================================================
    pkl_path = os.path.abspath(os.path.expanduser(pkl_path))
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"[PHASE_0_PARAMETRIC] pkl not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    # ============================================================
    # 2) problem setup (same as OTF)
    # ============================================================
    model_settings = {
        "L": 1.0,
        "N": 42,
        "Ux_left": 0.0,
        "Ux_right": 0.5,
        "Uy_left": 0.0,
        "Uy_right": 0.5,
    }

    L = float(model_settings["L"])
    N = int(model_settings["N"])

    fe_mesh = create_2D_square_mesh(L=L, N=N)

    bc_dict = {
        "Ux": {"left": model_settings["Ux_left"], "right": model_settings["Ux_right"]},
        "Uy": {"left": model_settings["Uy_left"], "right": model_settings["Uy_right"]},
    }

    material_dict = {"young_modulus": 1.0, "poisson_ratio": 0.3}
    mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad(
        "mechanical_loss_2d",
        loss_settings={"dirichlet_bc_dict": bc_dict, "num_gp": 2, "material_dict": material_dict},
        fe_mesh=fe_mesh,
    )

    fe_mesh.Initialize()
    mechanical_loss_2d.Initialize()

    n_nodes = fe_mesh.GetNumberOfNodes()
    if n_nodes != N * N:
        raise ValueError(f"Mesh nodes mismatch: fe_mesh has {n_nodes} nodes but N*N={N*N}")

    # ============================================================
    # 3) K_matrix from pkl OR coeffs_matrix→FourierControl
    # ============================================================
    if "K_matrix" in d:
        K_matrix = np.asarray(d["K_matrix"], dtype=np.float32)
        print(f"[PHASE_0_PARAMETRIC] Found K_matrix in pkl: {K_matrix.shape}")

    elif "coeffs_matrix" in d:
        coeffs_matrix = np.asarray(d["coeffs_matrix"], dtype=np.float32)
        print(f"[PHASE_0_PARAMETRIC] Found coeffs_matrix in pkl: {coeffs_matrix.shape}")

        x_freqs = np.asarray(d.get("x_freqs", np.array([2, 4, 6])))
        y_freqs = np.asarray(d.get("y_freqs", np.array([2, 4, 6])))
        z_freqs = np.asarray(d.get("z_freqs", np.array([0])))
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
        print(f"[PHASE_0_PARAMETRIC] Computed K_matrix from coeffs_matrix: {K_matrix.shape}")

    else:
        raise KeyError("[PHASE_0_PARAMETRIC] pkl must contain either 'K_matrix' or 'coeffs_matrix'")

    if K_matrix.ndim != 2:
        raise ValueError(f"[PHASE_0_PARAMETRIC] K_matrix must be 2D, got {K_matrix.shape}")

    num_samples, k_nodes = K_matrix.shape
    if k_nodes != n_nodes:
        raise ValueError(f"[PHASE_0_PARAMETRIC] K_matrix has {k_nodes} nodes, expected {n_nodes}. Check N mismatch.")

    labels_vec = None
    for key in ["labels", "sample_labels", "source_labels", "types"]:
        if key in d:
            labels_vec = list(d[key])
            if len(labels_vec) != num_samples:
                labels_vec = None
            break

    train_list = _parse_ids(train_ids, num_samples)
    test_list = _parse_ids(test_ids, num_samples)

    label = _sanitize_label(default_label)

    print(f"[PHASE_0_PARAMETRIC] out_root={out_root}")
    print(f"[PHASE_0_PARAMETRIC] train_ids={train_ids} -> {len(train_list)} samples")
    print(f"[PHASE_0_PARAMETRIC] test_ids={test_ids}  -> {len(test_list)} samples")

    pi_fno_param = None
    if not skip_training:
        # ============================================================
        # 4) training folder (single) — OTF-style naming
        # ============================================================
        train_case_dir = os.path.join(out_root, f"PHASE_0_PARAMETRIC_train_{label}")
        _clean_mkdir(train_case_dir)
        sys.stdout = Logger(os.path.join(train_case_dir, f"PHASE_0_PARAMETRIC_train_{label}.log"))

        X_train = K_matrix[train_list, :].astype(np.float32)
        X_test  = K_matrix[test_list, :].astype(np.float32)

        # ============================================================
        # 5) build model (same architecture as OTF)
        # ============================================================
        fno_model = bridge.ToNNX(
            FourierNeuralOperator2D(
                modes1=6,
                modes2=6,
                width=8,
                depth=4,
                channels_last_proj=32,
                out_channels=2,
                output_scale=0.1,
            ),
            rngs=nnx.Rngs(0),
        ).lazy_init(X_train[0].reshape(1, N, N, 1))

        params = nnx.state(fno_model, nnx.Param)
        total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
        print(f"[PHASE_0_PARAMETRIC] total number of FNO network parameters: {total_params}")

        adam_optimizer = optax.chain(optax.adam(1e-3))
        identity_control = IdentityControl("identity_control", {"num_vars": n_nodes}, fe_mesh)

        pi_fno_param = PhysicsInformedFourierParametricOperatorLearning(
            name="pi_fno_parametric_identity",
            control=identity_control,
            loss_function=mechanical_loss_2d,
            flax_neural_network=fno_model,
            optax_optimizer=adam_optimizer,
        )
        pi_fno_param.Initialize()

        # ============================================================
        # 6) PARAMETRIC TRAIN
        # ============================================================
        train_tag = f"parametric_identity_{label}"
        train_res_tracker = TrainingResidualTracker(out_dir=train_case_dir, tag=train_tag)

        print("[PHASE_0_PARAMETRIC] Starting parametric training...")
        t_train0 = time.time()

        pi_fno_param.Train(
            train_set=(jnp.asarray(X_train),),
            test_set=(jnp.asarray(X_test),),
            test_frequency=int(test_frequency),
            batch_size=int(batch_size),
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

        # restore best checkpoint (same behavior as OTF)
        pi_fno_param.RestoreState(restore_state_directory=os.path.join(train_case_dir, "flax_train_state"))
        print("[PHASE_0_PARAMETRIC] Restored best parametric checkpoint from flax_train_state.")

        # training stats summary
        train_csv = os.path.join(train_case_dir, f"{train_tag}_residual_rms.csv")
        if not os.path.exists(train_csv):
            train_csv = _find_first_existing([os.path.join(train_case_dir, "*residual_rms*.csv")])
        train_stats = _summarize_training(train_csv, residual_target=RESIDUAL_TARGET)
    else:
        print("[PHASE_0_PARAMETRIC] skip_training=True -> skipping FNO build/train/predict.")
        train_time_s = np.nan
        train_stats = {
            "epochs_completed": np.nan,
            "last_epoch": np.nan,
            "final_residual": np.nan,
            "min_residual": np.nan,
            "epoch_at_min": np.nan,
            "epoch_first_below_target": np.nan,
            "final_total_loss": np.nan,
        }

    # ============================================================
    # 7) EVALUATE test_list -> per-sample folders like OTF
    # ============================================================
    for sample_id in test_list:
        sample_label = _sanitize_label(labels_vec[sample_id]) if labels_vec is not None else label
        case_dir = os.path.join(out_root, f"PHASE_0_PARAMETRIC_sample_{sample_id}_{sample_label}")
        _clean_mkdir(case_dir)
        sys.stdout = Logger(os.path.join(case_dir, f"PHASE_0_PARAMETRIC_sample_{sample_id}_{sample_label}.log"))

        print(f"[PHASE_0_PARAMETRIC] sample_id={sample_id}, label={sample_label}")
        print(f"[PHASE_0_PARAMETRIC] pkl_path={pkl_path}")
        print(f"[PHASE_0_PARAMETRIC] output_dir={case_dir}")

        K_vec = K_matrix[sample_id, :].astype(np.float32)

        plt.figure(figsize=(4, 4))
        plt.imshow(K_vec.reshape(N, N), origin="lower")
        plt.title(f"Heterogeneity (sid={sample_id})")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, f"heterogeneity_sample_{sample_id}.png"), dpi=200)
        plt.close()

        # export fields (use a fresh mesh to avoid accumulating fields across samples)
        export_mesh = create_2D_square_mesh(L=L, N=N)
        export_mesh.Initialize()
        export_mesh[f"K_field_{sample_id}_param"] = K_vec.reshape((n_nodes,))

        FNO_UV = None
        if not skip_training:
            FNO_UV = np.array(pi_fno_param.Predict(K_vec.reshape(1, -1))).reshape(-1)
            export_mesh[f"U_FNO_{sample_id}_param"] = FNO_UV.reshape((n_nodes, 2))

        newton_total_iters = ""
        newton_final_residual = ""
        fe_time_s = ""

        ux_rms = np.nan; ux_max = np.nan
        uy_rms = np.nan; uy_max = np.nan
        uv_rms = np.nan; uv_max = np.nan

        if solve_FE:
            fe_setting = {
                "linear_solver_settings": {
                    "solver": "JAX-direct",
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
                mechanical_loss_2d,
                fe_setting,
            )
            nonlin_fe_solver.Initialize()

            ndofs = 2 * n_nodes
            if skip_training:
                initial_dofs = np.zeros(ndofs)
            else:
                initial_dofs = FNO_UV.copy() if effective_use_fno_warmstart else np.zeros(ndofs)

            t_fe0 = time.time()
            FE_UV_jax, residuals, total_iters = custom_newton_solve(
                fe_solver=nonlin_fe_solver,
                control_vars=K_vec,
                initial_dofs=initial_dofs,
                case_dir=case_dir,
                sample_tag=f"sample_{sample_id}_{sample_label}_param",
            )
            fe_time_s = time.time() - t_fe0

            FE_UV = np.array(FE_UV_jax).reshape(-1)
            export_mesh[f"U_FE_{sample_id}_param"] = FE_UV.reshape((n_nodes, 2))

            newton_total_iters = int(total_iters)
            newton_final_residual = float(residuals[-1]) if (residuals is not None and len(residuals) > 0) else ""

            iters = np.arange(1, len(residuals) + 1)
            plt.figure(figsize=(6, 4))
            plt.semilogy(iters, residuals, marker="o")
            plt.xlabel("Global Newton iteration")
            plt.ylabel(r"RMS residual $\|r\|_{\mathrm{rms}}$")
            plt.title(f"Newton convergence – sample {sample_id} (PARAM)")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, f"newton_residuals_sample_{sample_id}_param.png"))
            plt.close()

            if skip_training:
                plot_mesh_vec_data(
                    L,
                    [K_vec, FE_UV[0::2]],
                    subplot_titles=["Heterogeneity", "FE_Ux"],
                    fig_title=None,
                    cmap="viridis",
                    block_bool=True,
                    colour_bar=True,
                    colour_bar_name=None,
                    X_axis_name=None,
                    Y_axis_name=None,
                    show=False,
                    file_name=os.path.join(case_dir, f"plot_results_sample_{sample_id}_param.png"),
                )
            else:
                abs_err_uv = np.abs(FNO_UV - FE_UV)
                abs_err_ux = abs_err_uv[0::2]
                abs_err_uy = abs_err_uv[1::2]

                export_mesh[f"abs_error_{sample_id}_param"] = abs_err_uv.reshape((n_nodes, 2))
                export_mesh[f"abs_error_ux_{sample_id}_param"] = abs_err_ux.reshape((n_nodes,))
                export_mesh[f"abs_error_uy_{sample_id}_param"] = abs_err_uy.reshape((n_nodes,))

                ux_rms = float(np.sqrt(np.mean(abs_err_ux**2)))
                ux_max = float(np.max(abs_err_ux))
                uy_rms = float(np.sqrt(np.mean(abs_err_uy**2)))
                uy_max = float(np.max(abs_err_uy))

                ev = np.sqrt((FNO_UV[0::2] - FE_UV[0::2])**2 + (FNO_UV[1::2] - FE_UV[1::2])**2)
                uv_rms = float(np.sqrt(np.mean(ev**2)))
                uv_max = float(np.max(ev))

                plot_mesh_vec_data(
                    L,
                    [K_vec, FNO_UV[0::2], FE_UV[0::2], abs_err_ux],
                    subplot_titles=["Heterogeneity", "FNO_Ux", "FE_Ux", "absolute_error_Ux"],
                    fig_title=None,
                    cmap="viridis",
                    block_bool=True,
                    colour_bar=True,
                    colour_bar_name=None,
                    X_axis_name=None,
                    Y_axis_name=None,
                    show=False,
                    file_name=os.path.join(case_dir, f"plot_results_sample_{sample_id}_param.png"),
                )
        else:
            if skip_training:
                raise ValueError("skip_training=True requires solve_FE=True")
            plot_mesh_vec_data(
                L,
                [K_vec, FNO_UV[0::2]],
                subplot_titles=["Heterogeneity", "FNO_Ux"],
                fig_title=None,
                cmap="viridis",
                block_bool=True,
                colour_bar=True,
                colour_bar_name=None,
                X_axis_name=None,
                Y_axis_name=None,
                show=False,
                file_name=os.path.join(case_dir, f"plot_results_sample_{sample_id}_param.png"),
            )

        export_mesh.Finalize(export_dir=case_dir)

        row = {
            "phase": "PHASE_FE_ONLY" if skip_training else "PHASE_0_PARAMETRIC",
            "sample_id": int(sample_id),
            "label": sample_label,
            "pkl_path": pkl_path,
            "case_dir": case_dir,
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
            "batch_size": int(batch_size),
            "solve_FE": bool(solve_FE),
            "skip_training": bool(skip_training),
            "use_fno_warmstart": bool(effective_use_fno_warmstart),
            "newton_total_iters": newton_total_iters,
            "newton_final_residual": newton_final_residual,
            "fe_time_s": fe_time_s,
            "ux_rms": ux_rms,
            "ux_max": ux_max,
            "uy_rms": uy_rms,
            "uy_max": uy_max,
            "uv_rms": uv_rms,
            "uv_max": uv_max,
        }

        per_sample_csv = os.path.join(case_dir, f"stats_param_sample_{sample_id}.csv")
        _write_single_row_csv(per_sample_csv, STAT_FIELDS, row)
        _append_row_csv(master_stats_csv, STAT_FIELDS, row)

        print(f"[PHASE_0_PARAMETRIC] Wrote per-sample stats: {per_sample_csv}")
        print(f"[PHASE_0_PARAMETRIC] Appended master stats: {master_stats_csv}")

        if clean_dir:
            shutil.rmtree(case_dir)
            print(f"[PHASE_0_PARAMETRIC] Cleaned directory {case_dir}.")


if __name__ == "__main__":
    fol_num_epochs = 500
    solve_FE = True
    clean_dir = False
    use_fno_warmstart = False
    skip_training = False

    batch_size = 16
    test_frequency = 10
    save_frequency = 10

    script_dir = Path(__file__).resolve().parent
    pkl_path = str(script_dir / "fourier_control_dict.pkl")

    train_ids = "0-10"
    test_ids = "11-19"
    default_label = "fourier_control"
    out_root = str(script_dir / "PHASE_0_PARAMETRIC_2D_Fourier_strain_25")

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
        elif arg.startswith("skip_training="):
            skip_training = arg.split("=")[1].lower() == "true"
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
            default_label = arg.split("=", 1)[1]
        elif arg.startswith("out_root="):
            out_root = arg.split("=", 1)[1]
        else:
            print(
                "Usage:\n"
                "  python PHASE_0_PARAMETRIC_NEW.py "
                "fol_num_epochs=5000 batch_size=16 "
                "train_ids=0-150 test_ids=150-200 "
                "solve_FE=True skip_training=False use_fno_warmstart=False "
                "pkl_path=/path/to/fourier_control_dict.pkl "
                "out_root=/path/to/out\n"
            )
            sys.exit(1)

    main(
        fol_num_epochs=fol_num_epochs,
        solve_FE=solve_FE,
        clean_dir=clean_dir,
        use_fno_warmstart=use_fno_warmstart,
        skip_training=skip_training,
        pkl_path=pkl_path,
        train_ids=train_ids,
        test_ids=test_ids,
        default_label=default_label,
        out_root=out_root,
        batch_size=batch_size,
        test_frequency=test_frequency,
        save_frequency=save_frequency,
    )