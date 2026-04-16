"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: August, 2025
 License: FOL/LICENSE

 iFOL (Meta-Alpha-Meta-Implicit Parametric Operator Learning) script
 transformed to follow the PI-FNO 2D_PARAMETRIC.py shell structure.

 Core Architecture (iFOL):
   - MetaAlphaMetaImplicitParametricOperatorLearning wrapper
   - HyperNetwork (MLP synthesizer + MLP modulator)
   - FourierControl (trains/predicts on coeffs_matrix)
   - Two optimizers: main_loop + latent_step (both Adam)
   - latent_step_size, num_latent_iterations

 Shell Structure (PI-FNO):
   - Typed main() with CLI arg parsing
   - out_root with training folder + per-sample eval folders
   - PKL loading, FourierControl setup
   - FE comparison via custom_newton_solve
   - Per-sample + master stats CSV, VTU export
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import optax
from functools import partial
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pickle
import csv
import time
import glob

from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import (
    MetaAlphaMetaImplicitParametricOperatorLearning,
)
from fol.deep_neural_networks.nns import HyperNetwork, MLP
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.tools.decoration_functions import *

from fol.tools.newton_residual_tracker import custom_newton_solve
from training_residual_tracker import TrainingResidualTracker

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

jax.config.update("jax_default_matmul_precision", "high")

# ---------------------------------------------------------------------------
# Monkey-patch MetaAlphaMetaImplicit: add missing TrainStepMetrics/TestStepMetrics
# ---------------------------------------------------------------------------
# The base DeepNetwork.TrainStepMetrics does `nn, opt = state` which crashes for
# MetaAlphaMetaImplicit's 4-tuple state (nn, opt, latent_model, latent_opt).
# The class only overrides TrainStep/TestStep but not the *Metrics variants
# that the Train() scan-loop requires.  We patch them here at the class level.

@partial(nnx.jit, static_argnums=(0,))
def _meta_alpha_train_step_metrics(self, meta_state, data):
    nn_model, main_optimizer, latent_step_model, latent_optimizer = meta_state
    (_, batch_dict), meta_grads = nnx.value_and_grad(
        self.ComputeBatchLossValue, argnums=1, has_aux=True
    )(data, (nn_model, latent_step_model))
    main_optimizer.update(nn_model, meta_grads[0])
    latent_optimizer.update(latent_step_model, meta_grads[1])
    return batch_dict

@partial(nnx.jit, static_argnums=(0,))
def _meta_alpha_test_step_metrics(self, meta_state, data):
    nn_model, _, latent_step_model, _ = meta_state
    (_, batch_dict) = self.ComputeBatchLossValue(data, (nn_model, latent_step_model))
    return batch_dict

MetaAlphaMetaImplicitParametricOperatorLearning.TrainStepMetrics = _meta_alpha_train_step_metrics
MetaAlphaMetaImplicitParametricOperatorLearning.TestStepMetrics = _meta_alpha_test_step_metrics

# ---------------------------------------------------------------------------
# Patch ComputeBatchLossValue: add residual_rms_batch_mean metric
# ---------------------------------------------------------------------------
# The parent class ImplicitParametricOperatorLearning.ComputeBatchLossValue only
# returns {loss_name}_min/max/avg + total_loss.  PI-FNO adds residual_rms_* by
# calling loss_function.ComputeResidualNorm per sample.  We do the same here so
# that the training history plot, convergence criterion, and
# TrainingResidualTracker all have the residual_rms_batch_mean metric.

_original_compute_batch_loss = MetaAlphaMetaImplicitParametricOperatorLearning.ComputeBatchLossValue

def _patched_compute_batch_loss_value(self, batch, nn_model):
    """Wraps the original to append residual_rms_* metrics."""
    # --- original computation ---
    control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
    batch_predictions = self.ComputeBatchPredictions(batch[0], nn_model)
    batch_loss, (batch_min, batch_max, batch_avg) = self.loss_function.ComputeBatchLoss(
        control_outputs, batch_predictions
    )
    loss_name = self.loss_function.GetName()

    # --- residual RMS per sample (same as PI-FNO) ---
    def residual_for_sample(ctrl_sample, dofs_sample):
        ctrl_flat = ctrl_sample.reshape(-1)
        dofs_flat = dofs_sample.reshape(-1)
        full_dof_vec = self.loss_function.GetFullDofVector(
            ctrl_flat[jnp.newaxis, :],
            dofs_flat[jnp.newaxis, :],
        )[0]
        return self.loss_function.ComputeResidualNorm(
            ctrl_flat, full_dof_vec, relative=True,
        )

    residual_rms_per_sample = jax.vmap(residual_for_sample)(control_outputs, batch_predictions)

    metrics = {
        loss_name + "_min": batch_min,
        loss_name + "_max": batch_max,
        loss_name + "_avg": batch_avg,
        "total_loss": batch_loss,
        "residual_rms_batch_mean": jnp.mean(residual_rms_per_sample),
        "residual_rms_batch_min": jnp.min(residual_rms_per_sample),
        "residual_rms_batch_max": jnp.max(residual_rms_per_sample),
    }
    return batch_loss, metrics

MetaAlphaMetaImplicitParametricOperatorLearning.ComputeBatchLossValue = _patched_compute_batch_loss_value
# ---------------------------------------------------------------------------

# =========================
RESIDUAL_TARGET = 1e-2
MASTER_STATS_NAME = "stats_ifol_parametric.csv"
# =========================


def main(
    ifol_num_epochs: int = 5000,
    solve_FE: bool = True,
    clean_dir: bool = False,
    skip_training: bool = False,
    pkl_path: str = "fourier_control_dict_ifol.pkl",
    train_ids: str = "0-150",
    test_ids: str = "150-200",
    default_label: str = "fourier_control",
    out_root: str = "iFOL_PARAMETRIC_Fourier",
    batch_size: int = 350,
    test_frequency: int = 100,
    save_frequency: int = 100,
    # iFOL-specific hyperparameters
    characteristic_length: int = 64,
    latent_step_size: float = 1e-2,
    num_latent_iterations: int = 3,
    main_lr: float = 1e-5,
    latent_lr: float = 1e-5,
):
    """
    iFOL parametric training + per-sample evaluation, following the
    PI-FNO 2D_PARAMETRIC.py shell structure.

    Key difference from PI-FNO:
      - Trains/predicts on coeffs_matrix (Fourier coefficients), NOT K_matrix.
      - K_matrix is computed from coeffs_matrix for FE comparison & plotting.
      - Uses ``total_loss`` as convergence criterion (no residual_rms_batch_mean).
      - TrainingResidualTracker is NOT used (iFOL never produces the metric it
        needs); train stats are filled from training logs.
    """
    out_root = os.path.abspath(os.path.expanduser(out_root))
    os.makedirs(out_root, exist_ok=True)

    if skip_training and not solve_FE:
        raise ValueError("skip_training=True requires solve_FE=True")

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
        "ifol_num_epochs_requested",
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
        raise FileNotFoundError(f"[iFOL_PARAMETRIC] pkl not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    # ============================================================
    # 2) problem setup
    # ============================================================
    model_settings = {
        "L": 1.0,
        "N": 41,
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
        loss_settings={"dirichlet_bc_dict": bc_dict, "material_dict": material_dict},
        fe_mesh=fe_mesh,
    )

    fe_mesh.Initialize()
    mechanical_loss_2d.Initialize()

    n_nodes = fe_mesh.GetNumberOfNodes()
    if n_nodes != N * N:
        raise ValueError(f"Mesh nodes mismatch: fe_mesh has {n_nodes} nodes but N*N={N*N}")

    # ============================================================
    # 3) coeffs_matrix + K_matrix from PKL via FourierControl
    # ============================================================
    # iFOL always needs coeffs_matrix for training/prediction.
    # K_matrix is derived for FE solve & plotting.
    if "coeffs_matrix" not in d:
        raise KeyError(
            "[iFOL_PARAMETRIC] pkl must contain 'coeffs_matrix'. "
            "iFOL trains on Fourier coefficients, not raw K fields."
        )

    coeffs_matrix = np.asarray(d["coeffs_matrix"], dtype=np.float32)
    print(f"[iFOL_PARAMETRIC] Found coeffs_matrix in pkl: {coeffs_matrix.shape}")

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

    # K_matrix for FE comparison & heterogeneity plots
    K_matrix_jax = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)
    K_matrix = np.asarray(K_matrix_jax, dtype=np.float32)
    print(f"[iFOL_PARAMETRIC] Computed K_matrix from coeffs_matrix: {K_matrix.shape}")

    num_samples, k_nodes = K_matrix.shape
    if k_nodes != n_nodes:
        raise ValueError(
            f"[iFOL_PARAMETRIC] K_matrix has {k_nodes} nodes, expected {n_nodes}. Check N mismatch."
        )

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

    print(f"[iFOL_PARAMETRIC] out_root={out_root}")
    print(f"[iFOL_PARAMETRIC] train_ids={train_ids} -> {len(train_list)} samples")
    print(f"[iFOL_PARAMETRIC] test_ids={test_ids}  -> {len(test_list)} samples")

    ifol_model = None
    if not skip_training:
        # ============================================================
        # 4) training folder (single)
        # ============================================================
        train_case_dir = os.path.join(out_root, f"iFOL_PARAMETRIC_train_{label}")
        _clean_mkdir(train_case_dir)
        sys.stdout = Logger(os.path.join(train_case_dir, f"iFOL_PARAMETRIC_train_{label}.log"))

        X_train_coeffs = coeffs_matrix[train_list, :].astype(np.float32)
        X_test_coeffs = coeffs_matrix[test_list, :].astype(np.float32)

        # ============================================================
        # 5) build iFOL model (HyperNetwork + MetaAlphaMetaImplicit)
        # ============================================================
        synthesizer_nn = MLP(
            name="synthesizer_nn",
            input_size=3,
            output_size=2,
            hidden_layers=[characteristic_length] * 4,
            activation_settings={
                "type": "sin",
                "prediction_gain": 30,
                "initialization_gain": 1.0,
            },
            skip_connections_settings={"active": False, "frequency": 1},
        )

        latent_size = 8 * characteristic_length
        modulator_nn = MLP(
            name="modulator_nn",
            input_size=latent_size,
            use_bias=False,
        )

        hyper_network = HyperNetwork(
            name="hyper_nn",
            modulator_nn=modulator_nn,
            synthesizer_nn=synthesizer_nn,
            coupling_settings={
                "modulator_to_synthesizer_coupling_mode": "one_modulator_per_synthesizer_layer",
            },
        )

        main_loop_transform = optax.chain(optax.adam(main_lr))
        latent_step_optimizer = optax.chain(optax.adam(latent_lr))

        ifol_model = MetaAlphaMetaImplicitParametricOperatorLearning(
            name="meta_implicit_fol",
            control=fourier_control,
            loss_function=mechanical_loss_2d,
            flax_neural_network=hyper_network,
            main_loop_optax_optimizer=main_loop_transform,
            latent_step_optax_optimizer=latent_step_optimizer,
            latent_step_size=latent_step_size,
            num_latent_iterations=num_latent_iterations,
        )
        ifol_model.Initialize()

        # ============================================================
        # 6) PARAMETRIC TRAIN
        # ============================================================
        train_tag = f"parametric_ifol_{label}"
        train_res_tracker = TrainingResidualTracker(out_dir=train_case_dir, tag=train_tag)

        print("[iFOL_PARAMETRIC] Starting parametric training...")
        t_train0 = time.time()

        ifol_model.Train(
            train_set=(jnp.asarray(X_train_coeffs),),
            test_set=(jnp.asarray(X_test_coeffs),),
            test_frequency=int(test_frequency),
            batch_size=int(batch_size),
            convergence_settings={
                "num_epochs": int(ifol_num_epochs),
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

        # restore best checkpoint
        # iFOL RestoreState expects {dir}/nn + {dir}/latent subdirectories
        ifol_model.RestoreState(
            restore_state_directory=os.path.join(train_case_dir, "flax_train_state")
        )
        print("[iFOL_PARAMETRIC] Restored best iFOL checkpoint from flax_train_state.")

        # training stats summary from residual tracker CSV
        train_csv = os.path.join(train_case_dir, f"{train_tag}_residual_rms.csv")
        if not os.path.exists(train_csv):
            train_csv = _find_first_existing([os.path.join(train_case_dir, "*residual_rms*.csv")])
        train_stats = _summarize_training(train_csv, residual_target=RESIDUAL_TARGET)
    else:
        print("[iFOL_PARAMETRIC] skip_training=True -> skipping iFOL build/train/predict.")
        train_time_s = np.nan
        train_stats = {
            "train_csv": None,
            "epochs_completed": np.nan,
            "last_epoch": np.nan,
            "final_residual": np.nan,
            "min_residual": np.nan,
            "epoch_at_min": np.nan,
            "epoch_first_below_target": np.nan,
            "final_total_loss": np.nan,
        }

    # ============================================================
    # 7) EVALUATE test_list -> per-sample folders (PI-FNO style)
    # ============================================================
    for sample_id in test_list:
        sample_label = _sanitize_label(labels_vec[sample_id]) if labels_vec is not None else label
        case_dir = os.path.join(out_root, f"iFOL_PARAMETRIC_sample_{sample_id}_{sample_label}")
        _clean_mkdir(case_dir)
        sys.stdout = Logger(
            os.path.join(case_dir, f"iFOL_PARAMETRIC_sample_{sample_id}_{sample_label}.log")
        )

        print(f"[iFOL_PARAMETRIC] sample_id={sample_id}, label={sample_label}")
        print(f"[iFOL_PARAMETRIC] pkl_path={pkl_path}")
        print(f"[iFOL_PARAMETRIC] output_dir={case_dir}")

        K_vec = K_matrix[sample_id, :].astype(np.float32)

        # heterogeneity plot
        plt.figure(figsize=(4, 4))
        plt.imshow(K_vec.reshape(N, N), origin="lower")
        plt.title(f"Heterogeneity (sid={sample_id})")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, f"heterogeneity_sample_{sample_id}.png"), dpi=200)
        plt.close()

        # export fields (fresh mesh per sample to avoid accumulating fields)
        export_mesh = create_2D_square_mesh(L=L, N=N)
        export_mesh.Initialize()
        export_mesh[f"K_field_{sample_id}_ifol"] = K_vec.reshape((n_nodes,))

        iFOL_UV = None
        if not skip_training:
            # iFOL predicts from coeffs_matrix (not K_matrix)
            iFOL_UV = np.array(
                ifol_model.Predict(coeffs_matrix[sample_id, :].reshape(-1, 1).T)
            ).reshape(-1)
            export_mesh[f"U_iFOL_{sample_id}_param"] = iFOL_UV.reshape((n_nodes, 2))

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
                f"nonlin_fe_solver_ifol_{sample_id}",
                mechanical_loss_2d,
                fe_setting,
            )
            nonlin_fe_solver.Initialize()

            ndofs = 2 * n_nodes
            if skip_training:
                initial_dofs = np.zeros(ndofs)
            else:
                initial_dofs = np.zeros(ndofs)

            t_fe0 = time.time()
            FE_UV_jax, residuals, total_iters = custom_newton_solve(
                fe_solver=nonlin_fe_solver,
                control_vars=K_vec,
                initial_dofs=initial_dofs,
                case_dir=case_dir,
                sample_tag=f"sample_{sample_id}_{sample_label}_ifol",
            )
            fe_time_s = time.time() - t_fe0

            FE_UV = np.array(FE_UV_jax).reshape(-1)
            export_mesh[f"U_FE_{sample_id}_ifol"] = FE_UV.reshape((n_nodes, 2))

            newton_total_iters = int(total_iters)
            newton_final_residual = (
                float(residuals[-1]) if (residuals is not None and len(residuals) > 0) else ""
            )

            # Newton convergence plot
            iters = np.arange(1, len(residuals) + 1)
            plt.figure(figsize=(6, 4))
            plt.semilogy(iters, residuals, marker="o")
            plt.xlabel("Global Newton iteration")
            plt.ylabel(r"RMS residual $\|r\|_{\mathrm{rms}}$")
            plt.title(f"Newton convergence – sample {sample_id} (iFOL)")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, f"newton_residuals_sample_{sample_id}_ifol.png"))
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
                    file_name=os.path.join(case_dir, f"plot_results_sample_{sample_id}_ifol.png"),
                )
            else:
                abs_err_uv = np.abs(iFOL_UV - FE_UV)
                abs_err_ux = abs_err_uv[0::2]
                abs_err_uy = abs_err_uv[1::2]

                export_mesh[f"abs_error_{sample_id}_ifol"] = abs_err_uv.reshape((n_nodes, 2))
                export_mesh[f"abs_error_ux_{sample_id}_ifol"] = abs_err_ux.reshape((n_nodes,))
                export_mesh[f"abs_error_uy_{sample_id}_ifol"] = abs_err_uy.reshape((n_nodes,))

                ux_rms = float(np.sqrt(np.mean(abs_err_ux**2)))
                ux_max = float(np.max(abs_err_ux))
                uy_rms = float(np.sqrt(np.mean(abs_err_uy**2)))
                uy_max = float(np.max(abs_err_uy))

                ev = np.sqrt(
                    (iFOL_UV[0::2] - FE_UV[0::2]) ** 2
                    + (iFOL_UV[1::2] - FE_UV[1::2]) ** 2
                )
                uv_rms = float(np.sqrt(np.mean(ev**2)))
                uv_max = float(np.max(ev))

                plot_mesh_vec_data(
                    L,
                    [K_vec, iFOL_UV[0::2], FE_UV[0::2], abs_err_ux],
                    subplot_titles=["Heterogeneity", "iFOL_Ux", "FE_Ux", "absolute_error_Ux"],
                    fig_title=None,
                    cmap="viridis",
                    block_bool=True,
                    colour_bar=True,
                    colour_bar_name=None,
                    X_axis_name=None,
                    Y_axis_name=None,
                    show=False,
                    file_name=os.path.join(case_dir, f"plot_results_sample_{sample_id}_ifol.png"),
                )
        else:
            if skip_training:
                raise ValueError("skip_training=True requires solve_FE=True")
            plot_mesh_vec_data(
                L,
                [K_vec, iFOL_UV[0::2]],
                subplot_titles=["Heterogeneity", "iFOL_Ux"],
                fig_title=None,
                cmap="viridis",
                block_bool=True,
                colour_bar=True,
                colour_bar_name=None,
                X_axis_name=None,
                Y_axis_name=None,
                show=False,
                file_name=os.path.join(case_dir, f"plot_results_sample_{sample_id}_ifol.png"),
            )

        export_mesh.Finalize(export_dir=case_dir)

        # ---- per-sample + master stats CSV (PI-FNO style) ----
        row = {
            "phase": "iFOL_FE_ONLY" if skip_training else "iFOL_PARAMETRIC",
            "sample_id": int(sample_id),
            "label": sample_label,
            "pkl_path": pkl_path,
            "case_dir": case_dir,
            "train_ids": train_ids,
            "test_ids": test_ids,
            "train_count": len(train_list),
            "test_count": len(test_list),
            "ifol_num_epochs_requested": int(ifol_num_epochs),
            "epochs_completed": train_stats["epochs_completed"],
            "last_epoch": train_stats["last_epoch"],
            "train_residual_final": train_stats["final_residual"],
            "train_residual_min": train_stats["min_residual"],
            "epoch_at_min": train_stats["epoch_at_min"],
            "epoch_first_below_target": train_stats["epoch_first_below_target"],
            "residual_target": float(RESIDUAL_TARGET),
            "final_total_loss": train_stats["final_total_loss"],
            "train_time_s": float(train_time_s) if not np.isnan(train_time_s) else "",
            "batch_size": int(batch_size),
            "solve_FE": bool(solve_FE),
            "skip_training": bool(skip_training),
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

        per_sample_csv = os.path.join(case_dir, f"stats_ifol_sample_{sample_id}.csv")
        _write_single_row_csv(per_sample_csv, STAT_FIELDS, row)
        _append_row_csv(master_stats_csv, STAT_FIELDS, row)

        print(f"[iFOL_PARAMETRIC] Wrote per-sample stats: {per_sample_csv}")
        print(f"[iFOL_PARAMETRIC] Appended master stats: {master_stats_csv}")

        if clean_dir:
            shutil.rmtree(case_dir)
            print(f"[iFOL_PARAMETRIC] Cleaned directory {case_dir}.")


if __name__ == "__main__":
    ifol_num_epochs = 5000
    solve_FE = True
    clean_dir = False
    skip_training = False

    batch_size = 350
    test_frequency = 100
    save_frequency = 100

    # iFOL-specific hyperparameters
    characteristic_length = 64
    latent_step_size = 1e-2
    num_latent_iterations = 3
    main_lr = 1e-5
    latent_lr = 1e-5

    script_dir = Path(__file__).resolve().parent
    pkl_path = str(script_dir / "fourier_control_dict.pkl")

    train_ids = "0-80"
    test_ids = "78-79"
    default_label = "fourier_control"
    out_root = str(script_dir / "iFOL_PARAMETRIC_2D_Fourier")

    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("ifol_num_epochs="):
            ifol_num_epochs = int(arg.split("=")[1])
        elif arg.startswith("solve_FE="):
            solve_FE = arg.split("=")[1].lower() == "true"
        elif arg.startswith("clean_dir="):
            clean_dir = arg.split("=")[1].lower() == "true"
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
        elif arg.startswith("characteristic_length="):
            characteristic_length = int(arg.split("=")[1])
        elif arg.startswith("latent_step_size="):
            latent_step_size = float(arg.split("=")[1])
        elif arg.startswith("num_latent_iterations="):
            num_latent_iterations = int(arg.split("=")[1])
        elif arg.startswith("main_lr="):
            main_lr = float(arg.split("=")[1])
        elif arg.startswith("latent_lr="):
            latent_lr = float(arg.split("=")[1])
        else:
            print(
                "Usage:\n"
                "  python meta_implicit_mechanical_2D_nonlin_pr_pifno_shell.py "
                "ifol_num_epochs=5000 batch_size=350 "
                "train_ids=0-150 test_ids=150-200 "
                "solve_FE=True skip_training=False "
                "pkl_path=/path/to/fourier_control_dict.pkl "
                "out_root=/path/to/out "
                "characteristic_length=64 latent_step_size=1e-2 "
                "num_latent_iterations=3 main_lr=1e-5 latent_lr=1e-5\n"
            )
            sys.exit(1)

    main(
        ifol_num_epochs=ifol_num_epochs,
        solve_FE=solve_FE,
        clean_dir=clean_dir,
        skip_training=skip_training,
        pkl_path=pkl_path,
        train_ids=train_ids,
        test_ids=test_ids,
        default_label=default_label,
        out_root=out_root,
        batch_size=batch_size,
        test_frequency=test_frequency,
        save_frequency=save_frequency,
        characteristic_length=characteristic_length,
        latent_step_size=latent_step_size,
        num_latent_iterations=num_latent_iterations,
        main_lr=main_lr,
        latent_lr=latent_lr,
    )
