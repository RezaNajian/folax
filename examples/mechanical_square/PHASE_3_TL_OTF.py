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

from newton_residual_tracker import custom_newton_solve
from training_residual_tracker import TrainingResidualTracker


# =========================
RESIDUAL_TARGET = 1e-4
MASTER_STATS_NAME = "stats_phase3_tl_otf.csv"
# =========================


def safe_rename(src: str, dst: str):
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        os.replace(src, dst)


def _parse_ids(ids_str: str, n: int):
    s = str(ids_str).strip().lower()
    if s in ["all", "*"]:
        return list(range(n))

    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a = int(a.strip())
            b = int(b.strip())
            if a > b:
                a, b = b, a
            for i in range(a, b + 1):
                if not (0 <= i < n):
                    raise ValueError(f"sample id {i} out of range [0, {n-1}]")
                out.append(i)
        else:
            i = int(tok)
            if not (0 <= i < n):
                raise ValueError(f"sample id {i} out of range [0, {n-1}]")
            out.append(i)

    if not out:
        raise ValueError("eval_ids parsed to empty list.")

    # de-dup preserve order
    seen = set()
    out2 = []
    for i in out:
        if i not in seen:
            out2.append(i)
            seen.add(i)
    return out2


def _sanitize_label(s: str):
    s = str(s).strip().lower()
    s = s.replace(" ", "_")
    s = "".join(ch for ch in s if (ch.isalnum() or ch in ["_", "-"]))
    return s or "pkl"


def _clean_mkdir(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _find_first_existing(patterns: list[str]) -> str | None:
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def _read_training_residual_csv(csv_path: str):
    if (csv_path is None) or (not os.path.exists(csv_path)):
        return None

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return None

    cols = list(rows[0].keys())
    epoch_col = "epoch" if "epoch" in cols else cols[0]

    residual_candidates = [
        "residual_rms_batch_mean",
        "residual_rms",
        "residual",
    ]
    residual_col = None
    for c in residual_candidates:
        if c in cols:
            residual_col = c
            break
    if residual_col is None:
        for c in cols:
            cl = c.lower()
            if ("residual" in cl) and ("rms" in cl):
                residual_col = c
                break
    if residual_col is None:
        return None

    total_loss_col = "total_loss" if "total_loss" in cols else None

    epoch = np.array([float(r[epoch_col]) for r in rows], dtype=float)
    resid = np.array([float(r[residual_col]) for r in rows], dtype=float)
    tloss = None
    if total_loss_col is not None:
        tloss = np.array([float(r[total_loss_col]) for r in rows], dtype=float)

    return epoch, resid, tloss


def _summarize_training(csv_path: str, residual_target: float):
    out = {
        "train_csv": csv_path,
        "epochs_completed": np.nan,
        "last_epoch": np.nan,
        "final_residual": np.nan,
        "min_residual": np.nan,
        "epoch_at_min": np.nan,
        "epoch_first_below_target": np.nan,
        "final_total_loss": np.nan,
    }

    data = _read_training_residual_csv(csv_path)
    if data is None:
        return out

    epoch, resid, tloss = data
    if len(epoch) == 0:
        return out

    out["epochs_completed"] = int(epoch[-1] + 1)
    out["last_epoch"] = int(epoch[-1])
    out["final_residual"] = float(resid[-1])
    out["min_residual"] = float(np.min(resid))
    out["epoch_at_min"] = float(epoch[int(np.argmin(resid))])

    idx = np.where(resid <= residual_target)[0]
    if len(idx) > 0:
        out["epoch_first_below_target"] = float(epoch[int(idx[0])])

    if tloss is not None and len(tloss) == len(epoch):
        out["final_total_loss"] = float(tloss[-1])

    return out


def _append_row_csv(csv_path: str, fieldnames: list[str], row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_single_row_csv(csv_path: str, fieldnames: list[str], row: dict):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _copy_rms_plot_as_training_history(case_dir: str, sample_id: int, tag: str):
    """
    TL-OTF training image = tracker RMS plot (best for stitchers).
    Copy to training_history_sample_<id>.png for consistency with OTF tooling.
    """
    dst = os.path.join(case_dir, f"training_history_sample_{sample_id}.png")

    preferred = os.path.join(case_dir, f"{tag}_residual_rms.png")
    if os.path.exists(preferred):
        shutil.copyfile(preferred, dst)
        return

    candidates = sorted(glob.glob(os.path.join(case_dir, "*residual_rms*.png")))
    if candidates:
        shutil.copyfile(candidates[0], dst)


def _build_fno_model(N: int, K_vec_for_lazy_init: np.ndarray):
    """
    Build FNO same as OTF script (with RNG-key patch for checkpoint safety).
    """
    def merge_state(dst: nnx.State, src: nnx.State):
        for k, v in src.items():
            if isinstance(v, nnx.State):
                merge_state(dst[k], v)
            else:
                dst[k] = v

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
    ).lazy_init(K_vec_for_lazy_init.reshape(1, N, N, 1))

    graph_def, state = nnx.split(fno_model)
    rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
    merge_state(state, rngs_key)
    fno_model = nnx.merge(graph_def, state)
    return fno_model


def main(
    fol_num_epochs: int = 20000,
    solve_FE: bool = True,
    clean_dir: bool = False,
    use_fno_warmstart: bool = False,

    pkl_path: str = "fourier_control_dict.pkl",
    train_ids: str = "0-80",
    test_ids: str = "81-99",
    default_label: str = "fourier_control",

    out_root: str = "PHASE_3_TL_OTF_Fourier",

    # IMPORTANT: point to your parametric training folder (the one that contains flax_train_state/)
    param_case_dir: str = "PHASE_0_PARAMETRIC_Fourier/PHASE_0_PARAMETRIC_train_fourier_control",
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
        "param_case_dir",
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
        "solve_FE",
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
        raise FileNotFoundError(f"[PHASE_3_TL_OTF] pkl not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    # ============================================================
    # 2) problem setup (MUST match OTF/parametric setup)
    # ============================================================
    model_settings = {
        "L": 1.0,
        "N": 42,
        "Ux_left": 0.0,
        "Ux_right": 0.1,
        "Uy_left": 0.0,
        "Uy_right": 0.1,
    }

    L = float(model_settings["L"])
    N = int(model_settings["N"])

    fe_mesh_base = create_2D_square_mesh(L=L, N=N)
    fe_mesh_base.Initialize()
    n_nodes = fe_mesh_base.GetNumberOfNodes()
    if n_nodes != N * N:
        raise ValueError(f"Mesh nodes mismatch: fe_mesh has {n_nodes} nodes but N*N={N*N}")

    bc_dict = {
        "Ux": {"left": model_settings["Ux_left"], "right": model_settings["Ux_right"]},
        "Uy": {"left": model_settings["Uy_left"], "right": model_settings["Uy_right"]},
    }
    material_dict = {"young_modulus": 1.0, "poisson_ratio": 0.3}

    # ============================================================
    # 3) get K_matrix from pkl (preferred) or compute from coeffs_matrix
    # ============================================================
    if "K_matrix" in d:
        K_matrix = np.asarray(d["K_matrix"], dtype=np.float32)
        print(f"[PHASE_3_TL_OTF] Found K_matrix in pkl: {K_matrix.shape}")

    elif "coeffs_matrix" in d:
        coeffs_matrix = np.asarray(d["coeffs_matrix"], dtype=np.float32)
        print(f"[PHASE_3_TL_OTF] Found coeffs_matrix in pkl: {coeffs_matrix.shape}")

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
        fourier_control = FourierControl("fourier_control", fourier_control_settings, fe_mesh_base)
        fourier_control.Initialize()

        K_matrix_jax = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)
        K_matrix = np.asarray(K_matrix_jax, dtype=np.float32)
        print(f"[PHASE_3_TL_OTF] Computed K_matrix from coeffs_matrix: {K_matrix.shape}")

    else:
        raise KeyError("[PHASE_3_TL_OTF] pkl must contain either 'K_matrix' or 'coeffs_matrix'")

    num_samples, k_nodes = K_matrix.shape
    if k_nodes != n_nodes:
        raise ValueError(f"[PHASE_3_TL_OTF] K_matrix has {k_nodes} nodes, expected {n_nodes}. Check N mismatch.")

    # optional labels
    labels_vec = None
    for key in ["labels", "sample_labels", "source_labels", "types"]:
        if key in d:
            labels_vec = list(d[key])
            if len(labels_vec) != num_samples:
                labels_vec = None
            break

    train_list = _parse_ids(train_ids, num_samples)
    test_list = _parse_ids(test_ids, num_samples)
    run_list = test_list
    print(f"[PHASE_3_TL_OTF] out_root={out_root}")
    print(f"[PHASE_3_TL_OTF] train_ids={train_ids} -> {len(train_list)} samples")
    print(f"[PHASE_3_TL_OTF] test_ids={test_ids} -> {len(test_list)} samples")
    print(f"[PHASE_3_TL_OTF] Will run TL-OTF for {len(run_list)} test samples: {run_list[:10]}{'...' if len(run_list)>10 else ''}")

    # ============================================================
    # 4) parametric checkpoint dir (expects flax_train_state/)
    # ============================================================
    param_case_dir = os.path.abspath(os.path.expanduser(param_case_dir))
    param_state_dir = os.path.join(param_case_dir, "flax_train_state")
    if not os.path.isdir(param_state_dir):
        raise FileNotFoundError(f"[PHASE_3_TL_OTF] param flax_train_state not found: {param_state_dir}")

    # ============================================================
    # 5) per-sample TL-OTF loop
    # ============================================================
    for sample_id in run_list:
        label = _sanitize_label(labels_vec[sample_id]) if labels_vec is not None else _sanitize_label(default_label)

        working_directory_name = f"PHASE_3_TL_OTF_sample_{sample_id}_{label}"
        case_dir = os.path.join(out_root, working_directory_name)

        _clean_mkdir(case_dir)
        sys.stdout = Logger(os.path.join(case_dir, f"{working_directory_name}.log"))

        print(f"[PHASE_3_TL_OTF] sample_id={sample_id}, label={label}")
        print(f"[PHASE_3_TL_OTF] pkl_path={pkl_path}")
        print(f"[PHASE_3_TL_OTF] param_case_dir={param_case_dir}")
        print(f"[PHASE_3_TL_OTF] output_dir={case_dir}")

        # fresh mesh + loss per sample (avoids state bleed)
        fe_mesh = create_2D_square_mesh(L=L, N=N)
        mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad(
            "mechanical_loss_2d",
            loss_settings={"dirichlet_bc_dict": bc_dict, "num_gp": 2, "material_dict": material_dict},
            fe_mesh=fe_mesh,
        )
        fe_mesh.Initialize()
        mechanical_loss_2d.Initialize()

        # sample K
        K_vec = K_matrix[sample_id, :].astype(np.float32)

        # heterogeneity image
        plt.figure(figsize=(4, 4))
        plt.imshow(K_vec.reshape(N, N), origin="lower")
        plt.title(f"Heterogeneity (sid={sample_id})")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, f"heterogeneity_sample_{sample_id}.png"), dpi=200)
        plt.close()

        # build model (same as OTF) + wrapper
        fno_model = _build_fno_model(N, K_vec)

        params = nnx.state(fno_model, nnx.Param)
        total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
        print(f"[PHASE_3_TL_OTF] total number of FNO network parameters: {total_params}")

        adam_optimizer = optax.chain(optax.adam(1e-3))
        identity_control = IdentityControl("identity_control", {"num_vars": n_nodes}, fe_mesh)

        pi_fno_tl_otf = PhysicsInformedFourierParametricOperatorLearning(
            name="pi_fno_tl_otf_identity",
            control=identity_control,
            loss_function=mechanical_loss_2d,
            flax_neural_network=fno_model,
            optax_optimizer=adam_optimizer,
        )
        pi_fno_tl_otf.Initialize()

        # ---- restore parametric weights before OTF fine-tune
        pi_fno_tl_otf.RestoreState(restore_state_directory=param_state_dir)
        print("[PHASE_3_TL_OTF] Restored parametric checkpoint. Starting TL-OTF...")

        # TL-OTF tracker
        tl_tag = f"tl_otf_identity_{label}_sid{sample_id}"
        tl_res_tracker = TrainingResidualTracker(out_dir=case_dir, tag=tl_tag)

        # one-sample training
        train_set = K_vec.reshape(1, -1)
        test_set = K_vec.reshape(1, -1)

        t_train0 = time.time()
        pi_fno_tl_otf.Train(
            train_set=(train_set,),
            test_set=(test_set,),
            test_frequency=100,
            batch_size=1,
            convergence_settings={
                "num_epochs": int(fol_num_epochs),
                "convergence_criterion": "residual_rms_batch_mean",
                "relative_error": 1e-100,
                "absolute_error": 1e-4,
            },
            plot_settings={
                "plot_list": ["total_loss", "residual_rms_batch_mean"],
                "plot_frequency": 1,
                "save_frequency": 100,
                "save_directory": case_dir,
                "test_frequency": 100,
            },
            train_checkpoint_settings={
                "least_loss_checkpointing": True,
                "frequency": 100,
                "state_directory": os.path.join(case_dir, "flax_train_state"),
            },
            working_directory=case_dir,
            training_residual_tracker=tl_res_tracker,
        )
        train_time_s = time.time() - t_train0
        tl_res_tracker.finalize()

        # training-history alias (use RMS plot)
        _copy_rms_plot_as_training_history(case_dir, sample_id, tl_tag)

        # summarize TL training
        train_csv = os.path.join(case_dir, f"{tl_tag}_residual_rms.csv")
        if not os.path.exists(train_csv):
            train_csv = _find_first_existing([os.path.join(case_dir, "*residual_rms*.csv")])
        train_stats = _summarize_training(train_csv, residual_target=RESIDUAL_TARGET)

        # restore best TL checkpoint
        pi_fno_tl_otf.RestoreState(restore_state_directory=os.path.join(case_dir, "flax_train_state"))
        print("[PHASE_3_TL_OTF] Restored best TL-OTF checkpoint from this sample.")

        # predict
        FNO_UV = np.array(pi_fno_tl_otf.Predict(K_vec.reshape(1, -1))).reshape(-1)

        fe_mesh[f"U_FNO_{sample_id}_tl_otf"] = FNO_UV.reshape((n_nodes, 2))
        fe_mesh[f"K_field_{sample_id}_tl_otf"] = K_vec.reshape((n_nodes,))

        newton_total_iters = ""
        newton_final_residual = ""
        fe_time_s = ""

        ux_rms = np.nan; ux_max = np.nan
        uy_rms = np.nan; uy_max = np.nan
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
                f"nonlin_fe_solver_tl_otf_{sample_id}",
                mechanical_loss_2d,
                fe_setting,
            )
            nonlin_fe_solver.Initialize()

            ndofs = 2 * n_nodes
            initial_dofs = FNO_UV.copy() if use_fno_warmstart else np.zeros(ndofs)

            t_fe0 = time.time()
            FE_UV_jax, residuals, total_iters = custom_newton_solve(
                fe_solver=nonlin_fe_solver,
                control_vars=K_vec,
                initial_dofs=initial_dofs,
                case_dir=case_dir,
                sample_tag=f"sample_{sample_id}_{label}_tl_otf",
            )
            fe_time_s = time.time() - t_fe0

            FE_UV = np.array(FE_UV_jax).reshape(-1)
            fe_mesh[f"U_FE_{sample_id}_tl_otf"] = FE_UV.reshape((n_nodes, 2))

            newton_total_iters = int(total_iters)
            newton_final_residual = float(residuals[-1]) if (residuals is not None and len(residuals) > 0) else ""

            # Newton plot
            iters = np.arange(1, len(residuals) + 1)
            plt.figure(figsize=(6, 4))
            plt.semilogy(iters, residuals, marker="o")
            plt.xlabel("Global Newton iteration")
            plt.ylabel(r"RMS residual $\|r\|_{\mathrm{rms}}$")
            plt.title(f"Newton convergence – sample {sample_id} (TL-OTF)")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, f"newton_residuals_sample_{sample_id}_tl_otf.png"))
            plt.close()

            # absolute errors
            abs_err_uv = np.abs(FNO_UV - FE_UV)
            abs_err_ux = abs_err_uv[0::2]
            abs_err_uy = abs_err_uv[1::2]

            fe_mesh[f"abs_error_{sample_id}_tl_otf"] = abs_err_uv.reshape((n_nodes, 2))
            fe_mesh[f"abs_error_ux_{sample_id}_tl_otf"] = abs_err_ux.reshape((n_nodes,))
            fe_mesh[f"abs_error_uy_{sample_id}_tl_otf"] = abs_err_uy.reshape((n_nodes,))

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
                file_name=os.path.join(case_dir, f"plot_results_sample_{sample_id}_tl_otf.png"),
            )
        else:
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
                file_name=os.path.join(case_dir, f"plot_results_sample_{sample_id}_tl_otf.png"),
            )

        fe_mesh.Finalize(export_dir=case_dir)

        # write stats
        row = {
            "phase": "PHASE_3_TL_OTF",
            "sample_id": int(sample_id),
            "label": label,
            "pkl_path": pkl_path,
            "case_dir": case_dir,
            "param_case_dir": param_case_dir,
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
            "solve_FE": bool(solve_FE),
            "use_fno_warmstart": bool(use_fno_warmstart),
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

        per_sample_csv = os.path.join(case_dir, f"stats_tl_otf_sample_{sample_id}.csv")
        _write_single_row_csv(per_sample_csv, STAT_FIELDS, row)
        _append_row_csv(master_stats_csv, STAT_FIELDS, row)

        print(f"[PHASE_3_TL_OTF] Wrote per-sample stats: {per_sample_csv}")
        print(f"[PHASE_3_TL_OTF] Appended master stats: {master_stats_csv}")

        if clean_dir:
            shutil.rmtree(case_dir)
            print(f"[PHASE_3_TL_OTF] Cleaned directory {case_dir}.")


if __name__ == "__main__":
    fol_num_epochs = 20000
    solve_FE = True
    clean_dir = False
    use_fno_warmstart = False

    script_dir = Path(__file__).resolve().parent

    pkl_path = str(script_dir / "fourier_control_dict.pkl")
    train_ids = "0-80"
    test_ids = "81-99"
    default_label = "fourier_control"

    # IMPORTANT: this should be your parametric training folder that contains flax_train_state/
    # If you used my parametric script, it's:
    #   PHASE_0_PARAMETRIC_Fourier/PHASE_0_PARAMETRIC_train_fourier_control/
    param_case_dir = str(script_dir / "PHASE_0_PARAMETRIC_Fourier" / "PHASE_0_PARAMETRIC_train_fourier_control")

    out_root = str(script_dir / "PHASE_3_TL_OTF_Fourier")

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
        elif arg.startswith("pkl_path="):
            pkl_path = arg.split("=", 1)[1]
        elif arg.startswith("train_ids="):
            train_ids = arg.split("=", 1)[1]
        elif arg.startswith("test_ids="):
            test_ids = arg.split("=", 1)[1]
        elif arg.startswith("default_label="):
            default_label = arg.split("=", 1)[1]
        elif arg.startswith("param_case_dir="):
            param_case_dir = arg.split("=", 1)[1]
        elif arg.startswith("out_root="):
            out_root = arg.split("=", 1)[1]
        else:
            print(
                "Usage:\n"
                "  python PHASE_3_TL_OTF_NEW.py [options]\n\n"
                "Options:\n"
                "  fol_num_epochs=20000\n"
                "  solve_FE=True|False\n"
                "  use_fno_warmstart=True|False\n"
                "  pkl_path=/path/to/*.pkl\n"
                "  train_ids=all|0-10|0,1,5\n"
                "  test_ids=all|11-20|11,12,15\n"
                "  param_case_dir=/path/to/PHASE_0_PARAMETRIC_train_*/\n"
                "  out_root=/path/to/PHASE_3_TL_OTF_*/\n"
            )
            sys.exit(1)

    main(
        fol_num_epochs=fol_num_epochs,
        solve_FE=solve_FE,
        clean_dir=clean_dir,
        use_fno_warmstart=use_fno_warmstart,
        pkl_path=pkl_path,
        train_ids=train_ids,
        test_ids=test_ids,
        default_label=default_label,
        out_root=out_root,
        param_case_dir=param_case_dir,
    )