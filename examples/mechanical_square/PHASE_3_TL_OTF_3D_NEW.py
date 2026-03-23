#!/usr/bin/env python3
# ============================================================
# PHASE_3_TL_OTF_3D_NEW.py
#
# 3D TL-OTF version created by mapping the working 2D TL-OTF
# logic onto the working 3D OTF / 3D parametric backbone.
#
# Main behavior:
#   - sets up the 3D Neo-Hookean hexa problem
#   - loads (or optionally generates) the 3D Fourier PKL
#   - restores a pretrained 3D parametric checkpoint
#   - runs TL-OTF independently for each selected test sample
#   - restores the best per-sample TL checkpoint
#   - predicts, optionally runs FE, exports VTU, plots, and writes stats
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
        raise ValueError("ids parsed to empty list.")

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

    residual_candidates = ["residual_rms_batch_mean", "residual_rms", "residual"]
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
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
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


# ---------- 3D plotting helpers (mid-Z slice) ----------
def _midz_slice_scalar(vec_nodes: np.ndarray, Nx: int, Ny: int, Nz: int):
    vol = vec_nodes.reshape(Nx, Ny, Nz)
    k = Nz // 2
    return vol[:, :, k]


def _save_slice_png(vec_nodes: np.ndarray, Nx: int, Ny: int, Nz: int, out_path: str, title: str):
    sl = _midz_slice_scalar(vec_nodes, Nx, Ny, Nz)
    plt.figure(figsize=(4, 4))
    plt.imshow(sl.T, origin="lower")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_2x2_panel(K_vec, fno_ux, fe_ux, abs_err_ux, Nx, Ny, Nz, out_path):
    Ksl = _midz_slice_scalar(K_vec, Nx, Ny, Nz).T
    Fsl = _midz_slice_scalar(fno_ux, Nx, Ny, Nz).T
    Esl = _midz_slice_scalar(fe_ux, Nx, Ny, Nz).T
    Asl = _midz_slice_scalar(abs_err_ux, Nx, Ny, Nz).T

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs = axs.ravel()

    im0 = axs[0].imshow(Ksl, origin="lower")
    axs[0].set_title("Heterogeneity (mid-z)")
    fig.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(Fsl, origin="lower")
    axs[1].set_title("FNO_Ux (mid-z)")
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(Esl, origin="lower")
    axs[2].set_title("FE_Ux (mid-z)")
    fig.colorbar(im2, ax=axs[2], fraction=0.046)

    im3 = axs[3].imshow(Asl, origin="lower")
    axs[3].set_title("absolute_error_Ux")
    fig.colorbar(im3, ax=axs[3], fraction=0.046)

    for a in axs:
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------- NNX RNG compatibility patch ----------
def _patch_rng_keys(fno_model):
    def merge_state(dst: nnx.State, src: nnx.State):
        for k, v in src.items():
            if isinstance(v, nnx.State):
                merge_state(dst[k], v)
            else:
                dst[k] = v

    graph_def, state = nnx.split(fno_model)
    rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
    merge_state(state, rngs_key)
    return nnx.merge(graph_def, state)


# ---------- ensure .vtu export ----------
def _ensure_vtu_in_dir(export_dir: str):
    vtus = sorted(glob.glob(os.path.join(export_dir, "*.vtu")))
    if vtus:
        return

    vtks = sorted(glob.glob(os.path.join(export_dir, "*.vtk")))
    if not vtks:
        return

    try:
        import meshio
    except Exception:
        print("[EXPORT] WARN: meshio not available, cannot convert .vtk -> .vtu")
        return

    src = vtks[0]
    dst = os.path.splitext(src)[0] + ".vtu"
    try:
        m = meshio.read(src)
        meshio.write(dst, m, file_format="vtu")
        print(f"[EXPORT] Converted {os.path.basename(src)} -> {os.path.basename(dst)}")
    except Exception as e:
        print(f"[EXPORT] WARN: conversion failed: {e}")


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


def _build_fno_model_3d(Nx: int, Ny: int, Nz: int, K_vec_for_lazy_init: np.ndarray):
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
    ).lazy_init(K_vec_for_lazy_init.reshape(1, Nx, Ny, Nz, 1))

    fno_model = _patch_rng_keys(fno_model)
    return fno_model


def main(
    fol_num_epochs: int = 20000,
    solve_FE: bool = True,
    clean_dir: bool = False,
    use_fno_warmstart: bool = False,

    # PKL / sample generation controls
    create_new_samples: bool = False,
    num_samples: int = 200,
    random_seed: int = 0,
    pkl_path: str = "fourier_control_dict3D.pkl",

    # bookkeeping / selection
    train_ids: str = "0-47",
    test_ids: str = "48-49",
    default_label: str = "fourier_control3D",

    out_root: str = "PHASE_3_TL_OTF_3D_Fourier",

    # IMPORTANT: point to your 3D parametric training folder (contains flax_train_state/)
    param_case_dir: str = "PHASE_0_PARAMETRIC_3D_Fourier/PHASE_0_PARAMETRIC_train_fourier_control3d",
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
    # 1) base problem setup (3D)
    # ============================================================
    model_settings = {
        "L": 1.0,
        "N": 20,
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

    fe_mesh_base = create_3D_box_mesh_structured(Nx=Nx, Ny=Ny, Nz=Nz, Lx=L, Ly=L, Lz=L)
    fe_mesh_base.Initialize()

    n_nodes = fe_mesh_base.GetNumberOfNodes()
    if n_nodes != Nx * Ny * Nz:
        raise ValueError(f"Mesh nodes mismatch: fe_mesh has {n_nodes} nodes but Nx*Ny*Nz={Nx*Ny*Nz}")

    bc_dict = {
        "Ux": {"left": model_settings["Ux_left"], "right": model_settings["Ux_right"]},
        "Uy": {"left": model_settings["Uy_left"], "right": model_settings["Uy_right"]},
        "Uz": {"left": model_settings["Uz_left"], "right": model_settings["Uz_right"]},
    }
    material_dict = {"young_modulus": 1.0, "poisson_ratio": 0.3}

    # ============================================================
    # 2) load PKL OR generate PKL
    # ============================================================
    pkl_path = os.path.abspath(os.path.expanduser(pkl_path))

    if (not create_new_samples) and (not os.path.exists(pkl_path)):
        raise FileNotFoundError(
            f"[PHASE_3_TL_OTF_3D] pkl not found: {pkl_path}\n"
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
        fourier_control = FourierControl("fourier_control", fourier_control_settings, fe_mesh_base)
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

        print(f"[PHASE_3_TL_OTF_3D] Wrote NEW pkl: {pkl_path}")
        print(f"[PHASE_3_TL_OTF_3D] coeffs_matrix shape={coeffs_matrix.shape}, K_matrix shape={K_matrix.shape}")

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
        print(f"[PHASE_3_TL_OTF_3D] Found K_matrix in pkl: {K_matrix.shape}")

    elif "coeffs_matrix" in d:
        coeffs_matrix = np.asarray(d["coeffs_matrix"], dtype=np.float32)
        if num_samples is not None:
            coeffs_matrix = coeffs_matrix[: int(num_samples)]
        print(f"[PHASE_3_TL_OTF_3D] Found coeffs_matrix in pkl: {coeffs_matrix.shape}")

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
        fourier_control = FourierControl("fourier_control", fourier_control_settings, fe_mesh_base)
        fourier_control.Initialize()

        K_matrix_jax = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)
        K_matrix = np.asarray(K_matrix_jax, dtype=np.float32)
        print(f"[PHASE_3_TL_OTF_3D] Computed K_matrix from coeffs_matrix: {K_matrix.shape}")

    else:
        raise KeyError("[PHASE_3_TL_OTF_3D] pkl must contain either 'K_matrix' or 'coeffs_matrix'")

    if K_matrix.ndim != 2:
        raise ValueError(f"[PHASE_3_TL_OTF_3D] K_matrix must be 2D, got {K_matrix.shape}")

    num_samples_eff, k_nodes = K_matrix.shape
    if k_nodes != n_nodes:
        raise ValueError(
            f"[PHASE_3_TL_OTF_3D] K_matrix has {k_nodes} nodes, expected {n_nodes}. "
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
    run_list = test_list
    default_label = _sanitize_label(default_label)

    print(f"[PHASE_3_TL_OTF_3D] out_root={out_root}")
    print(f"[PHASE_3_TL_OTF_3D] train_ids={train_ids} -> {len(train_list)} samples")
    print(f"[PHASE_3_TL_OTF_3D] test_ids={test_ids}  -> {len(test_list)} samples")
    print(
        f"[PHASE_3_TL_OTF_3D] Will run TL-OTF for {len(run_list)} test samples: "
        f"{run_list[:10]}{'...' if len(run_list) > 10 else ''}"
    )

    # ============================================================
    # 4) parametric checkpoint dir (expects flax_train_state/)
    # ============================================================
    param_case_dir = os.path.abspath(os.path.expanduser(param_case_dir))
    param_state_dir = os.path.join(param_case_dir, "flax_train_state")
    if not os.path.isdir(param_state_dir):
        raise FileNotFoundError(f"[PHASE_3_TL_OTF_3D] param flax_train_state not found: {param_state_dir}")

    # ============================================================
    # 5) per-sample TL-OTF loop
    # ============================================================
    for sample_id in run_list:
        label = _sanitize_label(labels_vec[sample_id]) if labels_vec is not None else default_label

        case_dir = os.path.join(out_root, f"PHASE_3_TL_OTF_sample_{sample_id}_{label}")
        _clean_mkdir(case_dir)
        sys.stdout = Logger(os.path.join(case_dir, f"PHASE_3_TL_OTF_sample_{sample_id}_{label}.log"))

        print(f"[PHASE_3_TL_OTF_3D] sample_id={sample_id}, label={label}")
        print(f"[PHASE_3_TL_OTF_3D] pkl_path={pkl_path}")
        print(f"[PHASE_3_TL_OTF_3D] param_case_dir={param_case_dir}")
        print(f"[PHASE_3_TL_OTF_3D] output_dir={case_dir}")

        # fresh mesh + loss per sample (avoids state bleed)
        fe_mesh = create_3D_box_mesh_structured(Nx=Nx, Ny=Ny, Nz=Nz, Lx=L, Ly=L, Lz=L)
        mechanical_loss_3d = NeoHookeMechanicalLoss3DHexa(
            "mechanical_loss_3d",
            loss_settings={"dirichlet_bc_dict": bc_dict, "material_dict": material_dict},
            fe_mesh=fe_mesh,
        )
        fe_mesh.Initialize()
        mechanical_loss_3d.Initialize()

        K_vec = K_matrix[sample_id, :].astype(np.float32)

        _save_slice_png(
            K_vec,
            Nx,
            Ny,
            Nz,
            os.path.join(case_dir, f"heterogeneity_sample_{sample_id}.png"),
            title=f"Heterogeneity mid-z (sid={sample_id})",
        )

        fno_model = _build_fno_model_3d(Nx, Ny, Nz, K_vec)

        params = nnx.state(fno_model, nnx.Param)
        total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
        print(f"[PHASE_3_TL_OTF_3D] total number of FNO network parameters: {total_params}")

        adam_optimizer = optax.chain(optax.adam(1e-3))
        identity_control = IdentityControl("identity_control", {"num_vars": n_nodes}, fe_mesh)

        pi_fno_tl_otf = PhysicsInformedFourierParametricOperatorLearning3D(
            name="pi_fno_tl_otf_identity_3d",
            control=identity_control,
            loss_function=mechanical_loss_3d,
            flax_neural_network=fno_model,
            optax_optimizer=adam_optimizer,
        )
        pi_fno_tl_otf.Initialize()

        # ---- restore parametric weights before TL-OTF fine-tune
        pi_fno_tl_otf.RestoreState(restore_state_directory=param_state_dir)
        print("[PHASE_3_TL_OTF_3D] Restored parametric checkpoint. Starting TL-OTF...")

        tl_tag = f"tl_otf_identity_{label}_sid{sample_id}"
        tl_res_tracker = TrainingResidualTracker(out_dir=case_dir, tag=tl_tag)

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
                "absolute_error": float(RESIDUAL_TARGET),
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

        _copy_rms_plot_as_training_history(case_dir, sample_id, tl_tag)

        train_csv = os.path.join(case_dir, f"{tl_tag}_residual_rms.csv")
        if not os.path.exists(train_csv):
            train_csv = _find_first_existing([os.path.join(case_dir, "*residual_rms*.csv")])
        train_stats = _summarize_training(train_csv, residual_target=RESIDUAL_TARGET)

        pi_fno_tl_otf.RestoreState(restore_state_directory=os.path.join(case_dir, "flax_train_state"))
        print("[PHASE_3_TL_OTF_3D] Restored best TL-OTF checkpoint from this sample.")

        FNO_UVW = np.array(pi_fno_tl_otf.Predict(K_vec.reshape(1, -1))).reshape(-1)

        export_mesh = create_3D_box_mesh_structured(Nx=Nx, Ny=Ny, Nz=Nz, Lx=L, Ly=L, Lz=L)
        export_mesh.Initialize()
        export_mesh[f"U_FNO_{sample_id}_tl_otf"] = FNO_UVW.reshape((n_nodes, 3))
        export_mesh[f"K_field_{sample_id}_tl_otf"] = K_vec.reshape((n_nodes,))

        newton_total_iters = ""
        newton_final_residual = ""
        fe_time_s = ""

        ux_rms = np.nan
        ux_max = np.nan
        uy_rms = np.nan
        uy_max = np.nan
        uz_rms = np.nan
        uz_max = np.nan
        uv_rms = np.nan
        uv_max = np.nan

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
                sample_tag=f"sample_{sample_id}_{label}_tl_otf",
            )
            fe_time_s = time.time() - t_fe0

            FE_UVW = np.array(FE_UVW_jax).reshape(-1)
            export_mesh[f"U_FE_{sample_id}_tl_otf"] = FE_UVW.reshape((n_nodes, 3))

            newton_total_iters = int(total_iters)
            newton_final_residual = float(residuals[-1]) if (residuals is not None and len(residuals) > 0) else ""

            iters = np.arange(1, len(residuals) + 1)
            plt.figure(figsize=(6, 4))
            plt.semilogy(iters, residuals, marker="o")
            plt.xlabel("Global Newton iteration")
            plt.ylabel(r"RMS residual $\|r\|_{\mathrm{rms}}$")
            plt.title(f"Newton convergence – sample {sample_id} (TL-OTF-3D)")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, f"newton_residuals_sample_{sample_id}_tl_otf.png"))
            plt.close()

            abs_err_uvw = np.abs(FNO_UVW - FE_UVW)
            abs_err_ux = abs_err_uvw[0::3]
            abs_err_uy = abs_err_uvw[1::3]
            abs_err_uz = abs_err_uvw[2::3]

            export_mesh[f"abs_error_{sample_id}_tl_otf"] = abs_err_uvw.reshape((n_nodes, 3))
            export_mesh[f"abs_error_ux_{sample_id}_tl_otf"] = abs_err_ux.reshape((n_nodes,))
            export_mesh[f"abs_error_uy_{sample_id}_tl_otf"] = abs_err_uy.reshape((n_nodes,))
            export_mesh[f"abs_error_uz_{sample_id}_tl_otf"] = abs_err_uz.reshape((n_nodes,))

            ux_rms = float(np.sqrt(np.mean(abs_err_ux**2)))
            ux_max = float(np.max(abs_err_ux))
            uy_rms = float(np.sqrt(np.mean(abs_err_uy**2)))
            uy_max = float(np.max(abs_err_uy))
            uz_rms = float(np.sqrt(np.mean(abs_err_uz**2)))
            uz_max = float(np.max(abs_err_uz))

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
                Nx=Nx,
                Ny=Ny,
                Nz=Nz,
                out_path=os.path.join(case_dir, f"plot_results_sample_{sample_id}_tl_otf.png"),
            )
        else:
            Ksl = _midz_slice_scalar(K_vec, Nx, Ny, Nz).T
            Usl = _midz_slice_scalar(FNO_UVW[0::3], Nx, Ny, Nz).T
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            im0 = axs[0].imshow(Ksl, origin="lower")
            axs[0].set_title("Heterogeneity")
            plt.colorbar(im0, ax=axs[0], fraction=0.046)
            im1 = axs[1].imshow(Usl, origin="lower")
            axs[1].set_title("FNO_Ux")
            plt.colorbar(im1, ax=axs[1], fraction=0.046)
            for a in axs:
                a.set_xticks([])
                a.set_yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, f"plot_results_sample_{sample_id}_tl_otf.png"), dpi=200)
            plt.close()

        export_mesh.Finalize(export_dir=case_dir)
        _ensure_vtu_in_dir(case_dir)

        row = {
            "phase": "PHASE_3_TL_OTF_3D",
            "sample_id": int(sample_id),
            "label": label,
            "pkl_path": pkl_path,
            "case_dir": case_dir,
            "param_case_dir": param_case_dir,
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
            "solve_FE": bool(solve_FE),
            "use_fno_warmstart": bool(use_fno_warmstart),
            "newton_total_iters": newton_total_iters,
            "newton_final_residual": newton_final_residual,
            "fe_time_s": fe_time_s,
            "ux_rms": ux_rms,
            "ux_max": ux_max,
            "uy_rms": uy_rms,
            "uy_max": uy_max,
            "uz_rms": uz_rms,
            "uz_max": uz_max,
            "uv_rms": uv_rms,
            "uv_max": uv_max,
        }

        per_sample_csv = os.path.join(case_dir, f"stats_tl_otf_sample_{sample_id}.csv")
        _write_single_row_csv(per_sample_csv, STAT_FIELDS, row)
        _append_row_csv(master_stats_csv, STAT_FIELDS, row)

        print(f"[PHASE_3_TL_OTF_3D] Wrote per-sample stats: {per_sample_csv}")
        print(f"[PHASE_3_TL_OTF_3D] Appended master stats: {master_stats_csv}")

        if clean_dir:
            shutil.rmtree(case_dir)
            print(f"[PHASE_3_TL_OTF_3D] Cleaned directory {case_dir}.")


if __name__ == "__main__":
    fol_num_epochs = 20000
    solve_FE = True
    clean_dir = False
    use_fno_warmstart = False

    create_new_samples = False
    num_samples = 200
    random_seed = 0

    script_dir = Path(__file__).resolve().parent
    pkl_path = str(script_dir / "fourier_control_dict3D.pkl")

    train_ids = "0-47"
    test_ids = "48-49"
    default_label = "fourier_control3D"

    # IMPORTANT: this should be your 3D parametric training folder that contains flax_train_state/
    # If you used the corresponding 3D parametric script, it is:
    #   PHASE_0_PARAMETRIC_3D_Fourier/PHASE_0_PARAMETRIC_train_fourier_control3d/
    param_case_dir = str(script_dir / "PHASE_0_PARAMETRIC_3D_Fourier" / "PHASE_0_PARAMETRIC_train_fourier_control3d")

    out_root = str(script_dir / "PHASE_3_TL_OTF_3D_Fourier")

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
                "  python PHASE_3_TL_OTF_3D_NEW.py "
                "create_new_samples=False num_samples=200 random_seed=0 "
                "fol_num_epochs=20000 "
                "train_ids=0-47 test_ids=48-49 "
                "solve_FE=True use_fno_warmstart=False "
                "pkl_path=/path/to/fourier_control_dict3D.pkl "
                "param_case_dir=/path/to/PHASE_0_PARAMETRIC_train_*/ "
                "out_root=/path/to/out\n"
                "\n"
                "Notes:\n"
                "  - train_ids/test_ids ranges are INCLUSIVE.\n"
                "  - TL-OTF is run only on test_ids (one fresh model per selected test sample).\n"
                "  - The model first restores parametric weights from param_case_dir/flax_train_state.\n"
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
        param_case_dir=param_case_dir,
    )
