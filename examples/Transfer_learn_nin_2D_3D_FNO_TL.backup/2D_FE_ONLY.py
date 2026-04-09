import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import shutil
import pickle
import time

from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.fourier_control import FourierControl
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.tools.newton_residual_tracker import custom_newton_solve

from common.script_utils import (
    _parse_ids,
    _sanitize_label,
    _clean_mkdir,
    _append_row_csv,
    _write_single_row_csv,
)


RESIDUAL_TARGET = 1e-4
MASTER_STATS_NAME = "stats_phase0_parametric.csv"


def main(
    clean_dir: bool = False,
    pkl_path: str = "fourier_control_dict.pkl",
    train_ids: str = "0-80",
    test_ids: str = "97-99",
    default_label: str = "fourier_control",
    out_root: str = "PHASE_0_PARAMETRIC_2D_Fourier_strain_50",
    nl_maxiter: int = 30,
    nl_load_incr: int = 61,
    growth_tol: float = 1e3,
    ls_max_backtracks: int = 8,
    ls_shrink: float = 0.5,
    ls_accept_ratio: float = 1.001,
    plateau_max_consecutive: int = 5,
    plateau_rel_improve_tol: float = 1e-4,
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
        "fe_converged",
        "reached_full_load",
        "failed_load_step",
        "reached_load_factor",
        "guard_return_mode",
        "line_search_used",
        "failure_mode",
        "ux_rms", "ux_max",
        "uy_rms", "uy_max",
        "uv_rms", "uv_max",
    ]

    pkl_path = os.path.abspath(os.path.expanduser(pkl_path))
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"[PHASE_FE_ONLY] pkl not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    model_settings = {
        "L": 1.0,
        "N": 42,
        "Ux_left": 0.0,
        "Ux_right": 0.3,
        "Uy_left": 0.0,
        "Uy_right": 0.3,
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

    if "K_matrix" in d:
        K_matrix = np.asarray(d["K_matrix"], dtype=np.float32)
        print(f"[PHASE_FE_ONLY] Found K_matrix in pkl: {K_matrix.shape}")

    elif "coeffs_matrix" in d:
        coeffs_matrix = np.asarray(d["coeffs_matrix"], dtype=np.float32)
        print(f"[PHASE_FE_ONLY] Found coeffs_matrix in pkl: {coeffs_matrix.shape}")

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
        print(f"[PHASE_FE_ONLY] Computed K_matrix from coeffs_matrix: {K_matrix.shape}")

    else:
        raise KeyError("[PHASE_FE_ONLY] pkl must contain either 'K_matrix' or 'coeffs_matrix'")

    if K_matrix.ndim != 2:
        raise ValueError(f"[PHASE_FE_ONLY] K_matrix must be 2D, got {K_matrix.shape}")

    num_samples, k_nodes = K_matrix.shape
    if k_nodes != n_nodes:
        raise ValueError(f"[PHASE_FE_ONLY] K_matrix has {k_nodes} nodes, expected {n_nodes}. Check N mismatch.")

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

    print(f"[PHASE_FE_ONLY] out_root={out_root}")
    print(f"[PHASE_FE_ONLY] train_ids={train_ids} -> {len(train_list)} samples")
    print(f"[PHASE_FE_ONLY] test_ids={test_ids}  -> {len(test_list)} samples")

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

    for sample_id in test_list:
        sample_label = _sanitize_label(labels_vec[sample_id]) if labels_vec is not None else label
        case_dir = os.path.join(out_root, f"PHASE_0_PARAMETRIC_sample_{sample_id}_{sample_label}")
        _clean_mkdir(case_dir)
        sys.stdout = Logger(os.path.join(case_dir, f"PHASE_0_PARAMETRIC_sample_{sample_id}_{sample_label}.log"))

        print(f"[PHASE_FE_ONLY] sample_id={sample_id}, label={sample_label}")
        print(f"[PHASE_FE_ONLY] pkl_path={pkl_path}")
        print(f"[PHASE_FE_ONLY] output_dir={case_dir}")

        K_vec = K_matrix[sample_id, :].astype(np.float32)

        plt.figure(figsize=(4, 4))
        plt.imshow(K_vec.reshape(N, N), origin="lower")
        plt.title(f"Heterogeneity (sid={sample_id})")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, f"heterogeneity_sample_{sample_id}.png"), dpi=200)
        plt.close()

        export_mesh = create_2D_square_mesh(L=L, N=N)
        export_mesh.Initialize()
        export_mesh[f"K_field_{sample_id}_param"] = K_vec.reshape((n_nodes,))

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
                "maxiter": int(nl_maxiter),
                "load_incr": int(nl_load_incr),
            },
        }

        nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver(
            f"nonlin_fe_solver_param_{sample_id}",
            mechanical_loss_2d,
            fe_setting,
        )
        nonlin_fe_solver.Initialize()

        ndofs = 2 * n_nodes
        initial_dofs = np.zeros(ndofs)

        t_fe0 = time.time()
        FE_UV_jax, residuals, total_iters, run_info = custom_newton_solve(
            fe_solver=nonlin_fe_solver,
            control_vars=K_vec,
            initial_dofs=initial_dofs,
            case_dir=case_dir,
            sample_tag=f"sample_{sample_id}_{sample_label}_param",
            growth_tol=float(growth_tol),
            use_line_search=True,
            ls_max_backtracks=int(ls_max_backtracks),
            ls_shrink=float(ls_shrink),
            ls_accept_ratio=float(ls_accept_ratio),
            guard_return_mode="latest",
            plateau_max_consecutive=int(plateau_max_consecutive),
            plateau_rel_improve_tol=float(plateau_rel_improve_tol),
            return_run_info=True,
        )
        fe_time_s = time.time() - t_fe0

        FE_UV = np.array(FE_UV_jax).reshape(-1)
        FE_nodes = FE_UV.reshape((n_nodes, 2))

        left_ids = export_mesh.GetNodeSet("left")
        right_ids = export_mesh.GetNodeSet("right")

        print("LEFT  Ux min/max:", FE_nodes[left_ids, 0].min(), FE_nodes[left_ids, 0].max())
        print("RIGHT Ux min/max:", FE_nodes[right_ids, 0].min(), FE_nodes[right_ids, 0].max())
        print("LEFT  Uy min/max:", FE_nodes[left_ids, 1].min(), FE_nodes[left_ids, 1].max())
        print("RIGHT Uy min/max:", FE_nodes[right_ids, 1].min(), FE_nodes[right_ids, 1].max())
        print("GLOBAL Ux min/max:", FE_nodes[:, 0].min(), FE_nodes[:, 0].max())
        print("GLOBAL Uy min/max:", FE_nodes[:, 1].min(), FE_nodes[:, 1].max())

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

        export_mesh.Finalize(export_dir=case_dir)

        row = {
            "phase": "PHASE_FE_ONLY",
            "sample_id": int(sample_id),
            "label": sample_label,
            "pkl_path": pkl_path,
            "case_dir": case_dir,
            "train_ids": train_ids,
            "test_ids": test_ids,
            "train_count": len(train_list),
            "test_count": len(test_list),
            "fol_num_epochs_requested": 5000,
            "epochs_completed": train_stats["epochs_completed"],
            "last_epoch": train_stats["last_epoch"],
            "train_residual_final": train_stats["final_residual"],
            "train_residual_min": train_stats["min_residual"],
            "epoch_at_min": train_stats["epoch_at_min"],
            "epoch_first_below_target": train_stats["epoch_first_below_target"],
            "residual_target": float(RESIDUAL_TARGET),
            "final_total_loss": train_stats["final_total_loss"],
            "train_time_s": float(train_time_s),
            "batch_size": 16,
            "solve_FE": True,
            "skip_training": True,
            "use_fno_warmstart": False,
            "newton_total_iters": newton_total_iters,
            "newton_final_residual": newton_final_residual,
            "fe_time_s": fe_time_s,
            "fe_converged": bool(run_info.get("fe_converged", False)),
            "reached_full_load": bool(run_info.get("reached_full_load", False)),
            "failed_load_step": run_info.get("failed_load_step", ""),
            "reached_load_factor": float(run_info.get("reached_load_factor", 0.0)),
            "guard_return_mode": str(run_info.get("guard_return_mode", "latest")),
            "line_search_used": bool(run_info.get("line_search_used", True)),
            "failure_mode": str(run_info.get("failure_mode", "")),
            "ux_rms": np.nan,
            "ux_max": np.nan,
            "uy_rms": np.nan,
            "uy_max": np.nan,
            "uv_rms": np.nan,
            "uv_max": np.nan,
        }

        per_sample_csv = os.path.join(case_dir, f"stats_param_sample_{sample_id}.csv")
        _write_single_row_csv(per_sample_csv, STAT_FIELDS, row)
        _append_row_csv(master_stats_csv, STAT_FIELDS, row)

        print(f"[PHASE_FE_ONLY] Wrote per-sample stats: {per_sample_csv}")
        print(f"[PHASE_FE_ONLY] Appended master stats: {master_stats_csv}")

        if clean_dir:
            shutil.rmtree(case_dir)
            print(f"[PHASE_FE_ONLY] Cleaned directory {case_dir}.")


if __name__ == "__main__":
    clean_dir = False

    script_dir = Path(__file__).resolve().parent
    pkl_path = str(script_dir / "fourier_control_dict.pkl")

    train_ids = "0-80"
    test_ids = "97-99"
    default_label = "fourier_control"
    out_root = str(script_dir / "PHASE_0_PARAMETRIC_2D_Fourier_strain_50")
    nl_maxiter = 30
    nl_load_incr = 61
    growth_tol = 1e3
    ls_max_backtracks = 8
    ls_shrink = 0.5
    ls_accept_ratio = 1.001
    plateau_max_consecutive = 5
    plateau_rel_improve_tol = 1e-4

    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("clean_dir="):
            clean_dir = arg.split("=")[1].lower() == "true"
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
        elif arg.startswith("nl_maxiter="):
            nl_maxiter = int(arg.split("=", 1)[1])
        elif arg.startswith("nl_load_incr="):
            nl_load_incr = int(arg.split("=", 1)[1])
        elif arg.startswith("growth_tol="):
            growth_tol = float(arg.split("=", 1)[1])
        elif arg.startswith("ls_max_backtracks="):
            ls_max_backtracks = int(arg.split("=", 1)[1])
        elif arg.startswith("ls_shrink="):
            ls_shrink = float(arg.split("=", 1)[1])
        elif arg.startswith("ls_accept_ratio="):
            ls_accept_ratio = float(arg.split("=", 1)[1])
        elif arg.startswith("plateau_max_consecutive="):
            plateau_max_consecutive = int(arg.split("=", 1)[1])
        elif arg.startswith("plateau_rel_improve_tol="):
            plateau_rel_improve_tol = float(arg.split("=", 1)[1])
        else:
            print(
                "Usage:\n"
                "  python 2D_FE_ONLY.py "
                "train_ids=0-80 test_ids=97-99 "
                "pkl_path=/path/to/fourier_control_dict.pkl "
                "out_root=/path/to/out clean_dir=False "
                "nl_maxiter=30 nl_load_incr=61 growth_tol=1e3 "
                "ls_max_backtracks=8 ls_shrink=0.5 ls_accept_ratio=1.001 "
                "plateau_max_consecutive=5 plateau_rel_improve_tol=1e-4\n"
            )
            sys.exit(1)

    main(
        clean_dir=clean_dir,
        pkl_path=pkl_path,
        train_ids=train_ids,
        test_ids=test_ids,
        default_label=default_label,
        out_root=out_root,
        nl_maxiter=nl_maxiter,
        nl_load_incr=nl_load_incr,
        growth_tol=growth_tol,
        ls_max_backtracks=ls_max_backtracks,
        ls_shrink=ls_shrink,
        ls_accept_ratio=ls_accept_ratio,
        plateau_max_consecutive=plateau_max_consecutive,
        plateau_rel_improve_tol=plateau_rel_improve_tol,
    )
