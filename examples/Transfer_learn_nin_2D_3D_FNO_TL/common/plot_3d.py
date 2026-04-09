"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: August, 2025
 License: FOL/LICENSE

plot_3d.py
~~~~~~~~~~
Shared 3-D plotting utilities for the mechanical_square phase-scripts.

All plots visualise a mid-Z cross-section of a 3-D voxel field stored as a
flat ``(Nx * Ny * Nz,)`` node-valued numpy array.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def _midz_slice_scalar(vec_nodes: np.ndarray, Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """Return the mid-Z slice of *vec_nodes* reshaped to ``(Nx, Ny)``."""
    return vec_nodes.reshape(Nx, Ny, Nz)[:, :, Nz // 2]


def _save_slice_png(
    vec_nodes: np.ndarray,
    Nx: int,
    Ny: int,
    Nz: int,
    out_path: str,
    title: str,
) -> None:
    """Save a heat-map PNG of the mid-Z slice of *vec_nodes*."""
    sl = _midz_slice_scalar(vec_nodes, Nx, Ny, Nz)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(sl.T, origin="lower")
    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_2x2_panel(
    K_vec: np.ndarray,
    fno_ux: np.ndarray,
    fe_ux: np.ndarray,
    abs_err_ux: np.ndarray,
    Nx: int,
    Ny: int,
    Nz: int,
    out_path: str,
) -> None:
    """
    Save a 2×2 panel PNG showing (mid-Z slices of):
      - heterogeneity field K
      - FNO Ux prediction
      - FE reference Ux
      - absolute error |FNO_Ux − FE_Ux|
    """
    Ksl = _midz_slice_scalar(K_vec, Nx, Ny, Nz).T
    Fsl = _midz_slice_scalar(fno_ux, Nx, Ny, Nz).T
    Esl = _midz_slice_scalar(fe_ux, Nx, Ny, Nz).T
    Asl = _midz_slice_scalar(abs_err_ux, Nx, Ny, Nz).T

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs = axs.ravel()

    titles = [
        "Heterogeneity (mid-z)",
        "FNO_Ux (mid-z)",
        "FE_Ux (mid-z)",
        "absolute_error_Ux",
    ]
    for ax, data, title in zip(axs, [Ksl, Fsl, Esl, Asl], titles):
        im = ax.imshow(data, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
