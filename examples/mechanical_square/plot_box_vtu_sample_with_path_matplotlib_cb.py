
"""
Standalone plotting utility for 3D box .vtu sample files exported by the
PHASE_0_PARAMETRIC_3D / PHASE_2_OTF_3D / PHASE_3_TL_OTF_3D scripts.

It supports field names like:
    K_field_<sid>_param
    U_FNO_<sid>_param
    U_FE_<sid>_param

    K_field_<sid>_otf
    U_FNO_<sid>_otf
    U_FE_<sid>_otf

    K_field_<sid>_tl_otf
    U_FNO_<sid>_tl_otf
    U_FE_<sid>_tl_otf
"""

import os
from pathlib import Path
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Figure style
# ============================================================
FIG_SUPTITLE_SIZE = 26
PANEL_TITLE_SIZE = 20
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
LEGEND_SIZE = 14
LINE_WIDTH = 2.8

COLORBAR_TICK_SIZE = 28
PYVISTA_WINDOW_SIZE = (2400, 1800)
# ============================================================


def _find_existing_name(mesh, candidates):
    for name in candidates:
        if name in mesh.point_data:
            return name
    return None


def _diag_points_from_bounds(bounds, n=100):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    p0 = np.array([xmin, ymin, zmin], dtype=float)
    p1 = np.array([xmax, ymax, zmax], dtype=float)
    return np.linspace(p0, p1, n)


def _zoom_ylim(data, zoom_factor):
    data = np.asarray(data, dtype=float)
    mn = float(np.min(data))
    mx = float(np.max(data))
    ctr = 0.5 * (mn + mx)
    hr = 0.5 * (mx - mn) / max(float(zoom_factor), 1e-12)
    if hr == 0.0:
        hr = max(abs(ctr) * 0.05, 1e-8)
    return [ctr - hr, ctr + hr]


def _clip_mesh(mesh, cut_fraction=0.5):
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cut_size = float(cut_fraction) * (xmax - xmin)
    return mesh.clip_box(
        bounds=(
            xmax - cut_size, xmax,
            ymax - cut_size, ymax,
            zmax - cut_size, zmax,
        ),
        invert=True,
    )


def _style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=PANEL_TITLE_SIZE, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)


def _render_panel(
    mesh_obj,
    field,
    out_path,
    title="",
    cmap="viridis",
    clim=None,
    zoom=0.9,
    show_edges=False,
    window_size=PYVISTA_WINDOW_SIZE,
    scalar_bar_args=None,   # kept for compatibility, not used
):
    """
    Render a PyVista panel WITHOUT the built-in scalar bar, then add a
    Matplotlib colorbar next to the screenshot. This avoids VTK shrinking
    the scalar-bar labels.
    """
    # 1) Render PyVista screenshot without scalar bar
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(
        mesh_obj,
        scalars=field,
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        edge_color="white",
        line_width=0.2,
        show_scalar_bar=False,
    )
    plotter.camera_position = [(2, 2, 2), (0.5, 0.5, 0.5), (0, 0, 1)]
    plotter.camera.zoom(float(zoom))
    plotter.add_axes()

    img = plotter.screenshot(return_img=True)
    plotter.close()

    # 2) Build Matplotlib figure with image + colorbar
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1.8], wspace=0.05)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])

    ax_img.imshow(img)
    ax_img.axis("off")
    if title:
        ax_img.set_title(title, fontsize=PANEL_TITLE_SIZE, fontweight="bold")

    if clim is None:
        vals = np.asarray(mesh_obj[field], dtype=float)
        clim = [float(np.min(vals)), float(np.max(vals))]

    norm = plt.Normalize(vmin=clim[0], vmax=clim[1])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax_cb, orientation="vertical")
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    cbar.ax.yaxis.get_offset_text().set_size(COLORBAR_TICK_SIZE)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_box_vtu_sample(
    vtu_path,
    sample_id,
    phase=None,
    output_dir=None,
    warp_factor=1.0,
    clip=True,
    cut_fraction=0.5,
    zoom=0.9,
    cmap="viridis",
    show_edges=False,
    diag_points=100,
    disp_zoom=1.2,
    elasticity_zoom=1.0,
):
    """
    Plot one chosen sample from a 3D box .vtu file.

    Parameters
    ----------
    vtu_path : str
        Path to the .vtu file.
    sample_id : int
        Sample index written in the field names.
    phase : str | None
        One of {"param", "otf", "tl_otf"}.
        If None, the function auto-detects the first matching phase.
    output_dir : str | None
        Folder to save output PNGs. Defaults to the .vtu folder.
    warp_factor : float
        Deformation scale for warped displacement views.
    clip : bool
        Whether to show a clipped cube view.
    cut_fraction : float
        Fraction of the box length used for the clip box.
    zoom : float
        PyVista camera zoom factor.
    cmap : str
        Colormap.
    show_edges : bool
        Whether to show mesh edges in PyVista panels.
    diag_points : int
        Number of points for diagonal sampling.
    disp_zoom : float
        Vertical zoom factor for displacement line plot.
    elasticity_zoom : float
        Vertical zoom factor for elasticity line plot.

    Returns
    -------
    dict
        Dictionary with detected field names and output image paths.
    """
    vtu_path = os.path.abspath(os.path.expanduser(str(vtu_path)))
    if output_dir is None:
        output_dir = os.path.dirname(vtu_path)
    output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
    os.makedirs(output_dir, exist_ok=True)

    mesh = pv.read(vtu_path)
    sid = int(sample_id)

    # More robust phase search
    base_phases = ["param", "otf", "tl_otf", ""]
    if phase is not None:
        phases = [phase] + [p for p in base_phases if p != phase]
    else:
        phases = base_phases

    detected = None
    for ph in phases:
        suffix = f"_{ph}" if ph else ""
        k_name = _find_existing_name(mesh, [
            f"K_field_{sid}{suffix}",
            f"K_{sid}{suffix}",
            f"K_field_{sid}",
            f"K_{sid}",
        ])
        u_fno_name = _find_existing_name(mesh, [
            f"U_FNO_{sid}{suffix}",
            f"U_FOL_{sid}{suffix}",
            f"U_FNO_{sid}",
            f"U_FOL_{sid}",
        ])
        u_fe_name = _find_existing_name(mesh, [
            f"U_FE_{sid}{suffix}",
            f"U_FE_{sid}",
        ])
        if k_name is not None and u_fno_name is not None:
            detected = {
                "phase": ph if ph else "unknown",
                "K_field": k_name,
                "U_FNO": u_fno_name,
                "U_FE": u_fe_name,
            }
            break

    if detected is None:
        raise KeyError(
            f"Could not find fields for sample_id={sid}. "
            f"Available point-data keys: {list(mesh.point_data.keys())}"
        )

    k_name = detected["K_field"]
    u_fno_name = detected["U_FNO"]
    u_fe_name = detected["U_FE"]

    # Derived scalar fields
    fno_mag_name = f"{u_fno_name}_mag"
    if fno_mag_name not in mesh.point_data:
        mesh[fno_mag_name] = np.linalg.norm(np.asarray(mesh[u_fno_name]), axis=1)

    fe_mag_name = None
    error_name = None
    if u_fe_name is not None:
        fe_mag_name = f"{u_fe_name}_mag"
        if fe_mag_name not in mesh.point_data:
            mesh[fe_mag_name] = np.linalg.norm(np.asarray(mesh[u_fe_name]), axis=1)
        error_name = f"abs_error_mag_sid{sid}"
        mesh[error_name] = np.abs(mesh[fno_mag_name] - mesh[fe_mag_name])

    base_mesh = _clip_mesh(mesh, cut_fraction=cut_fraction) if clip else mesh

    k_clim = [float(np.min(mesh[k_name])), float(np.max(mesh[k_name]))]
    fno_clim = [0.0, float(np.max(mesh[fno_mag_name]))]
    if fe_mag_name is not None:
        u_max = max(float(np.max(mesh[fno_mag_name])), float(np.max(mesh[fe_mag_name])))
        disp_clim = [0.0, u_max]
        err_clim = [0.0, float(np.max(mesh[error_name]))]
    else:
        disp_clim = fno_clim
        err_clim = None

    # Panel 1: elasticity
    panel1 = Path(output_dir) / f"sample_{sid}_{detected['phase']}_elasticity.png"
    _render_panel(
        base_mesh,
        k_name,
        panel1,
        title=f"Elasticity ({k_name})",
        cmap=cmap,
        clim=k_clim,
        zoom=zoom,
        show_edges=show_edges,
    )

    # Panel 2: FNO warped
    mesh_fno = mesh.copy(deep=True)
    mesh_fno.active_vectors_name = u_fno_name
    warped_fno = mesh_fno.warp_by_vector(factor=float(warp_factor))
    warped_fno[fno_mag_name] = mesh[fno_mag_name]
    if clip:
        warped_fno = _clip_mesh(warped_fno, cut_fraction=cut_fraction)

    panel2 = Path(output_dir) / f"sample_{sid}_{detected['phase']}_fno.png"
    _render_panel(
        warped_fno,
        fno_mag_name,
        panel2,
        title=f"FNO displacement ({u_fno_name})",
        cmap=cmap,
        clim=disp_clim,
        zoom=zoom,
        show_edges=False,
    )

    panel3 = None
    panel4 = None

    if u_fe_name is not None:
        # Panel 3: FE warped
        mesh_fe = mesh.copy(deep=True)
        mesh_fe.active_vectors_name = u_fe_name
        warped_fe = mesh_fe.warp_by_vector(factor=float(warp_factor))
        warped_fe[fe_mag_name] = mesh[fe_mag_name]
        if clip:
            warped_fe = _clip_mesh(warped_fe, cut_fraction=cut_fraction)

        panel3 = Path(output_dir) / f"sample_{sid}_{detected['phase']}_fe.png"
        _render_panel(
            warped_fe,
            fe_mag_name,
            panel3,
            title=f"FE displacement ({u_fe_name})",
            cmap=cmap,
            clim=disp_clim,
            zoom=zoom,
            show_edges=False,
        )

        # Panel 4: error
        panel4 = Path(output_dir) / f"sample_{sid}_{detected['phase']}_error.png"
        _render_panel(
            base_mesh,
            error_name,
            panel4,
            title="Abs. diff. of |U|",
            cmap=cmap,
            clim=err_clim,
            zoom=zoom,
            show_edges=False,
        )

    # Separate diagonal plots
    diag_xyz = _diag_points_from_bounds(mesh.bounds, n=int(diag_points))
    sampled = pv.PolyData(diag_xyz).sample(mesh)
    x = np.linspace(0.0, 1.0, int(diag_points))
    k_diag = np.asarray(sampled[k_name], dtype=float)
    fno_diag = np.asarray(sampled[fno_mag_name], dtype=float)

    diag_elasticity_path = Path(output_dir) / f"sample_{sid}_{detected['phase']}_diag_elasticity.png"
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(x, k_diag, linewidth=LINE_WIDTH)
    _style_axis(ax, "Elasticity along diagonal", "Normalized distance", "Elasticity")
    ax.set_ylim(_zoom_ylim(k_diag, elasticity_zoom))
    ax.grid(False)
    fig.savefig(diag_elasticity_path, dpi=250)
    plt.close(fig)

    diag_displacement_path = Path(output_dir) / f"sample_{sid}_{detected['phase']}_diag_displacement.png"
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if fe_mag_name is not None:
        fe_diag = np.asarray(sampled[fe_mag_name], dtype=float)
        ax.plot(x, fno_diag, linewidth=LINE_WIDTH, label="FNO")
        ax.plot(x, fe_diag, "--", linewidth=LINE_WIDTH, label="FE")
        ax.set_ylim(_zoom_ylim(np.concatenate([fno_diag, fe_diag]), disp_zoom))
        ax.legend(fontsize=LEGEND_SIZE)
    else:
        ax.plot(x, fno_diag, linewidth=LINE_WIDTH, label="FNO")
        ax.set_ylim(_zoom_ylim(fno_diag, disp_zoom))
        ax.legend(fontsize=LEGEND_SIZE)

    _style_axis(ax, "Displacement magnitude along diagonal", "Normalized distance", "|U|")
    ax.grid(False)
    fig.savefig(diag_displacement_path, dpi=250)
    plt.close(fig)

    # Stitched summary: fixed 2x3 layout
    images = [
        panel1,
        panel2,
        panel3,
        panel4,
        diag_elasticity_path,
        diag_displacement_path,
    ]

    fig, axs = plt.subplots(2, 3, figsize=(7.2 * 3, 5.6 * 2))
    axs = np.atleast_1d(axs).ravel()

    for ax, img_path in zip(axs, images):
        if img_path is not None and Path(img_path).exists():
            img = plt.imread(img_path)
            ax.imshow(img)
        ax.axis("off")

    stitched = Path(output_dir) / f"sample_{sid}_{detected['phase']}_combined.png"
    fig.suptitle(
        f"box.vtu sample {sid} ({detected['phase']})",
        fontsize=FIG_SUPTITLE_SIZE,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(stitched, dpi=250)
    plt.close(fig)

    return {
        "sample_id": sid,
        "phase": detected["phase"],
        "fields": detected,
        "images": {
            "elasticity": str(panel1),
            "fno": str(panel2),
            "fe": str(panel3) if panel3 is not None else None,
            "error": str(panel4) if panel4 is not None else None,
            "diag_elasticity": str(diag_elasticity_path),
            "diag_displacement": str(diag_displacement_path),
            "combined": str(stitched),
        },
    }


if __name__ == "__main__":
    # ============================================================
    # EDIT ONLY THESE DEFAULTS
    # ============================================================
    vtu_path = r"/home/jerry-paul/Access/folax_main_jan_venv/3D_fourier_results/3D_Samples/PHASE_0_PARAMETRIC_3D_Fourier/PHASE_0_PARAMETRIC_sample_80_fourier_control3d/box.vtu"
    sample_id = 80
    phase = None   # "param", "otf", "tl_otf", or None for auto-detect

    output_dir = None          # None -> save next to the .vtu
    warp_factor = 1.0
    clip = True
    cut_fraction = 0.5
    zoom = 0.9
    cmap = "coolwarm"
    show_edges = False
    diag_points = 100
    disp_zoom = 1.2
    elasticity_zoom = 1.0

    # Optional CLI override:
    #   python plot_box_vtu_sample_with_path_matplotlib_cb.py /path/to/file.vtu 48 tl_otf
    import sys
    args = sys.argv[1:]
    if len(args) >= 1:
        vtu_path = args[0]
    if len(args) >= 2:
        sample_id = int(args[1])
    if len(args) >= 3:
        phase = args[2]

    out = plot_box_vtu_sample(
        vtu_path=vtu_path,
        sample_id=sample_id,
        phase=phase,
        output_dir=output_dir,
        warp_factor=warp_factor,
        clip=clip,
        cut_fraction=cut_fraction,
        zoom=zoom,
        cmap=cmap,
        show_edges=show_edges,
        diag_points=diag_points,
        disp_zoom=disp_zoom,
        elasticity_zoom=elasticity_zoom,
    )
    print(out)
