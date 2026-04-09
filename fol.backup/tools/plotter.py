"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: August, 2025 (revised)
 License: FOL/LICENSE

 A configurable 3D plotter for FOL/iFOL inference results saved in VTK files.
 - Auto-selects best FOL–FEM match (lowest L2 error on |U|)
 - Warped displacement panels (FOL/FEM)
 - Elasticity and error visualizations
 - Contour and orthogonal slice plots
 - Combined figure stitching via Matplotlib

 New in this revision:
 - Prefixes for elasticity and displacement fields are fully configurable via `config`:
     * elasticity_prefix (default: "K_")
     * u_fol_prefix      (default: "U_FOL_")
     * u_fe_prefix       (default: "U_FE_")
 - Optional fixed color limits for elasticity via `config["fixed_K_clim"]`.
 - Optional fixed color limits for error via `config["fixed_error_clim"]`.
 - Stable deformation scale in overview via `warp_factor_overview` (defaults to 1.0).
 - Global `show_edges` toggle in config.
 - Ensures **overview/combined use the same error field**: |‖U_FOL‖ − ‖U_FE‖|.

 Usage example:

    from fol.inference.plotter import Plotter3D
    import os, glob

    config = {
        "clip": True,
        "zoom": 0.9,
        "cmap": "coolwarm",
        "window_size": (1600, 1000),
        "scalar_bar_args": {"title": "", "vertical": True, "label_font_size": 22},
        "matplotlib_panel_zoom": {"displacement": 1.3, "elasticity": 1.0},
        "elasticity_prefix": "K_",
        "u_fol_prefix": "U_FOL_",
        "u_fe_prefix": "U_FE_",
        # "fixed_K_clim": [0.1, 1.0],
        # "fixed_error_clim": [0.0, 0.18],
        "show_edges": True,
        "warp_factor_overview": 1.0,
        "output_image": "overview.png",  # stitched figure filename
    }

    vtk_files = glob.glob(os.path.join(case_dir, "tested_samples", "*.vtk"))
    vtk_path = vtk_files[0]

    plotter = Plotter3D(vtk_path=vtk_path, warp_factor=1.0, config=config)
    plotter.render_all_panels()
"""

import os
import re
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use("Agg")  # headless-safe backend


class Plotter3D:
    """
    Class-based 3D plotter for FOL inference results saved in VTK files,
    with warped, contour-slice, and diagonal sampling views.

    All output PNGs are saved alongside the input `.vtk` file.
    """

    def __init__(self,
                 vtk_path: str,
                 warp_factor: float = 1.0,
                 config: dict | None = None):
        # Default configuration
        default_config = {
            "clip": True,
            "zoom": 0.8,
            "cmap": "coolwarm",
            "window_size": (1200, 800),
            "scalar_bar_args": {
                "title": "",
                "vertical": True,
                "title_font_size": 25,
                "label_font_size": 29,
                "position_x": 0.82,
                "position_y": 0.1,
                "width": 0.08,
                "height": 0.8,
                "font_family": "times",
            },
            "title_font_size": 24,
            "diag_points": 100,
            "final_figsize": (20, 10),
            "output_image": "combined_figure.png",
            "matplotlib_panel_zoom": {
                "displacement": 1,
                "elasticity": 0.8,
            },
            # prefixes (configurable)
            "elasticity_prefix": "K_",
            "u_fol_prefix": "U_FOL_",
            "u_fe_prefix": "U_FE_",
            # optional fixed clims for cross-run comparability
            "fixed_K_clim": None,
            "fixed_error_clim": None,
            # global toggles
            "show_edges": True,
            # overview-only warp factor (keeps overview scale stable)
            "warp_factor_overview": 1.0,
        }
        self.config = default_config if config is None else {**default_config, **config}
        self.do_clip = bool(self.config.get("clip", True))

        # Paths
        self.vtk_path = os.path.abspath(vtk_path)
        self.output_dir = os.path.dirname(self.vtk_path)
        os.makedirs(self.output_dir, exist_ok=True)

        # Load mesh & compute bounds
        self.mesh = pv.read(self.vtk_path)
        xmin, xmax, ymin, ymax, zmin, zmax = self.mesh.bounds
        self.cut_size = 0.5 * (xmax - xmin)
        self.xmax, self.ymax, self.zmax = xmax, ymax, zmax

        # Fields and parameters
        self.best_id = None
        self.fields = {}
        self.mag_names = {}
        self.warp_factor = float(warp_factor)
        self.warp_factor_overview = float(self.config.get("warp_factor_overview", self.warp_factor))
        self.diag_points = int(self.config["diag_points"])

        # Shared settings
        self.shared_scalar_bar_args = self.config["scalar_bar_args"]
        self.shared_zoom = float(self.config["zoom"])
        self.camera_position = [(2, 2, 2), (0.5, 0.5, 0.5), (0, 0, 1)]
        self.show_edges_default = bool(self.config.get("show_edges", True))

        # Name map (prefixes configurable)
        self.name_map = {
            "elasticity_prefix": str(self.config.get("elasticity_prefix", "K_")),
            "u_fol_prefix": str(self.config.get("u_fol_prefix", "U_FOL_")),
            "u_fe_prefix": str(self.config.get("u_fe_prefix", "U_FE_")),
        }

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def find_best_sample(self):
        """Find the sample index with minimum L2(|U_FOL|-|U_FE|)."""
        keys = list(self.mesh.point_data.keys())
        pf = re.escape(self.name_map["u_fol_prefix"])  # e.g., U_FOL_
        pe = re.escape(self.name_map["u_fe_prefix"])   # e.g., U_FE_

        fol_ids = {m.group(1) for k in keys if (m := re.match(pf + r"(\d+)$", k))}
        fe_ids  = {m.group(1) for k in keys if (m := re.match(pe + r"(\d+)$", k))}
        ids = sorted(fol_ids & fe_ids)
        if not ids:
            raise ValueError("No matching FOL/FE displacement field pairs found. Check prefixes and VTK fields.")

        best, min_err = None, float("inf")
        for i in ids:
            uf = self.mesh[f"{self.name_map['u_fol_prefix']}{i}"]
            ue = self.mesh[f"{self.name_map['u_fe_prefix']}{i}"]
            err = np.linalg.norm(np.linalg.norm(uf, axis=1) - np.linalg.norm(ue, axis=1))
            if err < min_err:
                min_err, best = err, i

        self.best_id = best
        self.fields = {
            "K_field": f"{self.name_map['elasticity_prefix']}{best}",
            "U_FOL":   f"{self.name_map['u_fol_prefix']}{best}",
            "U_FE":    f"{self.name_map['u_fe_prefix']}{best}",
        }
        self.mag_names = {
            "U_FOL_mag": f"{self.fields['U_FOL']}_mag",
            "U_FE_mag":  f"{self.fields['U_FE']}_mag",
        }
        print(f"Best sample = {best}, L2 error = {min_err:.6f}")

    def compute_derived_fields(self):
        fol, fe = self.fields['U_FOL'], self.fields['U_FE']
        self.mesh[self.mag_names['U_FOL_mag']] = np.linalg.norm(self.mesh[fol], axis=1)
        self.mesh[self.mag_names['U_FE_mag']]  = np.linalg.norm(self.mesh[fe], axis=1)

        # *** Single source of truth for error field used everywhere ***
        # absolute difference of magnitudes: |‖U_FOL‖ − ‖U_FE‖|
        self.mesh['abs_error'] = np.abs(
            self.mesh[self.mag_names['U_FOL_mag']] - self.mesh[self.mag_names['U_FE_mag']]
        )
        self.error_field = 'abs_error'
        self.error_title = r'| |U_FE| - |U_FOL| |'

    def apply_cut(self, mesh_obj: pv.DataSet) -> pv.DataSet:
        if not self.do_clip:
            return mesh_obj
        return mesh_obj.clip_box(
            bounds=(
                self.xmax - self.cut_size, self.xmax,
                self.ymax - self.cut_size, self.ymax,
                self.zmax - self.cut_size, self.zmax,
            ),
            invert=True,
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def render_panel(self, mesh_obj, field, clim, title, fname, show_edges=None):
        if show_edges is None:
            show_edges = self.show_edges_default
        plotter = pv.Plotter(off_screen=True, window_size=self.config['window_size'])
        plotter.add_mesh(
            mesh_obj, scalars=field, cmap=self.config['cmap'], clim=clim,
            show_edges=show_edges, edge_color='white', line_width=0.2,
            scalar_bar_args=self.shared_scalar_bar_args,
        )
        plotter.camera_position = self.camera_position
        plotter.camera.zoom(self.shared_zoom)
        plotter.add_axes()
        if title:
            plotter.add_text(title, font_size=self.config['title_font_size'], position='upper_edge')
        out = os.path.join(self.output_dir, fname)
        plotter.screenshot(out)
        plotter.close()
        print(f"Saved panel: {out}")

    def render_contour_slice(self):
        mesh = self.mesh
        fe_mag = self.mag_names['U_FE_mag']
        fol_mag = self.mag_names['U_FOL_mag']

        fe_contour = mesh.contour(scalars=fe_mag, isosurfaces=5)
        fol_contour = mesh.contour(scalars=fol_mag, isosurfaces=5)

        mesh.set_active_scalars(fe_mag)
        fe_slices = mesh.slice_orthogonal()
        mesh.set_active_scalars(fol_mag)
        fol_slices = mesh.slice_orthogonal()

        plotter = pv.Plotter(shape=(2, 2), off_screen=True,
                             window_size=(3200, 2400), border=False)

        plotter.subplot(0, 0)
        plotter.add_text(f"FEM: Contour {fe_mag}", font_size=12)
        plotter.add_mesh(fe_contour, scalars=fe_mag, cmap=self.config['cmap'])
        plotter.add_axes()

        plotter.subplot(0, 1)
        plotter.add_text("FEM: Orthogonal Slices", font_size=12)
        plotter.add_mesh(fe_slices, scalars=fe_mag, cmap=self.config['cmap'])
        plotter.add_axes()

        plotter.subplot(1, 0)
        plotter.add_text(f"FOL: Contour {fol_mag}", font_size=12)
        plotter.add_mesh(fol_contour, scalars=fol_mag, cmap=self.config['cmap'])
        plotter.add_axes()

        plotter.subplot(1, 1)
        plotter.add_text("FOL: Orthogonal Slices", font_size=12)
        plotter.add_mesh(fol_slices, scalars=fol_mag, cmap=self.config['cmap'])
        plotter.add_axes()

        plotter.link_views()
        plotter.view_isometric()
        screenshot = os.path.join(self.output_dir, "fol_fem_contour_grid.png")
        plotter.screenshot(screenshot)
        plotter.close()
        print(f"Saved contour-slice grid: {screenshot}")

    def render_diagonal_plot(self):
        n = self.diag_points
        diag_pts = np.linspace([0, 0, 0], [1, 1, 1], n)
        probe = pv.PolyData(diag_pts).sample(self.mesh)
        fol_mag = probe[self.mag_names['U_FOL_mag']]
        fe_mag = probe[self.mag_names['U_FE_mag']]
        K_diag = probe.point_data[self.fields['K_field']]

        fig = plt.figure(figsize=self.config['final_figsize'])
        gs = GridSpec(2, 2, figure=fig)

        clipped_png = os.path.join(self.output_dir, 'panel8.png')
        ax0 = fig.add_subplot(gs[0, 0])
        img = plt.imread(clipped_png)
        ax0.imshow(img)
        ax0.axis('off')
        ax0.set_title('Clipped Elasticity View', fontsize=self.config['title_font_size'])

        ax1 = fig.add_subplot(gs[0, 1])
        x = np.linspace(0, 1, n)
        ax1.plot(x, K_diag, linewidth=2)
        ax1.set_title('Elasticity Along Diagonal', fontsize=self.config['title_font_size'])
        ax1.set_xlabel('Normalized Distance')
        ax1.set_ylabel('Elasticity')
        e_zoom = float(self.config['matplotlib_panel_zoom']['elasticity'])
        ax1.set_ylim(self._zoom_ylim(K_diag, e_zoom))
        ax1.grid(False)

        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(x, fol_mag, label='FOL Disp', linewidth=2)
        ax2.plot(x, fe_mag, label='FEM Disp', linestyle='--', linewidth=2)
        ax2.set_title('Displacement Magnitude Along Diagonal', fontsize=self.config['title_font_size'])
        ax2.set_xlabel('Normalized Distance')
        ax2.set_ylabel('Displacement')
        d_zoom = float(self.config['matplotlib_panel_zoom']['displacement'])
        ax2.set_ylim(self._zoom_ylim(np.concatenate([fol_mag, fe_mag]), d_zoom))
        ax2.legend()
        ax2.grid(False)

        plt.tight_layout()
        out_diag = os.path.join(self.output_dir, 'diagonal_plot.png')
        plt.savefig(out_diag, dpi=300)
        plt.close(fig)
        print(f"Saved diagonal comparison plot: {out_diag}")

    @staticmethod
    def _zoom_ylim(data, zoom_factor: float):
        mn, mx = float(np.min(data)), float(np.max(data))
        ctr = 0.5 * (mn + mx)
        hr = 0.5 * (mx - mn) / max(zoom_factor, 1e-12)
        return [ctr - hr, ctr + hr]

    # ------------------------------------------------------------------
    # Pipelines
    # ------------------------------------------------------------------
    def render_all_panels(self):
        # full pipeline
        self.find_best_sample()
        self.compute_derived_fields()

        panels = []
        base = self.apply_cut(self.mesh)
        K = self.fields['K_field']

        # allow fixed_K_clim from config (for cross-run comparability)
        fixed = self.config.get("fixed_K_clim")
        if fixed is not None:
            K_clim = list(fixed)
        else:
            K_clim = [float(self.mesh[K].min()), float(self.mesh[K].max())]

        U_max = max(float(self.mesh[self.mag_names['U_FOL_mag']].max()),
                    float(self.mesh[self.mag_names['U_FE_mag']].max()))
        U_clim = [0.0, U_max]

        panels.append((base, K, K_clim, 'Elasticity Morphology', 'panel1.png', False))
        panels.append((base, K, K_clim, 'E(x,y,z) with Mesh', 'panel2.png', True))

        # clipped elasticity thumbnail used in stitched figure
        p5, p4, p2 = np.array([1, 0, 1]), np.array([0, 0, 1]), np.array([0, 1, 0])
        normal = np.cross(p4 - p5, p2 - p5).astype(float)
        normal /= np.linalg.norm(normal)
        panels.append((self.mesh.clip(normal=normal, origin=p5, invert=False), K, None, '', 'panel8.png', False))

        # FOL warped (stable overview warp)
        mf = self.mesh.copy(deep=True)
        mf.active_vectors_name = self.fields['U_FOL']
        wf = mf.warp_by_vector(factor=self.warp_factor_overview)
        wf[self.mag_names['U_FOL_mag']] = self.mesh[self.mag_names['U_FOL_mag']]
        panels.append((self.apply_cut(wf), self.mag_names['U_FOL_mag'], U_clim, 'FOL Deformation', 'panel5_warped.png', False))

        # FEM warped (stable overview warp)
        mf = self.mesh.copy(deep=True)
        mf.active_vectors_name = self.fields['U_FE']
        wf = mf.warp_by_vector(factor=self.warp_factor_overview)
        wf[self.mag_names['U_FE_mag']] = self.mesh[self.mag_names['U_FE_mag']]
        panels.append((self.apply_cut(wf), self.mag_names['U_FE_mag'], U_clim, 'FEM Deformation', 'panel6_warped.png', False))

        # error (same field used everywhere + optional fixed clim)
        err_clim = (list(self.config["fixed_error_clim"]) if self.config.get("fixed_error_clim") is not None
                    else [0.0, float(self.mesh[self.error_field].max())])
        panels.append((base, self.error_field, err_clim, self.error_title, 'panel7.png', False))

        for mesh_obj, field, clim, title, fname, edges in panels:
            self.render_panel(mesh_obj, field, clim, title, fname, show_edges=edges)

        # stitch
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig)
        for i, (_, _, _, _, fname, _) in enumerate(panels):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            img = plt.imread(os.path.join(self.output_dir, fname))
            ax.imshow(img)
            ax.axis('off')
        combined = os.path.join(self.output_dir, self.config['output_image'])
        plt.tight_layout()
        plt.savefig(combined, dpi=300)
        plt.close(fig)
        print(f"Saved combined figure: {combined}")

        # extras
        self.render_contour_slice()
        self.render_diagonal_plot()

    # ------------------------------------------------------------------
    # Optional: panels for a specific FEM sample index
    # ------------------------------------------------------------------
    def render_sample_panels(self, idx: int):
        K_field = f"{self.name_map['elasticity_prefix']}{idx}"
        U_FE_field = f"{self.name_map['u_fe_prefix']}{idx}"

        # Compute magnitude (if not already present)
        mag_field = f"{U_FE_field}_mag"
        if mag_field not in self.mesh.point_data:
            self.mesh[mag_field] = np.linalg.norm(self.mesh[U_FE_field], axis=1)

        # Panel 1: Elasticity field (PyVista)
        K_clim = [float(self.mesh[K_field].min()), float(self.mesh[K_field].max())]
        fixed = self.config.get("fixed_K_clim")
        if fixed is not None:
            K_clim = list(fixed)
        self.render_panel(
            self.apply_cut(self.mesh), K_field, K_clim,
            f'Elasticity ({K_field})', f'sample{idx}_elasticity.png'
        )

        # Panel 2: Warped deformation (PyVista)
        U_clim = [0.0, float(self.mesh[mag_field].max())]
        mf = self.mesh.copy(deep=True)
        mf.active_vectors_name = U_FE_field
        wf = mf.warp_by_vector(factor=self.warp_factor)
        wf[mag_field] = self.mesh[mag_field]
        self.render_panel(
            self.apply_cut(wf), mag_field, U_clim,
            f'Warped Displacement ({U_FE_field})', f'sample{idx}_warped.png'
        )

        # Panel 3: Displacement along diagonal (matplotlib)
        n = self.diag_points
        diag_pts = np.linspace([0, 0, 0], [1, 1, 1], n)
        probe = pv.PolyData(diag_pts).sample(self.mesh)
        fe_mag = probe.point_data[mag_field]
        x = np.linspace(0, 1, n)
        fig_line, ax_line = plt.subplots(figsize=(6, 6))
        ax_line.plot(x, fe_mag, label=f'{U_FE_field} along diagonal', linewidth=2)
        ax_line.set_xlabel('Normalized Distance')
        ax_line.set_ylabel('Displacement')
        ax_line.set_title(f'Displacement Along Diagonal ({U_FE_field})')
        ax_line.grid(True)
        plt.tight_layout()
        diag_plot_path = os.path.join(self.output_dir, f'sample{idx}_diag_line.png')
        fig_line.savefig(diag_plot_path, dpi=200)
        plt.close(fig_line)

        # Combine the three panels into a single PNG
        import matplotlib.image as mpimg
        img_paths = [
            f'sample{idx}_elasticity.png',
            f'sample{idx}_warped.png',
            f'sample{idx}_diag_line.png',
        ]
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for i, img_path in enumerate(img_paths):
            img = mpimg.imread(os.path.join(self.output_dir, img_path))
            axs[i].imshow(img)
            axs[i].axis('off')
        fig.suptitle(f"Sample {idx}", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        panel_path = os.path.join(self.output_dir, f'sample{idx}_panel.png')
        plt.savefig(panel_path, dpi=200)
        plt.close(fig)
        print(f"Saved combined panel for sample {idx}: {panel_path}")


# ----------------------------------------------------------------------
# Utility: solver convergence plot (kept as-is)
# ----------------------------------------------------------------------

def plot_solver_convergence(residual_norms_history, save_path=None, show=False):
    """
    Plots the nonlinear solver residual norm vs. iteration for each sample.

    Args:
        residual_norms_history: List of lists, each containing the residual norms for a sample.
        save_path: If given, the plot is saved to this path.
        show: If True, the plot is displayed.
    """
    plt.figure(figsize=(8, 6))
    for idx, residual_norms in enumerate(residual_norms_history):
        plt.semilogy(residual_norms, '-o', label=f"Sample {idx}")
    plt.xlabel("Iteration number", fontsize=12)
    plt.ylabel("Residual norm (log scale)", fontsize=12)
    plt.title("Nonlinear Solver Convergence (Neo-Hookean)", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Convergence plot saved to: {save_path}")
    if show:
        plt.show()
    plt.close()