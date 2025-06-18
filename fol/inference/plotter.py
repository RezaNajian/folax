import os
import re
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class Plotter3D:
    """
    Class-based 3D plotter for FOL inference results saved in VTK files,
    with warped, contour-slice, and diagonal sampling views.

    Usage:
        config = {
            "zoom": 0.8,
            "cmap": "coolwarm",
            "window_size": (1200, 800),
            "scalar_bar_args": {
                "title": "",
                "vertical": True,
                "title_font_size": 25,
                "label_font_size": 28,
                "position_x": 0.82,
                "position_y": 0.1,
                "width": 0.05,
                "height": 0.8,
                "font_family": "times",
            },
            "title_font_size": 24,
            "diag_points": 100,
            "final_figsize": (20, 10),
            "output_image": "combined_figure.png",
            "matplotlib_panel_zoom": {
                "displacement": 1,
                "elasticity": 0.8
            }
        }
        plotter = Plotter3D(vtk_path="file.vtk", warp_factor=10.0, config=config)
        plotter.render_all_panels()

    All output PNGs are saved alongside the input `.vtk` file.
    """

    def __init__(self,
                 vtk_path: str,
                 warp_factor: float = 10.0,
                 config: dict = None):
        # Default configuration
        default_config = {
            "zoom": 0.8,
            "cmap": "coolwarm",
            "window_size": (1200, 800),
            "scalar_bar_args": {
                "title": "",
                "vertical": True,
                "title_font_size": 25,
                "label_font_size": 28,
                "position_x": 0.82,
                "position_y": 0.1,
                "width": 0.05,
                "height": 0.8,
                "font_family": "times",
            },
            "title_font_size": 24,
            "diag_points": 100,
            "final_figsize": (20, 10),
            "output_image": "combined_figure.png",
            "matplotlib_panel_zoom": {
                "displacement": 1,
                "elasticity": 0.8
            }
        }
        self.config = default_config if config is None else {**default_config, **config}

        # Paths
        self.vtk_path = os.path.abspath(vtk_path)
        self.output_dir = os.path.dirname(self.vtk_path)
        os.makedirs(self.output_dir, exist_ok=True)

        # Load mesh & compute bounds
        self.mesh = pv.read(self.vtk_path)
        xmin, xmax, ymin, ymax, zmin, zmax = self.mesh.bounds
        self.cut_size = 0.7 * (xmax - xmin)
        self.xmax, self.ymax, self.zmax = xmax, ymax, zmax

        # Fields and parameters
        self.best_id = None
        self.fields = {}
        self.warp_factor = warp_factor
        self.diag_points = self.config["diag_points"]

        # Shared settings
        self.shared_scalar_bar_args = self.config["scalar_bar_args"]
        self.shared_zoom = self.config["zoom"]
        self.camera_position = [(2, 2, 2), (0.5, 0.5, 0.5), (0, 0, 1)]

    def find_best_sample(self):
        keys = self.mesh.point_data.keys()
        ufol = [k for k in keys if re.match(r"U_FOL_\d+", k)]
        ufe  = [k for k in keys if re.match(r"U_FE_\d+", k)]
        ids = sorted(set(k.split("_")[-1] for k in ufol) & set(k.split("_")[-1] for k in ufe))
        best, min_err = None, float('inf')
        for i in ids:
            uf = self.mesh[f"U_FOL_{i}"]
            ue = self.mesh[f"U_FE_{i}"]
            err = np.linalg.norm(np.linalg.norm(uf, axis=1) - np.linalg.norm(ue, axis=1))
            if err < min_err:
                min_err, best = err, i
        self.best_id = best
        self.fields = {
            "K_field": f"K_{best}",
            "U_FOL":   f"U_FOL_{best}",
            "U_FE":    f"U_FE_{best}"
        }
        print(f"Best sample = {best}, L2 error = {min_err:.6f}")

    def compute_derived_fields(self):
        fol, fe = self.fields['U_FOL'], self.fields['U_FE']
        self.mesh['U_FOL_mag'] = np.linalg.norm(self.mesh[fol], axis=1)
        self.mesh['U_FE_mag']  = np.linalg.norm(self.mesh[fe], axis=1)
        self.mesh['abs_error']  = np.abs(self.mesh['U_FOL_mag'] - self.mesh['U_FE_mag'])

    def apply_cut(self, mesh_obj):
        return mesh_obj.clip_box(
            bounds=(
                self.xmax - self.cut_size, self.xmax,
                self.ymax - self.cut_size, self.ymax,
                self.zmax - self.cut_size, self.zmax
            ),
            invert=True
        )

    def render_panel(self, mesh_obj, field, clim, title, fname, show_edges=False):
        plotter = pv.Plotter(off_screen=True, window_size=self.config['window_size'])
        plotter.add_mesh(
            mesh_obj, scalars=field, cmap=self.config['cmap'], clim=clim,
            show_edges=show_edges, edge_color='white', line_width=0.2,
            scalar_bar_args=self.shared_scalar_bar_args
        )
        plotter.camera_position = self.camera_position
        plotter.camera.zoom(self.shared_zoom)
        plotter.add_axes()
        plotter.add_text(title, font_size=self.config['title_font_size'], position='upper_edge')
        out = os.path.join(self.output_dir, fname)
        plotter.screenshot(out)
        plotter.close()
        print(f"Saved panel: {out}")

    def render_contour_slice(self):
        mesh = self.mesh
        fe_contour = mesh.contour(scalars="U_FE_mag", isosurfaces=5)
        fol_contour = mesh.contour(scalars="U_FOL_mag", isosurfaces=5)
        mesh.set_active_scalars("U_FE_mag")
        fe_slices = mesh.slice_orthogonal()
        mesh.set_active_scalars("U_FOL_mag")
        fol_slices = mesh.slice_orthogonal()
        plotter = pv.Plotter(
            shape=(2,2), off_screen=True,
            window_size=(3200,2400), border=False
        )
        plotter.subplot(0,0)
        plotter.add_text("FEM: Contour U_FE_mag", font_size=12)
        plotter.add_mesh(fe_contour, scalars="U_FE_mag", cmap=self.config['cmap'])
        plotter.add_axes()
        plotter.subplot(0,1)
        plotter.add_text("FEM: Orthogonal Slices", font_size=12)
        plotter.add_mesh(fe_slices, scalars="U_FE_mag", cmap=self.config['cmap'])
        plotter.add_axes()
        plotter.subplot(1,0)
        plotter.add_text("FOL: Contour U_FOL_mag", font_size=12)
        plotter.add_mesh(fol_contour, scalars="U_FOL_mag", cmap=self.config['cmap'])
        plotter.add_axes()
        plotter.subplot(1,1)
        plotter.add_text("FOL: Orthogonal Slices", font_size=12)
        plotter.add_mesh(fol_slices, scalars="U_FOL_mag", cmap=self.config['cmap'])
        plotter.add_axes()
        plotter.link_views()
        plotter.view_isometric()
        screenshot = os.path.join(self.output_dir, "fol_fem_contour_grid.png")
        plotter.screenshot(screenshot)
        plotter.close()
        print(f"Saved contour-slice grid: {screenshot}")

    def render_diagonal_plot(self):
        n = self.diag_points
        diag_pts = np.linspace([0,0,0], [1,1,1], n)
        probe = pv.PolyData(diag_pts).sample(self.mesh)
        fol_mag = probe['U_FOL_mag']
        fe_mag  = probe['U_FE_mag']
        K_diag  = probe.point_data[self.fields['K_field']]
        def compute_zoomed_ylim(data, zoom_factor):
            mn, mx = data.min(), data.max()
            ctr = 0.5*(mn+mx)
            hr  = 0.5*(mx-mn)/zoom_factor
            return [ctr-hr, ctr+hr]
        fig = plt.figure(figsize=self.config['final_figsize'])
        gs  = GridSpec(2,2,figure=fig)
        clipped_png = os.path.join(self.output_dir, 'panel8.png')
        ax0 = fig.add_subplot(gs[0,0])
        img = plt.imread(clipped_png)
        ax0.imshow(img); ax0.axis('off')
        ax0.set_title('Clipped Elasticity View', fontsize=self.config['title_font_size'])
        ax1 = fig.add_subplot(gs[0,1])
        x = np.linspace(0,1,n)
        ax1.plot(x, K_diag, linewidth=2)
        ax1.set_title('Elasticity Along Diagonal', fontsize=self.config['title_font_size'])
        ax1.set_xlabel('Normalized Distance')
        ax1.set_ylabel('Elasticity')
        ax1.set_ylim(compute_zoomed_ylim(K_diag, self.config['matplotlib_panel_zoom']['elasticity']))
        ax1.grid(False)
        ax2 = fig.add_subplot(gs[1,:])
        ax2.plot(x, fol_mag, label='FOL Disp', linewidth=2)
        ax2.plot(x, fe_mag,  label='FEM Disp', linestyle='--', linewidth=2)
        ax2.set_title('Displacement Magnitude Along Diagonal', fontsize=self.config['title_font_size'])
        ax2.set_xlabel('Normalized Distance')
        ax2.set_ylabel('Displacement')
        ax2.legend()
        combined = np.concatenate([fol_mag, fe_mag])
        ax2.set_ylim(compute_zoomed_ylim(combined, self.config['matplotlib_panel_zoom']['displacement']))
        ax2.grid(False)
        plt.tight_layout()
        out_diag = os.path.join(self.output_dir, 'diagonal_plot.png')
        plt.savefig(out_diag, dpi=300)
        plt.close(fig)
        print(f"Saved diagonal comparison plot: {out_diag}")

    def render_all_panels(self):
        # full pipeline
        self.find_best_sample()
        self.compute_derived_fields()
        # generate and save all panels
        panels = []
        base = self.apply_cut(self.mesh)
        K = self.fields['K_field']
        K_clim = [self.mesh[K].min(), self.mesh[K].max()]
        U_clim = [0, max(self.mesh['U_FOL_mag'].max(), self.mesh['U_FE_mag'].max())]
        panels.append((base, K, K_clim, 'Elasticity Morphology', 'panel1.png', False))
        panels.append((base, K, K_clim, 'E(x,y,z) with Mesh', 'panel2.png', True))
        p5, p4, p2 = np.array([1,0,1]), np.array([0,0,1]), np.array([0,1,0])
        normal = np.cross(p4-p5, p2-p5).astype(float); normal /= np.linalg.norm(normal)
        panels.append((self.mesh.clip(normal=normal, origin=p5, invert=False), K, None, '', 'panel8.png', False))
        mf = self.mesh.copy(deep=True); mf.active_vectors_name = self.fields['U_FOL']
        wf = mf.warp_by_vector(factor=self.warp_factor); wf['U_FOL_mag'] = mf['U_FOL_mag']
        panels.append((self.apply_cut(wf), 'U_FOL_mag', U_clim, 'FOL Deformation', 'panel5_warped.png', False))
        mf = self.mesh.copy(deep=True); mf.active_vectors_name = self.fields['U_FE']
        wf = mf.warp_by_vector(factor=self.warp_factor); wf['U_FE_mag'] = mf['U_FE_mag']
        panels.append((self.apply_cut(wf), 'U_FE_mag', U_clim, 'FEM Deformation', 'panel6_warped.png', False))
        panels.append((base, 'abs_error', [0, self.mesh['abs_error'].max()], '|U_FE - U_FOL|', 'panel7.png', False))
        for mesh_obj, field, clim, title, fname, edges in panels:
            self.render_panel(mesh_obj, field, clim, title, fname, show_edges=edges)
        fig = plt.figure(figsize=(20,12)); gs = GridSpec(2,3,figure=fig)
        for i, (_, _, _, _, fname, _) in enumerate(panels):
            ax = fig.add_subplot(gs[i//3, i%3])
            img = plt.imread(os.path.join(self.output_dir, fname))
            ax.imshow(img); ax.axis('off')
        combined = os.path.join(self.output_dir, self.config['output_image'])
        plt.tight_layout(); plt.savefig(combined, dpi=300); plt.close(fig)
        print(f"Saved combined figure: {combined}")
        # extra views
        self.render_contour_slice()
        self.render_diagonal_plot()
