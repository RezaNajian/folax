import os
import re
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ------------------------
# CONFIGURATION
# ------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
vtk_file = os.path.join(script_dir, "Box_3D_Tetra_20.vtk")
output_dir = script_dir
os.makedirs(output_dir, exist_ok=True)

config = {
    "zoom": 1.1,
    "cmap": "coolwarm",
    "window_size": (1000, 800),
    "scalar_bar_args": {
        "title": "",
        "vertical": True,
        "title_font_size": 25,
        "label_font_size": 28,
        "position_x": 0.82,
        "position_y": 0.1,
        "width": 0.09,
        "height": 0.8,
        "font_family": "times",
    },
    "title_font_size": 24,
    "diag_points": 100,
    "final_figsize": (20, 10),
    "output_image": "2D_displacement_panels_with_shared_style.png",
    "matplotlib_panel_zoom": {
        "displacement": 1,
        "elasticity": 0.8
    }
}

# ------------------------
# LOAD MESH & DETECT BEST ID
# ------------------------
mesh = pv.read(vtk_file)

ufol_keys = [k for k in mesh.point_data if re.match(r"U_FOL_\d+", k)]
ufe_keys = [k for k in mesh.point_data if re.match(r"U_FE_\d+", k)]
common_ids = sorted(set(k.split("_")[-1] for k in ufol_keys) & set(k.split("_")[-1] for k in ufe_keys))

min_error = float("inf")
best_id = None
for i in common_ids:
    u_fol = mesh[f"U_FOL_{i}"]
    u_fe = mesh[f"U_FE_{i}"]
    err = np.linalg.norm(np.linalg.norm(u_fol, axis=1) - np.linalg.norm(u_fe, axis=1))
    if err < min_error:
        min_error = err
        best_id = i

print(f" Best prediction index = {best_id} with L2 error = {min_error:.6f}")

# ------------------------
# FIELD NAMES BASED ON BEST ID
# ------------------------
fields = {
    "K_field": f"K_{best_id}",
    "U_FOL": f"U_FOL_{best_id}",
    "U_FE": f"U_FE_{best_id}",
}
K_field = fields["K_field"]

# ------------------------
# CLIP & COMPUTE DERIVED FIELDS
# ------------------------
p5, p4, p2 = np.array([1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
normal = np.cross(p4 - p5, p2 - p5).astype(float)
normal /= np.linalg.norm(normal)
clipped = mesh.clip(normal=normal, origin=p5, invert=False)

clipped["U_FOL_mag"] = np.linalg.norm(clipped[fields["U_FOL"]], axis=1)
clipped["U_FE_mag"] = np.linalg.norm(clipped[fields["U_FE"]], axis=1)
clipped["abs_error"] = np.abs(clipped["U_FOL_mag"] - clipped["U_FE_mag"])

# ------------------------
# PyVista RENDER FUNCTION
# ------------------------
def render_projected_panel(mesh, field, clim, title, filename):
    surf = mesh.extract_surface().triangulate()
    surf.points[:, 2] = 0
    plotter = pv.Plotter(off_screen=True, window_size=config["window_size"])
    plotter.add_mesh(
        surf,
        scalars=field,
        cmap=config["cmap"],
        clim=clim,
        show_edges=False,
        scalar_bar_args=config["scalar_bar_args"]
    )
    plotter.camera_position = "xy"
    plotter.camera.zoom(config["zoom"])
    plotter.add_text(title, font_size=config["title_font_size"], position="upper_edge")
    plotter.show(screenshot=os.path.join(output_dir, filename))

# ------------------------
# RENDER PANELS
# ------------------------
u_clim = [0, max(clipped["U_FOL_mag"].max(), clipped["U_FE_mag"].max())]
e_clim = [0, clipped["abs_error"].max()]
render_projected_panel(clipped, "U_FOL_mag", u_clim, "2D Displacement – FOL", "panel_2d_fol.png")
render_projected_panel(clipped, "U_FE_mag", u_clim, "2D Displacement – FEM", "panel_2d_fe.png")
render_projected_panel(clipped, "abs_error", e_clim, "Absolute Error |FOL - FEM|", "panel_2d_error.png")

# ------------------------
# DIAGONAL LINE DATA
# ------------------------
mesh["U_FOL_mag"] = np.linalg.norm(mesh[fields["U_FOL"]], axis=1)
mesh["U_FE_mag"] = np.linalg.norm(mesh[fields["U_FE"]], axis=1)

N = config["diag_points"]
diagonal_points = np.linspace([0, 0, 0], [1, 1, 1], N)
probe_line = pv.PolyData(diagonal_points)
probed = probe_line.sample(mesh)
fol_mag = np.linalg.norm(probed[fields["U_FOL"]], axis=1)
fe_mag = np.linalg.norm(probed[fields["U_FE"]], axis=1)
K_diag = probed.point_data.get(K_field)

# ------------------------
# PANEL ZOOM FUNCTION
# ------------------------
def compute_zoomed_ylim(data, zoom_factor):
    data_min, data_max = np.min(data), np.max(data)
    center = (data_min + data_max) / 2
    half_range = (data_max - data_min) / 2
    half_range /= zoom_factor
    return [center - half_range, center + half_range]

# ------------------------
# COMBINE FINAL FIGURE
# ------------------------
fig = plt.figure(figsize=config["final_figsize"])
gs = GridSpec(2, 3, height_ratios=[0.8, 0.6], figure=fig)

# First row: image panels
panel_files = ["panel_2d_fol.png", "panel_2d_fe.png", "panel_2d_error.png"]
for i, fname in enumerate(panel_files):
    ax = fig.add_subplot(gs[0, i])
    img = plt.imread(os.path.join(output_dir, fname))
    ax.imshow(img)
    ax.axis('off')

# Second row: displacement and elasticity plots
ax_disp = fig.add_subplot(gs[1, 0:2])
ax_disp.plot(np.linspace(0, 1, N), fol_mag, label="FOL Displacement", linewidth=2)
ax_disp.plot(np.linspace(0, 1, N), fe_mag, label="FEM Displacement", linestyle="--", linewidth=2)
ax_disp.set_xlabel("Normalized Distance Along Diagonal", fontsize=13)
ax_disp.set_ylabel("Displacement Magnitude", fontsize=13)
ax_disp.set_title("Displacement Magnitude Along Diagonal (0,0,0) → (1,1,1)", fontsize=13)
ax_disp.legend()
ax_disp.grid(False)

ylim_disp = compute_zoomed_ylim(np.concatenate([fol_mag, fe_mag]), config["matplotlib_panel_zoom"]["displacement"])
ax_disp.set_ylim(ylim_disp)

ax_k = fig.add_subplot(gs[1, 2])
ax_k.plot(np.linspace(0, 1, N), K_diag, color="black", linewidth=2, label="Elasticity")
ax_k.set_xlabel("Normalized Distance", fontsize=13)
ax_k.set_ylabel("Elasticity", fontsize=13)
ax_k.set_title("Elasticity Along Diagonal", fontsize=13)
ax_k.grid(False)
ax_k.legend()

ylim_K = compute_zoomed_ylim(K_diag, config["matplotlib_panel_zoom"]["elasticity"])
ax_k.set_ylim(ylim_K)

plt.tight_layout()
fig_path = os.path.join(output_dir, config["output_image"])
plt.savefig(fig_path, dpi=300)
plt.show()

print(f"\n Final figure saved with Matplotlib panel zoom at:\n{fig_path}")
