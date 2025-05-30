import os
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import re

# ------------------------
# CONFIGURATION
# ------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
vtk_file = os.path.join(script_dir, "Box_3D_Tetra_20.vtk")
output_dir = script_dir
os.makedirs(output_dir, exist_ok=True)

# ------------------------
# LOAD MESH
# ------------------------
mesh = pv.read(vtk_file)
xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
cut_size = 0.5 * (xmax - xmin)

# ------------------------
# FIND BEST TEST ID BASED ON MIN ABS ERROR
# ------------------------
ufol_keys = [k for k in mesh.point_data if re.match(r"U_FOL_\d+", k)]
ufe_keys = [k for k in mesh.point_data if re.match(r"U_FE_\d+", k)]

common_ids = sorted(set(k.split("_")[-1] for k in ufol_keys) & set(k.split("_")[-1] for k in ufe_keys))

min_error = float("inf")
best_id = None

for i in common_ids:
    u_fol = mesh[f"U_FOL_{i}"]
    u_fe = mesh[f"U_FE_{i}"]
    error = np.linalg.norm(np.linalg.norm(u_fol, axis=1) - np.linalg.norm(u_fe, axis=1))
    if error < min_error:
        min_error = error
        best_id = i

print(f" Best prediction index = {best_id} with L2 error = {min_error:.6f}")

# ------------------------
# SELECT BEST FIELDS
# ------------------------
fields = {
    "K_field": f"K_{best_id}",
    "U_FOL": f"U_FOL_{best_id}",
    "U_FE": f"U_FE_{best_id}",
    "abs_error": f"abs_error_{best_id}"  # not used, recomputed below
}

# ------------------------
# SHARED SETTINGS
# ------------------------
shared_scalar_bar_args = {
    "title": "",
    "vertical": True,
    "title_font_size": 25,
    "label_font_size": 21,
    "position_x": 0.88,
    "position_y": 0.2,
    "width": 0.04,
    "height": 0.6,
    "font_family": "times",
}

shared_zoom = 0.6

# ------------------------
# DERIVED FIELDS
# ------------------------
mesh["U_FOL_mag"] = np.linalg.norm(mesh[fields["U_FOL"]], axis=1)
mesh["U_FE_mag"] = np.linalg.norm(mesh[fields["U_FE"]], axis=1)
mesh["abs_error"] = np.abs(mesh["U_FOL_mag"] - mesh["U_FE_mag"])

# ------------------------
# CUT FUNCTION
# ------------------------
def apply_cut(m):
    return m.clip_box(
        bounds=(xmax - cut_size, xmax,
                ymax - cut_size, ymax,
                zmax - cut_size, zmax),
        invert=True)

# ------------------------
# RENDER UTILITY
# ------------------------
def render_panel(mesh, field, cmap, clim, title, fname,
                 show_edges=False,
                 zoom=shared_zoom,
                 scalar_bar_args=shared_scalar_bar_args):
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.add_mesh(
        mesh,
        scalars=field,
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        edge_color="white",
        line_width=0.2,
        scalar_bar_args=scalar_bar_args
    )
    plotter.camera_position = [(2, 2, 2), (0.5, 0.5, 0.5), (0, 0, 1)]
    plotter.camera.zoom(zoom)
    plotter.add_axes()
    plotter.add_text(title, font_size=24, position="upper_edge")
    plotter.show(screenshot=os.path.join(output_dir, fname))

# ------------------------
# PANEL GENERATION
# ------------------------
K_field = fields["K_field"]
K_clim = [mesh[K_field].min(), mesh[K_field].max()]
render_panel(apply_cut(mesh), K_field, "coolwarm", K_clim, "Elasticity Morphology", "panel1.png")
render_panel(apply_cut(mesh), K_field, "coolwarm", K_clim, "E(x,y,z) with Mesh", "panel2.png", show_edges=True)

# Panel 8 - clipped
p5 = np.array([1, 0, 1])
p4 = np.array([0, 0, 1])
p2 = np.array([0, 1, 0])
v1 = p4 - p5
v2 = p2 - p5
normal = np.cross(v1, v2).astype(float)
normal /= np.linalg.norm(normal)
origin = p5
clipped_mesh = mesh.clip(normal=normal, origin=origin, invert=False)

plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
plotter.add_mesh(
    clipped_mesh,
    scalars=K_field,
    cmap="coolwarm",
    show_edges=False,
    scalar_bar_args=shared_scalar_bar_args
)
plotter.add_mesh(mesh, color="black", style="wireframe", line_width=1.0, opacity=0.4)
plotter.camera_position = [(2, 2, 2), (0.5, 0.5, 0.5), (0, 0, 1)]
plotter.camera.zoom(shared_zoom)
plotter.add_axes()
plotter.add_text("Clipped Elasticity View", font_size=24, position="upper_edge")
plotter.show(screenshot=os.path.join(output_dir, "panel8.png"))

U_clim = [0, max(mesh["U_FOL_mag"].max(), mesh["U_FE_mag"].max())]
render_panel(apply_cut(mesh), "U_FOL_mag", "coolwarm", U_clim, "FOL Displacement", "panel5.png")
render_panel(apply_cut(mesh), "U_FE_mag", "coolwarm", U_clim, "FEM Displacement", "panel6.png")
render_panel(apply_cut(mesh), "abs_error", "coolwarm", [0, mesh["abs_error"].max()], "|U_FE - U_FOL|", "panel7.png")

# ------------------------
# COMBINE SELECTED PANELS INTO GRID
# ------------------------
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 3, figure=fig)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[1, 2])
]

panel_files = [
    "panel1.png",
    "panel2.png",
    "panel8.png",
    "panel5.png",
    "panel6.png",
    "panel7.png"
]

labels = ["", "", "", "", "", ""]

for ax, fname, label in zip(axes, panel_files, labels):
    img = plt.imread(os.path.join(output_dir, fname))
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.15, 0.2, label, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top')

plt.tight_layout()
final_path = os.path.join(output_dir, "combined_selected_panels.png")
plt.savefig(final_path, dpi=300)
plt.show()

print(f"\n Selected panels (1,2,8,5,6,7) saved to '{final_path}'")
