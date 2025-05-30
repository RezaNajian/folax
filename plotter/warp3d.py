import os
import re
import pyvista as pv
import numpy as np

# ------------------------
# CONFIGURATION
# ------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
vtk_file = os.path.join(script_dir, "Box_3D_Tetra_20.vtk")  # <-- update if needed
output_dir = script_dir
os.makedirs(output_dir, exist_ok=True)

output_image = os.path.join(output_dir, "warped_displacement_dual.png")
warp_factor = 20  #  fixed magnification

# ------------------------
# LOAD MESH
# ------------------------
mesh = pv.read(vtk_file)

# ------------------------
# FIND BEST PREDICTION ID BASED ON L2 ERROR
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
# DEFINE FIELD NAMES
# ------------------------
U_FOL_field = f"U_FOL_{best_id}"
U_FE_field = f"U_FE_{best_id}"

# ------------------------
# COMPUTE MAGNITUDES FOR COLORING
# ------------------------
mesh.point_data["U_FOL_mag"] = np.linalg.norm(mesh[U_FOL_field], axis=1)
mesh.point_data["U_FE_mag"] = np.linalg.norm(mesh[U_FE_field], axis=1)

# ------------------------
# WARP BY VECTOR FIELD
# ------------------------
mesh.active_vectors_name = U_FOL_field
warped_fol = mesh.warp_by_vector(factor=warp_factor)
warped_fol["magnitude"] = mesh["U_FOL_mag"]

mesh.active_vectors_name = U_FE_field
warped_fe = mesh.warp_by_vector(factor=warp_factor)
warped_fe["magnitude"] = mesh["U_FE_mag"]

# ------------------------
# CUSTOM CAMERA (rotated ISO manually)
# ------------------------
custom_camera = [(2, 5, 3),       # eye position (camera location)
                 (0.8, 0.5, 0.5),   # focal point (look at center of domain)
                 (0, 0, 1.5)]         # up direction (z-up)

# ------------------------
# RENDER 1×2 PANEL OF WARPED RESULTS
# ------------------------
plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800), off_screen=True)

# FOL Panel
plotter.subplot(0, 0)
plotter.add_text("Warped by U_FOL", font_size=14)
plotter.add_mesh(warped_fol, scalars="magnitude", cmap="coolwarm", show_edges=False)
plotter.camera_position = custom_camera
plotter.camera.zoom(1)
plotter.add_axes()

# FEM Panel
plotter.subplot(0, 1)
plotter.add_text("Warped by U_FE", font_size=14)
plotter.add_mesh(warped_fe, scalars="magnitude", cmap="coolwarm", show_edges=False)
plotter.camera_position = custom_camera
plotter.camera.zoom(1)
plotter.add_axes()

plotter.show(screenshot=output_image)
print(f"\n Warped displacement panels saved to:\n{output_image}")
