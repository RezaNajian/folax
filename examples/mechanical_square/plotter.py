import glob,os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from fol.tools.plotter import Plotter2D  # <-- adjust this path if needed

case_dir = os.path.join('.', "2D_tri_nonlin_meta_alpha_pr_single_mesh_rectangle")
vtk_files = glob.glob(os.path.join(case_dir, "*.vtk"))
if vtk_files:
    vtk_path = vtk_files[0]
    config = {
        "cmap": "coolwarm",
        "u_fol_prefix": "U_FOL_",
        "u_fe_prefix": "U_HFE_",
        "output_image": "overview2d.png",
        "warp_factor_overview": 1.5,
        "scalar_bar_args": {
            "title": "", "vertical": True, "label_font_size": 20, "height": .9, "width": 0.09,
        },
        "zoom": 1.2,
        "clip": False,
        "show_edges": True,
        "window_size": (1600, 1000),
    }

    plotter = Plotter2D(vtk_path=vtk_path, config=config)
    plotter.render_all_panels()
else:
    print("No .vtk file found to visualize.")