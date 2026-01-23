import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import jax.numpy as jnp
import subprocess, sys, os, glob, shutil, base64
from PIL import Image
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import cv2
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pyvista as pv

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(layout="wide")
st.title("FOLAX INTERACTIVE MICROSTRUCTURE SIMULATION")

# =========================================================
# SESSION STATE
# =========================================================
st.session_state.setdefault("running_solver", False)

# =========================================================
# LOGO
# =========================================================
if os.path.exists("logo.png"):
    with open("logo.png", "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style="position:fixed;bottom:20px;right:20px;width:90px;z-index:9999;">
            <img src="data:image/png;base64,{logo_base64}" style="width:100%;">
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# SOLVER RUNNER
# =========================================================
def run_solver(cmd, results_folder):
    st.session_state.running_solver = True
    with st.spinner("Running solver..."):
        proc = subprocess.run(cmd, capture_output=True, text=True)

    st.subheader("Solver Output")
    # st.text_area("stdout", proc.stdout, height=200)
    # st.text_area("stderr", proc.stderr, height=200)

    if proc.returncode != 0:
        st.error("Solver failed")
    else:
        st.success("Solver finished")

        imgs = glob.glob(os.path.join(results_folder, "*.png"))
        cols = st.columns(3)
        for i, img in enumerate(imgs):
            cols[i % 3].image(img, use_container_width=True)

        shutil.make_archive("FOL_results", "zip", results_folder)
        with open("FOL_results.zip", "rb") as f:
            st.download_button("Download Results", f, "FOL_results.zip")

    st.session_state.running_solver = False

def save_microstructure_image(fig, filename, folders):
    import os

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        fig.savefig(path, bbox_inches="tight", dpi=300)


# =========================================================
# TABS
# =========================================================
tabs = st.tabs(["2D", "3D", "Upload Microstructure", "Paint"])

# =========================================================
# 2D TAB
# =========================================================
# =========================================================
# Voronoi / Periodic Voronoi Microstructures (HORIZONTAL)
# =========================================================
with tabs[0]:

    controls, output = st.columns([1, 1.4])

    # =====================================================
    # LEFT COLUMN — CONTROLS
    # =====================================================
    with controls:
        st.subheader("2D Microstructures")

        selection_2d = st.selectbox(
            "Select Microstructure Type",
            ["Voronoi", "Periodic Voronoi", "Fourier"],
            key="select_2d"
        )

        # =========================
        # VORONOI CONTROLS
        # =========================
        if selection_2d == "Voronoi":
            L = 1.0
            N_voronoi = st.slider("Grid Size", 10, 150, 50, key="N_voronoi")
            num_seed_points = st.slider("Number of Seeds", 5, 50, 10, key="seeds_voronoi")

            generate_voronoi = st.button("Generate Voronoi", use_container_width=True)

            st.divider()
            st.subheader("Deep Learning & Solver Options")

            epochs = st.slider("Number of Epochs", 100, 5000, 1000, step=100)
            run_fe = st.checkbox(
                "Run Finite Element Solver (compare results)",
                value=True,
                key="fe_voronoi"
            )

            run_otf = st.button(
                "Run OTF Deep Learning Model",
                disabled=st.session_state.running_solver,
                use_container_width=True
            )

            run_neo = st.button(
                "Run Pre-Trained NeoHookean Deep Learning Model",
                key="neo_voronoi",
                disabled=st.session_state.running_solver,
                use_container_width=True
            )

        # =========================
        # PERIODIC VORONOI CONTROLS
        # =========================
        elif selection_2d == "Periodic Voronoi":
            L = 1.0
            N_periodic = st.slider("Grid Size", 10, 150, 50, key="N_p")
            num_seed_points = st.slider("Number of Seeds", 5, 50, 10, key="seeds_p")

            generate_periodic = st.button(
                "Generate Periodic Voronoi",
                key="run_periodic",
                use_container_width=True
            )

            st.divider()
            st.subheader("Deep Learning & Solver Options")

            epochs_periodic = st.slider(
                "Number of Epochs", 100, 5000, 2000, step=100, key="epochs_periodic"
            )

            run_fe = st.checkbox(
                "Run Finite Element Solver (compare results)",
                value=True,
                key="fe_periodic"
            )

            run_otf = st.button(
                "Run OTF Deep Learning Model",
                key="fol_periodic",
                disabled=st.session_state.running_solver,
                use_container_width=True
            )

            run_neo = st.button(
                "Run Pre-Trained NeoHookean Deep Learning Model",
                key="neo_periodic",
                disabled=st.session_state.running_solver,
                use_container_width=True
            )

    # =====================================================
    # RIGHT COLUMN — OUTPUTS
    # =====================================================
    with output:

        # =========================
        # VORONOI OUTPUT
        # =========================
        if selection_2d == "Voronoi":

            if generate_voronoi:
                x_coord = np.random.rand(num_seed_points) * L
                y_coord = np.random.rand(num_seed_points) * L
                feature_values = np.random.rand(num_seed_points)

                X, Y = np.meshgrid(
                    np.linspace(0, L, N_voronoi),
                    np.linspace(0, L, N_voronoi)
                )

                seed_points = np.vstack((x_coord, y_coord)).T
                tree = KDTree(seed_points)
                grid_points = np.vstack([X.ravel(), Y.ravel()]).T
                _, regions = tree.query(grid_points)

                K = np.zeros_like(X)
                for i, region in enumerate(regions):
                    K.ravel()[i] = feature_values[region]

                coeffs_matrix = np.concatenate(
                    [x_coord, y_coord, feature_values]
                ).reshape(1, -1)

                K_matrix = jnp.array(K.reshape(1, -1))
                np.save("K_matrix.npy", np.array(K_matrix))

                st.session_state["voronoi"] = (K, coeffs_matrix, K_matrix)

            if "voronoi" in st.session_state:
                K, coeffs_matrix, K_matrix = st.session_state["voronoi"]

                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(
                    K, origin="upper", aspect="equal", extent=(0, L, 0, L)
                )
                plt.colorbar(im, ax=ax, label="Young's Modulus (E)")
                st.pyplot(fig)

                save_microstructure_image(
                    fig,
                    filename="microstructure.png",
                    folders=[
                        "./meta_implicit_mechanical_2D",
                        "./mechanical_2d_base_from_ifol_meta",
                    ],
                )
                plt.close(fig)

            # -------- Solver execution --------
            if run_otf:
                if "voronoi" not in st.session_state:
                    st.error("Generate Voronoi microstructure first!")
                else:
                    _, _, K_matrix = st.session_state["voronoi"]
                    np.save("K_matrix.npy", np.array(K_matrix))
                    run_solver(
                        [
                            sys.executable,
                            "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control2.py",
                            f"N={N_voronoi}",
                            f"ifol_num_epochs={epochs}",
                            f"fe_solver={run_fe}",
                            "clean_dir=False",
                        ],
                        results_folder="./meta_implicit_mechanical_2D",
                    )

            if run_neo:
                if "voronoi" not in st.session_state:
                    st.error("Generate Voronoi microstructure first!")
                else:
                    _, _, K_matrix = st.session_state["voronoi"]
                    np.save("K_matrix.npy", np.array(K_matrix))
                    run_solver(
                        [
                            sys.executable,
                            "run_pretrained_neohookean.py",
                            f"N={N_voronoi}",
                        ],
                        results_folder="./mechanical_2d_base_from_ifol_meta",
                    )

        # =========================
        # PERIODIC VORONOI OUTPUT
        # =========================
        elif selection_2d == "Periodic Voronoi":

            if generate_periodic:
                x_coord = np.random.rand(num_seed_points) * L
                y_coord = np.random.rand(num_seed_points) * L
                feature_values = np.random.rand(num_seed_points)

                X, Y = np.meshgrid(
                    np.linspace(0, L, N_periodic),
                    np.linspace(0, L, N_periodic)
                )

                K = np.zeros_like(X)
                seed_points = np.vstack((x_coord, y_coord)).T

                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        point = np.array([X[i, j], Y[i, j]])
                        distances = []
                        for dx in [-L, 0, L]:
                            for dy in [-L, 0, L]:
                                for sx, sy in seed_points:
                                    distances.append(
                                        np.sqrt(
                                            (point[0] - (sx + dx)) ** 2
                                            + (point[1] - (sy + dy)) ** 2
                                        )
                                    )
                        K[i, j] = feature_values[
                            np.argmin(distances) % len(feature_values)
                        ]

                st.session_state["periodic_2d"] = K

            if "periodic_2d" in st.session_state:
                K = st.session_state["periodic_2d"]

                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(
                    K,
                    extent=(0, L, 0, L),
                    origin="upper",
                    aspect="equal",
                    vmin=0,
                    vmax=1,
                )
                plt.colorbar(im, ax=ax, label="Young's Modulus (E)")
                st.pyplot(fig)

                save_microstructure_image(
                    fig,
                    filename="microstructure.png",
                    folders=[
                        "./meta_implicit_mechanical_2D",
                        "./mechanical_2d_base_from_ifol_meta",
                    ],
                )
                plt.close(fig)

            if run_otf:
                K_matrix = np.array(K.reshape(1, -1))
                np.save("K_matrix.npy", K_matrix)
                run_solver(
                    [
                        sys.executable,
                        "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control2.py",
                        f"N={N_periodic}",
                        f"ifol_num_epochs={epochs_periodic}",
                        f"fe_solver={run_fe}",
                        "clean_dir=False",
                    ],
                    results_folder="./meta_implicit_mechanical_2D",
                )

            if run_neo:
                K_matrix = np.array(K.reshape(1, -1))
                np.save("K_matrix.npy", K_matrix)
                run_solver(
                    [
                        sys.executable,
                        "run_pretrained_neohookean.py",
                        f"N={N_periodic}",
                    ],
                    results_folder="./mechanical_2d_base_from_ifol_meta",
                )


        # =========================================================
        # Fourier 2D Microstructure (FIXED – USE LEFT/RIGHT COLUMNS)
        # =========================================================
        elif selection_2d == "Fourier":

            # ===============================
            # LEFT COLUMN — CONTROLS
            # ===============================
            with controls:  # use the same 'controls' column defined at the top
                L = 1.0

                N_fourier = st.slider("Grid Size", 10, 100, 50, key="N_f")

                x_freqs = st.text_input(
                    "x Frequencies (comma-separated)", "1,2", key="x_freqs"
                )
                y_freqs = st.text_input(
                    "y Frequencies (comma-separated)", "1,2", key="y_freqs"
                )

                K_max = st.number_input("K_max", 1.0)
                K_min = st.number_input("K_min", 0.1)
                beta = st.number_input("Beta", 1.0)

                # -------- Parse frequencies --------
                try:
                    x_freqs_list = list(map(float, x_freqs.split(",")))
                    y_freqs_list = list(map(float, y_freqs.split(",")))
                except Exception:
                    x_freqs_list, y_freqs_list = [], []

                # -------- Coefficients --------
                coeffs = []
                if x_freqs_list and y_freqs_list and len(x_freqs_list) == len(y_freqs_list):
                    st.markdown("### Coefficients")
                    for i in range(len(x_freqs_list) + 1):
                        coeffs.append(
                            st.slider(
                                f"Coefficient {i}",
                                -5.0, 5.0, 0.0, 0.1,
                                key=f"coeff_{i}",
                            )
                        )

                generate_fourier = st.button(
                    "Generate Fourier Field",
                    key="run_fourier",
                    use_container_width=True,
                )

                st.divider()
                st.subheader("Deep Learning & Solver Options")
                epochs_fourier = st.slider(
                    "Number of Epochs", 100, 5000, 2000, step=100, key="epochs_fourier"
                )

                run_fe = st.checkbox(
                    "Run Finite Element Solver (compare results)",
                    value=True,
                    key="fe_fourier_2d",
                )

                run_otf = st.button(
                    "Run OTF Deep Learning Solver",
                    key="fol_fourier",
                    disabled=st.session_state.running_solver,
                    use_container_width=True,
                )

                run_neo = st.button(
                    "Run Pre-Trained NeoHookean Deep Learning Model",
                    key="neo_fourier",
                    disabled=st.session_state.running_solver,
                    use_container_width=True,
                )

            # ===============================
            # RIGHT COLUMN — OUTPUT
            # ===============================
            with output:  # use the same 'output' column defined at the top
                if generate_fourier:
                    x = np.linspace(0, L, N_fourier)
                    y = np.linspace(0, L, N_fourier)
                    X, Y = np.meshgrid(x, y)

                    K = coeffs[0] / 2.0
                    for i, (xf, yf) in enumerate(zip(x_freqs_list, y_freqs_list)):
                        K += (
                            coeffs[i + 1]
                            * np.cos(2 * np.pi * xf * X / L)
                            * np.cos(2 * np.pi * yf * Y / L)
                        )

                    sigmoid = lambda x: 1 / (1 + np.exp(-x))
                    K_mapped = (K_max - K_min) * sigmoid(beta * (K - 0.5)) + K_min

                    st.session_state["fourier_2d"] = K_mapped
                    np.save("K_matrix.npy", K_mapped.reshape(1, -1))

                if "fourier_2d" in st.session_state:
                    K = st.session_state["fourier_2d"]

                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(
                        K,
                        extent=(0, L, 0, L),
                        origin="upper",
                        aspect="equal",
                        vmin=0,
                        vmax=1,
                    )
                    plt.colorbar(im, ax=ax, label="Young's Modulus (E)")
                    st.pyplot(fig)

                    save_microstructure_image(
                        fig,
                        filename="microstructure.png",
                        folders=[
                            "./meta_implicit_mechanical_2D",
                            "./mechanical_2d_base_from_ifol_meta",
                        ],
                    )
                    plt.close(fig)

                if run_otf:
                    if "fourier_2d" not in st.session_state:
                        st.error("Generate the Fourier field first!")
                    else:
                        run_solver(
                            [
                                sys.executable,
                                "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control2.py",
                                f"N={N_fourier}",
                                f"ifol_num_epochs={epochs_fourier}",
                                f"fe_solver={run_fe}",
                                "clean_dir=False",
                            ],
                            results_folder="./meta_implicit_mechanical_2D",
                        )

                if run_neo:
                    if "fourier_2d" not in st.session_state:
                        st.error("Generate the Fourier field first!")
                    else:
                        run_solver(
                            [
                                sys.executable,
                                "run_pretrained_neohookean.py",
                                f"N={N_fourier}",
                            ],
                            results_folder="./mechanical_2d_base_from_ifol_meta",
                        )





# =========================================================
# 3D TAB
# =========================================================
with tabs[1]:
    st.subheader("3D Microstructures")
    selection_3d = st.selectbox(
        "Select 3D Microstructure Type",
        ["Fourier 3D"],
        key="select_3d"
    )

    if selection_3d == "Fourier 3D":
        L3D = 1.0

        # ==============================
        # Layout: 1/3 controls | 2/3 visual
        # ==============================
        col_ctrl, col_vis = st.columns([1, 2])

        # ==============================
        # LEFT: CONTROLS
        # ==============================
        with col_ctrl:
            N3D = st.slider("Grid Size (N, 3D)", 5, 50, 20, key="N3D")

            x_freqs_3d = st.text_input("x Frequencies (comma-separated)", "1,2", key="x3d")
            y_freqs_3d = st.text_input("y Frequencies (comma-separated)", "1,2", key="y3d")
            z_freqs_3d = st.text_input("z Frequencies (comma-separated)", "1,2", key="z3d")

            K_max3d = st.number_input("K_max (3D)", 1.0)
            K_min3d = st.number_input("K_min (3D)", 0.0)
            beta3d = st.number_input("Beta", 0.1, 10.0, 1.0)
            shift3d = st.number_input("Shift", -5.0, 5.0, 0.0)

            try:
                xf_list = list(map(float, x_freqs_3d.split(',')))
                yf_list = list(map(float, y_freqs_3d.split(',')))
                zf_list = list(map(float, z_freqs_3d.split(',')))
            except:
                xf_list = yf_list = zf_list = []

            coeffs3d = []
            if xf_list and yf_list and zf_list and len(xf_list) == len(yf_list) == len(zf_list):
                st.markdown("### Coefficients")
                for i in range(len(xf_list) + 1):
                    coeffs3d.append(
                        st.slider(
                            f"Coefficient {i} (3D)",
                            -5.0, 5.0, 0.0, 0.1,
                            key=f"coeff3d_{i}"
                        )
                    )

            if st.button("Generate 3D Fourier", key="run_3d"):
                x = np.linspace(0, L3D, N3D)
                y = np.linspace(0, L3D, N3D)
                z = np.linspace(0, L3D, N3D)

                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

                K3D = coeffs3d[0] / 2.0
                for i, (xf, yf, zf) in enumerate(zip(xf_list, yf_list, zf_list)):
                    K3D += (
                        coeffs3d[i + 1]
                        * np.cos(2 * np.pi * xf * X / L3D)
                        * np.cos(2 * np.pi * yf * Y / L3D)
                        * np.cos(2 * np.pi * zf * Z / L3D)
                    )

                sigmoid = lambda x: 1 / (1 + np.exp(-x))
                K3D_mapped = (K_max3d - K_min3d) * sigmoid(
                    beta3d * (K3D - shift3d)
                ) + K_min3d

                st.session_state['fourier_3d'] = (X, Y, Z, K3D_mapped)

            st.divider()

            fe_3d = st.checkbox(
                "Run Finite Element Solver (compare results)",
                value=True,
                key="fe_3d"
            )

            epochs_3d = st.slider(
                "Number of Epochs",
                100, 2000, 1000,
                step=100,
                key="epochs_3d"
            )

            if st.button("Run OTF Deep Learning Solver", key="fol_3d"):
                if 'fourier_3d' not in st.session_state:
                    st.error("Generate the 3D Fourier field first!")
                else:
                    _, _, _, K3D_mapped = st.session_state['fourier_3d']
                    fol_result_3d = run_fol_async(
                        K3D_mapped,
                        fol_num_epochs=epochs_3d,
                        display_plot=True,
                        is_3d=True
                    )
                    st.session_state['fourier_3d_fol_result'] = fol_result_3d

        # ==============================
        # RIGHT: 3D VISUALIZATION
        # ==============================
        with col_vis:
            if 'fourier_3d' in st.session_state:
                X, Y, Z, K3D_mapped = st.session_state['fourier_3d']

                fig = go.Figure(
                    data=go.Isosurface(
                        x=X.flatten(),
                        y=Y.flatten(),
                        z=Z.flatten(),
                        value=K3D_mapped.flatten(),
                        isomin=0,
                        isomax=1,
                        surface_count=25,
                        colorscale="Viridis",
                        colorbar=dict(title="Young's Modulus (E)")
                    )
                )

                fig.update_layout(
                    title="3D Fourier Microstructure",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z"
                    ),
                    margin=dict(l=0, r=0, t=40, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Generate a 3D Fourier microstructure to visualize it.")


# =========================================================
# IMAGE UPLOAD TAB
# =========================================================
with tabs[2]:  # Image / VTK Upload tab
    st.subheader("Upload Microstructure Image or VTK")

    uploaded_file = st.file_uploader("Upload an image or VTK file", type=["png", "jpg", "jpeg", "vtk"])

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()

        # =======================
        # VTK Upload Path
        # =======================
        if file_ext == "vtk":
            vtk_path = f"temp_{uploaded_file.name}"
            with open(vtk_path, "wb") as f:
                f.write(uploaded_file.read())

            mesh = pv.read(vtk_path)

            # ---- scalar field ----
            scalar_name = list(mesh.point_data.keys())[0]
            scalar = mesh.point_data[scalar_name]
            scalar_norm = (scalar - scalar.min()) / (scalar.max() - scalar.min())

            # ---- bounds ----
            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

            # NORMALIZATION
            if hasattr(mesh, "spacing") and mesh.spacing is not None:
                dims = mesh.dimensions  # (nx, ny, nz)
                nx, ny, nz = dims[0] - 1, dims[1] - 1, dims[2] - 1
                new_spacing = (
                    1.0 / nx if nx > 0 else 1,
                    1.0 / ny if ny > 0 else 1,
                    1.0 / nz if nz > 0 else 1,
                )
                new_origin = (0.0, 0.0, 0.0)
                scaled_mesh = pv.ImageData(dimensions=dims, spacing=new_spacing, origin=new_origin)
                scaled_mesh.point_data[scalar_name] = scalar_norm
            else:
                pts = mesh.points.copy()
                pts[:, 0] = (pts[:, 0] - xmin) / (xmax - xmin)
                pts[:, 1] = (pts[:, 1] - ymin) / (ymax - ymin)
                pts[:, 2] = (pts[:, 2] - zmin) / (zmax - zmin)
                mesh.points = pts
                mesh.point_data[scalar_name] = scalar_norm
                scaled_mesh = mesh

            # 3D Render Screenshot
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(scaled_mesh, scalars=scalar_name, cmap="viridis", show_edges=False)
            plotter.view_isometric()
            screenshot = plotter.screenshot()

            # -------------------------------
            # LEFT & RIGHT layout
            # -------------------------------
            controls, output = st.columns([1, 2])

            # Right column: 3D visualization
            with output:
                st.image(screenshot, caption="Normalized 3D Visualization")

            # Left column: sliders & final normalized slice
            with controls:
                y_norm = st.slider("Y Slice (normalized)", 0.01, 1.0, 0.5)
                N_slice = st.slider("Output Resolution (N × N)", 10, 200, 50)

            # 2D slice
            y_real = ymin + y_norm * (ymax - ymin)
            slice_plane = mesh.slice(normal="y", origin=(0, y_real, 0))
            pts_slice = slice_plane.points.copy()
            pts_slice[:, 0] = (pts_slice[:, 0] - xmin) / (xmax - xmin)
            pts_slice[:, 1] = (pts_slice[:, 1] - ymin) / (ymax - ymin)
            pts_slice[:, 2] = (pts_slice[:, 2] - zmin) / (zmax - zmin)
            slice_values = slice_plane.point_data[scalar_name]
            slice_values_norm = (slice_values - slice_values.min()) / (slice_values.max() - slice_values.min())
            x = pts_slice[:, 0]
            z = pts_slice[:, 2]

            # Interpolate slice to grid
            from scipy.interpolate import griddata
            grid_x = np.linspace(0, 1, N_slice)
            grid_z = np.linspace(0, 1, N_slice)
            GX, GZ = np.meshgrid(grid_x, grid_z)
            K_slice = griddata(
                points=np.vstack([x, z]).T,
                values=slice_values_norm,
                xi=(GX, GZ),
                method="linear",
                fill_value=0.0
            )
            K_slice = np.flipud(K_slice)
            st.session_state["vtk_microstructure"] = K_slice

            # Show final normalized microstructure
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            im2 = ax2.imshow(K_slice, cmap="viridis", origin='upper')
            ax2.axis("off")
            with controls:
                st.pyplot(fig2, use_container_width=False)
            np.save("K_matrix.npy", K_slice.reshape(1, -1))
            save_microstructure_image(fig2, filename="microstructure.png",
                                      folders=["./meta_implicit_mechanical_2D",
                                               "./mechanical_2d_base_from_ifol_meta"])

            # -------------------------------
            # RUN SOLVERS
            # -------------------------------
            st.subheader("Deep Learning & Solver Options")
            epochs_vtk = st.slider("Number of Epochs", 100, 5000, 2000, step=100, key="epochs_vtk")
            run_fe_vtk = st.checkbox("Run Finite Element Solver (compare results)", value=True, key="fe_vtk")

            if st.button("Run OTF Deep Learning Solver (VTK)", disabled=st.session_state.running_solver):
                if "vtk_microstructure" not in st.session_state:
                    st.error("Generate the microstructure slice first!")
                else:
                    np.save("K_matrix.npy", K_slice.reshape(1, -1))
                    cmd = [
                        sys.executable,
                        "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control2.py",
                        f"N={N_slice}",
                        f"ifol_num_epochs={epochs_vtk}",
                        f"fe_solver={run_fe_vtk}",
                        "clean_dir=False"
                    ]
                    run_solver(cmd, results_folder="./meta_implicit_mechanical_2D")

            if st.button("Run Pre-Trained NeoHookean Model (VTK)", disabled=st.session_state.running_solver):
                if "vtk_microstructure" not in st.session_state:
                    st.error("Generate the microstructure slice first!")
                else:
                    np.save("K_matrix.npy", K_slice.reshape(1, -1))
                    cmd = [
                        sys.executable,
                        "run_pretrained_neohookean.py",
                        f"N={N_slice}"
                    ]
                    run_solver(cmd, results_folder="./mechanical_2d_base_from_ifol_meta")


        # =======================
        # IMAGE Upload Path
        # =======================
        else:
            image = Image.open(uploaded_file).convert("RGB")

            # -------------------------------
            # LEFT & RIGHT layout
            # -------------------------------
            controls, output = st.columns([1, 2])
            with output:
                # st.image(image, caption="Uploaded Image", use_container_width=True)
                st.image(image, caption="Uploaded Image", width=500)

            with controls:
                method = st.selectbox("Select Segmentation Method", ["KMeans", "Mask R-CNN", "Unet"])
                N_img = st.slider("Grid Size (reduce image to N×N)", 10, 200, 50)

            # Segmentation
            if method == "KMeans":
                image_resized = image.resize((N_img, N_img))
                img_array = np.array(image_resized)
                flat_pixels = img_array.reshape(-1, 3)

                # n_phases = st.slider("Number of Phases (clusters)", 2, 10, 3)
                # ---- LEFT: slider (short) ----
                with controls:
                    n_phases = st.slider(
                        "Number of Phases (clusters)",
                        2, 4, 3
                    )
                kmeans = KMeans(n_clusters=n_phases, n_init=10, random_state=0)
                labels = kmeans.fit_predict(flat_pixels)

                feature_values = np.linspace(0.1, 1.0, n_phases)
                K_matrix = feature_values[labels].reshape(N_img, N_img)
                st.session_state["uploaded_microstructure"] = K_matrix

                # Show segmented image
                fig, ax = plt.subplots(figsize=(3, 3))

                im = ax.imshow(K_matrix, cmap="viridis", origin="upper")
                ax.axis("off")
                # st.pyplot(fig)
                # st.pyplot(fig, use_container_width=False)
                with output:  # 👈 THIS is the key
                    st.pyplot(fig, use_container_width=False)
                np.save("K_matrix.npy", K_matrix.reshape(1, -1))
                save_microstructure_image(fig, filename="microstructure.png",
                                          folders=["./meta_implicit_mechanical_2D",
                                                   "./mechanical_2d_base_from_ifol_meta"])

            elif method == "Mask R-CNN":
                st.warning("Mask R-CNN segmentation not implemented yet. Coming soon!")

            elif method == "Unet":
                st.warning("Unet segmentation not implemented yet. Coming soon!")

            # -------------------------------
            # RUN SOLVERS for IMAGE
            # -------------------------------
            
            with controls:
                st.subheader("Deep Learning & Solver Options")
                epochs_img = st.slider("Number of Epochs", 100, 5000, 2000, step=100, key="epochs_img")
                run_fe_img = st.checkbox("Run Finite Element Solver (compare results)", value=True, key="fe_img")

                if st.button("Run OTF Deep Learning Solver (Image)", disabled=st.session_state.running_solver):
                    if "uploaded_microstructure" not in st.session_state:
                        st.error("Generate the segmented microstructure first!")
                    else:
                        np.save("K_matrix.npy", K_matrix.reshape(1, -1))
                        cmd = [
                            sys.executable,
                            "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control2.py",
                            f"N={N_img}",
                            f"ifol_num_epochs={epochs_img}",
                            f"fe_solver={run_fe_img}",
                            "clean_dir=False"
                        ]
                        run_solver(cmd, results_folder="./meta_implicit_mechanical_2D")

                if st.button("Run Pre-Trained NeoHookean Model (Image)", disabled=st.session_state.running_solver):
                    if "uploaded_microstructure" not in st.session_state:
                        st.error("Generate the segmented microstructure first!")
                    else:
                        np.save("K_matrix.npy", K_matrix.reshape(1, -1))
                        cmd = [
                            sys.executable,
                            "run_pretrained_neohookean.py",
                            f"N={N_img}"
                        ]
                        run_solver(cmd, results_folder="./mechanical_2d_base_from_ifol_meta")




with tabs[3]:
    st.subheader("Paint Your Own Microstructure")

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from streamlit_drawable_canvas import st_canvas
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import sys

    # -------------------------------
    # Session state init (CRITICAL)
    # -------------------------------
    if "canvas_key" not in st.session_state:
        st.session_state["canvas_key"] = 0

    # -------------------------------
    # Colormap helpers
    # -------------------------------
    viridis = cm.get_cmap("viridis")

    def viridis_hex(x):
        r, g, b, _ = viridis(x)
        return mcolors.to_hex((r, g, b))

    # -------------------------------
    # Config
    # -------------------------------
    N_paint = 20
    canvas_size = 400
    background_color = viridis_hex(0.0)

    # -------------------------------
    # Controls
    # -------------------------------
    drawing_mode = st.selectbox(
        "Choose Tool:",
        ("freedraw", "line", "rect", "circle", "transform"),
        format_func=lambda m: {
            "freedraw": "✏️ Brush",
            "line": "📏 Line",
            "rect": "▭ Rectangle (filled)",
            "circle": "◯ Circle (filled)",
            "transform": "✥ Move / Rotate"
        }[m],
        key="paint_tool_selectbox"
    )

    brush_size = st.slider("Brush size", 1, 50, 20, key="paint_brush_size")

    paint_value = 1.0
    stroke_color = viridis_hex(paint_value)

    # -------------------------------
    # Clear canvas
    # -------------------------------
    if st.button("Clear Canvas", key="paint_clear_canvas"):
        st.session_state["canvas_key"] += 1

    # -------------------------------
    # Layout
    # -------------------------------
    left_col, right_col = st.columns([1, 1.2])

    # -------------------------------
    # Canvas
    # -------------------------------
    with left_col:
        canvas_result = st_canvas(
            fill_color=stroke_color + "FF",
            stroke_width=brush_size,
            stroke_color=stroke_color,
            background_color=background_color,
            height=canvas_size,
            width=canvas_size,
            drawing_mode=drawing_mode,
            key=f"paint_canvas_{st.session_state['canvas_key']}",
            update_streamlit=True,
            display_toolbar=True,
        )

        convert_btn = st.button(
            "Convert Painting to Microstructure Field",
            key="paint_convert_btn"
        )

    # Persistent placeholder (RIGHT)
    microstructure_placeholder = right_col.empty()

    # -------------------------------
    # Conversion
    # -------------------------------
    if convert_btn:
        if canvas_result.image_data is None:
            st.error("Draw something first!")
        else:
            img = canvas_result.image_data[:, :, :3].astype(float) / 255.0
            small = cv2.resize(
                img, (N_paint, N_paint),
                interpolation=cv2.INTER_AREA
            )

            viridis_colors = viridis(np.linspace(0, 1, 256))[:, :3]
            K_matrix = np.zeros((N_paint, N_paint))

            for i in range(N_paint):
                for j in range(N_paint):
                    pixel = small[i, j, :]
                    idx = ((viridis_colors - pixel) ** 2).sum(axis=1).argmin()
                    K_matrix[i, j] = idx / 255.0

            K_matrix = 0.1 + 0.9 * K_matrix
            st.session_state["paint_microstructure"] = K_matrix
            np.save("K_matrix.npy", K_matrix.reshape(1, -1))

            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(K_matrix, cmap="viridis", origin="upper")
            ax.axis("off")
            plt.colorbar(im, ax=ax)
            microstructure_placeholder.pyplot(fig)
            plt.close(fig)

            save_microstructure_image(
                fig,
                filename="microstructure.png",
                folders=[
                    "./meta_implicit_mechanical_2D",
                    "./mechanical_2d_base_from_ifol_meta"
                ]
            )

    elif "paint_microstructure" in st.session_state:
        K_matrix = st.session_state["paint_microstructure"]
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(K_matrix, cmap="viridis", origin="upper")
        ax.axis("off")
        plt.colorbar(im, ax=ax)
        microstructure_placeholder.pyplot(fig)
        plt.close(fig)

    # -------------------------------
    # Solver Controls
    # -------------------------------
    if "paint_microstructure" in st.session_state:
        st.divider()
        st.subheader("Deep Learning & FE Solver")

        epochs = st.slider(
            "Number of Epochs",
            100, 5000, 1500, 100,
            key="paint_epochs"
        )

        run_fe = st.checkbox(
            "Run FE solver",
            value=True,
            key="paint_run_fe"
        )

        if st.button(
            "Run OTF DL Solver (Painted)",
            disabled=st.session_state.running_solver,
            key="paint_run_otf"
        ):
            K_matrix = st.session_state["paint_microstructure"]
            np.save("K_matrix.npy", K_matrix.reshape(1, -1))

            cmd = [
                sys.executable,
                "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control2.py",
                f"N={N_paint}",
                f"ifol_num_epochs={epochs}",
                f"fe_solver={run_fe}",
                "clean_dir=False"
            ]
            run_solver(cmd, results_folder="./meta_implicit_mechanical_2D")

        if st.button(
            "Run Pretrained NeoHookean (Painted)",
            disabled=st.session_state.running_solver,
            key="paint_run_neo"
        ):
            K_matrix = st.session_state["paint_microstructure"]
            np.save("K_matrix.npy", K_matrix.reshape(1, -1))

            cmd = [
                sys.executable,
                "run_pretrained_neohookean.py",
                f"N={N_paint}"
            ]
            run_solver(cmd, results_folder="./mechanical_2d_base_from_ifol_meta")


# with tabs[3]:
#     st.subheader("Paint Your Own Microstructure")

#     import matplotlib.cm as cm
#     import matplotlib.colors as mcolors
#     from streamlit_drawable_canvas import st_canvas

#     viridis = cm.get_cmap("viridis")
#     def viridis_hex(x):
#         r, g, b, _ = viridis(x)
#         return mcolors.to_hex((r, g, b))

#     N_paint = 20
#     canvas_size = 400
#     background_color = viridis_hex(0.0)
#     # st.write(f"Canvas background color: `{background_color}`")

#     # -------------------------------
#     # Unique keys for all widgets
#     # -------------------------------
#     drawing_mode = st.selectbox(
#         "Choose Tool:",
#         ("freedraw", "line", "rect", "circle", "transform"),
#         format_func=lambda m: {
#             "freedraw": "✏️ Brush",
#             "line": "📏 Line",
#             "rect": "▭ Rectangle (filled)",
#             "circle": "◯ Circle (filled)",
#             "transform": "✥ Move / Rotate"
#         }[m],
#         key="paint_tool_selectbox"
#     )

#     brush_size = st.slider("Brush size", 1, 50, 20, key="paint_brush_size")
#     # paint_value = st.slider("Paint Value (Viridis)", 0.0, 1.0, 1.0, 0.05, key="paint_value_slider")
#     paint_value = 1.0
#     stroke_color = viridis_hex(paint_value)

#     if st.button("Clear Canvas", key="paint_clear_canvas"):
#         st.session_state["canvas_key"] = (st.session_state.get("canvas_key", 0) + 1) % 10000
#     else:
#         st.session_state.setdefault("canvas_key", 0)

#     # -------------------------------
#     # Layout: canvas left, preview right
#     # -------------------------------
#     left_col, right_col = st.columns([1, 1.2])

#     # Canvas
#     with left_col:
#         canvas_result = st_canvas(
#             fill_color=stroke_color + "FF",
#             stroke_width=brush_size,
#             stroke_color=stroke_color,
#             background_color=background_color,
#             height=canvas_size,
#             width=canvas_size,
#             drawing_mode=drawing_mode,
#             key=f"paint_canvas_{st.session_state['canvas_key']}",
#             update_streamlit=True,
#             display_toolbar=True,
#         )
#         convert_btn = st.button("Convert Painting to Microstructure Field", key="paint_convert_btn")

#     # Persistent placeholder for microstructure
#     microstructure_placeholder = right_col.empty()

#     # Conversion
#     if convert_btn:
#         if canvas_result.image_data is None:
#             st.error("Draw something first!")
#         else:
#             img = canvas_result.image_data[:, :, :3].astype(float) / 255.0
#             small = cv2.resize(img, (N_paint, N_paint), interpolation=cv2.INTER_AREA)

#             viridis_colors = viridis(np.linspace(0, 1, 256))[:, :3]
#             K_matrix = np.zeros((N_paint, N_paint))
#             for i in range(N_paint):
#                 for j in range(N_paint):
#                     pixel = small[i, j, :]
#                     idx = ((viridis_colors - pixel) ** 2).sum(axis=1).argmin()
#                     K_matrix[i, j] = idx / 255.0
#             K_matrix = 0.1 + 0.9 * K_matrix
#             st.session_state["paint_microstructure"] = K_matrix
#             np.save("K_matrix.npy", K_matrix.reshape(1, -1))

#             # Show microstructure
#             fig, ax = plt.subplots(figsize=(5, 5))
#             im = ax.imshow(K_matrix, cmap="viridis", origin="upper")
#             ax.axis("off")
#             plt.colorbar(im, ax=ax)
#             microstructure_placeholder.pyplot(fig)
#             save_microstructure_image(
#                 fig,
#                 filename="microstructure.png",
#                 folders=["./meta_implicit_mechanical_2D", "./mechanical_2d_base_from_ifol_meta"]
#             )
#             plt.close(fig)

#     elif "paint_microstructure" in st.session_state:
#         K_matrix = st.session_state["paint_microstructure"]
#         fig, ax = plt.subplots(figsize=(5, 5))
#         im = ax.imshow(K_matrix, cmap="viridis", origin="upper")
#         ax.axis("off")
#         plt.colorbar(im, ax=ax)
#         microstructure_placeholder.pyplot(fig)
#         plt.close(fig)

#     # Solver Controls
#     if "paint_microstructure" in st.session_state:
#         st.divider()
#         st.subheader("Deep Learning & FE Solver")
#         epochs = st.slider("Number of Epochs", 100, 5000, 1500, 100, key="paint_epochs")
#         run_fe = st.checkbox("Run FE solver", value=True, key="paint_run_fe")

#         if st.button("Run OTF DL Solver (Painted)", disabled=st.session_state.running_solver, key="paint_run_otf"):
#             K_matrix = st.session_state["paint_microstructure"]
#             np.save("K_matrix.npy", K_matrix.reshape(1, -1))
#             cmd = [
#                 sys.executable,
#                 "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control2.py",
#                 f"N={N_paint}",
#                 f"ifol_num_epochs={epochs}",
#                 f"fe_solver={run_fe}",
#                 "clean_dir=False"
#             ]
#             run_solver(cmd, results_folder="./meta_implicit_mechanical_2D")

#         if st.button("Run Pretrained NeoHookean (Painted)", disabled=st.session_state.running_solver, key="paint_run_neo"):
#             K_matrix = st.session_state["paint_microstructure"]
#             np.save("K_matrix.npy", K_matrix.reshape(1, -1))
#             cmd = [
#                 sys.executable,
#                 "run_pretrained_neohookean.py",
#                 f"N={N_paint}"
#             ]
#             run_solver(cmd, results_folder="./mechanical_2d_base_from_ifol_meta")
