import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import jax.numpy as jnp
import subprocess
import sys
import os
import glob
import math
import time
import plotly.graph_objects as go
import shutil
from PIL import Image
from sklearn.cluster import KMeans

# Constants
IMAGE_WIDTH = 6
IMAGE_HEIGHT = 6
DPI = 200

st.title("FOLAX INTERACTIVE MICROSTRUCTURE SIMULATION")

# --- Initialize session state ---
if "running_solver" not in st.session_state:
    st.session_state.running_solver = False

# --- Tabs ---
tabs = st.tabs(["2D", "3D", "Image Upload"])

def run_solver(cmd):
    import shutil

    st.session_state.running_solver = True
    try:
        with st.spinner("Running FOL solver... please wait"):
            process = subprocess.run(cmd, capture_output=True, text=True)

        st.subheader("Solver Output")
        st.text_area("stdout", process.stdout, height=200)
        st.text_area("stderr", process.stderr, height=200)

        if process.returncode != 0:
            st.error("Solver failed. Check stderr.")
        else:
            st.success("FOL Solver finished!")


            
            # result_images = glob.glob("./meta_implicit_mechanical_2D/*.png")
            # for img in result_images:
            #     st.image(img, caption=os.path.basename(img), use_container_width=True)
            result_images = glob.glob("./meta_implicit_mechanical_2D/*.png")
            cols_per_row = 2  # number of images per row

            for i in range(0, len(result_images), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, img_path in enumerate(result_images[i:i+cols_per_row]):
                    cols[j].image(img_path, caption=os.path.basename(img_path), use_container_width=True)
            

            # Create ZIP of results folder
            results_folder = "./meta_implicit_mechanical_2D"
            zip_filename = "FOL_results.zip"
            shutil.make_archive("FOL_results", "zip", results_folder)

            # Streamlit button for downloading the ZIP
            with open(zip_filename, "rb") as f:
                st.download_button(
                    label="Download All Results",
                    data=f,
                    file_name=zip_filename,
                    mime="application/zip"
                )

    finally:
        st.session_state.running_solver = False

# =========================================================
# Voronoi Microstructure
# =========================================================
with tabs[0]:
    st.subheader("2D Microstructures")
    selection_2d = st.selectbox("Select Microstructure Type", ["Voronoi", "Periodic Voronoi", "Fourier"], key="select_2d")

    if selection_2d == "Voronoi":
        L = 1.0
        N_voronoi = st.slider("Grid Size", 10, 150, 50, key="N_voronoi")
        num_seed_points = st.slider("Number of Seeds", 5, 50, 10, key="seeds_voronoi")
        if st.button("Generate Voronoi"):
            x_coord = np.random.rand(num_seed_points) * L
            y_coord = np.random.rand(num_seed_points) * L
            feature_values = np.random.rand(num_seed_points)

            X, Y = np.meshgrid(np.linspace(0, L, N_voronoi), np.linspace(0, L, N_voronoi))
            seed_points = np.vstack((x_coord, y_coord)).T
            tree = KDTree(seed_points)
            grid_points = np.vstack([X.ravel(), Y.ravel()]).T
            _, regions = tree.query(grid_points)

            K = np.zeros_like(X)
            for i, region in enumerate(regions):
                K.ravel()[i] = feature_values[region]

            coeffs_matrix = np.concatenate([x_coord, y_coord, feature_values]).reshape(1, -1)
            K_matrix = jnp.array(K.reshape(1, -1))
            np.save("K_matrix.npy", np.array(K_matrix))

            st.session_state['voronoi'] = (K, coeffs_matrix, K_matrix)

        if 'voronoi' in st.session_state:
            K, coeffs_matrix, K_matrix = st.session_state['voronoi']
            fig, ax = plt.subplots(figsize=(6,6))
            im = ax.imshow(K, origin='upper', aspect='equal', extent=(0,L,0,L))
            plt.colorbar(im, ax=ax, label="Young's Modulus (E)")
            st.pyplot(fig)

        epochs = st.slider("Number of Epochs", 100, 5000, 1000, step=100)

        run_fe = st.checkbox("Run Finite Element Solver (compare with iFOL)", value=True, key="fe_voronoi")
        if st.button("Run OTF iFOL Solver", disabled=st.session_state.running_solver):
            if 'voronoi' not in st.session_state:
                st.error("Generate Voronoi microstructure first!")
            else:
                K, coeffs_matrix, K_matrix = st.session_state['voronoi']
                np.save("K_matrix.npy", np.array(K_matrix))
                cmd = [
                    sys.executable,
                    "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control.py",
                    f"N={N_voronoi}",
                    f"ifol_num_epochs={epochs}",
                    f"fe_solver={run_fe}",
                    "clean_dir=False"
                ]
                run_solver(cmd)


    # =========================================================
    # Periodic Voronoi
    # =========================================================
    elif selection_2d == "Periodic Voronoi":
        L = 1.0
        N_periodic = st.slider("Grid Size", 10, 150, 50, key="N_p")
        num_seed_points = st.slider("Number of Seeds", 5, 50, 10, key="seeds_p")

        if st.button("Generate Periodic Voronoi", key="run_periodic"):
            x_coord = np.random.rand(num_seed_points) * L
            y_coord = np.random.rand(num_seed_points) * L
            feature_values = np.random.rand(num_seed_points)

            X, Y = np.meshgrid(np.linspace(0,L,N_periodic), np.linspace(0,L,N_periodic))
            K = np.zeros_like(X)
            seed_points = np.vstack((x_coord, y_coord)).T

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    point = np.array([X[i,j], Y[i,j]])
                    distances = []
                    for dx in [-L,0,L]:
                        for dy in [-L,0,L]:
                            for sx, sy in seed_points:
                                distances.append(np.sqrt((point[0]-(sx+dx))**2 + (point[1]-(sy+dy))**2))
                    K[i,j] = feature_values[np.argmin(distances)%len(feature_values)]

            st.session_state['periodic_2d'] = K

        if 'periodic_2d' in st.session_state:
            K = st.session_state['periodic_2d']
            fig, ax = plt.subplots(figsize=(6,6))
            im = ax.imshow(K, extent=(0,L,0,L), origin='upper', aspect='equal', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label="Young's Modulus (E)")
            st.pyplot(fig)

            epochs_periodic = st.slider("Number of Epochs", 100, 5000, 2000, step=100, key="epochs_periodic")

            run_fe = st.checkbox("Run Finite Element Solver (compare with iFOL)", value=True, key="fe_periodic")

            if st.button("Run OTF iFOL Solver", key="fol_periodic", disabled=st.session_state.running_solver):
                K_matrix = np.array(K.reshape(1, -1))
                np.save("K_matrix.npy", K_matrix)
                cmd = [
                    sys.executable,
                    "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control.py",
                    f"N={N_periodic}",
                    f"ifol_num_epochs={epochs_periodic}",
                    f"fe_solver={run_fe}",
                    "clean_dir=False"
                ]
                run_solver(cmd)

    # =========================================================
    # Fourier
    # =========================================================
    elif selection_2d == "Fourier":
        L = 1.0
        N_fourier = st.slider("Grid Size", 10, 100, 50, key="N_f")
        x_freqs = st.text_input("x Frequencies (comma-separated)", "1,2", key="x_freqs")
        y_freqs = st.text_input("y Frequencies (comma-separated)", "1,2", key="y_freqs")
        K_max = st.number_input("K_max", 1.0)
        K_min = st.number_input("K_min", 0.1)
        beta = st.number_input("Beta", 1.0)

        try:
            x_freqs_list = list(map(float, x_freqs.split(',')))
            y_freqs_list = list(map(float, y_freqs.split(',')))
        except:
            x_freqs_list, y_freqs_list = [], []

        coeffs = []
        if x_freqs_list and y_freqs_list and len(x_freqs_list) == len(y_freqs_list):
            for i in range(len(x_freqs_list)+1):
                coeffs.append(st.slider(f"Coefficient {i}", -5.0, 5.0, 0.0, 0.1, key=f"coeff_{i}"))

        if st.button("Generate Fourier Field", key="run_fourier"):
            x = np.linspace(0, L, N_fourier)
            y = np.linspace(0, L, N_fourier)
            X, Y = np.meshgrid(x, y)

            K = coeffs[0] / 2.0
            for i, (xf, yf) in enumerate(zip(x_freqs_list, y_freqs_list)):
                K += coeffs[i+1] * np.cos(2*np.pi*xf*X/L) * np.cos(2*np.pi*yf*Y/L)

            sigmoid = lambda x: 1/(1+np.exp(-x))
            K_mapped = (K_max-K_min) * sigmoid(beta * (K - 0.5)) + K_min

            st.session_state['fourier_2d'] = K_mapped
            np.save("K_matrix.npy", K_mapped.reshape(1, -1))

        if 'fourier_2d' in st.session_state:
            K = st.session_state['fourier_2d']
            fig, ax = plt.subplots(figsize=(IMAGE_WIDTH, IMAGE_HEIGHT), dpi=DPI)
            im = ax.imshow(K, extent=(0,L,0,L), origin='upper', aspect='equal', vmin=0,vmax=1)
            plt.colorbar(im, ax=ax, label="Young's Modulus (E)")
            st.pyplot(fig)

            epochs_fourier = st.slider("Number of Epochs", 100, 5000, 2000, step=100, key="epochs_fourier")

            run_fe = st.checkbox("Run Finite Element Solver (compare with iFOL)", value=True, key="fe_fourier")
            if st.button("Run OTF iFOL Solver", key="fol_fourier", disabled=st.session_state.running_solver):
                if 'fourier_2d' not in st.session_state:
                    st.error("Generate the Fourier field first!")
                else:
                    cmd = [
                        sys.executable,
                        "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control.py",
                        f"N={N_fourier}",
                        f"ifol_num_epochs={epochs_fourier}",
                        f"fe_solver={run_fe}",
                        "clean_dir=False"
                    ]
                    run_solver(cmd)

# =========================================================
# ------------------------ 3D TAB -------------------------
# =========================================================
with tabs[1]:
    st.subheader("3D Microstructures")
    selection_3d = st.selectbox("Select 3D Microstructure Type", ["Fourier 3D"], key="select_3d")

    if selection_3d == "Fourier 3D":
        L3D = 1.0
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
            for i in range(len(xf_list)+1):
                coeffs3d.append(st.slider(f"Coefficient {i} (3D)", -5.0, 5.0, 0.0, 0.1, key=f"coeff3d_{i}"))

        if st.button("Generate 3D Fourier", key="run_3d"):
            x = np.linspace(0, L3D, N3D)
            y = np.linspace(0, L3D, N3D)
            z = np.linspace(0, L3D, N3D)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

            K3D = coeffs3d[0] / 2.0
            for i, (xf, yf, zf) in enumerate(zip(xf_list, yf_list, zf_list)):
                K3D += coeffs3d[i+1] * np.cos(2*np.pi*xf*X/L3D) * np.cos(2*np.pi*yf*Y/L3D) * np.cos(2*np.pi*zf*Z/L3D)

            sigmoid = lambda x: 1/(1+np.exp(-x))
            K3D_mapped = (K_max3d-K_min3d) * sigmoid(beta3d*(K3D-shift3d)) + K_min3d
            st.session_state['fourier_3d'] = (X, Y, Z, K3D_mapped)

        if 'fourier_3d' in st.session_state:
            X, Y, Z, K3D_mapped = st.session_state['fourier_3d']

            fig = go.Figure(data=go.Isosurface(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=K3D_mapped.flatten(),
                isomin=0, isomax=1,
                surface_count=25,
                colorscale='Viridis',
                colorbar=dict(title="Young's Modulus (E)")
            ))
            fig.update_layout(
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                title="3D Fourier Microstructure"
            )
            st.plotly_chart(fig)

            fe_3d = st.checkbox("Run Finite Element Solver (compare with iFOL)", value=True, key="fe_3d")
            epochs_3d = st.slider("Number of Epochs", 100, 2000, 1000, step=100, key="epochs_3d")

            if st.button("Run OTF iFOL Solver", key="fol_3d"):
                fol_result_3d = run_fol_async(K3D_mapped, fol_num_epochs=epochs_3d, display_plot=True, is_3d=True)
                st.session_state['fourier_3d_fol_result'] = fol_result_3d

            # # ASCII download
            # ascii_3d = ""
            # for i in range(int(N3D)):
            #     ascii_slice = generate_binary_ascii(K3D_mapped[:, :, i])
            #     ascii_3d += f"# Slice {i}\n" + ascii_slice + "\n\n"
            # display_ascii_download_link(ascii_3d, "3d_fourier_binary_ascii.txt")


with tabs[2]:  # Image Upload tab
    st.subheader("Upload Microstructure Image")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Open and show image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # User selects grid size
        N_img = st.slider("Grid Size (reduce image to N×N)", 10, 200, 50)

        # Resize to grid
        image_resized = image.resize((N_img, N_img))
        img_array = np.array(image_resized)

        # Flatten pixels
        flat_pixels = img_array.reshape(-1, 3)

        # --- Segmentation using KMeans ---
        n_phases = st.slider("Number of Phases (clusters)", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_phases, n_init=10, random_state=0)
        labels = kmeans.fit_predict(flat_pixels)

        # Map each cluster to a material property (Young’s modulus)
        feature_values = np.linspace(0.1, 1.0, n_phases)  # scale E between 0.1 and 1.0
        K_matrix = feature_values[labels].reshape(N_img, N_img)

        # Save K_matrix for solver
        np.save("K_matrix.npy", K_matrix.reshape(1, -1))
        st.session_state["uploaded_microstructure"] = K_matrix

        # --- Show segmented image ---
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(K_matrix, cmap="viridis", origin="upper")
        plt.colorbar(im, ax=ax, label="Young's Modulus (E)")
        st.pyplot(fig)

        # --- Option to run solver ---
        epochs_upload = st.slider("Number of Epochs", 100, 5000, 2000, step=100, key="epochs_upload")
        run_fe = st.checkbox("Run Finite Element Solver (compare with iFOL)", value=True, key="fe_upload")
        if st.button("Run OTF iFOL Solver", key="solver_upload", disabled=st.session_state.running_solver):
            cmd = [
                sys.executable,
                "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control.py",
                f"N={N_img}",
                f"ifol_num_epochs={epochs_upload}",
                f"fe_solver={run_fe}",
                "clean_dir=False"
            ]
            run_solver(cmd)
