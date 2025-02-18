from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt
from fol.mesh_input_output.mesh import Mesh
from matplotlib.colors import Normalize
import matplotlib.tri as tri
import warnings
warnings.filterwarnings("ignore")

fe_mesh = Mesh("fol_io","Li_battery_particle_scaled.med",'../meshes/')
fe_mesh.Initialize()
# create some random coefficients & K for training
def generate_random_smooth_patterns_from_mesh(coords, num_samples=10000,smoothness_levels=[0.15, 0.2, 0.3, 0.4, 0.5]):
    """
    Generate mixed random smooth patterns using a Gaussian Process with varying smoothness levels.
    Parameters:
        mesh (meshio.Mesh): A meshio object containing the coordinate information.
        num_samples (int): Total number of samples to generate (divided among smoothness levels).
        smoothness_levels (list): List of length scales for different smoothness levels.
    Returns:
        np.ndarray: A shuffled array of normalized samples from all smoothness levels.
    """
    # Extract coordinate points from the mesh
    X = coords[:, :2]  # Ensure only the first two coordinates are used if it's 3D
    all_samples = []
    for length_scale in smoothness_levels:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
        # Generate an equal number of samples per smoothness level
        num_per_level = num_samples // len(smoothness_levels)
        y_samples = gp.sample_y(X, n_samples=num_per_level, random_state=0)
        # Normalize each sample
        scaled_y_samples = np.array([
            2 * (y_sample - np.min(y_sample)) / (np.max(y_sample) - np.min(y_sample)) - 1
            for y_sample in y_samples.T
        ])
        all_samples.append(scaled_y_samples)
    # Concatenate all samples from different smoothness levels
    mixed_samples = np.vstack(all_samples)
    # Shuffle the samples randomly
    np.random.shuffle(mixed_samples)
    return mixed_samples

num_samples = 10000
coeffs_matrix = generate_random_smooth_patterns_from_mesh(fe_mesh.GetNodesCoordinates(),num_samples=num_samples)
N = fe_mesh.GetNodesCoordinates().shape[0]
np.save(f"particle_pf_2d_gaussian_N{N}_num{num_samples}.npy", coeffs_matrix)
# Update the plotting function to use `imshow` instead of `contourf`
def plot_triangulated(points, elements, values_list, titles=None, filename=None, value_range=None):
    """
    Plot multiple nodal solutions using triangular elements and save them in a single figure.
    Parameters:
    points (ndarray): (N, 2) array of node coordinates.
    elements (ndarray): (M, 3) array of triangular element connectivity.
    values_list (list of ndarray or single ndarray): List of (N,) arrays, each representing a nodal solution field.
    titles (list of str): Titles for each subplot.
    filename (str): If provided, saves the figure to the specified filename.
    value_range (tuple): (vmin, vmax) to fix the color scale. If None, automatically computed.
    """
    points = np.asarray(points, dtype=float)
    elements = np.asarray(elements, dtype=int)
    # Ensure values_list is a list of NumPy arrays
    if not isinstance(values_list, list):
        values_list = [values_list]
    values_list = [np.asarray(values, dtype=float).flatten() for values in values_list]
    # Compute global min/max for consistent color scaling
    if value_range is None:
        vmin, vmax = np.min(np.concatenate(values_list)), np.max(np.concatenate(values_list))
    else:
        vmin, vmax = value_range  # Use user-defined range
    norm = Normalize(vmin=vmin, vmax=vmax)
    num_plots = len(values_list)
    cols = min(5, num_plots)  # Up to 5 plots per row
    rows = (num_plots + cols - 1) // cols  # Compute number of rows dynamically
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True)
    axes = np.array(axes).reshape(-1)  # Flatten for easy indexing
    if titles is None:
        titles = [f"FE Solution {i}" for i in range(num_plots)]
    contour_plots = []  # Store contour plots for colorbar reference
    for i, (values, ax) in enumerate(zip(values_list, axes)):
        if len(values) != len(points):
            raise ValueError(f"Mismatch: {len(points)} nodes but {len(values)} values.")
        triang = tri.Triangulation(points[:, 0], points[:, 1], elements)
        contour = ax.tricontourf(triang, values, levels=1000, cmap="jet",vmin=vmin, vmax=vmax,norm=norm)
        # ax.triplot(triang, 'k-', alpha=0.3)  # Optional: mesh visualization
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.set_aspect('equal')  # Maintain correct aspect ratio
        contour_plots.append(contour)
    # Hide empty subplots (if num_plots < rows*cols)
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
    # Add a single colorbar spanning all subplots
    cbar_ax = fig.add_axes([1.05, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(contour_plots[0], ax=ax, orientation='vertical', fraction=0.05, pad=0.04,cax=cbar_ax)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_ticks(np.linspace(vmin, vmax, num=5))
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved figure as: {filename}")
    plt.show()

plot_samples = [[coeffs_matrix[i]] for i in range(25)]

# Plot the samples in a 5x5 grid using `imshow`
plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                                plot_samples,
                                filename=f"particle_pf_2d_gaussian_N{N}_num{num_samples}.png",value_range=(-1,1))

