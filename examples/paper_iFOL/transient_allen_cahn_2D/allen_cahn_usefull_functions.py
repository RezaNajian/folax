from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np

def generate_random_smooth_patterns(coords, num_samples=10000,smoothness_levels=[0.15, 0.2, 0.3, 0.4, 0.5]):
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

def generate_fixed_gaussian_basis_field(coords, num_samples=1, num_basis=25, length_scale=0.1, random_seed=1):
    """
    Generate multiple random smooth patterns using fixed Gaussian/RBF basis functions.

    Parameters:
        coords (np.ndarray): Array of shape (N, 2) representing mesh node coordinates.
        num_samples (int): Number of random field samples to generate.
        num_basis (int): Number of fixed RBF basis functions.
        length_scale (float): Controls the smoothness of the function.
        random_seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Generated fields of shape (num_samples, N)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Normalize mesh coordinates to [0,1]
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    norm_coords = (coords - min_coords) / (max_coords - min_coords)

    # Define fixed basis function centers
    basis_centers = np.random.rand(num_basis, 2)
    # Compute distances
    distances = cdist(norm_coords, basis_centers)
    basis_values = np.exp(- (distances ** 2) / (2 * length_scale ** 2))

    # Generate multiple sets of coefficients
    coefficients = np.random.randn(num_samples, num_basis)
    # Compute all fields
    fields = coefficients @ basis_values.T  # shape: (num_samples, N)

    # Normalize each field to [-1, 1]
    min_vals = fields.min(axis=1, keepdims=True)
    max_vals = fields.max(axis=1, keepdims=True)
    fields = 2 * (fields - min_vals) / (max_vals - min_vals) - 1

    return fields

def plot_triangulated(points, elements, values_list, titles=None, filename=None, value_range=None, row=False, show=False):
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
        if row:
            cols = num_plots
            rows = 1
        else:
            cols = min(2, num_plots)  # Up to 2 plots per row
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
            contour = ax.tricontourf(triang, values, levels=1000, cmap="jet",vmin=vmin, vmax=vmax)
            # ax.triplot(triang, 'k-', alpha=0.3)  # Optional: mesh visualization
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.set_aspect('equal')  # Maintain correct aspect ratio
            contour_plots.append(contour)
            cmap = cm.jet  # Choose a colormap
            # Create a ScalarMappable and set the norm
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Empty array since it's independent
            # Create the color bar
            cbar = fig.colorbar(sm, ax = ax, orientation='vertical', fraction=0.05, pad=0.04)
            # Add a single colorbar spanning all subplots
            cbar.ax.tick_params(labelsize=24)
            cbar.set_ticks(np.linspace(vmin, vmax, num=5))
    
        # Hide empty subplots (if num_plots < rows*cols)
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()

def plot_mesh_tri(points, elements, filename=None, show=False):
    """
    Plot the triangular mesh using given node coordinates and element connectivity. 
    Parameters:
    points (ndarray): (N, 2) array of node coordinates.
    elements (ndarray): (M, 3) array of triangular element connectivity.
    filename (str): If provided, saves the figure to the specified filename.
    """
    points = np.asarray(points, dtype=float)
    elements = np.asarray(elements, dtype=int)  
    fig, ax = plt.subplots(figsize=(6, 6))
    triang = tri.Triangulation(points[:, 0], points[:, 1], elements)
    ax.triplot(triang, 'k-',linewidth=0.5, alpha=1.0)  # Plot mesh edges
    # ax.set_frame_on(False)
    ax.set_aspect('equal')  # Maintain correct aspect ratio
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
