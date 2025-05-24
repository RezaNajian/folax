import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib.collections import PatchCollection

def generate_random_smooth_patterns(L, N, num_samples=9000, smoothness_levels=[0.025, 0.05, 0.1, 0.2, 0.3, 0.4]):
    """
    Generate mixed random smooth patterns using a Gaussian Process with varying smoothness levels.
    Parameters:
        L (float): Length of the domain.
        N (int): Number of grid points per dimension.
        num_samples (int): Total number of samples to generate (divided among smoothness levels).
        smoothness_levels (list): List of length scales for different smoothness levels.
    Returns:
        np.ndarray: A shuffled array of normalized samples from all smoothness levels.
    """
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X1, X2 = np.meshgrid(x, y)
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    all_samples = []
    for length_scale in smoothness_levels:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
        # Generate an equal number of samples per smoothness level
        num_per_level = num_samples // len(smoothness_levels)
        y_samples = gp.sample_y(X, n_samples=num_per_level, random_state=0)
        # Normalize each sample
        scaled_y_samples = np.array([(y_sample - np.min(y_sample)) / (np.max(y_sample) - np.min(y_sample))
                                     for y_sample in y_samples.T])
        all_samples.append(scaled_y_samples)
    # Concatenate all samples from different smoothness levels
    mixed_samples = np.vstack(all_samples)
    # Shuffle the samples randomly
    np.random.shuffle(mixed_samples)
    return mixed_samples

def generate_random_smooth_patterns_evaluation(L, N, num_samples=6, smoothness_levels=[0.05, 0.1]):
    """
    Generate mixed random smooth patterns using a Gaussian Process with varying smoothness levels.

    Parameters:
        L (float): Length of the domain.
        N (int): Number of grid points per dimension.
        num_samples (int): Total number of samples to generate (divided among smoothness levels).
        smoothness_levels (list): List of length scales for different smoothness levels.

    Returns:
        np.ndarray: A shuffled array of normalized samples from all smoothness levels.
    """
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X1, X2 = np.meshgrid(x, y)
    X = np.vstack([X1.ravel(), X2.ravel()]).T

    all_samples = []

    for length_scale in smoothness_levels:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)

        # Generate an equal number of samples per smoothness level
        num_per_level = num_samples // len(smoothness_levels)
        y_samples = gp.sample_y(X, n_samples=num_per_level, random_state=0)

        # Normalize each sample
        scaled_y_samples = np.array([(y_sample - np.min(y_sample)) / (np.max(y_sample) - np.min(y_sample))
                                        for y_sample in y_samples.T])

        all_samples.append(scaled_y_samples)

    # Concatenate all samples from different smoothness levels
    mixed_samples = np.vstack(all_samples)

    # Shuffle the samples randomly
    np.random.shuffle(mixed_samples)

    return mixed_samples
    
def generate_morph_pattern(N):
    # Initialize hetero_morph array
    hetero_morph = np.full((N * N), 1.0)

    # Generate physical coordinates for a square domain of edge length 1
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    # Define shapes with their respective parameters
    shapes = [
        {"type": "circle", "center": (0.6, 0.9), "radius": 0.25},
        {"type": "ellipse", "center": (0.5, 0.4), "radii": (0.35, 0.2), "rotation": np.pi / 6},
        {"type": "circle", "center": (0.5, 0.0), "radius": 0.2}#,
        # {"type": "rectangle", "center": (0.25, 0.25), "size": (0.2, 0.1), "rotation": 0}
    ]

    # Apply conditions for each shape
    for shape in shapes:
        if shape["type"] == "circle":
            mask = (X - shape["center"][0])**2 + ((1-Y) - shape["center"][1])**2 < shape["radius"]**2
        elif shape["type"] == "ellipse":
            a, b = shape["radii"]
            cx, cy = shape["center"]
            X_rot = (X - cx) * np.cos(shape["rotation"]) - ((1-Y) - cy) * np.sin(shape["rotation"])
            Y_rot = (X - cx) * np.sin(shape["rotation"]) + ((1-Y) - cy) * np.cos(shape["rotation"])
            mask = (X_rot**2 / a**2) + (Y_rot**2 / b**2) < 1
        elif shape["type"] == "rectangle":
            w, h = shape["size"]
            cx, cy = shape["center"]
            mask = (np.abs(X - cx) < w / 2) & (np.abs((1-Y) - cy) < h / 2)

        # Flatten masks and apply to hetero_morph
        hetero_morph[mask.ravel()] = 0.1

    return hetero_morph


def plot_mesh_vec_data_thermal_row(L, vectors_list, subplot_titles=None, fig_title=None, cmap='viridis',
                       block_bool=False, colour_bar=True, colour_bar_name=None,
                       X_axis_name=None, Y_axis_name=None, show=False, file_name=None):
    num_vectors = len(vectors_list)
    if num_vectors < 1 or num_vectors > 8:
        raise ValueError("vectors_list must contain between 1 and 8 elements.")

    if subplot_titles is not None and len(subplot_titles) != num_vectors:
        raise ValueError("subplot_titles must have the same number of elements as vectors_list if provided.")

    # Determine the grid size for the subplots
    # grid_size = math.ceil(math.sqrt(num_vectors))
    fig, axs = plt.subplots(1, num_vectors, figsize=(5*num_vectors, 5), squeeze=False)
    
    # Flatten the axs array and hide unused subplots if any
    axs = axs.flatten()
    for ax in axs[num_vectors:]:
        ax.axis('off')

    vmin = min([v.min() for v in vectors_list])
    vmax = max([v.max() for v in vectors_list])

    for i, squared_mesh_vec_data in enumerate(vectors_list):
        N = int((squared_mesh_vec_data.reshape(-1, 1).shape[0])**0.5)
        im = axs[i].imshow(squared_mesh_vec_data.reshape(N, N), cmap=cmap, extent=[0, L, 0, L],vmin =vmin, vmax = vmax)

        if subplot_titles is not None:
            axs[i].set_title(subplot_titles[i])
        else:
            axs[i].set_title(f'Plot {i+1}')

        if colour_bar and i == num_vectors-1:
            cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=28)
            cbar.set_ticks(np.linspace(vmin, vmax, 3))

        if X_axis_name is not None:
            axs[i].set_xlabel(X_axis_name)

        if Y_axis_name is not None:
            axs[i].set_ylabel(Y_axis_name)
        
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    if fig_title is not None:
        plt.suptitle(fig_title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if show:
        plt.show(block=block_bool)

    if file_name is not None:
        plt.savefig(file_name)

def plot_mesh_quad(points, elements, background=None, filename=None, show=False):
    """
    Plot the mesh using given node coordinates and element connectivity over a background image or contour.
    
    Parameters:
    points (ndarray): (N, 2) array of node coordinates.
    elements (ndarray): (M, 3) or (M, 4) array of element connectivity (triangles or quadrilaterals).
    background (ndarray, optional): 2D array to be plotted using imshow as a background.
    extent (tuple, optional): (xmin, xmax, ymin, ymax) extent for the background image.
    filename (str, optional): If provided, saves the figure to the specified filename.
    """
    points = np.asarray(points, dtype=float)
    elements = np.asarray(elements, dtype=int)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the background image if provided
    if background is not None:
        ax.imshow(background, extent=[0,1,0,1], origin='lower', cmap='viridis', alpha=1)

    patches_list = []
    for elem in elements:
        if len(elem) == 3:  # Triangle
            polygon = patches.Polygon(points[elem], edgecolor='k', facecolor='none', linewidth=0.5)
        elif len(elem) == 4:  # Quadrilateral
            polygon = patches.Polygon(points[elem], edgecolor='k', facecolor='none', linewidth=0.5)
        else:
            raise ValueError("Elements must have 3 (triangular) or 4 (quadrilateral) nodes.")
        patches_list.append(polygon)

    patch_collection = PatchCollection(patches_list, match_original=True)
    ax.add_collection(patch_collection)

    ax.set_aspect('equal')
    ax.autoscale()  # Adjust limits based on mesh

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    if show:
        plt.show()