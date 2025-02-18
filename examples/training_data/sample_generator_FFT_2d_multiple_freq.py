import sys
import os
import optax
import numpy as np
import scipy.interpolate
from fol.tools.usefull_functions import *
# problem setup
model_settings = {"L":1,"N":16,
                "T_left":1.0,"T_right":0.0}
working_directory_name = 'mesh'
case_dir = os.path.join('.', working_directory_name)

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create some random samples for training and store them

def generate_random_smooth_patterns_2d_fft(mesh, num_samples=1000, length_scale=0.1):
    """
    Generate random smooth patterns in 2D using Fourier-based methods on meshio node coordinates,
    with a single length scale.
    
    Parameters:
        mesh (np.ndarray): The input mesh containing node coordinates (num_nodes, 2).
        num_samples (int): The number of random patterns to generate.
        length_scale (float): Length scale to control smoothness.
    
    Returns:
        np.ndarray: An array of shape (num_samples, num_nodes) with generated patterns.
    """
    # Extract node coordinates
    coords = mesh[:, :2]
    num_nodes = coords.shape[0]
    
    # Determine domain size (bounding box dimensions)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    L = max_coords - min_coords
    
    # Choose an appropriate grid resolution based on the number of nodes
    N = int(np.sqrt(num_nodes))  # Approximate square root to define a grid
    
    # Create the frequency grid
    kx = np.fft.fftfreq(N, d=L[0] / N)
    ky = np.fft.fftfreq(N, d=L[1] / N)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    
    # Compute the squared frequency magnitudes
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # Avoid division by zero at the origin
    
    # Create the power spectrum for a Gaussian kernel at the given length scale
    power_spectrum = np.exp(-0.5 * (K2 * (length_scale * 2 * np.pi)**2))
    
    # Create the structured grid for interpolation
    x = np.linspace(min_coords[0], max_coords[0], N)
    y = np.linspace(min_coords[1], max_coords[1], N)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    
    # Generate random smooth patterns
    patterns = np.zeros((num_samples, num_nodes))
    for i in range(num_samples):
        # Generate random Fourier coefficients (complex with Hermitian symmetry)
        random_coeffs = (
            np.random.normal(size=(N, N)) + 
            1j * np.random.normal(size=(N, N))
        )
        random_coeffs[0, 0] = 0.0  # Remove the DC component
        random_coeffs *= np.sqrt(power_spectrum)
        
        # Perform inverse FFT to generate the spatial pattern
        pattern_grid = np.fft.ifftn(random_coeffs).real
        
        # Normalize the pattern to the range [0, 1]
        pattern_grid = (pattern_grid - pattern_grid.min()) / (pattern_grid.max() - pattern_grid.min())
        
        # Interpolate the pattern onto the meshio node coordinates
        interpolator = scipy.interpolate.RegularGridInterpolator((x, y), pattern_grid, bounds_error=False, fill_value=0)
        patterns[i, :] = interpolator(coords)
    
    return patterns


num_samples = 5000
# length_scale = 0.05
samples = np.zeros((num_samples,model_settings["N"]**2))
length_scale_list = [0.05,0.1,0.2,0.3,0.4]
num_samples_per_length_scale = num_samples//len(length_scale_list)
for i in range(5):
    samples[num_samples_per_length_scale*i:(num_samples_per_length_scale*(i+1))]= generate_random_smooth_patterns_2d_fft(fe_mesh.GetNodesCoordinates(),num_samples=1000,length_scale=length_scale_list[i])   

np.random.shuffle(samples)
np.save(f"2d_fft_N{model_settings['N']}_num{num_samples}_multiple_freq.npy", samples)

def plot_patterns(patterns, N,num_plots=5):
    """
    Plot a selection of generated smooth patterns using imshow.
    
    Parameters:
        patterns (np.ndarray): The generated patterns of shape (num_samples, N, N).
        x (np.ndarray): The x-coordinates of the grid.
        y (np.ndarray): The y-coordinates of the grid.
        num_plots (int): Number of patterns to plot.
    """
    num_plots = min(num_plots, patterns.shape[0])
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
    if num_plots == 1:
        axes = [axes]
    for i in range(num_plots):
        im = axes[i].imshow(patterns[i].reshape(N,N), extent=[0,1, 0,1], origin='lower', cmap='viridis')
        plt.colorbar(im, ax=axes[i])
        axes[i].set_title(f"Pattern {i+1}")
    plt.show()
    plt.savefig(f"2d_fft_N{model_settings['N']}_num{num_samples}.png")


plot_patterns(samples,model_settings["N"], num_plots=5)
