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
fe_mesh = create_3D_box_mesh(Nx=model_settings["N"]-1,
                             Ny=model_settings["N"]-1,
                             Nz=model_settings["N"]-1,
                             Lx=model_settings["L"],
                             Ly=model_settings["L"],
                             Lz=model_settings["L"],
                             case_dir=case_dir)

# create some random samples for training and store them


def generate_random_smooth_patterns_3d_fft(mesh, num_samples=1000, length_scale=0.1):
    """
    Generate random smooth patterns in 3D using Fourier-based methods on meshio node coordinates.
    
    Parameters:
        mesh (meshio.Mesh): The input mesh containing node coordinates.
        num_samples (int): The number of random patterns to generate.
        length_scale (float): Controls the smoothness of the patterns.
    
    Returns:
        np.ndarray: An array of shape (num_samples, num_nodes) with generated patterns.
    """
    # Extract node coordinates
    coords = mesh
    num_nodes = coords.shape[0]
    
    # Determine domain size (bounding box dimensions)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    L = max_coords - min_coords
    
    # Choose an appropriate grid resolution based on the number of nodes
    N = int(np.cbrt(num_nodes))  # Approximate cubic root to define a grid
    
    # Create the frequency grid
    kx = np.fft.fftfreq(N, d=L[0] / N)
    ky = np.fft.fftfreq(N, d=L[1] / N)
    kz = np.fft.fftfreq(N, d=L[2] / N)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    
    # Compute the squared frequency magnitudes
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0  # Avoid division by zero at the origin
    
    # Create the power spectrum for a Gaussian kernel
    power_spectrum = np.exp(-0.5 * (K2 * (length_scale * 2 * np.pi)**2))
    
    # Create the structured grid for interpolation
    x = np.linspace(min_coords[0], max_coords[0], N)
    y = np.linspace(min_coords[1], max_coords[1], N)
    z = np.linspace(min_coords[2], max_coords[2], N)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")
    
    # Generate random smooth patterns
    patterns = np.zeros((num_samples, num_nodes))
    for i in range(num_samples):
        # Generate random Fourier coefficients (complex with Hermitian symmetry)
        random_coeffs = (
            np.random.normal(size=(N, N, N)) + 
            1j * np.random.normal(size=(N, N, N))
        )
        random_coeffs[0, 0, 0] = 0.0  # Remove the DC component
        random_coeffs *= np.sqrt(power_spectrum)
        
        # Perform inverse FFT to generate the spatial pattern
        pattern_grid = np.fft.ifftn(random_coeffs).real
        
        # Normalize the pattern to the range [0, 1]
        pattern_grid = (pattern_grid - pattern_grid.min()) / (pattern_grid.max() - pattern_grid.min())
        
        # Interpolate the pattern onto the meshio node coordinates
        interpolator = scipy.interpolate.RegularGridInterpolator((x, y, z), pattern_grid, bounds_error=False, fill_value=0)
        patterns[i, :] = interpolator(coords)
    
    return patterns


num_samples = 1000*5
length_scale = 0.05
samples = np.zeros((num_samples,model_settings["N"]**3))
length_scale_list = [0.05,0.1,0.2,0.3,0.4]
for i in range(5):
    samples[1000*i:(1000*(i+1))]= generate_random_smooth_patterns_3d_fft(fe_mesh.GetNodesCoordinates(),num_samples=1000,length_scale=length_scale_list[i])   

samples = np.array(samples)
np.random.shuffle(samples)
print(samples.shape)
np.save(f"3d_fft_N{model_settings['N']}_num{num_samples}_multiple_freq.npy", samples)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
slice_idx = model_settings["N"] // 2  # Select the middle slice for visualization

for i in range(3):
    for j, axis in enumerate(['X-Y', 'Y-Z', 'X-Z']):
        if axis == 'X-Y':
            slice_data = samples[i][slice_idx, :, :]
        elif axis == 'Y-Z':
            slice_data = samples[i][:, slice_idx, :]
        elif axis == 'X-Z':
            slice_data = samples[i][:, :, slice_idx]

        axes[i, j].imshow(slice_data, cmap='viridis', origin='lower')
        axes[i, j].set_title(f'Sample {i+1}, {axis} Slice')
        axes[i, j].axis('off')

# plt.tight_layout()
plt.show()
plt.savefig(f"3d_fft_N{model_settings['N']}_num{num_samples}.png")