import sys
import os
import optax
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from fol.tools.usefull_functions import *
# problem setup
model_settings = {"L":1,"N":32,
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

def generate_random_smooth_patterns_3d_fft(L, N, num_samples=1000, length_scale=0.1):
    """
    Generate random smooth patterns in 3D using Fourier-based methods.
    
    Parameters:
        L (float): The length of the domain in each dimension.
        N (int): The number of grid points per dimension.
        num_samples (int): The number of random patterns to generate.
        length_scale (float): Controls the smoothness of the patterns.
    
    Returns:
        np.ndarray: An array of shape (num_samples, N, N, N) with generated patterns.
    """
    # Create the frequency grid
    kx = np.fft.fftfreq(N, d=L / N)
    ky = np.fft.fftfreq(N, d=L / N)
    kz = np.fft.fftfreq(N, d=L / N)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

    # Compute the squared frequency magnitudes
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0  # Avoid division by zero at the origin

    # Create the power spectrum for a Gaussian kernel
    power_spectrum = np.exp(-0.5 * (K2 * (length_scale * 2 * np.pi)**2))

    # Generate random smooth patterns
    patterns = []
    for _ in range(num_samples):
        # Generate random Fourier coefficients (complex with Hermitian symmetry)
        random_coeffs = (
            np.random.normal(size=(N, N, N)) + 
            1j * np.random.normal(size=(N, N, N))
        )
        random_coeffs[0, 0, 0] = 0.0  # Remove the DC component
        random_coeffs *= np.sqrt(power_spectrum)

        # Perform inverse FFT to generate the spatial pattern
        pattern = np.fft.ifftn(random_coeffs).real

        # Normalize the pattern to the range [0, 1]
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        patterns.append(pattern)

    return np.array(patterns)

num_samples = 5000
samples = generate_random_smooth_patterns_3d_fft(model_settings["L"],model_settings["N"],num_samples=num_samples,length_scale=0.1)
np.save(f"3d_fft_N{model_settings['N']}_num{num_samples}.npy", samples)

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