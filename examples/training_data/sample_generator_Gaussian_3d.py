import sys
import os
import optax
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
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

def generate_random_smooth_patterns_3d(L, N, num_samples=1000):
    # Define the kernel for the Gaussian Process
    kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)

    # Create the 3D grid
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    z = np.linspace(0, L, N)
    X1, X2, X3 = np.meshgrid(x, y, z)
    X = np.vstack([X1.ravel(), X2.ravel(), X3.ravel()]).T

    # Generate multiple samples
    y_samples = gp.sample_y(X, n_samples=num_samples, random_state=0)

    # Normalize each sample
    scaled_y_samples = np.array([
        ((y_sample - np.min(y_sample)) / (np.max(y_sample) - np.min(y_sample)))
        for y_sample in y_samples.T
    ])

    # Reshape each sample back to 3D
    scaled_y_samples_3d = scaled_y_samples.reshape(num_samples, N*N*N)

    return scaled_y_samples_3d
num_samples = 5000
samples = generate_random_smooth_patterns_3d(model_settings["L"],model_settings["N"],num_samples=num_samples)
np.save(f"3d_gaussian_N{model_settings['N']}_num{num_samples}.npy", samples)


