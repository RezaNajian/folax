import sys
import os
import optax
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from fol.tools.usefull_functions import *
# problem setup
model_settings = {"L":1,"N":51,
                "T_left":1.0,"T_right":0.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])


# create some random samples for training and store them

def generate_random_smooth_patterns(L, N, num_samples=1000):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
    # Create the grid
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X1, X2 = np.meshgrid(x, y)
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    # Generate multiple samples
    y_samples = gp.sample_y(X, n_samples=num_samples, random_state=0)
    # Normalize each sample
    scaled_y_samples = np.array([((y_sample - np.min(y_sample)) / (np.max(y_sample) - np.min(y_sample))) 
                                 for y_sample in y_samples.T])
    return scaled_y_samples
num_samples = 12000
samples = generate_random_smooth_patterns(model_settings["L"],model_settings["N"],num_samples=num_samples)
np.save(f"gaussian_N{model_settings['N']}_num{num_samples}.npy", samples)


