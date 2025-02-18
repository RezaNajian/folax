from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt
# directory & save handling

# problem setup
model_settings = {"L":1,"N":51,
                "T_left":1.0,"T_right":0.0}

# create some random coefficients & K for training
def generate_random_smooth_patterns(L, N, num_samples=7200, smoothness_levels=[0.025, 0.05, 0.1, 0.2, 0.3, 0.4]):
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
num_samples = 7200
coeffs_matrix = generate_random_smooth_patterns(model_settings["L"],model_settings["N"],num_samples=num_samples)
np.save(f"thermal_2d_gaussian_N{model_settings['N']}_num{num_samples}.npy", coeffs_matrix)
# Update the plotting function to use `imshow` instead of `contourf`
def plot_grid_of_samples_imshow(coeffs_matrix, L, N, num_samples, grid_size=10):
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X1, X2 = np.meshgrid(x, y)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    num_to_plot = grid_size ** 2
    for i, ax in enumerate(axes.flat):
        if i < coeffs_matrix.shape[0]:  # Ensure we don't exceed the number of samples
            sample = coeffs_matrix[i].reshape(X1.shape)
            im = ax.imshow(sample[::-1], cmap="jet", origin="lower", extent=(0, 1, 0, 1))
            # ax.set_title(f"Sample {i + 1}", fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")  # Hide unused subplots
    fig.tight_layout()
    plt.show()
    plt.savefig(f"thermal_2d_gaussian_N{model_settings['N']}_num{num_samples}.png")

# Plot the samples in a 5x5 grid using `imshow`
plot_grid_of_samples_imshow(coeffs_matrix,model_settings["L"],model_settings["N"], num_samples=num_samples,grid_size=10)

