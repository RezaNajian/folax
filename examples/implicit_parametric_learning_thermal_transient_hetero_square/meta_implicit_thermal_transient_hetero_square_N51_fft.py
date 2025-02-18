import sys
import os
import optax
import numpy as np

from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning_field_hetero import MetaImplicitParametricOperatorLearning
from fol.solvers.fe_linear_residual_based_solver_hetero import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
from fol.loss_functions.thermal_transient_hetero import ThermalTransientLoss2DQuad
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import scipy.interpolate
import jax
jax.config.update("jax_default_matmul_precision", "float32")
# directory & save handling
working_directory_name = 'siren_implicit_thermal_2D_dt_0005_N51_fft'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":51,
                "T_left":1.0,"T_right":0.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]}}

material_dict = {"rho":1.0,"cp":1.0,"dt":0.005}
thermal_loss_2d = ThermalTransientLoss2DQuad("thermal_transient_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)
no_control = NoControl("No_control",fe_mesh)

fe_mesh.Initialize()
thermal_loss_2d.Initialize()
no_control.Initialize()

# create some random coefficients & K for training
create_samples = True
if create_samples:
    def generate_random_smooth_patterns(L, N, num_samples=10000, smoothness_levels=[0.05, 0.1, 0.2, 0.3, 0.4]):
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

        # Define radii in terms of the physical domain (adjust based on desired coverage)
        radius1 = 0.25  # Example radius values in the domain's units
        radius2 = 0.2
        radius3 = 0.2

        # Define condition centers in terms of physical coordinates
        center1 = (0.3, 0.7)
        center2 = (0.7, 0.15)
        center3 = (0.2, 0.0)

        # Calculate distances for each condition and apply them to the array
        mask1 = (X - center1[0])**2 + ((1-Y) - center1[1])**2 < radius1**2
        mask2 = (X - center2[0])**2 + ((1-Y) - center2[1])**2 < radius2**2
        mask3 = (X - center3[0])**2 + ((1-Y) - center3[1])**2 < radius3**2

        # Flatten masks and apply to hetero_morph
        hetero_morph[mask1.ravel() | mask2.ravel() | mask3.ravel()] = 0.1

        return hetero_morph


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

    # Generate samples from FFT-based method
    num_samples = 10000
    coeffs_matrix = np.zeros((num_samples,model_settings["N"]**2))
    length_scale_list = [0.05,0.1,0.2,0.3,0.4]
    num_samples_per_length_scale = num_samples//len(length_scale_list)
    for i in range(len(length_scale_list)):
        coeffs_matrix[num_samples_per_length_scale*i:(num_samples_per_length_scale*(i+1))]= generate_random_smooth_patterns_2d_fft(fe_mesh.GetNodesCoordinates(),
                                                                                                                                   num_samples=num_samples_per_length_scale,
                                                                                                                                   length_scale=length_scale_list[i])   
    np.random.shuffle(coeffs_matrix)
    np.save(os.path.join(case_dir,"T_samples.npy"),coeffs_matrix)
    # coeffs_matrix = generate_random_smooth_patterns(model_settings["L"],model_settings["N"])
    hetero_info = generate_morph_pattern(model_settings["N"]).reshape(1,-1) #np.full((1,fe_mesh.GetNumberOfNodes()),1.0)#
    np.save(os.path.join(case_dir,"hetero_info.npy"),hetero_info)

else:
    pass

# hetero_matrix = np.tile(hetero_info,(coeffs_matrix.shape[0],1))
T_matrix = no_control.ComputeBatchControlledVariables(coeffs_matrix)
K_matrix = no_control.ComputeBatchControlledVariables(hetero_info)
# print(hetero_matrix.shape)

# design synthesizer & modulator NN for hypernetwork
characteristic_length = model_settings["N"]
characteristic_length = 256
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=3,
                     output_size=1,
                     hidden_layers=[characteristic_length] * 6,
                     activation_settings={"type":"sin",
                                          "prediction_gain":30,
                                          "initialization_gain":1.0},
                     skip_connections_settings={"active":False,"frequency":1})

latent_size = characteristic_length
modulator_nn = MLP(name="modulator_nn",
                   input_size=latent_size,
                   use_bias=False) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 10000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=no_control,
                                            loss_function=thermal_loss_2d,
                                            hetero_info=K_matrix.flatten(),
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3,
                                            checkpoint_settings={"restore_state":False,
                                            "state_directory":case_dir+"/flax_state"},
                                            working_directory=case_dir)
fol.Initialize()


train_start_id = 0
train_end_id = 6000
test_start_id = 1*train_end_id
test_end_id = int(1.2*train_end_id)
# here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
        #   test_set=(coeffs_matrix[test_start_id:test_end_id,:],),
# #         #   test_settings={"test_frequency":10},
fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
            batch_size=120,
            convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
            plot_settings={"plot_save_rate":100},
            save_settings={"save_nn_model":True,
                         "best_model_checkpointing":True,
                         "best_model_checkpointing_frequency":100})

# load teh best model
fol.RestoreCheckPoint(fol.checkpoint_settings)

# relative_L2_error = 0.0
num_steps = 50
eval_start_id = 0
eval_end_id = 1
FOL_T = np.zeros((fe_mesh.GetNumberOfNodes(),num_steps))
for test in range(eval_start_id,eval_end_id):
    eval_id = test
    FOL_T_temp  = coeffs_matrix[eval_id,:]
    for i in range(num_steps):
        FOL_T_temp = np.array(fol.Predict(FOL_T_temp.reshape(-1,1).T)).reshape(-1)
        FOL_T[:,i] = FOL_T_temp 

    fe_mesh['T_FOL'] = FOL_T#.reshape((fe_mesh.GetNumberOfNodes(), 1))

    # solve FE here
    fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu"},
                    "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                "maxiter":10,"load_incr":5}}
    linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",thermal_loss_2d,fe_setting)
    linear_fe_solver.Initialize()
    FE_T = np.zeros((fe_mesh.GetNumberOfNodes(),num_steps))
    FE_T_temp = coeffs_matrix[eval_id,:]
    for i in range(num_steps):
        FE_T_temp = np.array(linear_fe_solver.Solve(FE_T_temp,K_matrix,FE_T_temp))  #np.zeros(fe_mesh.GetNumberOfNodes())
        FE_T[:,i] = FE_T_temp    
    fe_mesh['T_FE'] = FE_T#.reshape((fe_mesh.GetNumberOfNodes(), 1))

    absolute_error = np.abs(FOL_T- FE_T)
    fe_mesh['abs_error'] = absolute_error#.reshape((fe_mesh.GetNumberOfNodes(), 1))
    
fe_mesh['Heterogeneity'] = K_matrix.reshape(-1,1)
fe_mesh['T_init'] = coeffs_matrix[eval_start_id,:].reshape(-1,1)

eval_id = eval_start_id
time_list = [int(num_steps/5) - 1,int(num_steps/2) - 1,num_steps - 1]
plot_mesh_vec_data_thermal(1,[coeffs_matrix[eval_id],FOL_T[:,time_list[0]],FOL_T[:,time_list[1]],FOL_T[:,time_list[2]]],
                   ["","","",""],
                   fig_title="Initial condition and implicit FOL solution",cmap = "jet",
                   file_name=os.path.join(case_dir,"FOL-T-dist.png"))

plot_mesh_vec_data_thermal(1,[coeffs_matrix[eval_id],FE_T[:,time_list[0]],FE_T[:,time_list[1]],FE_T[:,time_list[2]]],
                   ["","","",""],
                   fig_title="Initial condition and FEM solution",cmap = "jet",
                   file_name=os.path.join(case_dir,"FEM-T-dist.png"))
plot_mesh_vec_data(1,[coeffs_matrix[eval_id],absolute_error[:,time_list[0]],absolute_error[:,time_list[1]],absolute_error[:,time_list[2]]],
                   ["","","",""],
                   fig_title="Initial condition and iFOL error against FEM",cmap = "jet",
                   file_name=os.path.join(case_dir,"FOL-T-Error-dist.png"))

plot_mesh_vec_data(1,[hetero_info[0]],
                   [""],
                   fig_title="Heterogeneous microstructure",cmap = "viridis",
                   file_name=os.path.join(case_dir,"hetero_microstucture_fine.png"))

fe_mesh.Finalize(export_dir=case_dir)
