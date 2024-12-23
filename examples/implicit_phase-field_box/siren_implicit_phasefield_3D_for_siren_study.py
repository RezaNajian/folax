import sys
import os
import optax
import numpy as np
from flax import nnx
import jax
from fol.loss_functions.phasefield_3D_fe_hex import AllenCahnLoss3DHex
from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.implicit_transient_parametric_operator_learning_super_res import ImplicitParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver_phasefield import FiniteElementNonLinearResidualBasedSolverPhasefield
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import MLP
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import NearestNDInterpolator

import time 
from datetime import datetime

omega_list = [5,10,15,20,30,10,10,20,20,30,30]
alpha_list = [1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3]
plot_label = []
for i in range(len(alpha_list)):
    plot_label.append('Omega_'+str(omega_list[i])+' Alpha_'+str(alpha_list[i]))
num_steps = 100
relative_l2_error_list = np.zeros((len(alpha_list),num_steps+1))

for l, (omega, alpha) in enumerate(zip(omega_list,alpha_list)):
    # directory & save handling
    working_directory_name = 'siren_implicit_AllenCahn_3D'+'_'+str(omega)+'_'+str(alpha)
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,"N":50,
                    "T_left":1.0,"T_right":-1.0}
    num_steps = 100
    # creation of the model
    mesh_res_rate = 2
    fe_mesh = Mesh("fol_io","cube_hex_n50.med",'../meshes/')
    fe_mesh_pred = Mesh("fol_io","cube_hex_n100.med",'../meshes/')
    # create fe-based loss function
    bc_dict = {"T":{}}#"left":model_settings["T_left"],"right":model_settings["T_right"]
    Dirichlet_BCs = False

    material_dict = {"rho":1.0,"cp":1.0,"dt":0.0002,"epsilon":0.1}
    dt_res_rate = 5
    material_dict_pred = {"rho":material_dict["rho"],"cp":material_dict["cp"],"dt":material_dict["dt"]/dt_res_rate,"epsilon":material_dict["epsilon"]}
    phasefield_loss_3d = AllenCahnLoss3DHex("phasefield_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "num_gp":2,
                                                                                "material_dict":material_dict},
                                                                                fe_mesh=fe_mesh)
    phasefield_loss_3d_pred = AllenCahnLoss3DHex("phasefield_loss_3d_pred",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "num_gp":2,
                                                                                "material_dict":material_dict_pred},
                                                                                fe_mesh=fe_mesh_pred)
    no_control = NoControl("No_Control",fe_mesh)

    fe_mesh.Initialize()
    fe_mesh_pred.Initialize()
    phasefield_loss_3d.Initialize()
    phasefield_loss_3d_pred.Initialize()
    no_control.Initialize()

    # create some random coefficients & K for training
    create_random_coefficients = True
    if create_random_coefficients:
        # number_of_random_samples = 200
        # coeffs_matrix,K_matrix = create_random_fourier_samples(_control,number_of_random_samples)
        # export_dict = model_settings.copy()
        # export_dict["coeffs_matrix"] = coeffs_matrix
        # export_dict["x_freqs"] = fourier_control.x_freqs
        # export_dict["y_freqs"] = fourier_control.y_freqs
        # export_dict["z_freqs"] = fourier_control.z_freqs
        # with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'wb') as f:
        #     pickle.dump(export_dict,f)
        # def generate_random_smooth_pattern_3d(points, L, epsilon, subsample_size=10000):
        #     """
        #     Generate a random smooth pattern using Gaussian process sampling with subsampling.

        #     Parameters:
        #         points (np.ndarray): A (N, 3) array of coordinates from `meshio`.
        #         L (float): The length scale of the domain (assumed cubic).
        #         subsample_size (int): Number of points to use for subsampling.

        #     Returns:
        #         np.ndarray: A 1D array of smooth random values corresponding to the input points.
        #     """
        #     # Normalize the points to the domain [0, L]
        #     normalized_points = points / L

        #     # Subsample points for Gaussian process
        #     total_points = len(normalized_points)
        #     if total_points <= subsample_size:
        #         sampled_points = normalized_points
        #     else:
        #         indices = np.random.choice(total_points, size=subsample_size, replace=False)
        #         sampled_points = normalized_points[indices]

        #     # Define the kernel for the Gaussian process
        #     kernel = C(1.0, (1e-3, 1e3)) * RBF(epsilon, (1e-2, 1e2))
        #     gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=42)

        #     # Sample Gaussian process values at the subsampled points
        #     u_subsampled = gp.sample_y(sampled_points, n_samples=1, random_state=0).ravel()

        #     # Normalize the sampled values to the range [-1, 1]
        #     scaled_u_subsampled = 2.0 * (u_subsampled - np.min(u_subsampled)) / (np.max(u_subsampled) - np.min(u_subsampled)) - 1.0

        #     # Interpolate from subsampled points back to the full set of points
        #     interpolator = NearestNDInterpolator(sampled_points, scaled_u_subsampled)
        #     smooth_pattern = interpolator(normalized_points)

        #     return smooth_pattern.reshape(1,-1)
        def generate_random_smooth_pattern_3d_fft(points, L, epsilon):
            """
            Generate a random smooth pattern using FFT for efficiency.

            Parameters:
                points (np.ndarray): A (N, 3) array of coordinates from `meshio`.
                L (float): The length scale of the domain (assumed cubic).
                epsilon (float): Controls smoothness of the pattern.

            Returns:
                np.ndarray: A 1D array of smooth random values corresponding to the input points.
            """
            import numpy as np
            from scipy.fft import fftn, ifftn, fftfreq

            # Normalize the points to the domain [0, L]
            normalized_points = points / L

            # Create a grid for FFT
            grid_size = 128  # Choose an appropriate grid size
            x = np.linspace(0, 1, grid_size)
            y = np.linspace(0, 1, grid_size)
            z = np.linspace(0, 1, grid_size)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

            # Generate random field in Fourier domain
            kx = fftfreq(grid_size, d=1/grid_size)
            ky = fftfreq(grid_size, d=1/grid_size)
            kz = fftfreq(grid_size, d=1/grid_size)
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

            # Compute power spectrum for smoothness
            power_spectrum = np.exp(-epsilon * (KX**2 + KY**2 + KZ**2))
            noise = np.random.normal(size=(grid_size, grid_size, grid_size)) + 1j * np.random.normal(size=(grid_size, grid_size, grid_size))
            random_field_fft = noise * power_spectrum

            # Inverse FFT to get the spatial domain random field
            random_field = np.real(ifftn(random_field_fft))

            # Interpolate the random field to the original points
            from scipy.interpolate import RegularGridInterpolator

            interpolator = RegularGridInterpolator((x, y, z), random_field, bounds_error=False, fill_value=None)
            smooth_pattern = interpolator(normalized_points)

            # Normalize to [-1, 1]
            smooth_pattern = 2.0 * (smooth_pattern - np.min(smooth_pattern)) / (np.max(smooth_pattern) - np.min(smooth_pattern)) - 1.0

            return smooth_pattern.reshape(1, -1)



        def generate_double_bubble_3d(coords, L, epsilon):
            # Define centers of the bubbles
            center1 = np.array([0.4 * L, 0.5 * L, 0.5 * L])
            center2 = np.array([0.6 * L, 0.5 * L, 0.5 * L])

            # Compute distances from both centers
            distances1 = np.sqrt(np.sum((coords - center1)**2, axis=1))
            distances2 = np.sqrt(np.sum((coords - center2)**2, axis=1))

            # Compute bubble functions
            bubble1 = np.tanh((0.2 - distances1) / epsilon)
            bubble2 = np.tanh((0.2 - distances2) / epsilon)

            # Take the maximum value of the two bubbles
            double_bubble = np.maximum(bubble1, bubble2)

            return double_bubble

        coeffs_matrix = generate_double_bubble_3d(fe_mesh.GetNodesCoordinates(),
                                                  model_settings["L"],material_dict["epsilon"])
        coeffs_matrix_fine = generate_double_bubble_3d(fe_mesh_pred.GetNodesCoordinates(),
                                                       model_settings["L"],material_dict["epsilon"])
        # coeffs_matrix = generate_random_smooth_pattern_3d_fft(fe_mesh.GetNodesCoordinates(),
        #                                                   model_settings["L"],
        #                                                   material_dict["epsilon"])
        # coeffs_matrix_fine = coeffs_matrix 
        # generate_random_smooth_pattern_3d(fe_mesh_pred.GetNodesCoordinates(),
        #                                                        model_settings["L"],
        #                                                        material_dict["epsilon"])

    else:
        pass
        # with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'rb') as f:
        #     loaded_dict = pickle.load(f) 
        # coeffs_matrix = loaded_dict["coeffs_matrix"]
        # coeffs_matrix = jnp.load("gaussian_kernel_50000_N21.npy")


    # K_matrix = []
    T_matrix = no_control.ComputeBatchControlledVariables(coeffs_matrix)

    export_Ks = False
    # if export_Ks:
    #     for i in range(K_matrix.shape[0]):
    #         fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
    #     fe_mesh.Finalize(export_dir=case_dir)
    #     exit()

    # specify id of the K of interest
    eval_id = 0

    # design siren NN for learning
    hidden_layers = [100,100]
    siren_NN = MLP(name="SIREN_NN", 
                        input_size=3,
                        output_size=1,
                        hidden_layers=hidden_layers,
                        activation_settings={"type":"sin",
                                             "prediction_gain":omega,
                                             "initialization_gain":alpha},
                        skip_connections_settings={"active":False,"frequency":1})

    lr = 1e-4
    # create fol optax-based optimizer
    chained_transform = optax.chain(optax.normalize_by_update_norm(),
                                    optax.adam(lr))

    # create fol
    start_time = time.time()
    fol = ImplicitParametricOperatorLearning(name="dis_fol",control=no_control,
                                            loss_function=phasefield_loss_3d,
                                            loss_function_pred=phasefield_loss_3d_pred,
                                            flax_neural_network=siren_NN,
                                            optax_optimizer=chained_transform,
                                            checkpoint_settings={"restore_state":False,
                                            "state_directory":case_dir+"/flax_state"},
                                            working_directory=case_dir)

    fol.Initialize()

    t_init = 0.0
    t_current = t_init
    FOL_T_temp = coeffs_matrix.flatten()
    FOL_T = np.zeros((fe_mesh_pred.GetNumberOfNodes(),num_steps))
    # For the first time step
    fol.Train(train_set=(jnp.concatenate((jnp.array([t_current]),FOL_T_temp)).reshape(-1,1).T,),batch_size=1,
                convergence_settings={"num_epochs":300,"relative_error":1e-5},
                plot_settings={"plot_save_rate":1000},
                save_settings={"save_nn_model":True})
    FOL_T_temp_fine = np.array(fol.Predict_fine(jnp.array([t_current]))).reshape(-1)
    FOL_T_temp = np.array(fol.Predict(jnp.array([t_current]))).reshape(-1)
    FOL_T[:,0] = FOL_T_temp_fine
    # For the subsequent time steps the checkpoint function should be activated
    for i in range(num_steps-1):
        t_current += material_dict["dt"]
        chained_transform = optax.chain(optax.normalize_by_update_norm(),
                                    optax.adam(lr))
        fol = ImplicitParametricOperatorLearning(name="dis_fol",control=no_control,
                                                loss_function=phasefield_loss_3d,
                                                loss_function_pred=phasefield_loss_3d_pred,
                                                flax_neural_network=siren_NN,
                                                optax_optimizer=chained_transform,
                                                checkpoint_settings={"restore_state":True,
                                                "state_directory":case_dir+"/flax_state"},
                                                working_directory=case_dir)
        fol.Initialize()
        fol.Train(train_set=(jnp.concatenate((jnp.array([t_current]),FOL_T_temp)).reshape(-1,1).T,),batch_size=100,
                    convergence_settings={"num_epochs":300,"relative_error":1e-5},
                    plot_settings={"plot_save_rate":1000},
                    save_settings={"save_nn_model":True})
        FOL_T_temp_fine = np.array(fol.Predict_fine(jnp.array([t_current]))).reshape(-1)
        FOL_T[:,i+1] = FOL_T_temp_fine
        FOL_T_temp = np.array(fol.Predict(jnp.array([t_current]))).reshape(-1)

    end_time = time.time()
    execution_time = end_time - start_time
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time} - Info : iFOL part - finished in {execution_time:.4f} seconds")
    fe_mesh_pred['T_FOL'] = FOL_T
    fe_mesh_pred['T_init'] = coeffs_matrix_fine.reshape((fe_mesh_pred.GetNumberOfNodes(), 1))
    # solve FE here
    start_time = time.time()
    fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-7,"atol":1e-7,
                                                "maxiter":1000,"pre-conditioner":"none","Dirichlet_BCs":Dirichlet_BCs},
                    "nonlinear_solver_settings":{"rel_tol":1e-7,"abs_tol":1e-7,
                                                "maxiter":20,"load_incr":1}}
    nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolverPhasefield("nonlinear_fe_solver",phasefield_loss_3d_pred,fe_setting)
    nonlinear_fe_solver.Initialize()
    num_steps_FE = num_steps*dt_res_rate
    FE_T = np.zeros((fe_mesh_pred.GetNumberOfNodes(),num_steps_FE))
    FE_T_temp = coeffs_matrix_fine.flatten()
    for i in range(num_steps_FE):
        FE_T_temp = np.array(nonlinear_fe_solver.Solve(FE_T_temp,FE_T_temp))  #np.zeros(fe_mesh.GetNumberOfNodes())
        FE_T[:,i] = FE_T_temp   
    end_time = time.time()
    execution_time = end_time - start_time
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time} - Info : FEM part - finished in {execution_time:.4f} seconds")
    
    fe_mesh_pred['T_FE'] = FE_T
    absolute_error = np.abs(FOL_T- FE_T[:,(dt_res_rate-1)::dt_res_rate])
    fe_mesh_pred['abs_error'] = absolute_error#.reshape((fe_mesh.GetNumberOfNodes(), 1))
    # time_list = [int(num_steps/5) - 1,int(num_steps/2) - 1,num_steps - 1]
    # time_list_FE = [int(num_steps_FE/5)- 1,int(num_steps_FE/2)- 1,num_steps_FE-1] 
    # plot_mesh_vec_data_phasefield(1,[coeffs_matrix_fine[eval_id],FOL_T[:,time_list[0]],FOL_T[:,time_list[1]],FOL_T[:,time_list[2]]],#,absolute_error
    #                    ["","","",""],
    #                    fig_title="Initial condition and implicit FOL solution",cmap = "jet",
    #                    file_name=os.path.join(case_dir,"FOL-T-dist.png"))
    # plot_mesh_vec_data_phasefield(1,[coeffs_matrix_fine[eval_id],FE_T[:,time_list_FE[0]],FE_T[:,time_list_FE[1]],FE_T[:,time_list_FE[2]]],
    #                    ["","","",""],
    #                    fig_title="Initial condition and FEM solution",cmap = "jet",
    #                    file_name=os.path.join(case_dir,"FEM-T-dist.png"))
    # plot_mesh_vec_data(1,[coeffs_matrix_fine[eval_id],absolute_error[:,time_list[0]],absolute_error[:,time_list[1]],absolute_error[:,time_list[2]]],
    #                    ["","","",""],
    #                    fig_title="Initial condition and iFOL error against FEM",cmap = "jet",
    #                    file_name=os.path.join(case_dir,"FOL-T-Error-dist.png"))
    relative_l2_error = np.linalg.norm(FOL_T- FE_T[:,(dt_res_rate-1)::dt_res_rate],axis=0)/np.linalg.norm(FE_T[:,(dt_res_rate-1)::dt_res_rate],axis=0)
    relative_l2_error = np.insert(relative_l2_error,0,0)
    time_plot = np.linspace(0,t_current,num_steps+1)
    plot_relative_l2_error(time_plot,relative_l2_error,file_name=os.path.join(case_dir,"iFOL_FEM_relative_L2_error.png"))
    relative_l2_error_list[l] = relative_l2_error

    fe_mesh_pred.Finalize(export_dir=case_dir)

plot_relative_l2_error_multiple(time_plot,relative_l2_error_list,label_name=plot_label,file_name="iFOL_FEM_relative_L2_error_all.png")
np.savetxt("relative_l2_error.txt",np.hstack((time_plot.reshape(-1,1),relative_l2_error_list.T)))