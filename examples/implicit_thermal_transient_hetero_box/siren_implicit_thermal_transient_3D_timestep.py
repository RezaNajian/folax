import sys
import os
import optax
import numpy as np
from flax import nnx
import jax
from fol.loss_functions.thermal_transient_hetero_3D_fe_hex import ThermalTransientLossHetero3DHex
from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.implicit_transient_parametric_operator_learning_super_res_hetero import ImplicitParametricOperatorLearning
from fol.solvers.fe_linear_residual_based_solver_hetero import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import MLP
import pickle

timestep_list = [0.05,0.02,0.01,0.005,0.0025,0.001]
name_list = ["5","2","11","52","25","12",]

for k, timestep in enumerate(timestep_list):
    # directory & save handling
    working_directory_name = 'siren_implicit_thermal_3D'+'_test'+str(name_list[k])
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,"N":32,
                    "T_left":1.0,"T_right":0.0}

    # creation of the model
    mesh_res_rate = 1
    num_steps = int(0.5/timestep)
    fe_mesh = create_3D_box_mesh(Nx=model_settings["N"]-1,
                                 Ny=model_settings["N"]-1,
                                 Nz=model_settings["N"]-1,
                                 Lx=model_settings["L"],
                                 Ly=model_settings["L"],
                                 Lz=model_settings["L"],
                                 case_dir=case_dir)
    fe_mesh_pred = create_3D_box_mesh(Nx=model_settings["N"]*mesh_res_rate-1,
                                      Ny=model_settings["N"]*mesh_res_rate-1,
                                      Nz=model_settings["N"]*mesh_res_rate-1,
                                      Lx=model_settings["L"],
                                      Ly=model_settings["L"],
                                      Lz=model_settings["L"],
                                      case_dir=case_dir)

    # create fe-based loss function
    bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]}}#
    Dirichlet_BCs = True
    material_dict = {"rho":1.0,"cp":1.0,"dt":timestep}
    thermal_loss_3d = ThermalTransientLossHetero3DHex("thermal_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "num_gp":2,
                                                                                "material_dict":material_dict},
                                                                                fe_mesh=fe_mesh)
    thermal_loss_3d_pred = ThermalTransientLossHetero3DHex("thermal_loss_3d_pred",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "num_gp":2,
                                                                                "material_dict":material_dict},
                                                                                fe_mesh=fe_mesh_pred)

    no_control = NoControl("No_Control",fe_mesh)

    fe_mesh.Initialize()
    fe_mesh_pred.Initialize()
    thermal_loss_3d.Initialize()
    thermal_loss_3d_pred.Initialize()
    no_control.Initialize()

    # create some random coefficients & K for training
    create_random_coefficients = True
    if create_random_coefficients:
        def generate_random_smooth_patterns_3d_fft(
            points, num_samples=1, length_scale=0.1
        ):
            """
            Generate random smooth patterns in 3D using Fourier-based methods with meshio coordinates.

            Parameters:
                points (np.ndarray): Meshio coordinates array of shape (num_points, 3).
                num_samples (int): The number of random patterns to generate.
                length_scale (float): Controls the smoothness of the patterns.

            Returns:
                np.ndarray: An array of shape (num_samples, num_points) with generated patterns.
            """
            num_points = points.shape[0]

            # Compute distances between all points to approximate the frequency grid
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            Lx = x.max() - x.min()
            Ly = y.max() - y.min()
            Lz = z.max() - z.min()

            # Create the frequency grid (assuming a structured layout for simplicity)
            N = int(round(num_points**(1/3)))  # Estimate grid size (assumes near-cubic grid)
            kx = np.fft.fftfreq(N, d=Lx / N)
            ky = np.fft.fftfreq(N, d=Ly / N)
            kz = np.fft.fftfreq(N, d=Lz / N)
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

            # Compute the squared frequency magnitudes
            K2 = KX**2 + KY**2 + KZ**2
            K2[0, 0, 0] = 1.0  # Avoid division by zero at the origin

            # Create the power spectrum for a Gaussian kernel
            power_spectrum = np.exp(-0.5 * (K2 * (length_scale * 2 * np.pi) ** 2))

            # Generate random smooth patterns
            patterns = []
            for _ in range(num_samples):
                random_coeffs = (
                    np.random.normal(size=(N, N, N))
                    + 1j * np.random.normal(size=(N, N, N))
                )
                random_coeffs[0, 0, 0] = 0.0
                random_coeffs *= np.sqrt(power_spectrum)

                pattern_3d = np.fft.ifftn(random_coeffs).real

                # Interpolate the 3D grid onto the meshio points
                from scipy.interpolate import RegularGridInterpolator

                grid_x = np.linspace(x.min(), x.max(), N)
                grid_y = np.linspace(y.min(), y.max(), N)
                grid_z = np.linspace(z.min(), z.max(), N)
                interpolator = RegularGridInterpolator(
                    (grid_x, grid_y, grid_z), pattern_3d, bounds_error=False, fill_value=0
                )
                pattern = interpolator(points)
                pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
                patterns.append(pattern)

            return np.array(patterns)

        def generate_composite_material(
            points, 
            num_inclusions=10, 
            inclusion_radius=0.2, 
            inclusion_value=0.2, 
            matrix_value=1

        ):
            """
            Generate composite-like material samples with spherical inclusions using meshio coordinates.

            Parameters:
                points (np.ndarray): Meshio coordinates array of shape (num_points, 3).
                num_inclusions (int): Number of inclusions per sample.
                inclusion_radius (float): Radius of the spherical inclusions.
                inclusion_value (float): Value assigned to the inclusion phase.
                matrix_value (float): Value assigned to the matrix phase.

            Returns:
                np.ndarray: Array of shape (num_samples, num_points) with composite patterns.
            """
            num_points = points.shape[0]
            samples = []

            # Initialize the material array with the matrix value
            material = np.full(num_points, matrix_value, dtype=float)

            # List to store inclusion centers
            inclusion_centers = [np.array([0.3995, 0.8279, 0.9229]),
                                 np.array([0.0435, 0.2529, 0.7887]),
                                 np.array([0.4041, 0.0104, 0.1794]),
                                 np.array([0.8309, 0.3136, 0.2780]),
                                 np.array([0.5803, 0.9523, 0.2323]),
                                 np.array([0.1439, 0.2935, 0.3608]),
                                 np.array([0.5844, 0.2953, 0.9866]),
                                 np.array([0.9155, 0.0424, 0.6355]),
                                 np.array([0.9087, 0.7211, 0.7662]),
                                 np.array([0.1959, 0.9935, 0.1157])]

            # Generate random, non-overlapping centers
            # for _ in range(num_inclusions):
            #     while True:
            #         # Randomly choose the center of the inclusion within the coordinate range
            #         center = np.random.uniform(points.min(axis=0), points.max(axis=0), size=3)

            #         # Check for overlap with existing inclusions
            #         if all(np.linalg.norm(center - np.array(existing_center)) > 2 * inclusion_radius
            #                for existing_center in inclusion_centers):
            #             # Add the center to the list of inclusion centers
            #             inclusion_centers.append(center)
            #             break

            # Create inclusions based on the generated centers
            for center in inclusion_centers:
                # Compute the distance from the center for all points
                distances = np.linalg.norm(points - center, axis=1)

                # Assign inclusion value to points within the radius
                material[distances <= inclusion_radius] = inclusion_value

            # Add the result to samples
            samples.append(material)

            return np.array(samples)

        # coeffs_matrix = np.full((1,fe_mesh.GetNumberOfNodes()),0.1)#generate_random_smooth_patterns_3d_fft(fe_mesh.GetNodesCoordinates()).reshape(1,-1) 
        coeffs_matrix_fine = np.full((1,fe_mesh_pred.GetNumberOfNodes()),0.1)#generate_random_smooth_patterns_3d_fft(fe_mesh_pred.GetNodesCoordinates()).reshape(1,-1) 
        # hetero_info = generate_composite_material(fe_mesh.GetNodesCoordinates()).reshape(1,-1)
        hetero_info_fine = generate_composite_material(fe_mesh_pred.GetNodesCoordinates()).reshape(1,-1)

    else:
        pass


    # K_matrix = []
    # T_matrix = no_control.ComputeBatchControlledVariables(coeffs_matrix)
    # K_matrix = no_control.ComputeBatchControlledVariables(hetero_info)
    T_matrix_fine = no_control.ComputeBatchControlledVariables(coeffs_matrix_fine)
    K_matrix_fine = no_control.ComputeBatchControlledVariables(hetero_info_fine)

    # solve FE here
    fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu","Dirichlet_BCs":Dirichlet_BCs},
                    "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                "maxiter":10,"load_incr":5}}
    linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",thermal_loss_3d_pred,fe_setting)
    linear_fe_solver.Initialize()
    FE_T = np.zeros((fe_mesh_pred.GetNumberOfNodes(),num_steps))
    FE_T_temp = coeffs_matrix_fine[0]
    T_line = []
    for i in range(num_steps):
        FE_T_temp = np.array(linear_fe_solver.Solve(FE_T_temp,K_matrix_fine,FE_T_temp))  #,np.zeros(fe_mesh.GetNumberOfNodes())
        FE_T[:,i] = FE_T_temp
        
    
    distances = np.linalg.norm(fe_mesh.GetNodesCoordinates()[:,1:] - np.array([0.3125,0.3125]), axis=1)

    # Assign inclusion value to points within the radius
    mesh_size = 0.5/32
    FE_T_reshaped = FE_T[distances <= mesh_size,0] 
    FE_T_reshaped = np.concatenate((FE_T_reshaped[1:],FE_T_reshaped[:1]))
    T_line.append(FE_T_reshaped.flatten())
    fe_mesh['T_ini'] = T_matrix_fine.reshape(-1,1)
    fe_mesh['T_FE'] = FE_T
    fe_mesh['Heterogeneity'] = K_matrix_fine.reshape(-1,1)
    # absolute_error = np.abs(FOL_T- FE_T)
    # fe_mesh['abs_error'] = absolute_error
    fe_mesh.Finalize(export_dir=case_dir)
    fe_line = np.linspace(0,1,32)

    import matplotlib.pyplot as plt
    plt.plot(fe_line, FE_T_reshaped.flatten())

plt.legend(["0.05","0.02","0.01","0.005","0.0025","0.001"])
plt.savefig("cross_section_timestepstudy.png")
np.savetxt("T_line.txt",T_line)
