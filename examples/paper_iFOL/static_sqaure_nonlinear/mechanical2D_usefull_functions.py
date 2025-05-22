import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import math
import gmsh
import meshio
import os
import shutil
from fol.mesh_input_output.mesh import Mesh
import copy


def create_2D_square_mesh(L,N):

    # create empty fe mesh object
    fe_mesh = Mesh("square_io","square.")

    # FE init starts here
    Ne = N - 1  # Number of elements in each direction
    nx = Ne + 1  # Number of nodes in the x-direction
    ny = Ne + 1  # Number of nodes in the y-direction
    ne = Ne * Ne    # Total number of elements
    # Generate mesh coordinates
    x = jnp.linspace(0, L, nx)
    y = jnp.linspace(0, L, ny)
    X, Y = jnp.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    Z = jnp.zeros((Y.shape[-1]))

    fe_mesh.node_ids = jnp.arange(Y.shape[-1])
    fe_mesh.nodes_coordinates = jnp.stack((X,Y,Z), axis=1)

    # Create a matrix to store element nodal information
    elements_nodes = jnp.zeros((ne, 4), dtype=int)
    # Fill in the elements_nodes with element and node numbers
    for i in range(Ne):
        for j in range(Ne):
            e = i * Ne + j  # Element index
            # Define the nodes of the current element
            nodes = jnp.array([i * (Ne + 1) + j, i * (Ne + 1) + j + 1, (i + 1) * (Ne + 1) + j + 1, (i + 1) * (Ne + 1) + j])
            # Store element and node numbers in the matrix
            elements_nodes = elements_nodes.at[e].set(nodes) # Node numbers

    fe_mesh.elements_nodes = {"quad":elements_nodes}

    # Identify boundary nodes on the left and right edges
    left_boundary_nodes = jnp.arange(0, ny * nx, nx)  # Nodes on the left boundary
    right_boundary_nodes = jnp.arange(nx - 1, ny * nx, nx)  # Nodes on the right boundary

    fe_mesh.node_sets = {"left":left_boundary_nodes,
                         "right":right_boundary_nodes}
    
    fe_mesh.mesh_io = meshio.Mesh(fe_mesh.nodes_coordinates,fe_mesh.elements_nodes)

    fe_mesh.is_initialized = True

    return fe_mesh

def create_random_fourier_samples(fourier_control,numberof_sample):
    N = int(fourier_control.GetNumberOfControlledVariables()**0.5)
    num_coeffs = fourier_control.GetNumberOfVariables()
    coeffs_matrix = np.zeros((0,num_coeffs))
    for i in range (numberof_sample):
        coeff_vec = np.random.normal(size=num_coeffs)
        coeffs_matrix = np.vstack((coeffs_matrix,coeff_vec))

    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    # also add uniform dstibuted K of value 0.5
    coeff_vec = 1e-4 * np.zeros((num_coeffs))
    coeff_vec[0] = 10
    coeffs_matrix = np.vstack((coeffs_matrix,coeff_vec))
    K_matrix = np.vstack((K_matrix,fourier_control.ComputeControlledVariables(coeff_vec)))
    # plot_data_input(K_matrix,10,'K distributions')    

    return coeffs_matrix,K_matrix


def create_random_voronoi_samples(voronoi_control,number_of_sample,dim=2):
    number_seeds = voronoi_control.number_of_seeds
    rangeofValues = voronoi_control.E_values
    numberofVar = voronoi_control.num_control_vars
    coeffs_matrix = np.zeros((0,numberofVar))
    
    for _ in range(number_of_sample):
        x_coords = np.random.rand(number_seeds)
        y_coords = np.random.rand(number_seeds)
        if dim == 3:
            z_coords = np.random.rand(number_seeds)
        
        
        if isinstance(rangeofValues, tuple):
            E_values = np.random.uniform(rangeofValues[0],rangeofValues[-1],number_seeds)
        if isinstance(rangeofValues, list):
            E_values = np.random.choice(rangeofValues, size=number_seeds)
        
        Kcoeffs = np.zeros((0,numberofVar))
        if dim == 3:
            Kcoeffs = np.concatenate((x_coords.reshape(1,-1), y_coords.reshape(1,-1), 
                                  z_coords.reshape(1,-1), E_values.reshape(1,-1)), axis=1)
        else:
            Kcoeffs = np.concatenate((x_coords.reshape(1,-1), y_coords.reshape(1,-1), 
                                      E_values.reshape(1,-1)), axis=1)
        
        coeffs_matrix = np.vstack((coeffs_matrix,Kcoeffs))
    K_matrix = voronoi_control.ComputeBatchControlledVariables(coeffs_matrix)
    return coeffs_matrix,K_matrix

def create_clean_directory(case_dir):
    # Check if the directory exists
    if os.path.exists(case_dir):
        # Remove the directory and all its contents
        shutil.rmtree(case_dir)
    
    # Create the new directory
    os.makedirs(case_dir)

def plot_mesh_res(vectors_list:list, file_name:str="plot",dir:str="U"):
    fontsize = 16
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))  # Adjusted to 4 columns

    # Plot the first entity in the first row
    data = vectors_list[0]
    N = int((data.reshape(-1, 1).shape[0]) ** 0.5)
    im = axs[0, 0].imshow(data.reshape(N, N), cmap='viridis', aspect='equal')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_title('Elasticity Morph.', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 0], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot the same entity with mesh grid in the first row, second column
    im = axs[0, 1].imshow(data.reshape(N, N), cmap='bone', aspect='equal')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_xticklabels([])  # Remove text on x-axis
    axs[0, 1].set_yticklabels([])  # Remove text on y-axis
    axs[0, 1].set_title(f'Mesh Grid: {N} x {N}', fontsize=fontsize)
    axs[0, 1].grid(True, color='red', linestyle='-', linewidth=1)  # Adding solid grid lines with red color
    axs[0, 1].xaxis.grid(True)
    axs[0, 1].yaxis.grid(True)

    x_ticks = np.linspace(0, N, N)
    y_ticks = np.linspace(0, N, N)
    axs[0, 1].set_xticks(x_ticks)
    axs[0, 1].set_yticks(y_ticks)

    cbar = fig.colorbar(im, ax=axs[0, 1], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Zoomed-in region
    zoomed_min = int(0.2*N)
    zoomed_max = int(0.4*N)
    zoom_region = data.reshape(N, N)[zoomed_min:zoomed_max, zoomed_min:zoomed_max]
    im = axs[0, 2].imshow(zoom_region, cmap='bone', aspect='equal')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].set_xticklabels([])  # Remove text on x-axis
    axs[0, 2].set_yticklabels([])  # Remove text on y-axis
    axs[0, 2].set_title(f'Zoomed-in: $x \in [{zoomed_min/N:.2f}, {zoomed_max/N:.2f}], y \in [{zoomed_min/N:.2f}, {zoomed_max/N:.2f}]$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 2], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot the mesh grid
    axs[0, 2].xaxis.set_major_locator(plt.LinearLocator(21))
    axs[0, 2].yaxis.set_major_locator(plt.LinearLocator(21))
    axs[0, 2].grid(color='red', linestyle='-', linewidth=2)

    # Plot cross-sections along x-axis at y=0.5 for U (FOL and FEM) in the second row, fourth column
    y_idx = int(N * 0.5)
    U1 = vectors_list[0].reshape(N, N)
    axs[0, 3].plot(np.linspace(0, 1, N), U1[y_idx, :], label='Elasticity', color='black')
    axs[0, 3].set_xlim([0, 1])
    #axs[0, 3].set_ylim([min(U1[y_idx, :].min()), max(U1[y_idx, :].max())])
    axs[0, 3].set_aspect(aspect='auto')
    axs[0, 3].set_title('Cross-section of E at y=0.5', fontsize=fontsize)
    axs[0, 3].legend(fontsize=fontsize)
    axs[0, 3].grid(True)
    axs[0, 3].set_xlabel('x', fontsize=fontsize)
    axs[0, 3].set_ylabel('E', fontsize=fontsize)


    # Plot the second entity in the second row
    data = vectors_list[1]
    im = axs[1, 0].imshow(data.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 0].set_title(f'${dir}$, FOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 0], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot the fourth entity in the second row
    data = vectors_list[2]
    im = axs[1, 1].imshow(data.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_title(f'${dir}$, FEM', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 1], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot the absolute difference between vectors_list[1] and vectors_list[3] in the third row, second column
    diff_data_1 = np.abs(vectors_list[1] - vectors_list[2])
    im = axs[1, 2].imshow(diff_data_1.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    axs[1, 2].set_title(f'Abs. Difference ${dir}$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 2], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # Plot cross-sections along x-axis at y=0.5 for U (FOL and FEM) in the second row, fourth column
    y_idx = int(N * 0.5)
    U1 = vectors_list[1].reshape(N, N)
    U2 = vectors_list[2].reshape(N, N)
    axs[1, 3].plot(np.linspace(0, 1, N), U1[y_idx, :], label=f'{dir} FOL', color='blue')
    axs[1, 3].plot(np.linspace(0, 1, N), U2[y_idx, :], label=f'{dir} FEM', color='red')
    axs[1, 3].set_xlim([0, 1])
    axs[1, 3].set_ylim([min(U1[y_idx, :].min(), U2[y_idx, :].min()), max(U1[y_idx, :].max(), U2[y_idx, :].max())])
    axs[1, 3].set_aspect(aspect='auto')
    axs[1, 3].set_title(f'Cross-section of {dir} at y=0.5', fontsize=fontsize)
    axs[1, 3].legend(fontsize=fontsize)
    axs[1, 3].grid(True)
    axs[1, 3].set_xlabel('x', fontsize=fontsize)
    axs[1, 3].set_ylabel(f'{dir}', fontsize=fontsize)

    plt.tight_layout()

    # Save the figure in multiple formats
    plt.savefig(file_name, dpi=300)
    # plt.savefig(plot_name+'.pdf')


def plot_mesh_grad_res_mechanics(vectors_list:list, file_name:str="plot", loss_settings:dict={}):
    fontsize = 16
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))

    data = vectors_list[0]
    L = 1
    N = int((data.reshape(-1, 1).shape[0])**0.5)
    nu = loss_settings["poisson_ratio"]
    e = loss_settings["young_modulus"]
    mu = e / (2*(1+nu))
    lambdaa = nu * e / ((1+nu)*(1-2*nu))
    c1 = e / (1 - nu**2)

    dx = L / (N - 1)

    U_fem = vectors_list[2][::2]
    V_fem = vectors_list[2][1::2]
    domain_map_matrix = vectors_list[0].reshape(N, N)
    dU_dx_fem = np.gradient(U_fem.reshape(N, N), dx, axis=1)
    dV_dy_fem = np.gradient(V_fem.reshape(N, N), dx, axis=0)
    stress_xx_fem = domain_map_matrix * c1 * (dU_dx_fem + nu * dV_dy_fem) # plain stress condition
    stress_yy_fem = domain_map_matrix * c1 * (nu * dU_dx_fem + dV_dy_fem) # plain stress condition

    im = axs[0, 1].imshow(stress_xx_fem, cmap='plasma')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_title('$\sigma_{xx}$, FEM', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 0], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    im = axs[1, 1].imshow(stress_yy_fem, cmap='plasma')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_title('$\sigma_{yy}$, FEM', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 0], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)


    U_fol = vectors_list[1][::2]
    V_fol = vectors_list[1][1::2]
    dU_dx_fol = np.gradient(U_fol.reshape(N, N), dx, axis=1)
    dV_dy_fol = np.gradient(V_fol.reshape(N, N), dx, axis=0)
    stress_xx_fol = domain_map_matrix * c1 * (dU_dx_fol + nu * dV_dy_fol) # plain stress condition
    stress_yy_fol = domain_map_matrix * c1 * (nu * dU_dx_fol + dV_dy_fol) # plain stress condition

    min_v = np.min(stress_xx_fem)
    max_v = np.max(stress_xx_fem)
    im = axs[0, 0].imshow(stress_xx_fol, cmap='plasma', vmin=min_v, vmax=max_v)
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_title('$\sigma_{xx}$, FOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 1], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    min_v = np.min(stress_yy_fem)
    max_v = np.max(stress_yy_fem)
    im = axs[1, 0].imshow(stress_yy_fol, cmap='plasma', vmin=min_v, vmax=max_v)
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 0].set_title('$\sigma_{yy}$, FOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 1], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)


    diff_data_2 = np.abs(stress_xx_fem - stress_xx_fol)
    im = axs[0, 2].imshow(diff_data_2, cmap='plasma')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].set_title('Abs. Difference $\sigma_{xx}$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[0, 2], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    diff_data_2 = np.abs(stress_yy_fem - stress_yy_fol)
    im = axs[1, 2].imshow(diff_data_2, cmap='plasma')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    axs[1, 2].set_title('Abs. Difference $\sigma_{yy}$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[1, 2], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)


    # Extract cross-sections at y = 0.5
    y_index = N // 2
    stress_x_cross_fem = stress_xx_fem[y_index, :]
    stress_y_cross_fem = stress_yy_fem[y_index, :]
    stress_x_cross_fol = stress_xx_fol[y_index, :]
    stress_y_cross_fol = stress_yy_fol[y_index, :]

    # Plot cross-sections in the fourth column
    axs[0, 3].plot(np.linspace(0, L, N), stress_x_cross_fem, label='FEM', color='r')
    axs[0, 3].plot(np.linspace(0, L, N), stress_x_cross_fol, label='FOL', color='b')
    axs[0, 3].set_title('Cross-section $\sigma_{xx}$', fontsize=fontsize)
    axs[0, 3].legend()

    axs[1, 3].plot(np.linspace(0, L, N), stress_y_cross_fem, label='FEM', color='r')
    axs[1, 3].plot(np.linspace(0, L, N), stress_y_cross_fol, label='FOL', color='b')
    axs[1, 3].set_title('Cross-section $\sigma_{yy}$', fontsize=fontsize)
    axs[1, 3].legend()

    # Save cross-section data to a text file
    file_dir = os.path.join(os.path.dirname(os.path.abspath(file_name)),'cross_section_data.txt')
    with open(file_dir, 'w') as f:
        f.write('x, stress_x_fem, stress_x_fol, stress_y_fem, stress_y_fol, stress_xy_fem, stress_xy_fol\n')
        for i in range(N):
            f.write(f'{i*dx}, {stress_x_cross_fem[i]}, {stress_x_cross_fol[i]}, {stress_y_cross_fem[i]}, {stress_y_cross_fol[i]}\n')


    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    # plt.savefig(plot_name+'.pdf')