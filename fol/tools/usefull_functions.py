
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

def plot_mesh_vec_data(L, vectors_list, subplot_titles=None, fig_title=None, cmap='viridis',
                       block_bool=False, colour_bar=True, colour_bar_name=None,
                       X_axis_name=None, Y_axis_name=None, show=False, file_name=None):
    num_vectors = len(vectors_list)
    if num_vectors < 1 or num_vectors > 4:
        raise ValueError("vectors_list must contain between 1 and 4 elements.")

    if subplot_titles is not None and len(subplot_titles) != num_vectors:
        raise ValueError("subplot_titles must have the same number of elements as vectors_list if provided.")

    # Determine the grid size for the subplots
    grid_size = math.ceil(math.sqrt(num_vectors))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(5*grid_size, 5*grid_size), squeeze=False)
    
    # Flatten the axs array and hide unused subplots if any
    axs = axs.flatten()
    for ax in axs[num_vectors:]:
        ax.axis('off')

    for i, squared_mesh_vec_data in enumerate(vectors_list):
        N = int((squared_mesh_vec_data.reshape(-1, 1).shape[0])**0.5)
        im = axs[i].imshow(squared_mesh_vec_data.reshape(N, N), cmap=cmap, extent=[0, L, 0, L])

        if subplot_titles is not None:
            axs[i].set_title(subplot_titles[i])
        else:
            axs[i].set_title(f'Plot {i+1}')

        if colour_bar:
            fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

        if X_axis_name is not None:
            axs[i].set_xlabel(X_axis_name)

        if Y_axis_name is not None:
            axs[i].set_ylabel(Y_axis_name)

    if fig_title is not None:
        plt.suptitle(fig_title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if show:
        plt.show(block=block_bool)

    if file_name is not None:
        plt.savefig(file_name)

def plot_data_input(input_morph, num_columns, filename):

    N = int(input_morph.shape[1]**0.5)
    L = 1

    # Calculate the number of rows based on the number of columns and the length of input_morph
    num_rows = int(np.ceil(len(input_morph) / num_columns))

    # Create a new figure with variable subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns, num_rows))

    # Flatten the axes array to handle variable numbers of subplots
    axes = axes.flatten()

    # Loop through the input_morph rows and plot each row in a separate subplot
    for i in range(len(input_morph)):
        ax = axes[i] if i < len(axes) else None  # Handle cases with fewer subplots than data
        if ax:
            Z = input_morph[i].reshape(N, N)  # Reshape the vectorized Z to a 2D array
            min_val = np.min(Z)
            max_val = np.max(Z)
            im = ax.imshow(Z, cmap='viridis', extent=[0, L, 0, L], vmin=min_val, vmax=max_val)
            ax.set_title(f'Row {i+1}')
            ax.set_xticks([])
            ax.set_yticks([])

    # Add a color bar at the top of the figure
    cbar_ax = fig.add_axes([0.3, 1.02, 0.4, 0.005])  # Define position and size of the color bar
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

    # Remove any unused subplots
    for i in range(len(input_morph), len(axes)):
        fig.delaxes(axes[i])

    # Adjust subplot spacing
    plt.tight_layout()

    # Save the plot as a PDF and PNG file with user-defined filename
    plt.savefig(f'{filename}.png')

def create_2D_square_model_info_thermal(L,N,T_left,T_right):
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
    # Gauss quadrature points and weights (for a 2x2 integration)
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

    element_ids = jnp.arange(0,elements_nodes.shape[0])

    # Identify boundary nodes on the left and right edges
    left_boundary_nodes = jnp.arange(0, ny * nx, nx)  # Nodes on the left boundary
    left_boundary_nodes_values = T_left * jnp.ones(left_boundary_nodes.shape)
    right_boundary_nodes = jnp.arange(nx - 1, ny * nx, nx)  # Nodes on the right boundary
    right_boundary_nodes_values = T_right * jnp.ones(right_boundary_nodes.shape)
    boundary_nodes = jnp.concatenate([left_boundary_nodes, right_boundary_nodes])
    boundary_values = jnp.concatenate([left_boundary_nodes_values, right_boundary_nodes_values])
    non_boundary_nodes = []
    for i in range(N*N):
        if not jnp.any(boundary_nodes == i):
            non_boundary_nodes.append(i)
    non_boundary_nodes = jnp.array(non_boundary_nodes)

    nodes_dict = {"nodes_ids":jnp.arange(Y.shape[-1]),"X":X,"Y":Y,"Z":Z}
    elements_dict = {"elements_ids":element_ids,"elements_nodes":elements_nodes}
    dofs_dict = {"T":{"non_dirichlet_nodes_ids":non_boundary_nodes,"dirichlet_nodes_ids":boundary_nodes,"dirichlet_nodes_dof_value":boundary_values}}
    return {"nodes_dict":nodes_dict,"elements_dict":elements_dict,"dofs_dict":dofs_dict}

def box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, case_dir):

    cell_type = 'hexahedron'
    degree= 1
    msh_dir = case_dir
    os.makedirs(msh_dir, exist_ok=True)
    msh_file = os.path.join(msh_dir, 'box.msh')

    offset_x = 0.
    offset_y = 0.
    offset_z = 0.
    domain_x = Lx
    domain_y = Ly
    domain_z = Lz

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # save in old MSH format
    if cell_type.startswith('tetra'):
        Rec2d = False  # tris or quads
        Rec3d = False  # tets, prisms or hexas
    else:
        Rec2d = True
        Rec3d = True
    p = gmsh.model.geo.addPoint(offset_x, offset_y, offset_z)
    l = gmsh.model.geo.extrude([(0, p)], domain_x, 0, 0, [Nx], [1])
    s = gmsh.model.geo.extrude([l[1]], 0, domain_y, 0, [Ny], [1], recombine=Rec2d)
    v = gmsh.model.geo.extrude([s[1]], 0, 0, domain_z, [Nz], [1], recombine=Rec3d)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write(msh_file)
    gmsh.finalize()

    mesh = meshio.read(msh_file)
    points = mesh.points # (num_total_nodes, dim)
    cells =  mesh.cells_dict[cell_type] # (num_cells, num_nodes)
    meshio_obj = meshio.Mesh(points=points, cells={cell_type: cells})

    return meshio_obj

def create_3D_box_mesh(Nx,Ny,Nz,Lx,Ly,Lz,case_dir):

    # create empty fe mesh object
    fe_mesh = Mesh("box_io","box.")

    settings = box_mesh(Nx,Ny,Nz,Lx,Ly,Lz,case_dir)
    fe_mesh.node_ids = jnp.arange(len(settings.points))
    fe_mesh.nodes_coordinates = jnp.array(settings.points)

    left_mask = jnp.isclose(fe_mesh.nodes_coordinates[:,0], 0.0, atol=1e-5)
    right_mask = jnp.isclose(fe_mesh.nodes_coordinates[:,0], Lx, atol=1e-5)

    left_boundary_node_ids = fe_mesh.node_ids[left_mask]
    right_boundary_node_ids = fe_mesh.node_ids[right_mask]

    fe_mesh.elements_nodes = {"hexahedron":jnp.array(settings.cells_dict['hexahedron'])}

    fe_mesh.node_sets = {"left":left_boundary_node_ids,
                         "right":right_boundary_node_ids}
    
    fe_mesh.mesh_io = meshio.Mesh(fe_mesh.nodes_coordinates,fe_mesh.elements_nodes)

    fe_mesh.is_initialized = True

    return fe_mesh

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


def UpdateDefaultDict(default_dict:dict,given_dict:dict):
    filtered_update = {k: given_dict[k] for k in default_dict if k in given_dict}
    updated_dict = copy.deepcopy(default_dict)
    updated_dict.update(filtered_update)
    return updated_dict


"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: August, 2025 (revised)
 License: FOL/LICENSE

 A configurable 3D plotter for FOL/iFOL inference results saved in VTK files.
 - Auto-selects best FOL–FEM match (lowest L2 error on |U|)
 - Warped displacement panels (FOL/FEM)
 - Elasticity and error visualizations
 - Contour and orthogonal slice plots
 - Combined figure stitching via Matplotlib

 New in this revision:
 - Prefixes for elasticity and displacement fields are fully configurable via `config`:
     * elasticity_prefix (default: "K_")
     * u_fol_prefix      (default: "U_FOL_")
     * u_fe_prefix       (default: "U_FE_")
 - Optional fixed color limits for elasticity via `config["fixed_K_clim"]`.
 - Optional fixed color limits for error via `config["fixed_error_clim"]`.
 - Stable deformation scale in overview via `warp_factor_overview` (defaults to 1.0).
 - Global `show_edges` toggle in config.
 - Ensures **overview/combined use the same error field**: |‖U_FOL‖ − ‖U_FE‖|.

 Usage example:

    from fol.inference.plotter import Plotter3D
    import os, glob

    config = {
        "clip": True,
        "zoom": 0.9,
        "cmap": "coolwarm",
        "window_size": (1600, 1000),
        "scalar_bar_args": {"title": "", "vertical": True, "label_font_size": 22},
        "matplotlib_panel_zoom": {"displacement": 1.3, "elasticity": 1.0},
        "elasticity_prefix": "K_",
        "u_fol_prefix": "U_FOL_",
        "u_fe_prefix": "U_FE_",
        # "fixed_K_clim": [0.1, 1.0],
        # "fixed_error_clim": [0.0, 0.18],
        "show_edges": True,
        "warp_factor_overview": 1.0,
        "output_image": "overview.png",  # stitched figure filename
    }

    vtk_files = glob.glob(os.path.join(case_dir, "tested_samples", "*.vtk"))
    vtk_path = vtk_files[0]

    plotter = Plotter3D(vtk_path=vtk_path, warp_factor=1.0, config=config)
    plotter.render_all_panels()
"""

import os
import re
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use("Agg")  # headless-safe backend


class Plotter3D:
    """
    Class-based 3D plotter for FOL inference results saved in VTK files,
    with warped, contour-slice, and diagonal sampling views.

    All output PNGs are saved alongside the input `.vtk` file.
    """

    def __init__(self,
                 vtk_path: str,
                 warp_factor: float = 1.0,
                 config: dict | None = None):
        # Default configuration
        default_config = {
            "clip": True,
            "zoom": 0.8,
            "cmap": "coolwarm",
            "window_size": (1200, 800),
            "scalar_bar_args": {
                "title": "",
                "vertical": True,
                "title_font_size": 25,
                "label_font_size": 29,
                "position_x": 0.82,
                "position_y": 0.1,
                "width": 0.08,
                "height": 0.8,
                "font_family": "times",
            },
            "title_font_size": 24,
            "diag_points": 100,
            "final_figsize": (20, 10),
            "output_image": "combined_figure.png",
            "matplotlib_panel_zoom": {
                "displacement": 1,
                "elasticity": 0.8,
            },
            # prefixes (configurable)
            "elasticity_prefix": "K_",
            "u_fol_prefix": "U_FOL_",
            "u_fe_prefix": "U_FE_",
            # optional fixed clims for cross-run comparability
            "fixed_K_clim": None,
            "fixed_error_clim": None,
            # global toggles
            "show_edges": True,
            # overview-only warp factor (keeps overview scale stable)
            "warp_factor_overview": 1.0,
        }
        self.config = default_config if config is None else {**default_config, **config}
        self.do_clip = bool(self.config.get("clip", True))

        # Paths
        self.vtk_path = os.path.abspath(vtk_path)
        self.output_dir = os.path.dirname(self.vtk_path)
        os.makedirs(self.output_dir, exist_ok=True)

        # Load mesh & compute bounds
        self.mesh = pv.read(self.vtk_path)
        xmin, xmax, ymin, ymax, zmin, zmax = self.mesh.bounds
        self.cut_size = 0.5 * (xmax - xmin)
        self.xmax, self.ymax, self.zmax = xmax, ymax, zmax

        # Fields and parameters
        self.best_id = None
        self.fields = {}
        self.mag_names = {}
        self.warp_factor = float(warp_factor)
        self.warp_factor_overview = float(self.config.get("warp_factor_overview", self.warp_factor))
        self.diag_points = int(self.config["diag_points"])

        # Shared settings
        self.shared_scalar_bar_args = self.config["scalar_bar_args"]
        self.shared_zoom = float(self.config["zoom"])
        self.camera_position = [(2, 2, 2), (0.5, 0.5, 0.5), (0, 0, 1)]
        self.show_edges_default = bool(self.config.get("show_edges", True))

        # Name map (prefixes configurable)
        self.name_map = {
            "elasticity_prefix": str(self.config.get("elasticity_prefix", "K_")),
            "u_fol_prefix": str(self.config.get("u_fol_prefix", "U_FOL_")),
            "u_fe_prefix": str(self.config.get("u_fe_prefix", "U_FE_")),
        }

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def find_best_sample(self):
        """Find the sample index with minimum L2(|U_FOL|-|U_FE|)."""
        keys = list(self.mesh.point_data.keys())
        pf = re.escape(self.name_map["u_fol_prefix"])  # e.g., U_FOL_
        pe = re.escape(self.name_map["u_fe_prefix"])   # e.g., U_FE_

        fol_ids = {m.group(1) for k in keys if (m := re.match(pf + r"(\d+)$", k))}
        fe_ids  = {m.group(1) for k in keys if (m := re.match(pe + r"(\d+)$", k))}
        ids = sorted(fol_ids & fe_ids)
        if not ids:
            raise ValueError("No matching FOL/FE displacement field pairs found. Check prefixes and VTK fields.")

        best, min_err = None, float("inf")
        for i in ids:
            uf = self.mesh[f"{self.name_map['u_fol_prefix']}{i}"]
            ue = self.mesh[f"{self.name_map['u_fe_prefix']}{i}"]
            err = np.linalg.norm(np.linalg.norm(uf, axis=1) - np.linalg.norm(ue, axis=1))
            if err < min_err:
                min_err, best = err, i

        self.best_id = best
        self.fields = {
            "K_field": f"{self.name_map['elasticity_prefix']}{best}",
            "U_FOL":   f"{self.name_map['u_fol_prefix']}{best}",
            "U_FE":    f"{self.name_map['u_fe_prefix']}{best}",
        }
        self.mag_names = {
            "U_FOL_mag": f"{self.fields['U_FOL']}_mag",
            "U_FE_mag":  f"{self.fields['U_FE']}_mag",
        }
        print(f"Best sample = {best}, L2 error = {min_err:.6f}")

    def compute_derived_fields(self):
        fol, fe = self.fields['U_FOL'], self.fields['U_FE']
        self.mesh[self.mag_names['U_FOL_mag']] = np.linalg.norm(self.mesh[fol], axis=1)
        self.mesh[self.mag_names['U_FE_mag']]  = np.linalg.norm(self.mesh[fe], axis=1)

        # *** Single source of truth for error field used everywhere ***
        # absolute difference of magnitudes: |‖U_FOL‖ − ‖U_FE‖|
        self.mesh['abs_error'] = np.abs(
            self.mesh[self.mag_names['U_FOL_mag']] - self.mesh[self.mag_names['U_FE_mag']]
        )
        self.error_field = 'abs_error'
        self.error_title = r'| |U_FE| - |U_FOL| |'

    def apply_cut(self, mesh_obj: pv.DataSet) -> pv.DataSet:
        if not self.do_clip:
            return mesh_obj
        return mesh_obj.clip_box(
            bounds=(
                self.xmax - self.cut_size, self.xmax,
                self.ymax - self.cut_size, self.ymax,
                self.zmax - self.cut_size, self.zmax,
            ),
            invert=True,
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def render_panel(self, mesh_obj, field, clim, title, fname, show_edges=None):
        if show_edges is None:
            show_edges = self.show_edges_default
        plotter = pv.Plotter(off_screen=True, window_size=self.config['window_size'])
        plotter.add_mesh(
            mesh_obj, scalars=field, cmap=self.config['cmap'], clim=clim,
            show_edges=show_edges, edge_color='white', line_width=0.2,
            scalar_bar_args=self.shared_scalar_bar_args,
        )
        plotter.camera_position = self.camera_position
        plotter.camera.zoom(self.shared_zoom)
        plotter.add_axes()
        if title:
            plotter.add_text(title, font_size=self.config['title_font_size'], position='upper_edge')
        out = os.path.join(self.output_dir, fname)
        plotter.screenshot(out)
        plotter.close()
        print(f"Saved panel: {out}")

    def render_contour_slice(self):
        mesh = self.mesh
        fe_mag = self.mag_names['U_FE_mag']
        fol_mag = self.mag_names['U_FOL_mag']

        fe_contour = mesh.contour(scalars=fe_mag, isosurfaces=5)
        fol_contour = mesh.contour(scalars=fol_mag, isosurfaces=5)

        mesh.set_active_scalars(fe_mag)
        fe_slices = mesh.slice_orthogonal()
        mesh.set_active_scalars(fol_mag)
        fol_slices = mesh.slice_orthogonal()

        plotter = pv.Plotter(shape=(2, 2), off_screen=True,
                             window_size=(3200, 2400), border=False)

        plotter.subplot(0, 0)
        plotter.add_text(f"FEM: Contour {fe_mag}", font_size=12)
        plotter.add_mesh(fe_contour, scalars=fe_mag, cmap=self.config['cmap'])
        plotter.add_axes()

        plotter.subplot(0, 1)
        plotter.add_text("FEM: Orthogonal Slices", font_size=12)
        plotter.add_mesh(fe_slices, scalars=fe_mag, cmap=self.config['cmap'])
        plotter.add_axes()

        plotter.subplot(1, 0)
        plotter.add_text(f"FOL: Contour {fol_mag}", font_size=12)
        plotter.add_mesh(fol_contour, scalars=fol_mag, cmap=self.config['cmap'])
        plotter.add_axes()

        plotter.subplot(1, 1)
        plotter.add_text("FOL: Orthogonal Slices", font_size=12)
        plotter.add_mesh(fol_slices, scalars=fol_mag, cmap=self.config['cmap'])
        plotter.add_axes()

        plotter.link_views()
        plotter.view_isometric()
        screenshot = os.path.join(self.output_dir, "fol_fem_contour_grid.png")
        plotter.screenshot(screenshot)
        plotter.close()
        print(f"Saved contour-slice grid: {screenshot}")

    def render_diagonal_plot(self):
        n = self.diag_points
        diag_pts = np.linspace([0, 0, 0], [1, 1, 1], n)
        probe = pv.PolyData(diag_pts).sample(self.mesh)
        fol_mag = probe[self.mag_names['U_FOL_mag']]
        fe_mag = probe[self.mag_names['U_FE_mag']]
        K_diag = probe.point_data[self.fields['K_field']]

        fig = plt.figure(figsize=self.config['final_figsize'])
        gs = GridSpec(2, 2, figure=fig)

        clipped_png = os.path.join(self.output_dir, 'panel8.png')
        ax0 = fig.add_subplot(gs[0, 0])
        img = plt.imread(clipped_png)
        ax0.imshow(img)
        ax0.axis('off')
        ax0.set_title('Clipped Elasticity View', fontsize=self.config['title_font_size'])

        ax1 = fig.add_subplot(gs[0, 1])
        x = np.linspace(0, 1, n)
        ax1.plot(x, K_diag, linewidth=2)
        ax1.set_title('Elasticity Along Diagonal', fontsize=self.config['title_font_size'])
        ax1.set_xlabel('Normalized Distance')
        ax1.set_ylabel('Elasticity')
        e_zoom = float(self.config['matplotlib_panel_zoom']['elasticity'])
        ax1.set_ylim(self._zoom_ylim(K_diag, e_zoom))
        ax1.grid(False)

        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(x, fol_mag, label='FOL Disp', linewidth=2)
        ax2.plot(x, fe_mag, label='FEM Disp', linestyle='--', linewidth=2)
        ax2.set_title('Displacement Magnitude Along Diagonal', fontsize=self.config['title_font_size'])
        ax2.set_xlabel('Normalized Distance')
        ax2.set_ylabel('Displacement')
        d_zoom = float(self.config['matplotlib_panel_zoom']['displacement'])
        ax2.set_ylim(self._zoom_ylim(np.concatenate([fol_mag, fe_mag]), d_zoom))
        ax2.legend()
        ax2.grid(False)

        plt.tight_layout()
        out_diag = os.path.join(self.output_dir, 'diagonal_plot.png')
        plt.savefig(out_diag, dpi=300)
        plt.close(fig)
        print(f"Saved diagonal comparison plot: {out_diag}")

    @staticmethod
    def _zoom_ylim(data, zoom_factor: float):
        mn, mx = float(np.min(data)), float(np.max(data))
        ctr = 0.5 * (mn + mx)
        hr = 0.5 * (mx - mn) / max(zoom_factor, 1e-12)
        return [ctr - hr, ctr + hr]

    # ------------------------------------------------------------------
    # Pipelines
    # ------------------------------------------------------------------
    def render_all_panels(self):
        # full pipeline
        self.find_best_sample()
        self.compute_derived_fields()

        panels = []
        base = self.apply_cut(self.mesh)
        K = self.fields['K_field']

        # allow fixed_K_clim from config (for cross-run comparability)
        fixed = self.config.get("fixed_K_clim")
        if fixed is not None:
            K_clim = list(fixed)
        else:
            K_clim = [float(self.mesh[K].min()), float(self.mesh[K].max())]

        U_max = max(float(self.mesh[self.mag_names['U_FOL_mag']].max()),
                    float(self.mesh[self.mag_names['U_FE_mag']].max()))
        U_clim = [0.0, U_max]

        panels.append((base, K, K_clim, 'Elasticity Morphology', 'panel1.png', False))
        panels.append((base, K, K_clim, 'E(x,y,z) with Mesh', 'panel2.png', True))

        # clipped elasticity thumbnail used in stitched figure
        p5, p4, p2 = np.array([1, 0, 1]), np.array([0, 0, 1]), np.array([0, 1, 0])
        normal = np.cross(p4 - p5, p2 - p5).astype(float)
        normal /= np.linalg.norm(normal)
        panels.append((self.mesh.clip(normal=normal, origin=p5, invert=False), K, None, '', 'panel8.png', False))

        # FOL warped (stable overview warp)
        mf = self.mesh.copy(deep=True)
        mf.active_vectors_name = self.fields['U_FOL']
        wf = mf.warp_by_vector(factor=self.warp_factor_overview)
        wf[self.mag_names['U_FOL_mag']] = self.mesh[self.mag_names['U_FOL_mag']]
        panels.append((self.apply_cut(wf), self.mag_names['U_FOL_mag'], U_clim, 'FOL Deformation', 'panel5_warped.png', False))

        # FEM warped (stable overview warp)
        mf = self.mesh.copy(deep=True)
        mf.active_vectors_name = self.fields['U_FE']
        wf = mf.warp_by_vector(factor=self.warp_factor_overview)
        wf[self.mag_names['U_FE_mag']] = self.mesh[self.mag_names['U_FE_mag']]
        panels.append((self.apply_cut(wf), self.mag_names['U_FE_mag'], U_clim, 'FEM Deformation', 'panel6_warped.png', False))

        # error (same field used everywhere + optional fixed clim)
        err_clim = (list(self.config["fixed_error_clim"]) if self.config.get("fixed_error_clim") is not None
                    else [0.0, float(self.mesh[self.error_field].max())])
        panels.append((base, self.error_field, err_clim, self.error_title, 'panel7.png', False))

        for mesh_obj, field, clim, title, fname, edges in panels:
            self.render_panel(mesh_obj, field, clim, title, fname, show_edges=edges)

        # stitch
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig)
        for i, (_, _, _, _, fname, _) in enumerate(panels):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            img = plt.imread(os.path.join(self.output_dir, fname))
            ax.imshow(img)
            ax.axis('off')
        combined = os.path.join(self.output_dir, self.config['output_image'])
        plt.tight_layout()
        plt.savefig(combined, dpi=300)
        plt.close(fig)
        print(f"Saved combined figure: {combined}")

        # extras
        self.render_contour_slice()
        self.render_diagonal_plot()

    # ------------------------------------------------------------------
    # Optional: panels for a specific FEM sample index
    # ------------------------------------------------------------------
    def render_sample_panels(self, idx: int):
        K_field = f"{self.name_map['elasticity_prefix']}{idx}"
        U_FE_field = f"{self.name_map['u_fe_prefix']}{idx}"

        # Compute magnitude (if not already present)
        mag_field = f"{U_FE_field}_mag"
        if mag_field not in self.mesh.point_data:
            self.mesh[mag_field] = np.linalg.norm(self.mesh[U_FE_field], axis=1)

        # Panel 1: Elasticity field (PyVista)
        K_clim = [float(self.mesh[K_field].min()), float(self.mesh[K_field].max())]
        fixed = self.config.get("fixed_K_clim")
        if fixed is not None:
            K_clim = list(fixed)
        self.render_panel(
            self.apply_cut(self.mesh), K_field, K_clim,
            f'Elasticity ({K_field})', f'sample{idx}_elasticity.png'
        )

        # Panel 2: Warped deformation (PyVista)
        U_clim = [0.0, float(self.mesh[mag_field].max())]
        mf = self.mesh.copy(deep=True)
        mf.active_vectors_name = U_FE_field
        wf = mf.warp_by_vector(factor=self.warp_factor)
        wf[mag_field] = self.mesh[mag_field]
        self.render_panel(
            self.apply_cut(wf), mag_field, U_clim,
            f'Warped Displacement ({U_FE_field})', f'sample{idx}_warped.png'
        )

        # Panel 3: Displacement along diagonal (matplotlib)
        n = self.diag_points
        diag_pts = np.linspace([0, 0, 0], [1, 1, 1], n)
        probe = pv.PolyData(diag_pts).sample(self.mesh)
        fe_mag = probe.point_data[mag_field]
        x = np.linspace(0, 1, n)
        fig_line, ax_line = plt.subplots(figsize=(6, 6))
        ax_line.plot(x, fe_mag, label=f'{U_FE_field} along diagonal', linewidth=2)
        ax_line.set_xlabel('Normalized Distance')
        ax_line.set_ylabel('Displacement')
        ax_line.set_title(f'Displacement Along Diagonal ({U_FE_field})')
        ax_line.grid(True)
        plt.tight_layout()
        diag_plot_path = os.path.join(self.output_dir, f'sample{idx}_diag_line.png')
        fig_line.savefig(diag_plot_path, dpi=200)
        plt.close(fig_line)

        # Combine the three panels into a single PNG
        import matplotlib.image as mpimg
        img_paths = [
            f'sample{idx}_elasticity.png',
            f'sample{idx}_warped.png',
            f'sample{idx}_diag_line.png',
        ]
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for i, img_path in enumerate(img_paths):
            img = mpimg.imread(os.path.join(self.output_dir, img_path))
            axs[i].imshow(img)
            axs[i].axis('off')
        fig.suptitle(f"Sample {idx}", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        panel_path = os.path.join(self.output_dir, f'sample{idx}_panel.png')
        plt.savefig(panel_path, dpi=200)
        plt.close(fig)
        print(f"Saved combined panel for sample {idx}: {panel_path}")


# ----------------------------------------------------------------------
# Utility: solver convergence plot 
# ----------------------------------------------------------------------

def plot_solver_convergence(residual_norms_history, save_path=None, show=False):
    """
    Plots the nonlinear solver residual norm vs. iteration for each sample.

    Args:
        residual_norms_history: List of lists, each containing the residual norms for a sample.
        save_path: If given, the plot is saved to this path.
        show: If True, the plot is displayed.
    """
    plt.figure(figsize=(8, 6))
    for idx, residual_norms in enumerate(residual_norms_history):
        plt.semilogy(residual_norms, '-o', label=f"Sample {idx}")
    plt.xlabel("Iteration number", fontsize=12)
    plt.ylabel("Residual norm (log scale)", fontsize=12)
    plt.title("Nonlinear Solver Convergence (Neo-Hookean)", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Convergence plot saved to: {save_path}")
    if show:
        plt.show()
    plt.close()