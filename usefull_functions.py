
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import math
import gmsh
import meshio
import os

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

def create_3D_box_model_info_thermal(Nx,Ny,Nz,Lx,Ly,Lz,T_left,T_right,case_dir):

    settings = box_mesh(Nx,Ny,Nz,Lx,Ly,Lz,case_dir)
    X = settings.points[:,0]
    Y = settings.points[:,1]
    Z = settings.points[:,2]

    left_boundary_node_ids = []
    right_boundary_node_ids = []
    none_boundary_node_ids = []
    for node_id,node_corrds in enumerate(settings.points):
        if np.isclose(node_corrds[0], 0., atol=1e-5):
            left_boundary_node_ids.append(node_id)
        elif np.isclose(node_corrds[0], Lx, atol=1e-5):
            right_boundary_node_ids.append(node_id)
        else:
            none_boundary_node_ids.append(node_id)

    left_boundary_node_ids = jnp.array(left_boundary_node_ids)
    right_boundary_node_ids = jnp.array(right_boundary_node_ids)
    none_boundary_node_ids = jnp.array(none_boundary_node_ids)

    left_boundary_nodes_values = T_left * jnp.ones(left_boundary_node_ids.shape)
    right_boundary_nodes_values = T_right * jnp.ones(right_boundary_node_ids.shape)

    boundary_nodes = jnp.concatenate([left_boundary_node_ids, right_boundary_node_ids])
    boundary_values = jnp.concatenate([left_boundary_nodes_values, right_boundary_nodes_values])

    nodes_dict = {"nodes_ids":jnp.arange(Y.shape[-1]),"X":X,"Y":Y,"Z":Z}
    elements_dict = {"elements_ids":jnp.arange(len(settings.cells_dict['hexahedron'])),
                     "elements_nodes":jnp.array(settings.cells_dict['hexahedron'])}
    dofs_dict = {"T":{"non_dirichlet_nodes_ids":none_boundary_node_ids,"dirichlet_nodes_ids":boundary_nodes,"dirichlet_nodes_dof_value":boundary_values}}
    return {"nodes_dict":nodes_dict,"elements_dict":elements_dict,"dofs_dict":dofs_dict},settings
def create_3D_box_model_info_mechanical(model_settings,case_dir):

    settings = box_mesh(model_settings["Nx"],model_settings["Ny"],
                        model_settings["Nz"],model_settings["Lx"],
                        model_settings["Ly"],model_settings["Lz"],case_dir)
    X = settings.points[:,0]
    Y = settings.points[:,1]
    Z = settings.points[:,2]

    left_boundary_node_ids = []
    left_non_boundary_node_ids = []
    right_boundary_node_ids = []
    right_non_boundary_node_ids = []
    left_right_non_boundary_node_ids = []
    for node_id,node_corrds in enumerate(settings.points):
        if np.isclose(node_corrds[0], 0., atol=1e-5):
            left_boundary_node_ids.append(node_id)
        else:
            left_non_boundary_node_ids.append(node_id)

        if np.isclose(node_corrds[0], model_settings["Lx"], atol=1e-5):
            right_boundary_node_ids.append(node_id)
        else:
            right_non_boundary_node_ids.append(node_id)

        if not np.isclose(node_corrds[0], 0., atol=1e-5):
            if not np.isclose(node_corrds[0], model_settings["Lx"], atol=1e-5):
                left_right_non_boundary_node_ids.append(node_id)

    dofs_dict = {"Ux":{"non_dirichlet_nodes_ids":[],
                       "dirichlet_nodes_ids":[],
                       "dirichlet_nodes_dof_value":[]},
                 "Uy":{"non_dirichlet_nodes_ids":[],
                       "dirichlet_nodes_ids":[],
                       "dirichlet_nodes_dof_value":[]},
                 "Uz":{"non_dirichlet_nodes_ids":[],
                       "dirichlet_nodes_ids":[],
                       "dirichlet_nodes_dof_value":[]}}

    for dof in ["Ux","Uy","Uz"]:
        if model_settings[f"{dof}_left"] !="" and model_settings[f"{dof}_right"] !="":

            dofs_dict[dof]["non_dirichlet_nodes_ids"].extend(left_right_non_boundary_node_ids)

            dofs_dict[dof]["dirichlet_nodes_ids"].extend(left_boundary_node_ids)
            dof_values = [model_settings[f"{dof}_left"]] * len(left_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_dof_value"].extend(dof_values)

            dofs_dict[dof]["dirichlet_nodes_ids"].extend(right_boundary_node_ids)
            dof_values = [model_settings[f"{dof}_right"]] * len(right_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_dof_value"].extend(dof_values)

        elif model_settings[f"{dof}_right"] !="":
            dofs_dict[dof]["non_dirichlet_nodes_ids"].extend(right_non_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_ids"].extend(right_boundary_node_ids)
            dof_values = [model_settings[f"{dof}_right"]] * len(right_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_dof_value"].extend(dof_values)  

        elif model_settings[f"{dof}_left"] !="":
            dofs_dict[dof]["non_dirichlet_nodes_ids"].extend(left_non_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_ids"].extend(left_boundary_node_ids)
            dof_values = [model_settings[f"{dof}_left"]] * len(left_boundary_node_ids)
            dofs_dict[dof]["dirichlet_nodes_dof_value"].extend(dof_values) 

        dofs_dict[dof]["dirichlet_nodes_dof_value"] = np.array(dofs_dict[dof]["dirichlet_nodes_dof_value"])
        dofs_dict[dof]["non_dirichlet_nodes_ids"] = np.array(dofs_dict[dof]["non_dirichlet_nodes_ids"])
        dofs_dict[dof]["dirichlet_nodes_ids"] = np.array(dofs_dict[dof]["dirichlet_nodes_ids"])


    nodes_dict = {"nodes_ids":jnp.arange(Y.shape[-1]),"X":X,"Y":Y,"Z":Z}
    elements_dict = {"elements_ids":jnp.arange(len(settings.cells_dict['hexahedron'])),
                     "elements_nodes":jnp.array(settings.cells_dict['hexahedron'])}

    return {"nodes_dict":nodes_dict,"elements_dict":elements_dict,"dofs_dict":dofs_dict},settings

def create_2D_square_model_info_mechanical(L,N,Ux_left,Ux_right,Uy_left,Uy_right):
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
    nodes_dict = {"nodes_ids":jnp.arange(Y.shape[-1]),"X":X,"Y":Y,"Z":Z}

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
    elements_dict = {"elements_ids":element_ids,"elements_nodes":elements_nodes}

    # Identify boundary nodes on the left and right edges
    left_boundary_nodes = jnp.arange(0, ny * nx, nx)  # Nodes on the left boundary
    right_boundary_nodes = jnp.arange(nx - 1, ny * nx, nx)  # Nodes on the right boundary

    left_ux_values = Ux_left * jnp.ones(left_boundary_nodes.shape)
    right_ux_values = Ux_right * jnp.ones(right_boundary_nodes.shape)
    ux_boundary_nodes = jnp.concatenate([left_boundary_nodes, right_boundary_nodes])
    ux_boundary_values = jnp.concatenate([left_ux_values, right_ux_values])
    ux_non_boundary_nodes = []
    for i in range(N*N):
        if not (jnp.any(ux_boundary_nodes == i)):
            ux_non_boundary_nodes.append(i)
    ux_non_boundary_nodes = jnp.array(ux_non_boundary_nodes)

    dofs_dict = {"Ux":{"non_dirichlet_nodes_ids":ux_non_boundary_nodes,"dirichlet_nodes_ids":ux_boundary_nodes,"dirichlet_nodes_dof_value":ux_boundary_values}}

    left_uy_values = Uy_left * jnp.ones(left_boundary_nodes.shape)
    right_uy_values = Uy_right * jnp.ones(left_boundary_nodes.shape)
    uy_boundary_nodes = jnp.concatenate([left_boundary_nodes, right_boundary_nodes])
    uy_boundary_values = jnp.concatenate([left_uy_values, right_uy_values])
    uy_non_boundary_nodes = []
    for i in range(N*N):
        if not (jnp.any(uy_boundary_nodes == i)):
            uy_non_boundary_nodes.append(i)
    uy_non_boundary_nodes = jnp.array(uy_non_boundary_nodes)

    dofs_dict["Uy"] = {"non_dirichlet_nodes_ids":uy_non_boundary_nodes,"dirichlet_nodes_ids":uy_boundary_nodes,"dirichlet_nodes_dof_value":uy_boundary_values}
    
    return {"nodes_dict":nodes_dict,"elements_dict":elements_dict,"dofs_dict":dofs_dict}

def create_random_fourier_samples(fourier_control):
    N = int(fourier_control.GetNumberOfControlledVariables()**0.5)
    num_coeffs = fourier_control.GetNumberOfVariables()
    coeffs_matrix = np.zeros((0,num_coeffs))
    for i in range (1):
        coeff_vec = np.random.normal(size=num_coeffs)
        coeffs_matrix = np.vstack((coeffs_matrix,coeff_vec))

    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    # also add uniform dstibuted K of value 0.5
    coeff_vec = 1e-3 * np.zeros((num_coeffs))
    coeff_vec[0] = 1.0
    coeffs_matrix = np.vstack((coeffs_matrix,coeff_vec))
    K_matrix = np.vstack((K_matrix,fourier_control.ComputeControlledVariables(coeff_vec)))
    # plot_data_input(K_matrix,10,'K distributions')    

    return coeffs_matrix,K_matrix