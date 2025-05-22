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

    fe_mesh.node_ids = jnp.arange(Y.shape[-1])
    fe_mesh.nodes_coordinates = jnp.stack((X,Y,Z), axis=1)

    fe_mesh.elements_nodes = {"hexahedron":jnp.array(settings.cells_dict['hexahedron'])}

    fe_mesh.node_sets = {"left":left_boundary_node_ids,
                         "right":right_boundary_node_ids}
    
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