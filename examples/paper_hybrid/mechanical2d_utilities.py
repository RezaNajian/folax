import jax.numpy as jnp
import numpy as np
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.loss import Loss
import matplotlib.pyplot as plt
from typing import Callable
import os
import time
import json
from datetime import datetime

def create_tpms_gyroid(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]
    if tpms_settings.get("constant") is not None:
        const = tpms_settings["constant"]
    else:
        const = 0.

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    sin_pix = jnp.sin(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    sin_piy = jnp.sin(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)
    sin_piz = jnp.sin(fz * jnp.pi * z + phi_z)

    K = cos_piy * sin_pix + cos_piz * sin_piy + cos_pix * sin_piz - const
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K

def create_tpms_schwarz_P(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]
    if tpms_settings.get("constant") is not None:
        const = tpms_settings["constant"]
    else:
        const = 0.

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)


    K = cos_piy + cos_piz + cos_pix - const
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K

def create_tpms_diamond(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]
    if tpms_settings.get("constant") is not None:
        const = tpms_settings["constant"]
    else:
        const = 0.

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)
    sin_pix = jnp.sin(fx * jnp.pi * x + phi_x)
    sin_piy = jnp.sin(fy * jnp.pi * y + phi_y)
    sin_piz = jnp.sin(fz * jnp.pi * z + phi_z)



    K = sin_pix * sin_piy * sin_piz +\
          sin_pix * cos_piy * cos_piz +\
              cos_pix * sin_piy * cos_piz +\
                  cos_pix * cos_piy * sin_piz - const
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K

def create_tpms_lidinoid(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]
    if tpms_settings.get("constant") is not None:
        const = tpms_settings["constant"]
    else:
        const = -0.3

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)
    sin_pix = jnp.sin(fx * jnp.pi * x + phi_x)
    sin_piy = jnp.sin(fy * jnp.pi * y + phi_y)
    sin_piz = jnp.sin(fz * jnp.pi * z + phi_z)
    cos_2pix = jnp.cos(2*(fx * jnp.pi * x + phi_x))
    cos_2piy = jnp.cos(2*(fy * jnp.pi * y + phi_y))
    cos_2piz = jnp.cos(2*(fz * jnp.pi * z + phi_z))
    sin_2pix = jnp.sin(2*(fx * jnp.pi * x + phi_x))
    sin_2piy = jnp.sin(2*(fy * jnp.pi * y + phi_y))
    sin_2piz = jnp.sin(2*(fz * jnp.pi * z + phi_z))



    K = sin_2pix * cos_piy * sin_piz +\
          sin_pix * sin_2piy * cos_piz +\
              cos_pix * sin_piy * sin_2piz -\
                  cos_2pix * cos_2piy - cos_2piy * cos_2piz - cos_2piz * cos_2pix - const
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K

def create_tpms_split_p(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)
    sin_pix = jnp.sin(fx * jnp.pi * x + phi_x)
    sin_piy = jnp.sin(fy * jnp.pi * y + phi_y)
    sin_piz = jnp.sin(fz * jnp.pi * z + phi_z)
    cos_2pix = jnp.cos(2*(fx * jnp.pi * x + phi_x))
    cos_2piy = jnp.cos(2*(fy * jnp.pi * y + phi_y))
    cos_2piz = jnp.cos(2*(fz * jnp.pi * z + phi_z))
    sin_2pix = jnp.sin(2*(fx * jnp.pi * x + phi_x))
    sin_2piy = jnp.sin(2*(fy * jnp.pi * y + phi_y))
    sin_2piz = jnp.sin(2*(fz * jnp.pi * z + phi_z))



    K = 1.1*(sin_2pix * cos_piy * sin_piz +
            sin_pix * sin_2piy * cos_piz +
            cos_pix * sin_piy * sin_2piz) - 0.2*(
        cos_2pix * cos_2piy + 
            cos_2piy * cos_2piz + 
            cos_2piz * cos_2pix) - 0.4*(
        cos_2pix + cos_2piy + cos_2piz)
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K


def create_sphere_lattice(fe_mesh, tpms_settings):
    """
    Creates a periodic sphere lattice inside a 3D box.

    Parameters:
    - fe_mesh: Mesh object with GetNodesX/Y/Z()
    - tpms_settings: dict with keys:
        - "sphere_diameter": float
        - "min": value inside the sphere
        - "max": value outside spheres
        - "section_axis_value": optional, float (z-plane slice)
        - "tolerance": optional, float (fuzzy boundary around spheres)
    """
    D = tpms_settings.get("sphere_diameter", 0.4)
    r = D / 2.0
    min_val = tpms_settings.get("min", 0.1)
    max_val = tpms_settings.get("max", 1.)
    tol = tpms_settings.get("tolerance", 0.02 * D)

    # Get mesh node coordinates
    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    # Cross-section override
    if tpms_settings.get("section_axis_value") is not None:
        section_z = tpms_settings["section_axis_value"]
        z = jnp.ones_like(x) * section_z

    # Assume the domain is [0,1]^3
    box_min, box_max = 0.0, 1.0
    box_size = box_max - box_min

    # Determine number of spheres in each direction
    n_per_axis = int(jnp.ceil(box_size / D)) + 1

    # Create lattice of centers with periodic extension
    lin = jnp.linspace(0, box_size, n_per_axis)
    cx, cy, cz = jnp.meshgrid(lin, lin, lin, indexing="ij")
    centers = jnp.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=1)

    # Add center sphere if not already there
    center_sphere = jnp.array([[0.5, 0.5, 0.5]])
    centers = jnp.concatenate([centers, center_sphere], axis=0)

    # Wrap node coordinates and sphere centers for distance calculation
    coords = jnp.stack([x, y, z], axis=1)
    dist_min = jnp.full((coords.shape[0],), jnp.inf)

    # Calculate minimum periodic distance to all centers
    for center in centers:
        d = jnp.abs(coords - center)
        d = jnp.minimum(d, 1.0 - d)  # periodic distance
        dist = jnp.linalg.norm(d, axis=1)
        dist_min = jnp.minimum(dist_min, dist)

    # Apply tolerance band if needed
    inside = dist_min <= (r + tol)
    outside = dist_min > (r + tol)

    # Smooth transition (optional, not implemented here — binary only)
    result = jnp.where(inside, min_val, max_val)
    return result

def create_random_periodic_sphere_field(fe_mesh, tpms_settings):
    """
    Generate a periodic field with randomly distributed non-overlapping spheres.

    Parameters:
    - fe_mesh: finite element mesh with node positions.
    - tpms_settings: dict with:
        - "sphere_diameter": float
        - "min": value inside spheres
        - "max": value outside spheres
        - "tolerance": optional fuzzy boundary width
        - "num_spheres": optional int, default=30
    """

    D = tpms_settings["sphere_diameter"]
    r = D / 2.0
    tol = tpms_settings.get("tolerance", 0.02 * D)
    min_val = tpms_settings["min"]
    max_val = tpms_settings["max"]
    num_spheres = tpms_settings.get("num_spheres", 30)
    np.random.seed(52)

    # Mesh node coordinates
    if tpms_settings.get("section_axis_value") is not None:
        # FE init starts here
        N = int(fe_mesh.GetNumberOfNodes()**0.5)
        # Generate mesh coordinates
        x = jnp.linspace(0, 1, N)
        y = jnp.linspace(0, 1, N)
        z = jnp.linspace(0, 1, N)
        X, Y, Z = jnp.meshgrid(x, y, z)
        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()
        coords = jnp.stack([x, y, z], axis=1)
    else:
        x = fe_mesh.GetNodesX()
        y = fe_mesh.GetNodesY()
        z = fe_mesh.GetNodesZ()
        coords = jnp.stack([x, y, z], axis=1)

    # Generate non-overlapping centers
    centers = []
    max_attempts = 1000
    attempts = 0
    while len(centers) < num_spheres and attempts < max_attempts:
        candidate = jnp.array(np.random.rand(3))  # random in [0, 1]^3
        is_far_enough = True
        for c in centers:
            delta = jnp.abs(candidate - c)
            delta = jnp.minimum(delta, 1.0 - delta)  # periodic
            dist = jnp.linalg.norm(delta)
            if dist < D:  # overlapping
                is_far_enough = False
                break
        if is_far_enough:
            centers.append(candidate)
        attempts += 1

    centers = jnp.array(centers)

    # Compute minimum distance from each node to any sphere center (periodic)
    dist_min = jnp.full((coords.shape[0],), jnp.inf)
    for center in centers:
        delta = jnp.abs(coords - center)
        delta = jnp.minimum(delta, 1.0 - delta)  # periodic wrapping
        dist = jnp.linalg.norm(delta, axis=1)
        dist_min = jnp.minimum(dist_min, dist)

    # Thresholding
    inside = dist_min <= (r + tol)
    binary_K = jnp.where(inside, min_val, max_val)

    if tpms_settings.get("section_axis_value") is not None:
        z_val = tpms_settings.get("section_axis_value")
        Nz = len(jnp.unique(z))
        mask = jnp.isclose(z, z_val, atol=1.0 / Nz / 2)
        K_slice = binary_K[mask]
        return K_slice

    return binary_K



def create_random_fiber_field(fe_mesh, tpms_settings):
    """
    Create a field with elongated super-ellipsoids (fibers) inside a periodic box.
    
    Parameters:
    - fe_mesh: provides node positions.
    - tpms_settings: dict with:
        - "fiber_length": float (along main axis)
        - "fiber_radius": float (minor radius)
        - "min", "max": float
        - "num_fibers": int
        - "tolerance": optional, fuzzy boundary width
    """
    L = tpms_settings["fiber_length"]
    r = tpms_settings["fiber_radius"]
    min_val = tpms_settings["min"]
    max_val = tpms_settings["max"]
    tol = tpms_settings.get("tolerance", 0.02 * r)
    num_fibers = tpms_settings.get("num_fibers", 30)
    np.random.seed(52)


    # Mesh node coordinates
    if tpms_settings.get("section_axis_value") is not None:
        # FE init starts here
        N = int(fe_mesh.GetNumberOfNodes()**0.5)
        # Generate mesh coordinates
        x = jnp.linspace(0, 1, N)
        y = jnp.linspace(0, 1, N)
        z = jnp.linspace(0, 1, N)
        X, Y, Z = jnp.meshgrid(x, y, z)
        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()
        coords = jnp.stack([x, y, z], axis=1)
    else:
        x = fe_mesh.GetNodesX()
        y = fe_mesh.GetNodesY()
        z = fe_mesh.GetNodesZ()
        coords = jnp.stack([x, y, z], axis=1)

    # Each fiber has center + random unit direction
    centers = []
    directions = []
    max_attempts = 1000
    attempts = 0

    while len(centers) < num_fibers and attempts < max_attempts:
        center = np.random.rand(3)
        theta = np.random.uniform(0, jnp.pi)
        phi = np.random.uniform(0, 2 * jnp.pi)
        direction = jnp.array([
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta)
        ])
        # Skip overlap checking for simplicity (can be added)
        centers.append(jnp.array(center))
        directions.append(direction)
        attempts += 1

    centers = jnp.stack(centers)
    directions = jnp.stack(directions)

    # Distance function for super-ellipsoid (fiber)
    dist_min = jnp.full((coords.shape[0],), jnp.inf)
    for i in range(num_fibers):
        c = centers[i]
        d = directions[i]
        vec = coords - c
        vec = jnp.where(vec > 0.5, vec - 1.0, vec)
        vec = jnp.where(vec < -0.5, vec + 1.0, vec)

        # Project onto fiber axis
        t = jnp.dot(vec, d)
        t = jnp.clip(t, -L/2, L/2)
        proj = jnp.outer(t, d)
        radial = vec - proj
        radial_dist = jnp.linalg.norm(radial, axis=1)

        # inside if radial <= r and |t| <= L/2
        inside = (radial_dist <= (r + tol)) & (jnp.abs(t) <= (L/2 + tol))
        dist_min = jnp.where(inside, 0.0, dist_min)

    binary_K = jnp.where(dist_min == 0.0, min_val, max_val)

    if tpms_settings.get("section_axis_value") is not None:
        z_val = tpms_settings.get("section_axis_value")
        Nz = len(jnp.unique(z))
        mask = jnp.isclose(z, z_val, atol=1.0 / Nz / 2)
        K_slice = binary_K[mask]
        return K_slice
    
    return binary_K



def plot_tpms_2d(tpms_settings: dict, model_settings: dict, tpms_fn: Callable, fe_mesh: Mesh, file_name:str, section_values:tuple=(0.,0.5,1.)):
    N = model_settings["N"]  # number of nodes per side in 2D (assumed square)

    plt.figure(figsize=(15, 8))

    section_values = section_values
    for index, axis_value in enumerate(section_values, start=1):
        tpms_settings.update({"section_axis_value": axis_value})
        K_matrix = tpms_fn(fe_mesh, tpms_settings)

        plt.subplot(2, 3, index)
        plt.imshow(K_matrix.reshape(N, N), cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
        plt.title(f"Section at {axis_value}")
        plt.axis("off")

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300)
        print(f"Saved: {file_name}")
        plt.close()
    else:
        plt.show()

import numpy as np
import jax.numpy as jnp

def compute_nodal_stress_from_gauss(total_stress: jnp.ndarray, element_nodes: np.ndarray, num_nodes: int) -> jnp.ndarray:
    """
    Args:
        total_stress: jnp.ndarray of shape (num_elements, 4, 3) - stress at Gauss points
        element_nodes: np.ndarray of shape (num_elements, 4) - node indices per element
        num_nodes: total number of nodes in the mesh

    Returns:
        nodal_stress: jnp.ndarray of shape (num_nodes, 3) - averaged stress per node
    """
    # Accumulators
    stress_accumulator = np.zeros((num_nodes, 3), dtype=np.float32)
    count_accumulator = np.zeros((num_nodes,), dtype=np.int32)

    num_elements, num_gauss, _ = total_stress.shape

    # For each element
    for e in range(num_elements):
        nodes = element_nodes[e]  # (4,) node indices
        gauss_stresses = total_stress[e]  # (4, 3)

        # Assuming Gauss point i corresponds to node i (common in reduced integration)
        for i in range(4):
            node_id = nodes[i]
            stress_accumulator[node_id] += np.array(gauss_stresses[i])
            count_accumulator[node_id] += 1

    # Avoid divide-by-zero
    count_accumulator[count_accumulator == 0] = 1
    nodal_stress = stress_accumulator / count_accumulator[:, None]  # shape: (num_nodes, 3)
    return jnp.array(nodal_stress)


def get_stress(loss_function:Loss, fe_mesh:Mesh, disp_field_vec:jnp.array, K_matrix:jnp.array):
    element_nodes = fe_mesh.GetElementsNodes(loss_function.element_type)
    total_control_vars = K_matrix.reshape(-1,1)
    total_stress = loss_function.ComputeTotalStress(total_control_vars,disp_field_vec)
    num_nodes = fe_mesh.GetNumberOfNodes()
    nodal_stress = compute_nodal_stress_from_gauss(total_stress, element_nodes, num_nodes)

    return nodal_stress.flatten()

def plot_paper(topology_field:tuple[np.array, np.array], shape_tuple:tuple[int,int], file_name:str,
               fe_sol_field:tuple[np.array, np.array],ifol_sol_field:tuple[np.array, np.array], 
               fe_stress_field:tuple[np.array, np.array],ifol_stress_field:tuple[np.array, np.array], 
               sol_field_err:tuple[np.array, np.array], stress_field_err:tuple[np.array, np.array]):
    
    N_base, N_zssr = shape_tuple
    K_matrix_base , K_matrix_zssr = topology_field                          # shape: (N_base,), (N_zssr,)
    ifol_sol_field_base , ifol_sol_field_zssr = ifol_sol_field              # shape: (2*N_base,), (2*N_zssr,)
    fe_sol_field_base , fe_sol_field_zssr = fe_sol_field                    # shape: (2*N_base,), (2*N_zssr,)
    sol_field_err_base , sol_field_err_zssr = sol_field_err                 # shape: (2*N_base,), (2*N_zssr,)
    stress_field_err_base , stress_field_err_zssr = stress_field_err        # shape: (3*N_base,), (3*N_zssr,)
    fe_stress_field_base , fe_stress_field_zssr = fe_stress_field           # shape: (3*N_base,), (3*N_zssr,)
    ifol_stress_field_base , ifol_stress_field_zssr = ifol_stress_field     # shape: (3*N_base,), (3*N_zssr,)

    
    fontsize = 16
    dir = "u"
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))  # Adjusted to 4 columns and 5 rows

    #### first row ####
    row, col = 0, 0
    # Plot the morphology for the base resolution
    im = axs[row, col].imshow(K_matrix_base.reshape(N_base, N_base), cmap='viridis', aspect='equal')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title(f'Elasticity Morph. {N_base}x{N_base}', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)
    
    row, col = 0, 1
    # Plot ifol solution displacement field for the base resolution
    ifol_u_base = ifol_sol_field_base[::2]
    im = axs[row, col].imshow(ifol_u_base.reshape(N_base, N_base), cmap='coolwarm', aspect='equal')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title(f'${dir}$, iFOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    row, col = 0, 2
    # Plot fe solution displacement field for the base resolution
    fe_u_base = fe_sol_field_base[::2]
    im = axs[row, col].imshow(fe_u_base.reshape(N_base, N_base), cmap='coolwarm', aspect='equal')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title(f'${dir}$, FE', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    row, col = 0, 3
    # Plot fe solution displacement field for the base resolution
    u_error_base = sol_field_err_base[::2]
    im = axs[row, col].imshow(u_error_base.reshape(N_base, N_base), cmap='coolwarm', aspect='equal')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title(f'iFOL Abs. Difference ${dir}$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    
    #### second row ####
    row, col = 1, 0
    # Plot the morphology for the base resolution
    im = axs[row, col].imshow(K_matrix_zssr.reshape(N_zssr, N_zssr), cmap='viridis', aspect='equal')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title(f'Elasticity Morph. {N_zssr}x{N_zssr}', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)
    
    row, col = 1, 1
    # Plot ifol solution displacement field for the base resolution
    ifol_u_zssr = ifol_sol_field_zssr[::2]
    im = axs[row, col].imshow(ifol_u_zssr.reshape(N_zssr, N_zssr), cmap='coolwarm', aspect='equal')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title(f'${dir}$, iFOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    row, col = 1, 2
    # Plot fe solution displacement field for the base resolution
    fe_u_zssr = fe_sol_field_zssr[::2]
    im = axs[row, col].imshow(fe_u_zssr.reshape(N_zssr, N_zssr), cmap='coolwarm', aspect='equal')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title(f'${dir}$, HFE', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    row, col = 1, 3
    # Plot fe solution displacement error field for the base resolution
    u_error_zssr = sol_field_err_zssr[::2]
    im = axs[row, col].imshow(u_error_zssr.reshape(N_zssr, N_zssr), cmap='coolwarm', aspect='equal')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title(f'iFOL Abs. Difference ${dir}$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    # #### third row ####
    # row, col = 1, 0
    # # Plot ifol xx component of stress field for the base resolution
    # ifol_sxx_base = ifol_stress_field_base[::3]
    # im = axs[row, col].imshow(ifol_sxx_base.reshape(N_base, N_base), cmap='plasma')
    # axs[row, col].set_xticks([])
    # axs[row, col].set_yticks([])
    # axs[row, col].set_title('$\sigma_{xx}$, iFOL', fontsize=fontsize)
    # cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    # cbar.ax.tick_params(labelsize=fontsize)
    # cbar.ax.yaxis.labelpad = 5
    # cbar.ax.tick_params(length=5, width=1)

    # row, col = 1, 1
    # # Plot ifol yy component of stress field for the base resolution
    # ifol_syy_base = ifol_stress_field_base[1::3]
    # im = axs[row, col].imshow(ifol_syy_base.reshape(N_base, N_base), cmap='plasma')
    # axs[row, col].set_xticks([])
    # axs[row, col].set_yticks([])
    # axs[row, col].set_title('$\sigma_{yy}$, iFOL', fontsize=fontsize)
    # cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    # cbar.ax.tick_params(labelsize=fontsize)
    # cbar.ax.yaxis.labelpad = 5
    # cbar.ax.tick_params(length=5, width=1)

    # row, col = 1, 2
    # # Plot abs error of xx component of stress field for the base resolution
    # err_sxx_base = stress_field_err_base[::3]
    # im = axs[row, col].imshow(err_sxx_base.reshape(N_base, N_base), cmap='plasma')
    # axs[row, col].set_xticks([])
    # axs[row, col].set_yticks([])
    # axs[row, col].set_title('$\sigma_{xx}$, Abs. Difference', fontsize=fontsize)
    # cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    # cbar.ax.tick_params(labelsize=fontsize)
    # cbar.ax.yaxis.labelpad = 5
    # cbar.ax.tick_params(length=5, width=1)

    # row, col = 1, 3
    # # Plot abs error of yy component of stress field for the base resolution
    # err_syy_base = stress_field_err_base[1::3]
    # im = axs[row, col].imshow(err_syy_base.reshape(N_base, N_base), cmap='plasma')
    # axs[row, col].set_xticks([])
    # axs[row, col].set_yticks([])
    # axs[row, col].set_title('$\sigma_{yy}$, Abs. Difference', fontsize=fontsize)
    # cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    # cbar.ax.tick_params(labelsize=fontsize)
    # cbar.ax.yaxis.labelpad = 5
    # cbar.ax.tick_params(length=5, width=1)

    #### fourth row ####
    row, col = 2, 0
    # Plot ifol xx component of stress field for the zssr resolution
    ifol_sxx_zssr = ifol_stress_field_zssr[::3]
    im = axs[row, col].imshow(ifol_sxx_zssr.reshape(N_zssr, N_zssr), cmap='plasma')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title('$\sigma_{xx}$, iFOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    row, col = 2, 1
    # Plot ifol yy component of stress field for the zssr resolution
    ifol_syy_zssr = ifol_stress_field_zssr[1::3]
    im = axs[row, col].imshow(ifol_syy_zssr.reshape(N_zssr, N_zssr), cmap='plasma')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title('$\sigma_{yy}$, iFOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    row, col = 2, 2
    # Plot abs error of xx component of stress field for the zssr resolution
    err_sxx_zssr = stress_field_err_zssr[::3]
    im = axs[row, col].imshow(err_sxx_zssr.reshape(N_zssr, N_zssr), cmap='plasma')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title('$\sigma_{xx}$, Abs. Difference', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    row, col = 2, 3
    # Plot abs error of yy component of stress field for the zssr resolution
    err_syy_zssr = stress_field_err_zssr[1::3]
    im = axs[row, col].imshow(err_syy_zssr.reshape(N_zssr, N_zssr), cmap='plasma')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title('$\sigma_{yy}$, Abs. Difference', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    #### fifth row ####
    row, col = 3, 0
    # Plot fe xx component of stress field for the hybrid fe
    fe_sxx_zssr = fe_stress_field_zssr[::3]
    im = axs[row, col].imshow(fe_sxx_zssr.reshape(N_zssr, N_zssr), cmap='plasma')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title('$\sigma_{xx}$, HFE', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    row, col = 3, 1
    # Plot fe yy component of stress field for the hybrid fe
    fe_syy_zssr = fe_stress_field_zssr[1::3]
    im = axs[row, col].imshow(fe_syy_zssr.reshape(N_zssr, N_zssr), cmap='plasma')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title('$\sigma_{yy}$, HFE', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)    


    # Extract cross-sections at y = 0.5
    y_index = N_zssr // 2
    stress_x_cross_fem = fe_sxx_zssr.reshape(N_zssr, N_zssr)[y_index, :]
    stress_y_cross_fem = fe_syy_zssr.reshape(N_zssr, N_zssr)[y_index, :]
    stress_x_cross_ifol = ifol_sxx_zssr.reshape(N_zssr, N_zssr)[y_index, :]
    stress_y_cross_ifol = ifol_syy_zssr.reshape(N_zssr, N_zssr)[y_index, :]

    # Plot cross-sections in the fourth column
    row, col = 3, 2
    L = 1.
    axs[row, col].plot(np.linspace(0, L, N_zssr), stress_x_cross_ifol, label='iFOL', color='b')
    axs[row, col].plot(np.linspace(0, L, N_zssr), stress_x_cross_fem, label='HFEM', color='r')
    axs[row, col].set_title('Cross-section $\sigma_{xx}$', fontsize=fontsize)
    axs[row, col].legend()

    row, col = 3, 3
    axs[row, col].plot(np.linspace(0, L, N_zssr), stress_y_cross_fem, label='HFEM', color='r')
    axs[row, col].plot(np.linspace(0, L, N_zssr), stress_y_cross_ifol, label='iFOL', color='b')
    axs[row, col].set_title('Cross-section $\sigma_{yy}$', fontsize=fontsize)
    axs[row, col].legend()

    plt.tight_layout()
    plt.savefig(file_name, dpi=300)


def plot_iFOL_FE_HFE(topology_field:np.array, ifol_sol_field:np.array, fe_sol_field:np.array, hfe_sol_field:np.array,
                 err_sol_field:np.array, file_name:str):
    
    fontsize = 16
    dir = "u"
    N = int(topology_field.size**0.5)
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))  # Adjusted to 1 columns and 4 rows

    #### first row ####
    row, col = 0, 0
    # Plot the morphology for the base resolution
    im = axs[col].imshow(topology_field.reshape(N, N), cmap='viridis', aspect='equal')
    axs[col].set_xticks([])
    axs[col].set_yticks([])
    axs[col].set_title(f'Elasticity Morph. {N}x{N}', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)
    
    row, col = 0, 1
    # Plot ifol solution displacement field for the base resolution
    ifol_u_base = ifol_sol_field[::2]
    im = axs[col].imshow(ifol_u_base.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[col].set_xticks([])
    axs[col].set_yticks([])
    axs[col].set_title(f'${dir}$, iFOL', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    row, col = 0, 2
    # Plot fe solution displacement field for the base resolution
    fe_u_base = fe_sol_field[::2]
    im = axs[col].imshow(fe_u_base.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[col].set_xticks([])
    axs[col].set_yticks([])
    axs[col].set_title(f'${dir}$, FE', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    row, col = 0, 4
    # Plot fe solution displacement field for the base resolution
    hfe_u_base = hfe_sol_field[::2]
    im = axs[col].imshow(hfe_u_base.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[col].set_xticks([])
    axs[col].set_yticks([])
    axs[col].set_title(f'${dir}$, HFE', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    row, col = 0, 3
    # Plot fe solution displacement field for the base resolution
    u_error_base = err_sol_field[::2]
    im = axs[col].imshow(u_error_base.reshape(N, N), cmap='coolwarm', aspect='equal')
    axs[col].set_xticks([])
    axs[col].set_yticks([])
    axs[col].set_title(f'iFOL Abs. Difference ${dir}$', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300)


def dump_output(ifol_model_settings: dict,
                train_settings_dict: dict,
                model_settings: dict,
                material_dict: dict,
                plotter: Callable,
                plot_settings: dict,
                file_name:dict,
                case_dir: str,
                file_format: str = "txt"):
    
    # # Ensure the path exists
    # path = os.path.join('.', case_dir)
    # os.makedirs(path, exist_ok=True)
    
    # Unique output filename with timestamp
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Handle file formats correctly
    if file_format.lower() == "txt":
        # output_filename = os.path.join(path, f"{file_name}_{timestamp}.txt")
        output_filename = f"{file_name}_{timestamp}.txt"
        
        with open(output_filename, 'w') as f:
            f.write("Model Settings:\n")
            json.dump(model_settings, f, indent=4)
            f.write("\n\nMaterial Dictionary:\n")
            json.dump(material_dict, f, indent=4)
            f.write("\n\niFOL Model Settings:\n")
            json.dump(ifol_model_settings, f, indent=4)
            f.write("\n\nTraining Settings:\n")
            json.dump(train_settings_dict, f, indent=4)

    elif file_format.lower() == "json":
        # output_filename = os.path.join(path, f"{file_name}_{timestamp}.json")
        output_filename = f"{file_name}_{timestamp}.txt"

        data_to_dump = {
            "model_settings": model_settings,
            "material_dict": material_dict,
            "ifol_model_settings": ifol_model_settings,
            "train_settings_dict": train_settings_dict,
            "timestamp": timestamp
        }

        with open(output_filename, 'w') as f:
            json.dump(data_to_dump, f, indent=4)

    else:
        raise TypeError("file_format must be either 'txt' or 'json'")

    # Prepare and call the plotting function
    name = plot_settings.get("file_name", "plot.png")
    # plot_settings["file_name"] = os.path.join(path, f"{name}_{timestamp}.png")
    plot_settings["file_name"] = f"{name}_{timestamp}.png"
    plotter(**plot_settings)


def tpms_to_K_matrix(tpms_sample_dict:dict):
    K_matrix_list_all = []
    tpms_functions_list = list(tpms_sample_dict.keys())
    for tpms_function in tpms_functions_list:
        axis_values_list = list(tpms_sample_dict[tpms_function]["K_matrix"].keys())
        for axis_value in axis_values_list:
            K_matrix_list_all.append(tpms_sample_dict[tpms_function]["K_matrix"][axis_value])
    return np.array(K_matrix_list_all)

def fourier_to_K_matrix(fourier_sample_dict:dict):
    K_matrix_list_all = []
    coeffs_matrix_list_all = []
    sample_series_list = list(fourier_sample_dict.keys())
    for series in sample_series_list:
        K_matrix_array = fourier_sample_dict[series]["K_matrix"]
        coeffs_matrix_array = fourier_sample_dict[series]["coeffs_matrix"]
        for index in range(K_matrix_array.shape[0]):
            K_matrix_list_all.append(K_matrix_array[index])
            coeffs_matrix_list_all.append(coeffs_matrix_array[index])
    return np.array(coeffs_matrix_list_all), np.array(K_matrix_list_all)

def voronoi_to_K_matrix(voronoi_sample_dict:dict):
    K_matrix_list_all = []
    coeffs_matrix_list_all = []
    sample_series_list = list(voronoi_sample_dict.keys())
    for series in sample_series_list:
        K_matrix_array = voronoi_sample_dict[series]["K_matrix"]
        coeffs_matrix_array = voronoi_sample_dict[series]["coeffs_matrix"]
        for index in range(K_matrix_array.shape[0]):
            K_matrix_list_all.append(K_matrix_array[index])
            coeffs_matrix_list_all.append(coeffs_matrix_array[index])
    return np.array(coeffs_matrix_list_all), np.array(K_matrix_list_all)
 
def plot_paper_triple(topology_field:tuple[np.array, np.array, np.array], shape_tuple:tuple[int,int,int], 
               fe_sol_field:tuple[np.array, np.array, np.array],ifol_sol_field:tuple[np.array, np.array, np.array], 
               sol_field_err:tuple[np.array, np.array, np.array], file_name:str,
               fe_stress_field:np.array, ifol_stress_field:np.array, stress_field_err:np.array):
    
    N_base, N_zssr1, N_zssr2 = shape_tuple
    K_matrix_base , K_matrix_zssr1 , K_matrix_zssr2 = topology_field               # shape: (N_base,), (N_zssr,)
    ifol_sol_field_base , ifol_sol_field_zssr1 , ifol_sol_field_zssr2 = ifol_sol_field       # shape: (2*N_base,), (2*N_zssr,)
    fe_sol_field_base , fe_sol_field_zssr1 , fe_sol_field_zssr2 = fe_sol_field               # shape: (2*N_base,), (2*N_zssr,)
    sol_field_err_base , sol_field_err_zssr1 , sol_field_err_zssr2 = sol_field_err           # shape: (2*N_base,), (2*N_zssr,)
    stress_field_err_zssr = stress_field_err        # shape: (3*N_base,), (3*N_zssr,)
    fe_stress_field_zssr = fe_stress_field           # shape: (3*N_base,), (3*N_zssr,)
    ifol_stress_field_zssr = ifol_stress_field     # shape: (3*N_base,), (3*N_zssr,)

    
    fontsize = 16
    dir = "u"
    fig, axs = plt.subplots(5, 4, figsize=(16, 20))  # Adjusted to 4 columns and 5 rows
    N = [N_base,N_zssr1,N_zssr2]
    data_list = [
                [K_matrix_base, ifol_sol_field_base, sol_field_err_base, fe_sol_field_base],
                [K_matrix_zssr1, ifol_sol_field_zssr1, sol_field_err_zssr1, fe_sol_field_zssr1],
                [K_matrix_zssr2, ifol_sol_field_zssr2, sol_field_err_zssr2, fe_sol_field_zssr2]]
    cmap_list = ['viridis','coolwarm', 'coolwarm', 'coolwarm']
    title_list = ['Elasticity Morph.',f'iFOL',f'Abs. Difference',f'HFE']
    for row_index in range(3):
        for col_index in range(4):
            im = axs[row_index, col_index].imshow(data_list[row_index][col_index].reshape(N[row_index], N[row_index]), 
                                      cmap=cmap_list[col_index], aspect='equal')
            axs[row_index, col_index].set_xticks([])
            axs[row_index, col_index].set_yticks([])
            axs[row_index, col_index].set_title(f'{str(title_list[col_index])} {N[row_index]}x{N[row_index]}', fontsize=fontsize)
            cbar = fig.colorbar(im, ax=axs[row_index, col_index], pad=0.02, shrink=0.7)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.yaxis.labelpad = 5
            cbar.ax.tick_params(length=5, width=1)        
    
    data_list = [ifol_stress_field_zssr[::3], ifol_stress_field_zssr[1::3], stress_field_err_zssr[::3], stress_field_err_zssr[1::3]]
    stress_title_list = ['$\sigma_{xx}$, iFOL', '$\sigma_{yy}$, iFOL', '$\sigma_{xx}$, Abs. Difference', '$\sigma_{yy}$, Abs. Difference']
    vmin_xx = fe_stress_field_zssr[::3].min()
    vmax_xx = fe_stress_field_zssr[::3].max()
    vmin_yy = fe_stress_field_zssr[1::3].min()
    vmax_yy = fe_stress_field_zssr[1::3].max()
    cbar_scale_min = [vmin_xx, vmin_yy, None, None]
    cbar_scale_max = [vmax_xx, vmax_yy, None, None]
    for col_index in range(4):
        #### fourth row ####
        row = 3
        # Plot ifol xx component of stress field for the zssr resolution
        im = axs[row, col_index].imshow(data_list[col_index].reshape(N_zssr2, N_zssr2), cmap='plasma', 
                                        vmin=cbar_scale_min[col_index], vmax=cbar_scale_max[col_index])
        axs[row, col_index].set_xticks([])
        axs[row, col_index].set_yticks([])
        axs[row, col_index].set_title(stress_title_list[col_index], fontsize=fontsize)
        cbar = fig.colorbar(im, ax=axs[row, col_index], pad=0.02, shrink=0.7)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.yaxis.labelpad = 5
        cbar.ax.tick_params(length=5, width=1)
        

    #### fifth row ####
    row, col = 4, 0
    # Plot fe xx component of stress field for the hybrid fe
    fe_sxx_zssr = fe_stress_field_zssr[::3]
    im = axs[row, col].imshow(fe_sxx_zssr.reshape(N_zssr2, N_zssr2), cmap='plasma')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title('$\sigma_{xx}$, HFE', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)

    row, col = 4, 1
    # Plot fe yy component of stress field for the hybrid fe
    fe_syy_zssr = fe_stress_field_zssr[1::3]
    im = axs[row, col].imshow(fe_syy_zssr.reshape(N_zssr2, N_zssr2), cmap='plasma')
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])
    axs[row, col].set_title('$\sigma_{yy}$, HFE', fontsize=fontsize)
    cbar = fig.colorbar(im, ax=axs[row, col], pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.labelpad = 5
    cbar.ax.tick_params(length=5, width=1)    

    ifol_sxx_zssr = ifol_stress_field_zssr[::3]
    ifol_syy_zssr = ifol_stress_field_zssr[1::3]
    # Extract cross-sections at y = 0.5
    y_index = N_zssr2 // 2
    stress_x_cross_fem = fe_sxx_zssr.reshape(N_zssr2, N_zssr2)[y_index, :]
    stress_y_cross_fem = fe_syy_zssr.reshape(N_zssr2, N_zssr2)[y_index, :]
    stress_x_cross_ifol = ifol_sxx_zssr.reshape(N_zssr2, N_zssr2)[y_index, :]
    stress_y_cross_ifol = ifol_syy_zssr.reshape(N_zssr2, N_zssr2)[y_index, :]

    # Plot cross-sections in the fourth column
    row, col = 4, 2
    L = 1.
    axs[row, col].plot(np.linspace(0, L, N_zssr2), stress_x_cross_ifol, label='iFOL', color='b')
    axs[row, col].plot(np.linspace(0, L, N_zssr2), stress_x_cross_fem, label='HFEM', color='r')
    axs[row, col].set_title('Cross-section $\sigma_{xx}$', fontsize=fontsize)
    axs[row, col].legend()

    row, col = 4, 3
    axs[row, col].plot(np.linspace(0, L, N_zssr2), stress_y_cross_fem, label='HFEM', color='r')
    axs[row, col].plot(np.linspace(0, L, N_zssr2), stress_y_cross_ifol, label='iFOL', color='b')
    axs[row, col].set_title('Cross-section $\sigma_{yy}$', fontsize=fontsize)
    axs[row, col].legend()

    plt.tight_layout()
    plt.savefig(file_name, dpi=300)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

def plot_paper_triple2(topology_field:tuple[np.array, np.array, np.array], shape_tuple:tuple[int,int,int], 
               fe_sol_field:tuple[np.array, np.array, np.array],ifol_sol_field:tuple[np.array, np.array, np.array], 
               sol_field_err:tuple[np.array, np.array, np.array], file_name:str,
               fe_stress_field:np.array, ifol_stress_field:np.array, stress_field_err:np.array):
    
    N_base, N_zssr1, N_zssr2 = shape_tuple
    K_matrix_base , K_matrix_zssr1 , K_matrix_zssr2 = topology_field
    ifol_sol_field_base , ifol_sol_field_zssr1 , ifol_sol_field_zssr2 = ifol_sol_field
    fe_sol_field_base , fe_sol_field_zssr1 , fe_sol_field_zssr2 = fe_sol_field
    sol_field_err_base , sol_field_err_zssr1 , sol_field_err_zssr2 = sol_field_err
    stress_field_err_zssr = stress_field_err
    fe_stress_field_zssr = fe_stress_field
    ifol_stress_field_zssr = ifol_stress_field

    fontsize = 16
    fig = plt.figure(figsize=(18, 26))
    gs = gridspec.GridSpec(6, 4, figure=fig, wspace=0.2, hspace=0.2)
    
    axs = np.empty((6, 4), dtype=object)
    # Create axs array with GridSpec
    for r in range(6):
        for c in range(4):
            # Skip the slots that will be part of a spanning plot
            if (r == 3 and c == 2) or (r == 3 and c == 3) or (r == 4 and c == 2) or (r == 4 and c == 3) or (r == 5 and c == 2) or (r == 5 and c == 3):
                axs[r, c] = None
                continue
            axs[r, c] = fig.add_subplot(gs[r, c])
    # Replace specific cells with spanning subplots
    axs[3, 2] = fig.add_subplot(gs[3, 2:4])
    axs[4, 2] = fig.add_subplot(gs[4, 2:4])
    axs[5, 2] = fig.add_subplot(gs[5, 2:4])
    
    N = [N_base, N_zssr1, N_zssr2]
    data_list = [
        [K_matrix_base, ifol_sol_field_base, sol_field_err_base, fe_sol_field_base],
        [K_matrix_zssr1, ifol_sol_field_zssr1, sol_field_err_zssr1, fe_sol_field_zssr1],
        [K_matrix_zssr2, ifol_sol_field_zssr2, sol_field_err_zssr2, fe_sol_field_zssr2]
    ]
    cmap_list = ['viridis', 'coolwarm', 'coolwarm', 'coolwarm']
    title_list = ['Elasticity Morph.', 'iFOL', 'Abs. Difference', 'HFEM']

    # First 3 rows
    for row_index in range(3):
        for col_index in range(4):
            im = axs[row_index, col_index].imshow(
                data_list[row_index][col_index].reshape(N[row_index], N[row_index]),
                cmap=cmap_list[col_index], aspect='equal'
            )
            axs[row_index, col_index].set_xticks([])
            axs[row_index, col_index].set_yticks([])
            axs[row_index, col_index].set_title(
                f'{str(title_list[col_index])} {N[row_index]}x{N[row_index]}', fontsize=fontsize
            )
            cbar = fig.colorbar(im, ax=axs[row_index, col_index], pad=0.02, shrink=0.7)
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.yaxis.labelpad = 5
            cbar.ax.tick_params(length=5, width=1)

    # Extract stress components
    ifol_sxx_zssr = ifol_stress_field_zssr[::3]
    ifol_syy_zssr = ifol_stress_field_zssr[1::3]
    ifol_sxy_zssr = ifol_stress_field_zssr[2::3]
    fe_sxx_zssr = fe_stress_field_zssr[::3]
    fe_syy_zssr = fe_stress_field_zssr[1::3]
    fe_sxy_zssr = fe_stress_field_zssr[2::3]

    # Extract cross-sections at y = 0.5
    y_index = N_zssr2 // 2
    stress_x_cross_fem = fe_sxx_zssr.reshape(N_zssr2, N_zssr2)[y_index, :]
    stress_y_cross_fem = fe_syy_zssr.reshape(N_zssr2, N_zssr2)[y_index, :]
    stress_xy_cross_fem = fe_sxy_zssr.reshape(N_zssr2, N_zssr2)[y_index, :]
    stress_x_cross_ifol = ifol_sxx_zssr.reshape(N_zssr2, N_zssr2)[y_index, :]
    stress_y_cross_ifol = ifol_syy_zssr.reshape(N_zssr2, N_zssr2)[y_index, :]
    stress_xy_cross_ifol = ifol_sxy_zssr.reshape(N_zssr2, N_zssr2)[y_index, :]

    L = 1.
    
    # Row 4
    im = axs[3, 0].imshow(ifol_sxx_zssr.reshape(N_zssr2, N_zssr2), cmap='plasma',
                          vmin=fe_sxx_zssr.min(), vmax=fe_sxx_zssr.max())
    axs[3, 0].set_xticks([]); axs[3, 0].set_yticks([])
    axs[3, 0].set_title(f'$P_{{11}}$, iFOL {N[2]}x{N[2]}', fontsize=fontsize)
    fig.colorbar(im, ax=axs[3, 0], pad=0.02, shrink=0.7)

    im = axs[3, 1].imshow(fe_sxx_zssr.reshape(N_zssr2, N_zssr2), cmap='plasma')
    axs[3, 1].set_xticks([]); axs[3, 1].set_yticks([])
    axs[3, 1].set_title(f'$P_{{11}}$, HFEM {N[2]}x{N[2]}', fontsize=fontsize)
    fig.colorbar(im, ax=axs[3, 1], pad=0.02, shrink=0.7)

    axs[3, 2].plot(np.linspace(0, L, N_zssr2), stress_x_cross_ifol, label='iFOL', color='b')
    axs[3, 2].plot(np.linspace(0, L, N_zssr2), stress_x_cross_fem, label='HFEM', color='r')
    axs[3, 2].set_title('Cross-section $P_{11}$', fontsize=fontsize)
    axs[3, 2].legend()
    # pos = axs[3, 2].get_position()
    # axs[3, 2].set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height * 0.8])

    # Row 5
    im = axs[4, 0].imshow(ifol_syy_zssr.reshape(N_zssr2, N_zssr2), cmap='plasma',
                          vmin=fe_syy_zssr.min(), vmax=fe_syy_zssr.max())
    axs[4, 0].set_xticks([]); axs[4, 0].set_yticks([])
    axs[4, 0].set_title(f'$P_{{22}}$, iFOL {N[2]}x{N[2]}', fontsize=fontsize)
    fig.colorbar(im, ax=axs[4, 0], pad=0.02, shrink=0.7)

    im = axs[4, 1].imshow(fe_syy_zssr.reshape(N_zssr2, N_zssr2), cmap='plasma')
    axs[4, 1].set_xticks([]); axs[4, 1].set_yticks([])
    axs[4, 1].set_title(f'$P_{{22}}$, HFEM {N[2]}x{N[2]}', fontsize=fontsize)
    fig.colorbar(im, ax=axs[4, 1], pad=0.02, shrink=0.7)

    axs[4, 2].plot(np.linspace(0, L, N_zssr2), stress_y_cross_fem, label='HFEM', color='r')
    axs[4, 2].plot(np.linspace(0, L, N_zssr2), stress_y_cross_ifol, label='iFOL', color='b')
    axs[4, 2].set_title('Cross-section $P_{22}$', fontsize=fontsize)
    axs[4, 2].legend()
    # pos = axs[4, 2].get_position()
    # axs[4, 2].set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height * 0.8])

    # Row 6
    im = axs[5, 0].imshow(ifol_sxy_zssr.reshape(N_zssr2, N_zssr2), cmap='plasma',
                          vmin=fe_sxy_zssr.min(), vmax=fe_sxy_zssr.max())
    axs[5, 0].set_xticks([]); axs[5, 0].set_yticks([])
    axs[5, 0].set_title(f'$P_{{12}}$, iFOL {N[2]}x{N[2]}', fontsize=fontsize)
    fig.colorbar(im, ax=axs[5, 0], pad=0.02, shrink=0.7)

    im = axs[5, 1].imshow(fe_sxy_zssr.reshape(N_zssr2, N_zssr2), cmap='plasma')
    axs[5, 1].set_xticks([]); axs[5, 1].set_yticks([])
    axs[5, 1].set_title(f'$P_{{12}}$, HFEM {N[2]}x{N[2]}', fontsize=fontsize)
    fig.colorbar(im, ax=axs[5, 1], pad=0.02, shrink=0.7)

    axs[5, 2].plot(np.linspace(0, L, N_zssr2), stress_xy_cross_fem, label='HFEM', color='r')
    axs[5, 2].plot(np.linspace(0, L, N_zssr2), stress_xy_cross_ifol, label='iFOL', color='b')
    axs[5, 2].set_title('Cross-section $P_{12}$', fontsize=fontsize)
    axs[5, 2].legend()
    # pos = axs[5, 2].get_position()
    # axs[5, 2].set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height * 0.8])

    plt.subplots_adjust(
    left=0.05,   # space on the left of the figure
    right=0.95,  # space on the right
    top=0.95,    # space on top
    bottom=0.05, # space at the bottom
    # wspace=0.2,
    # hspace=0.2
    )
    plt.savefig(file_name, dpi=300)
    plt.close()
