import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.fe_loss import FiniteElementLoss
import jax
from jax import jit
import os 
import math
import gmsh
import meshio


def reshape_T_U_to_nodewise3D(FE_TUV, num_nodes):
    T = FE_TUV[0:num_nodes]
    UV = FE_TUV[num_nodes:]
    UV = UV.reshape((num_nodes, 3))  # [[Ux0, Uy0], ..., [UxN, UyN]]
    return jnp.concatenate([T[:, None], UV], axis=1)

def reshape_nodewise3D_to_T_U(FE_nodewise):
    # Split into T and UVW
    num_nodes = FE_nodewise.shape[0]
    TUVW_all = []
    for i in range(num_nodes):
        T = FE_nodewise[i,:, 0]  # shape: (num_nodes,)
        UVW = FE_nodewise[i,:, 1:]  # shape: (num_nodes, 3)
        TUVW = jnp.concatenate([T, UVW.reshape(-1)]) 
        TUVW_all.append(TUVW)
    return np.array(TUVW_all)


def generate_morph_pattern(points):
    """
    Generate heterogeneity pattern based on meshio mesh points.

    Parameters:
        points (ndarray): meshio.points, shape = (num_points, 2) or (num_points, 3)

    Returns:
        hetero_morph (ndarray): Array of heterogeneity values, shape = (num_points,)
    """
    num_points = points.shape[0]
    hetero_morph = np.full((num_points,), 0.3)  # Initialize with zeros

    X = points[:, 0]
    Y = points[:, 1]

    # Define pattern centers and radii
    center2 = (0.6, 0.6)
    radius2 = 0.2

    # Apply condition based on Euclidean distance
    dist2 = (X - center2[0])**2 + ((1 - Y) - center2[1])**2
    mask2 = dist2 < radius2**2

    # Assign lower value inside the region
    hetero_morph[mask2] = 1.0

    # # Second inclusion
    # center3 = (0.3, 0.7)
    # radius3 = 0.15
    # dist3 = (X - center3[0])**2 + ((1 - Y) - center3[1])**2
    # mask3 = dist3 < radius3**2
    # hetero_morph[mask3] = 1.0  

    return hetero_morph

def generate_morph_pattern2(points):
    """
    Generate heterogeneity pattern based on meshio mesh points.

    Parameters:
        points (ndarray): meshio.points, shape = (num_points, 2) or (num_points, 3)

    Returns:
        hetero_morph (ndarray): Array of heterogeneity values, shape = (num_points,)
    """
    num_points = points.shape[0]
    hetero_morph = np.full((num_points,), 1.0)  # Initialize with zeros

    X = points[:, 0]
    Y = points[:, 1]

    # Define pattern centers and radii
    center2 = (0.6, 0.4)
    radius2 = 0.2

    # Apply condition based on Euclidean distance
    dist2 = (X - center2[0])**2 + ((1 - Y) - center2[1])**2
    mask2 = dist2 < radius2**2

    # Assign lower value inside the region
    hetero_morph[mask2] = 0.3

    # # Second inclusion
    # center3 = (0.3, 0.7)
    # radius3 = 0.15
    # dist3 = (X - center3[0])**2 + ((1 - Y) - center3[1])**2
    # mask3 = dist3 < radius3**2
    # hetero_morph[mask3] = 1.0  

    return hetero_morph


def sigmoid(x, sharpness=50):
    return 1 / (1 + np.exp(-sharpness * x))

def generate_morph_pattern_smooth(points):
    """
    Generate smooth heterogeneity pattern based on meshio mesh points.

    Parameters:
        points (ndarray): meshio.points, shape = (num_points, 2) or (num_points, 3)

    Returns:
        hetero_morph (ndarray): Array of heterogeneity values, shape = (num_points,)
    """
    num_points = points.shape[0]
    hetero_morph = np.full((num_points,), 0.3)  # background value

    X = points[:, 0]
    Y = points[:, 1]

    # First inclusion (center2)
    center2 = (0.6, 0.4)
    radius2 = 0.2
    dist2 = np.sqrt((X - center2[0])**2 + ((1 - Y) - center2[1])**2)
    smooth2 = sigmoid(radius2 - dist2, sharpness=100)  # 0→1 inside radius

    # Combine inclusions smoothly (e.g., max or sum)
    inclusion = smooth2
    hetero_morph += (1.0 - 0.3) * inclusion  # smoothly go from 0.3 → 1.0

    return hetero_morph


def GetStressVector2D(loss_function: FiniteElementLoss,fe_mesh: Mesh, DeT: jnp.array,
                       UVWT: jnp.array, TeT: jnp.array, Te_initT:jnp.array):
    UVW = jnp.array(UVWT)
    De = jnp.array(DeT)
    Te = jnp.array(TeT)
    Te_init = jnp.array(Te_initT)
    element_type = loss_function.element_type
    element_nodes = fe_mesh.GetElementsNodes(element_type)
    XYZ = fe_mesh.GetNodesCoordinates()
    # g_points = loss_function.g_points
    # dim = loss_function.dim
    # num_gp = loss_function.num_gp
    e = loss_function.loss_settings["material_dict"]["young_modulus"]
    v = loss_function.loss_settings["material_dict"]["poisson_ratio"]
    # compute elasticity matrix
    def ComputeElement2D(xyze,ke,se,te,te_init,body_force=0):
        # Mechanics loss
        # de: conductivity
        # te: temperature
        # ke: stiffness
        # se: displacement
        # te_init: initial temperature
        # te = te.reshape(-1,1)
        se = se.reshape(-1,1)
        ke = ke.reshape(-1,1)
        te = jax.lax.stop_gradient(te.reshape(-1,1))
        te_init = jax.lax.stop_gradient(te_init.reshape(-1,1))
        @jit
        def compute_at_gauss_point(gp_point,gp_weight):
            # Mechanical part
            N_vec = loss_function.fe_element.ShapeFunctionsValues(gp_point)
            e_at_gauss = jnp.dot(N_vec, ke.squeeze())
            DN_DX = loss_function.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point)
            B_mat = loss_function.CalculateBMatrix(DN_DX)
            J = loss_function.fe_element.Jacobian(xyze,gp_point)
            temp_at_gauss = jnp.dot(N_vec,te.squeeze())
            total_strain_at_gauss = B_mat@se
            thermal_strain_vec = loss_function.thermal_loss_settings["alpha"] * (temp_at_gauss - jnp.dot(N_vec, te_init.squeeze())) * loss_function.thermal_st_vec
            elastic_strain = total_strain_at_gauss - thermal_strain_vec
            D = loss_function.CalculateDMatrix2D(e*e_at_gauss,v)
            gp_stress = D @ elastic_strain
            return gp_stress
        gp_points,gp_weights = loss_function.fe_element.GetIntegrationData()
        stress_at_gauss = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        return  stress_at_gauss

    def ComputeElementNodalStress2D(element_node):
        dofs_ids = jnp.zeros(2*4,dtype=int)
        dofs_ids = dofs_ids.at[::2].set(2*element_node)
        dofs_ids = dofs_ids.at[1::2].set(2*element_node + 1)

        uvwe = UVW[dofs_ids]
        te = Te[element_node]
        te_init = Te_init[element_node]
        de = De[element_node]
        xyze = XYZ[element_node, :]
        se = ComputeElement2D(xyze, de, uvwe, te, te_init)
        return element_node, se

    def AccumulateNodalStress(element_nodes):
        num_nodes = XYZ.shape[0]  
        stress_shape = (num_nodes, 3)  # Shape for stress accumulation array
        count_shape = (num_nodes,)  # Shape for count accumulation array

        element_indices, element_stresses = jax.vmap(ComputeElementNodalStress2D)(element_nodes)

        flat_indices = element_indices.flatten()  # Shape (num_elements * num_nodes_per_elem,)
        flat_stresses = element_stresses.reshape(-1,3)  # Shape (num_elements * num_nodes_per_elem, 3)
        flat_counts = jnp.ones(flat_indices.shape)  # To keep track of contributions

        # Scatter-add stresses to each node and count contributions
        nodal_stress = jnp.zeros(stress_shape)
        contribution_count = jnp.zeros(count_shape)

        nodal_stress = nodal_stress.at[flat_indices].add(flat_stresses)
        contribution_count = contribution_count.at[flat_indices].add(flat_counts)

        # Compute the average stress at each node
        nodal_stress = jnp.where(contribution_count[:, None] > 0, 
                                nodal_stress / contribution_count[:, None], 
                                0)       
        return nodal_stress
    return AccumulateNodalStress(element_nodes)

def GetHeatFluxVector2D(loss_function: FiniteElementLoss, fe_mesh: Mesh, 
                        conductivity: jnp.array, temperature: jnp.array):
    """
    Compute nodal heat flux vector in 2D from temperature and conductivity field.

    Parameters:
        loss_function: FiniteElementLoss object with FE settings
        fe_mesh: Mesh object
        conductivity: array of shape (num_nodes,) or (num_elements,)
        temperature: array of shape (2*num_nodes,) or (num_nodes,)

    Returns:
        nodal_heat_flux: array of shape (num_nodes, 2)
    """
    # Prepare mesh and element info
    element_type = loss_function.element_type
    element_nodes = fe_mesh.GetElementsNodes(element_type)
    XYZ = fe_mesh.GetNodesCoordinates()
    conductivity = jnp.array(conductivity) if conductivity.ndim == 1 else jnp.array(conductivity).squeeze()
    T = jnp.array(temperature) if temperature.ndim == 1 else jnp.array(temperature).squeeze()

    # Element-wise heat flux computation
    def ComputeElement2DHeatFlux(xyze, ke, te):
        te = te.reshape(-1, 1)

        @jit
        def compute_at_gauss_point(gp_point, gp_weight):
            DN_DX = loss_function.fe_element.ShapeFunctionsGlobalGradients(xyze, gp_point)
            N_vec = loss_function.fe_element.ShapeFunctionsValues(gp_point)
            conductivity_at_gauss = jnp.dot(N_vec.reshape(1,-1),ke)
            temp_grad = DN_DX.T @ te
            q = -conductivity_at_gauss * temp_grad  # Fourier's law
            return q.squeeze()

        gp_points, gp_weights = loss_function.fe_element.GetIntegrationData()
        heat_flux_at_gauss = jax.vmap(compute_at_gauss_point, in_axes=(0, 0))(gp_points, gp_weights)
        return heat_flux_at_gauss

    def ComputeElementNodalHeatFlux2D(element_node):
        te = T[element_node]
        ke = conductivity[element_node]
        xyze = XYZ[element_node, :]
        qe = ComputeElement2DHeatFlux(xyze, ke, te)
        return element_node, qe

    def AccumulateNodalHeatFlux(element_nodes):
        num_nodes = XYZ.shape[0]
        heat_shape = (num_nodes, 2)
        count_shape = (num_nodes,)

        element_indices, element_fluxes = jax.vmap(ComputeElementNodalHeatFlux2D)(element_nodes)

        flat_indices = element_indices.flatten()
        flat_flux = element_fluxes.reshape(-1, 2)
        flat_counts = jnp.ones(flat_indices.shape)

        nodal_flux = jnp.zeros(heat_shape)
        contribution_count = jnp.zeros(count_shape)

        nodal_flux = nodal_flux.at[flat_indices].add(flat_flux)
        contribution_count = contribution_count.at[flat_indices].add(flat_counts)

        nodal_flux = jnp.where(contribution_count[:, None] > 0,
                               nodal_flux / contribution_count[:, None], 0)
        return nodal_flux

    return AccumulateNodalHeatFlux(element_nodes)

def GetStressVector3D(loss_function: FiniteElementLoss, fe_mesh: Mesh, DeT: jnp.array,
                      UVWT: jnp.array, TeT: jnp.array, Te_initT: jnp.array):
    UVW = jnp.array(UVWT)
    De = jnp.array(DeT)
    Te = jnp.array(TeT)
    Te_init = jnp.array(Te_initT)
    element_type = loss_function.element_type
    element_nodes = fe_mesh.GetElementsNodes(element_type)
    XYZ = fe_mesh.GetNodesCoordinates()
    e = loss_function.loss_settings["material_dict"]["young_modulus"]
    v = loss_function.loss_settings["material_dict"]["poisson_ratio"]

    def ComputeElement3D(xyze, ke, se, te, te_init, body_force=0):
        se = se.reshape(-1, 1)
        ke = ke.reshape(-1, 1)
        te = jax.lax.stop_gradient(te.reshape(-1, 1))
        te_init = jax.lax.stop_gradient(te_init.reshape(-1, 1))

        @jit
        def compute_at_gauss_point(gp_point, gp_weight):
            N_vec = loss_function.fe_element.ShapeFunctionsValues(gp_point)
            e_at_gauss = jnp.dot(N_vec, ke.squeeze())
            DN_DX = loss_function.fe_element.ShapeFunctionsGlobalGradients(xyze, gp_point)
            B_mat = loss_function.CalculateBMatrix3D(DN_DX)  # <- 3D B matrix
            J = loss_function.fe_element.Jacobian(xyze, gp_point)
            temp_at_gauss = jnp.dot(N_vec, te.squeeze())
            total_strain_at_gauss = B_mat @ se
            thermal_strain_vec = loss_function.thermal_loss_settings["alpha"] * (
                temp_at_gauss - jnp.dot(N_vec, te_init.squeeze())) * loss_function.thermal_st_vec
            elastic_strain = total_strain_at_gauss - thermal_strain_vec
            D = loss_function.CalculateDMatrix3D(e * e_at_gauss, v)  # <- 3D D matrix
            gp_stress = D @ elastic_strain
            return gp_stress

        gp_points, gp_weights = loss_function.fe_element.GetIntegrationData()
        stress_at_gauss = jax.vmap(compute_at_gauss_point, in_axes=(0, 0))(gp_points, gp_weights)
        return stress_at_gauss

    def ComputeElementNodalStress3D(element_node):
        dofs_ids = jnp.zeros(3 * 8, dtype=int)
        dofs_ids = dofs_ids.at[::3].set(3 * element_node)
        dofs_ids = dofs_ids.at[1::3].set(3 * element_node + 1)
        dofs_ids = dofs_ids.at[2::3].set(3 * element_node + 2)

        uvwe = UVW[dofs_ids]
        te = Te[element_node]
        te_init = Te_init[element_node]
        de = De[element_node]
        xyze = XYZ[element_node, :]
        se = ComputeElement3D(xyze, de, uvwe, te, te_init)
        return element_node, se

    def AccumulateNodalStress(element_nodes):
        num_nodes = XYZ.shape[0]
        stress_shape = (num_nodes, 6)  # σxx, σyy, σzz, σxy, σyz, σzx
        count_shape = (num_nodes,)

        element_indices, element_stresses = jax.vmap(ComputeElementNodalStress3D)(element_nodes)

        flat_indices = element_indices.flatten()
        flat_stresses = element_stresses.reshape(-1, 6)
        flat_counts = jnp.ones(flat_indices.shape)

        nodal_stress = jnp.zeros(stress_shape)
        contribution_count = jnp.zeros(count_shape)

        nodal_stress = nodal_stress.at[flat_indices].add(flat_stresses)
        contribution_count = contribution_count.at[flat_indices].add(flat_counts)

        nodal_stress = jnp.where(contribution_count[:, None] > 0,
                                 nodal_stress / contribution_count[:, None],
                                 0)
        return nodal_stress

    return AccumulateNodalStress(element_nodes)

def GetHeatFluxVector3D(loss_function: FiniteElementLoss, fe_mesh: Mesh,
                        conductivity: jnp.array, temperature: jnp.array):
    """
    Compute nodal heat flux vector in 3D from temperature and conductivity field.

    Parameters:
        loss_function: FiniteElementLoss object with FE settings
        fe_mesh: Mesh object
        conductivity: array of shape (num_nodes,) or (num_elements,)
        temperature: array of shape (num_nodes,)

    Returns:
        nodal_heat_flux: array of shape (num_nodes, 3)
    """
    element_type = loss_function.element_type
    element_nodes = fe_mesh.GetElementsNodes(element_type)
    XYZ = fe_mesh.GetNodesCoordinates()
    conductivity = jnp.array(conductivity).squeeze()
    T = jnp.array(temperature).squeeze()

    # Element-wise heat flux computation
    def ComputeElement3DHeatFlux(xyze, ke, te):
        te = te.reshape(-1, 1)

        @jit
        def compute_at_gauss_point(gp_point, gp_weight):
            DN_DX = loss_function.fe_element.ShapeFunctionsGlobalGradients(xyze, gp_point)  # shape (n_nodes, 3)
            N_vec = loss_function.fe_element.ShapeFunctionsValues(gp_point)  # shape (n_nodes,)
            conductivity_at_gauss = jnp.dot(N_vec.reshape(1, -1), ke)  # shape (1,)

            temp_grad = DN_DX.T @ te  # shape (3, 1)
            q = -conductivity_at_gauss * temp_grad  # Fourier’s law
            return q.squeeze()  # shape (3,)

        gp_points, gp_weights = loss_function.fe_element.GetIntegrationData()
        heat_flux_at_gauss = jax.vmap(compute_at_gauss_point, in_axes=(0, 0))(gp_points, gp_weights)
        return heat_flux_at_gauss  # shape (num_gp, 3)

    def ComputeElementNodalHeatFlux3D(element_node):
        te = T[element_node]
        ke = conductivity[element_node]
        xyze = XYZ[element_node, :]
        qe = ComputeElement3DHeatFlux(xyze, ke, te)
        return element_node, qe

    def AccumulateNodalHeatFlux(element_nodes):
        num_nodes = XYZ.shape[0]
        heat_shape = (num_nodes, 3)
        count_shape = (num_nodes,)

        element_indices, element_fluxes = jax.vmap(ComputeElementNodalHeatFlux3D)(element_nodes)

        flat_indices = element_indices.flatten()
        flat_flux = element_fluxes.reshape(-1, 3)
        flat_counts = jnp.ones(flat_indices.shape)

        nodal_flux = jnp.zeros(heat_shape)
        contribution_count = jnp.zeros(count_shape)

        nodal_flux = nodal_flux.at[flat_indices].add(flat_flux)
        contribution_count = contribution_count.at[flat_indices].add(flat_counts)

        nodal_flux = jnp.where(contribution_count[:, None] > 0,
                               nodal_flux / contribution_count[:, None], 0)
        return nodal_flux

    return AccumulateNodalHeatFlux(element_nodes)


def plot_mesh_vec_data_hetero(L, vectors_list, subplot_titles=None, fig_title=None, cmap='viridis',
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
        im = axs[i].imshow(squared_mesh_vec_data.reshape(N, N), cmap=cmap, extent=[0, L, 0, L],vmin =0, vmax = 1)

        if subplot_titles is not None:
            axs[i].set_title(subplot_titles[i])
        else:
            axs[i].set_title(f'Plot {i+1}')

        if colour_bar:
            cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=20)
            cbar.set_ticks(np.linspace(0, 1, 3))


        if X_axis_name is not None:
            axs[i].set_xlabel(X_axis_name)

        if Y_axis_name is not None:
            axs[i].set_ylabel(Y_axis_name)
        
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    if fig_title is not None:
        plt.suptitle(fig_title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if show:
        plt.show(block=block_bool)

    if file_name is not None:
        plt.savefig(file_name)


def create_3D_box_mesh_structured(Nx, Ny, Nz, Lx, Ly, Lz):
    """
    Nx, Ny, Nz : 各方向の節点数（= FNO の解像度）
    Lx, Ly, Lz : 領域長さ [0, Lx] × [0, Ly] × [0, Lz]
    """

    # --- 1. 節点座標を FNO と同じ順序で作る (i,j,k) = (x,y,z) ---
    xs = jnp.linspace(0.0, Lx, Nx)  # i = 0..Nx-1
    ys = jnp.linspace(0.0, Ly, Ny)  # j = 0..Ny-1
    zs = jnp.linspace(0.0, Lz, Nz)  # k = 0..Nz-1

    X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")  # shape: (Nx, Ny, Nz)

    # nodes_coordinates: (num_nodes, 3)
    # flatten は C-order なので idx(i,j,k) = i*Ny*Nz + j*Nz + k になる
    nodes_coordinates = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    num_nodes = nodes_coordinates.shape[0]

    node_ids = jnp.arange(num_nodes, dtype=jnp.int32)

    # --- 2. 要素 (hexahedron) の connectivity を作る ---
    # ノード番号の対応ルール（FNO と一致）:
    # idx(i,j,k) = i*Ny*Nz + j*Nz + k
    def node_idx(i, j, k):
        return i * Ny * Nz + j * Nz + k

    hex_elems = []
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            for k in range(Nz - 1):
                # 8つの頂点
                n000 = node_idx(i,   j,   k)
                n100 = node_idx(i+1, j,   k)
                n110 = node_idx(i+1, j+1, k)
                n010 = node_idx(i,   j+1, k)
                n001 = node_idx(i,   j,   k+1)
                n101 = node_idx(i+1, j,   k+1)
                n111 = node_idx(i+1, j+1, k+1)
                n011 = node_idx(i,   j+1, k+1)

                # meshio の "hexahedron" でよく使う順序
                hex_elems.append([n000, n100, n110, n010,
                                  n001, n101, n111, n011])

    hex_elems = jnp.array(hex_elems, dtype=jnp.int32)
    elements_nodes = {"hexahedron": hex_elems}

    # --- 3. 境界節点セット (left/right/top/bottom/front/back) ---
    x = nodes_coordinates[:, 0]
    y = nodes_coordinates[:, 1]
    z = nodes_coordinates[:, 2]

    atol = 1e-8
    left_boundary_node_ids   = jnp.where(jnp.isclose(x, 0.0, atol=atol),  size=None)[0]
    right_boundary_node_ids  = jnp.where(jnp.isclose(x, Lx,  atol=atol), size=None)[0]
    bottom_boundary_node_ids = jnp.where(jnp.isclose(y, 0.0, atol=atol),  size=None)[0]
    top_boundary_node_ids    = jnp.where(jnp.isclose(y, Ly,  atol=atol), size=None)[0]
    front_boundary_node_ids  = jnp.where(jnp.isclose(z, 0.0, atol=atol),  size=None)[0]
    back_boundary_node_ids   = jnp.where(jnp.isclose(z, Lz,  atol=atol), size=None)[0]

    node_sets = {
        "left":   left_boundary_node_ids,
        "right":  right_boundary_node_ids,
        "bottom": bottom_boundary_node_ids,
        "top":    top_boundary_node_ids,
        "front":  front_boundary_node_ids,
        "back":   back_boundary_node_ids,
    }

    # --- 4. fe_mesh オブジェクトに詰める ---
    fe_mesh = Mesh("box_io", "box.")  # あなたの既存クラスを利用

    fe_mesh.node_ids = node_ids
    fe_mesh.nodes_coordinates = nodes_coordinates
    fe_mesh.elements_nodes = elements_nodes
    fe_mesh.node_sets = node_sets

    # meshio 用には numpy 配列に変換
    fe_mesh.mesh_io = meshio.Mesh(
        points=np.asarray(nodes_coordinates),
        cells={"hexahedron": np.asarray(hex_elems, dtype=np.int64)}
    )

    fe_mesh.is_initialized = True
    return fe_mesh
