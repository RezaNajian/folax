import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.fe_loss import FiniteElementLoss
import jax
from jax import jit
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def reshape_T_U_to_nodewise(FE_TUV, num_nodes):
    T = FE_TUV[0:num_nodes]
    UV = FE_TUV[num_nodes:]
    UV = UV.reshape((num_nodes, 2))  # [[Ux0, Uy0], ..., [UxN, UyN]]
    return jnp.concatenate([T[:, None], UV], axis=1)

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

def plot_mesh_vec_data_thermal_2Dthermomech(
    L, vectors_list, subplot_titles=None, fig_title=None, cmap='viridis',
    block_bool=False, colour_bar=True, colour_bar_name=None,
    X_axis_name=None, Y_axis_name=None, show=False, file_name=None):

    num_vectors = int(np.ceil(np.sqrt(len(vectors_list)))) 
    if num_vectors < 1 or num_vectors > 8:
        raise ValueError("vectors_list must contain between 1 and 64 elements.")

    # if subplot_titles is not None and len(subplot_titles) != len(vectors_list):
    #     raise ValueError("subplot_titles must have the same number of elements as vectors_list if provided.")

    fig, axs = plt.subplots(num_vectors, num_vectors, figsize=(5*num_vectors,4*num_vectors), squeeze=False)
    axs = axs.flatten()

    vmin = [vec.min() for vec in vectors_list]
    vmax = [vec.max() for vec in vectors_list]

    for i, squared_mesh_vec_data in enumerate(vectors_list):
        N = int((squared_mesh_vec_data.size)**0.5)
        im = axs[i].imshow(squared_mesh_vec_data.reshape(N, N), cmap=cmap,
                           extent=[0, L, 0, L], vmin=vmin[i], vmax=vmax[i])

        if subplot_titles is not None and i<3:
            axs[i].set_title(subplot_titles[i],fontsize=13, pad=6)

        if colour_bar:
            cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=16)
            cbar.set_ticks(np.linspace(vmin[i], vmax[i], 3))
            if colour_bar_name:
                cbar.set_label(colour_bar_name, fontsize=18)

        axs[i].set_xlabel(X_axis_name or "")
        axs[i].set_ylabel(Y_axis_name or "")
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    for j in range(len(vectors_list), len(axs)):
        axs[j].axis('off')

    if fig_title:
        plt.suptitle(fig_title, fontsize=22)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.12, wspace=0.08, hspace=0.08)
    if show:
        plt.show(block=block_bool)

    if file_name:
        plt.savefig(file_name)

def plot_mesh_vec_data_thermal_2Dheatflux(
    L, vectors6,
    col_titles=('iFOL', 'FEM', 'Absolute error'),  # 上の列タイトル
    cmap='inferno',
    share_scale_per_col=False,                 # Trueなら列ごとにvmin/vmax共有
    fig_size=(14, 8),
    file_name=None, show=True
):
    assert len(vectors6) == 6 
    fig, axs = plt.subplots(2, 3, figsize=fig_size, constrained_layout=True, squeeze=False)

    # vmin/vmax の決定（列ごとに共有 or 独立）
    if share_scale_per_col:
        col_vmins, col_vmaxs = [], []
        for j in range(3):
            col_data = [vectors6[i*3+j] for i in range(2)]
            col_vmins.append(min([v.min() for v in col_data]))
            col_vmaxs.append(max([v.max() for v in col_data]))

    for i in range(2):          # row
        for j in range(3):      # col
            k = i*3 + j
            data = vectors6[k]
            N = int(np.sqrt(data.size))
            ax = axs[i, j]
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([]); ax.set_yticks([])

            if share_scale_per_col:
                vmin, vmax = col_vmins[j], col_vmaxs[j]
            else:
                vmin, vmax = data.min(), data.max()

            im = ax.imshow(data.reshape(N, N), cmap=cmap, extent=[0, L, 0, L],
                           vmin=vmin, vmax=vmax)

            # 軸内カラーバー（はみ出さない）
            # cax = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=0.5)
            cbar = fig.colorbar(im)
            cbar.ax.tick_params(labelsize=16)

            # 一番上の行に列タイトル
            if i == 0 and col_titles:
                ax.set_title(col_titles[j], fontsize=13, pad=6)

            # # 一番左の列に行タイトル
            # if j == 0 and row_titles:
            #     ax.text(-0.08, 0.5, row_titles[i], va='center', ha='right',
            #             rotation=90, transform=ax.transAxes, fontsize=13)
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.12, wspace=0.08, hspace=0.08)
    # 保存または表示
    if file_name:
        plt.savefig(file_name, dpi=300)
    if show:
        plt.show()

    return fig, axs


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