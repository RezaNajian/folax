import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.fe_loss import FiniteElementLoss
import jax
from jax import jit


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
