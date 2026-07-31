"""Utilities used by the thermo-mechanical casting examples."""

import jax.numpy as jnp
import numpy as np

from fol.loss_functions.fe_loss import FiniteElementLoss
from fol.mesh_input_output.mesh import Mesh


def create_uniform_dist_bc_samples(displ_control, numberof_sample, low=0.0, high=1.0):
    """Generate uniformly distributed boundary-control samples."""
    num_control_vars = displ_control.num_control_vars
    bc_matrix = np.random.uniform(
        low=low, high=high, size=(numberof_sample, num_control_vars)
    )
    bc_nodal_value_matrix = displ_control.ComputeBatchControlledVariables(bc_matrix)
    return bc_matrix, bc_nodal_value_matrix


def reshape_T_U_to_nodewise3D(FE_TUV, num_nodes):
    """Return solver DOFs as rows of ``[T, Ux, Uy, Uz]``.

    The current finite-element assembly stores all four fields interleaved by
    node.  Consequently this operation is a direct reshape; the former blocked
    ``[all T, all U]`` conversion would scramble current solver output.
    """
    values = jnp.asarray(FE_TUV)
    expected_size = 4 * num_nodes
    if values.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} thermo-mechanical DOFs for {num_nodes} "
            f"nodes, received {values.size}."
        )
    return values.reshape(num_nodes, 4)


def generate_morph_pattern(points):
    """Generate a sharp circular heterogeneity pattern."""
    points = np.asarray(points)
    hetero_morph = np.full((points.shape[0],), 0.3)
    distance_squared = (
        (points[:, 0] - 0.6) ** 2 + ((1.0 - points[:, 1]) - 0.6) ** 2
    )
    hetero_morph[distance_squared < 0.2**2] = 1.0
    return hetero_morph


def generate_morph_pattern2(points):
    """Generate the complementary sharp circular heterogeneity pattern."""
    points = np.asarray(points)
    hetero_morph = np.full((points.shape[0],), 1.0)
    distance_squared = (
        (points[:, 0] - 0.6) ** 2 + ((1.0 - points[:, 1]) - 0.4) ** 2
    )
    hetero_morph[distance_squared < 0.2**2] = 0.3
    return hetero_morph


def sigmoid(x, sharpness=50):
    """Evaluate a logistic transition with configurable sharpness."""
    return 1.0 / (1.0 + np.exp(-sharpness * x))


def generate_morph_pattern_smooth(points):
    """Generate a smoothly varying circular heterogeneity pattern."""
    points = np.asarray(points)
    distance = np.sqrt(
        (points[:, 0] - 0.6) ** 2 + ((1.0 - points[:, 1]) - 0.4) ** 2
    )
    return 0.3 + 0.7 * sigmoid(0.2 - distance, sharpness=100)


def _validate_postprocessing_inputs(
    loss_function: FiniteElementLoss,
    fe_mesh: Mesh,
    conductivity,
    temperature,
    displacement=None,
):
    """Normalize and validate nodal post-processing fields."""
    num_nodes = fe_mesh.GetNumberOfNodes()
    if loss_function.fe_mesh.GetNumberOfNodes() != num_nodes:
        raise ValueError("The loss function and mesh have different node counts.")

    conductivity = jnp.asarray(conductivity).reshape(-1)
    temperature = jnp.asarray(temperature).reshape(-1)
    if conductivity.size != num_nodes:
        raise ValueError(
            f"Expected {num_nodes} nodal conductivity values, received "
            f"{conductivity.size}."
        )
    if temperature.size != num_nodes:
        raise ValueError(
            f"Expected {num_nodes} nodal temperatures, received {temperature.size}."
        )

    if displacement is None:
        return conductivity, temperature

    displacement = jnp.asarray(displacement).reshape(-1)
    expected_displacements = num_nodes * loss_function.dim
    if displacement.size != expected_displacements:
        raise ValueError(
            f"Expected {expected_displacements} displacement DOFs, received "
            f"{displacement.size}."
        )
    return conductivity, temperature, displacement.reshape(num_nodes, loss_function.dim)


def _get_stress_vector(
    loss_function: FiniteElementLoss,
    fe_mesh: Mesh,
    conductivity,
    displacement,
    temperature,
    initial_temperature,
    expected_dim: int,
):
    """Adapt separate nodal fields to ``ThermoMechanicsLoss.ComputeStress``."""
    if loss_function.dim != expected_dim:
        raise ValueError(
            f"Expected a {expected_dim}D loss function, got {loss_function.dim}D."
        )

    conductivity, temperature, displacement = _validate_postprocessing_inputs(
        loss_function, fe_mesh, conductivity, temperature, displacement
    )
    initial_temperature = jnp.asarray(initial_temperature).reshape(-1)
    if initial_temperature.size != fe_mesh.GetNumberOfNodes():
        raise ValueError(
            "Initial temperature must contain one value for every mesh node."
        )

    # ComputeStress obtains the reference temperature from the initialized loss
    # object.  Keep the legacy argument in this adapter for call compatibility.
    nodal_tuvw = jnp.concatenate(
        (temperature[:, None], displacement), axis=1
    ).reshape(-1)
    return loss_function.ComputeStress(conductivity, nodal_tuvw)


def GetStressVector2D(
    loss_function: FiniteElementLoss,
    fe_mesh: Mesh,
    DeT,
    UVWT,
    TeT,
    Te_initT,
):
    """Compute nodal 2D stress in Voigt order using the current loss API."""
    return _get_stress_vector(
        loss_function, fe_mesh, DeT, UVWT, TeT, Te_initT, expected_dim=2
    )


def GetHeatFluxVector2D(
    loss_function: FiniteElementLoss, fe_mesh: Mesh, conductivity, temperature
):
    """Compute the nodal 2D heat-flux vector using the current loss API."""
    if loss_function.dim != 2:
        raise ValueError(f"Expected a 2D loss function, got {loss_function.dim}D.")
    conductivity, temperature = _validate_postprocessing_inputs(
        loss_function, fe_mesh, conductivity, temperature
    )
    return loss_function.ComputeHeatFlux(conductivity, temperature)


def GetStressVector3D(
    loss_function: FiniteElementLoss,
    fe_mesh: Mesh,
    DeT,
    UVWT,
    TeT,
    Te_initT,
):
    """Compute nodal 3D stress in Voigt order using the current loss API."""
    return _get_stress_vector(
        loss_function, fe_mesh, DeT, UVWT, TeT, Te_initT, expected_dim=3
    )


def GetHeatFluxVector3D(
    loss_function: FiniteElementLoss, fe_mesh: Mesh, conductivity, temperature
):
    """Compute the nodal 3D heat-flux vector using the current loss API."""
    if loss_function.dim != 3:
        raise ValueError(f"Expected a 3D loss function, got {loss_function.dim}D.")
    conductivity, temperature = _validate_postprocessing_inputs(
        loss_function, fe_mesh, conductivity, temperature
    )
    return loss_function.ComputeHeatFlux(conductivity, temperature)
