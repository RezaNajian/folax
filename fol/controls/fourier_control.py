"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial
from jax.nn import sigmoid
from fol.mesh_input_output.mesh import Mesh
from fol.tools.decoration_functions import *

class FourierControl(Control):
    """
    Fourier-based parametric control defined over a finite element mesh.

    This control maps a compact set of design variables (Fourier coefficients)
    to a spatially varying scalar field evaluated at mesh nodes. The nodal field
    is represented as a truncated cosine Fourier series in the x-, y-, and
    z-directions, then smoothly projected to user-defined bounds using a
    sigmoid-based scaling.

    Formulation
    -----------
    Let the mesh contain nodes with coordinates ``(x_i, y_i, z_i)``. The user
    provides one-dimensional frequency arrays ``x_freqs``, ``y_freqs``, and
    ``z_freqs``. A Cartesian product of these arrays defines a collection of
    frequency triplets ``(f_x, f_y, f_z)``, constructed by calling
    ``meshgrid(x_freqs, y_freqs, z_freqs, indexing="ij")`` and flattening the
    result. Each triplet corresponds to exactly one cosine basis function.

    Given a control vector ``a`` of length
    ``len(x_freqs) * len(y_freqs) * len(z_freqs) + 1``, the unbounded nodal
    field value at node ``i`` is computed as:

    - The first entry ``a[0]`` defines a constant (mean) mode and contributes
      ``a[0] / 2`` to every node.
    - Each remaining entry ``a[j]`` multiplies one cosine basis function
      associated with a frequency triplet ``(f_x, f_y, f_z)`` and contributes

      ``a[j] * cos(pi * f_x * x_i) * cos(pi * f_y * y_i) * cos(pi * f_z * z_i)``

      to the nodal value.

    The unbounded nodal field ``K_i`` is obtained by summing the constant
    contribution and all cosine-mode contributions.

    The nodal field is then mapped to user-defined bounds via a smooth sigmoid
    projection:

    ``u_i = (max - min) * sigmoid(beta * (K_i - 0.5)) + min``

    This projection enforces ``u_i`` in ``[min, max]`` while preserving
    differentiability for gradient-based optimization.

    Args:
        control_name (str):
            Name identifier for the control instance.
        control_settings (dict):
            Dictionary defining the Fourier parameterization. Required keys:
            ``"beta"``, ``"x_freqs"``, ``"y_freqs"``, ``"z_freqs"``.
            Optional keys: ``"min"`` and ``"max"`` defining output bounds.
        fe_mesh (Mesh):
            Finite element mesh providing nodal coordinates via
            ``GetNodesX()``, ``GetNodesY()``, and ``GetNodesZ()``.

    Attributes:
        settings (dict):
            Control configuration dictionary provided at construction.
        fe_mesh (Mesh):
            Finite element mesh used to evaluate basis functions.
        min (float):
            Lower bound of the controlled field (default: ``1e-6`` if not set).
        max (float):
            Upper bound of the controlled field (default: ``1.0`` if not set).
        beta (float):
            Sigmoid sharpness parameter controlling projection steepness.
        x_freqs (jax.numpy.ndarray):
            Frequency values in the x-direction.
        y_freqs (jax.numpy.ndarray):
            Frequency values in the y-direction.
        z_freqs (jax.numpy.ndarray):
            Frequency values in the z-direction.
        frquencies_vec (jax.numpy.ndarray):
            Flattened array of frequency triplets with shape ``(n_modes, 3)``,
            where ``n_modes = len(x_freqs) * len(y_freqs) * len(z_freqs)``.
        num_control_vars (int):
            Number of control variables (Fourier coefficients), equal to
            ``n_modes + 1``.
        num_controlled_vars (int):
            Number of controlled variables, equal to the number of mesh nodes.
        initialized (bool):
            Whether :meth:`Initialize` has been called.

    Notes:
        The first coefficient ``a[0]`` is treated as a mean mode and scaled by
        ``1/2`` to match the implemented definition:

        ``K = a[0]/2 + sum_j a[j] * cos_x * cos_y * cos_z``

        where each ``a[j]`` for ``j >= 1`` corresponds to one frequency triplet.

    """
    def __init__(self,control_name: str,control_settings: dict, fe_mesh: Mesh):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh
        self.scale_min = 0.0
        self.scale_max = 1.0

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize the Fourier basis and control dimensions.

        This method prepares the frequency grid, determines the number of
        control variables, and caches frequency combinations for efficient
        evaluation. Initialization is performed once unless ``reinitialize``
        is set to ``True``.

        Args:
            reinitialize (bool, optional):
                If ``True``, forces reinitialization even if already initialized.
                Default is ``False``.

        Returns:
            None

        Raises:
            KeyError:
                If required frequency arrays or parameters are missing from
                ``control_settings``.
        """
        if self.initialized and not reinitialize:
            return

        if "min" in self.settings.keys():
            self.min = self.settings["min"]
        else:
            self.min = 1e-6
        if "max" in self.settings.keys():
            self.max = self.settings["max"]
        else:
            self.max = 1.0
        self.beta = self.settings["beta"]
        self.x_freqs = self.settings["x_freqs"]
        self.y_freqs = self.settings["y_freqs"]
        self.z_freqs = self.settings["z_freqs"]
        self.num_x_freqs = self.x_freqs.shape[-1]
        self.num_y_freqs = self.y_freqs.shape[-1]
        self.num_z_freqs = self.z_freqs.shape[-1]
        self.num_control_vars = self.num_x_freqs * self.num_y_freqs * self.num_z_freqs + 1
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
        mesh_x, mesh_y, mesh_z = jnp.meshgrid(self.x_freqs, self.y_freqs, self.z_freqs, indexing='ij')
        self.frquencies_vec = jnp.vstack([mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()]).T
        self.initialized = True

    def ComputeControlledVariables(self,variable_vector:jnp.array):
        """
        Evaluate the controlled field at mesh nodes from Fourier coefficients.

        This method maps the input control vector to a nodal scalar field by
        evaluating a truncated cosine Fourier series at each mesh node. The
        resulting field is passed through a sigmoid transformation to enforce
        lower and upper bounds.

        The first entry of ``variable_vector`` corresponds to the constant
        (mean) mode. The remaining entries correspond to cosine modes defined
        by the Cartesian product of ``x_freqs``, ``y_freqs``, and ``z_freqs``.

        Args:
            variable_vector (jax.numpy.ndarray):
                One-dimensional array of Fourier coefficients. Its length must
                equal ``num_control_vars``.

        Returns:
            jax.numpy.ndarray:
                Array of controlled values evaluated at mesh nodes, with shape
                ``(num_nodes,)``.

        Raises:
            ValueError:
                If ``Initialize`` has not been called or if the input vector has
                an incompatible length.
        """
        variable_vector *= (self.scale_max-self.scale_min)
        variable_vector += self.scale_min
        def evaluate_at_frequencies(freqs,coeff):
            cos_x = jnp.cos(freqs[0] * jnp.pi * self.fe_mesh.GetNodesX())
            cos_y = jnp.cos(freqs[1] * jnp.pi * self.fe_mesh.GetNodesY())
            cos_z = jnp.cos(freqs[2] * jnp.pi * self.fe_mesh.GetNodesZ())
            return coeff * cos_x * cos_y * cos_z
        K = (variable_vector[0]/2.0) + jnp.sum(vmap(evaluate_at_frequencies, in_axes=(0, 0))
                                               (self.frquencies_vec,variable_vector[1:]),axis=0)
        return (self.max-self.min) * sigmoid(self.beta*(K-0.5)) + self.min

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass