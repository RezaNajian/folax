"""
 Authors: (you) based on VoronoiControl3D + TPMS idea
 Date: Feb, 2026
 License: FOL/License.txt

VoronoiTPMSControl3D:
- Voronoi seeds partition the domain.
- Each seed has a region "high modulus" chosen from E_values (strong macro contrast).
- Inside each Voronoi region, TPMS level-set creates a two-phase microstructure
  (strong micro contrast): {min, region_E} (binary) or smooth blend (sigmoid).

Settings (same as VoronoiControl3D + optional TPMS knobs):
  required:
    number_of_seeds: int
    E_values: tuple/list of modulus levels (e.g., (0.05, 1.0) or [0.1,0.5,1.0])

  optional:
    min: float (soft phase), default 1e-3
    L: float, default 1.0
    fc_min: int, default 1
    fc_max: int, default 3 (inclusive)
    tau_min: float, default -0.25
    tau_max: float, default +0.25
    binary: bool, default True
    beta: float, default 20.0  (used only if binary=False)
    solid_is_high: bool, default True
"""

from .control import Control
import jax.numpy as jnp
from jax import jit
from functools import partial
from jax.nn import sigmoid
from fol.mesh_input_output.mesh import Mesh
from fol.tools.decoration_functions import *

class VoronoiTPMSControl3D(Control):
    def __init__(self, control_name: str, control_settings, fe_mesh: Mesh):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh

    @print_with_timestamp_and_execution_time
    def Initialize(self, reinitialize=False) -> None:
        if self.initialized and not reinitialize:
            return

        self.number_of_seeds = int(self.settings["number_of_seeds"])

        if not isinstance(self.settings["E_values"], (tuple, list)):
            raise ValueError("'E_values' should be either tuple or list")
        if len(self.settings["E_values"]) < 2:
            raise ValueError("'E_values' must have at least 2 values for strong contrast")

        self.E_values = jnp.array(self.settings["E_values"], dtype=jnp.float32)
        self.num_E = int(self.E_values.shape[0])

        # strong contrast: soft phase baseline (keep >0 for stability later in Neo-Hooke)
        self.min = float(self.settings.get("min", 1e-3))
        self.L   = float(self.settings.get("L", 1.0))

        self.fc_min  = int(self.settings.get("fc_min", 1))
        self.fc_max  = int(self.settings.get("fc_max", 3))   # inclusive
        self.tau_min = float(self.settings.get("tau_min", -0.25))
        self.tau_max = float(self.settings.get("tau_max",  0.25))

        self.binary = bool(self.settings.get("binary", True))
        self.beta   = float(self.settings.get("beta", 20.0))
        self.solid_is_high = bool(self.settings.get("solid_is_high", True))

        # 5 TPMS types fixed:
        self.n_types = 5  # 0..4

        # Control vector layout PER SEED (10 vars):
        #   x,y,z,  k_sel, type_sel, fc_sel, sx,sy,sz, tau_sel
        self.num_control_vars = self.number_of_seeds * 10
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()

        self.initialized = True

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self, variable_vector: jnp.array):
        n = self.number_of_seeds
        v = jnp.clip(variable_vector, 0.0, 1.0)

        # unpack
        i = 0
        x_seed = v[i:i+n] * self.L; i += n
        y_seed = v[i:i+n] * self.L; i += n
        z_seed = v[i:i+n] * self.L; i += n

        k_sel    = v[i:i+n]; i += n
        type_sel = v[i:i+n]; i += n
        fc_sel   = v[i:i+n]; i += n
        sx_sel   = v[i:i+n]; i += n
        sy_sel   = v[i:i+n]; i += n
        sz_sel   = v[i:i+n]; i += n
        tau_sel  = v[i:i+n]; i += n

        # mesh points
        X = self.fe_mesh.GetNodesX()
        Y = self.fe_mesh.GetNodesY()
        Z = self.fe_mesh.GetNodesZ()
        grid = jnp.stack([X, Y, Z], axis=1)                 # (n_nodes,3)
        seeds = jnp.stack([x_seed, y_seed, z_seed], axis=1) # (n_seeds,3)

        # nearest seed (Voronoi)
        d2 = jnp.sum((grid[:, None, :] - seeds[None, :, :])**2, axis=2)  # (n_nodes,n_seeds)
        nearest = jnp.argmin(d2, axis=1)                                 # (n_nodes,)

        # region modulus: choose from E_values (discrete)
        k_idx = jnp.floor(k_sel * self.num_E).astype(jnp.int32)
        k_idx = jnp.clip(k_idx, 0, self.num_E - 1)
        E_seed = self.E_values[k_idx]     # (n_seeds,)
        E_node = E_seed[nearest]          # (n_nodes,)

        # TPMS type per node
        type_id = jnp.floor(type_sel * self.n_types).astype(jnp.int32)
        type_id = jnp.clip(type_id, 0, self.n_types - 1)
        type_node = type_id[nearest]      # (n_nodes,)

        # fc integer per node
        span = (self.fc_max - self.fc_min + 1)
        fc_int = self.fc_min + jnp.floor(fc_sel * span).astype(jnp.int32)
        fc_int = jnp.clip(fc_int, self.fc_min, self.fc_max).astype(jnp.float32)
        fc_node = fc_int[nearest]

        # shifts per node in [0,L]
        sx = (sx_sel * self.L)[nearest]
        sy = (sy_sel * self.L)[nearest]
        sz = (sz_sel * self.L)[nearest]

        # tau per node
        tau = (self.tau_min + (self.tau_max - self.tau_min) * tau_sel)[nearest]

        # periodic shift in physical space
        Xs = jnp.mod(X + sx, self.L)
        Ys = jnp.mod(Y + sy, self.L)
        Zs = jnp.mod(Z + sz, self.L)

        two_pi = 2.0 * jnp.pi
        XX = two_pi * fc_node * (Xs / self.L)
        YY = two_pi * fc_node * (Ys / self.L)
        ZZ = two_pi * fc_node * (Zs / self.L)

        # 5 TPMS level-sets
        phi0 = jnp.sin(XX) * jnp.cos(YY) + jnp.sin(YY) * jnp.cos(ZZ) + jnp.sin(ZZ) * jnp.cos(XX)  # Gyroid
        phi1 = jnp.cos(XX) + jnp.cos(YY) + jnp.cos(ZZ)                                             # Schwarz_P
        phi2 = (
            jnp.sin(XX) * jnp.sin(YY) * jnp.sin(ZZ)
            + jnp.sin(XX) * jnp.cos(YY) * jnp.cos(ZZ)
            + jnp.cos(XX) * jnp.sin(YY) * jnp.cos(ZZ)
            + jnp.cos(XX) * jnp.cos(YY) * jnp.sin(ZZ)
        )                                                                                           # Schwarz_D
        phi3 = 3.0 * (jnp.cos(XX) + jnp.cos(YY) + jnp.cos(ZZ)) + 4.0 * jnp.cos(XX) * jnp.cos(YY) * jnp.cos(ZZ)  # Neovius
        phi4 = jnp.cos(XX) * jnp.sin(YY) + jnp.cos(YY) * jnp.sin(ZZ) + jnp.cos(ZZ) * jnp.sin(XX)                # Fischer-Koch-S

        phi_all = jnp.stack([phi0, phi1, phi2, phi3, phi4], axis=0)  # (5,n_nodes)
        node_ids = jnp.arange(self.num_controlled_vars)
        phi = phi_all[type_node, node_ids]

        level = phi - tau

        # map to K: {min, E_node} (strong contrast)
        if self.binary:
            solid = (level >= 0.0)
            if self.solid_is_high:
                K = jnp.where(solid, E_node, self.min)
            else:
                K = jnp.where(solid, self.min, E_node)
        else:
            s = sigmoid(self.beta * level)
            if self.solid_is_high:
                K = self.min + (E_node - self.min) * s
            else:
                K = self.min + (E_node - self.min) * (1.0 - s)

        return K

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass