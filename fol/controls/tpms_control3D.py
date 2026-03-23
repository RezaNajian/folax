"""
TPMSControl3D - nodal material field generator on an existing 3D mesh.

Control vector (normalized in [0,1] unless you override):
  v[0] -> TPMS type selector (mapped to {0..4})
  v[1] -> frequency fc (mapped to integer in [fc_min, fc_max])
  v[2] -> sx (shift in [0,1])
  v[3] -> sy (shift in [0,1])
  v[4] -> sz (shift in [0,1])
  v[5] -> tau (threshold shift in [tau_min, tau_max])

Returns:
  K_nodes : (n_nodes,) in [min, max]
"""

from .control import Control
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from jax.nn import sigmoid

from fol.mesh_input_output.mesh import Mesh
from fol.tools.decoration_functions import *


class TPMSControl3D(Control):

    def __init__(self, control_name: str, control_settings: dict, fe_mesh: Mesh):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh
        self.scale_min = 0.0
        self.scale_max = 1.0

        # fixed map (id -> name) for debugging / saving metadata
        self.tpms_type_map = {
            0: "Gyroid",
            1: "Schwarz_P",
            2: "Schwarz_D",
            3: "Neovius",
            4: "Fischer_Koch_S",
        }

    @print_with_timestamp_and_execution_time
    def Initialize(self, reinitialize=False) -> None:
        if self.initialized and not reinitialize:
            return

        # output range
        self.min = float(self.settings.get("min", 1e-6))
        self.max = float(self.settings.get("max", 1.0))

        # cube size used for scaling coords (your mesh is typically [0,1])
        self.L = float(self.settings.get("L", 1.0))

        # TPMS parameter ranges
        self.fc_min  = int(self.settings.get("fc_min", 1))
        self.fc_max  = int(self.settings.get("fc_max", 3))   # inclusive
        self.tau_min = float(self.settings.get("tau_min", -0.25))
        self.tau_max = float(self.settings.get("tau_max",  0.25))

        # mapping options
        self.binary = bool(self.settings.get("binary", True))
        self.beta   = float(self.settings.get("beta", 20.0))  # only used if binary=False

        self.n_types = int(self.settings.get("n_types", 5))
        if self.n_types != 5:
            raise ValueError("This implementation currently supports exactly 5 TPMS types (n_types=5).")

        # control vector length
        self.num_control_vars = 6
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()

        self.initialized = True

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self, variable_vector: jnp.array):
        """
        variable_vector is expected in [0,1] (like FourierControl), then we map to physical ranges.
        """
        # rescale like FourierControl does
        v = variable_vector * (self.scale_max - self.scale_min) + self.scale_min
        v = jnp.clip(v, 0.0, 1.0)

        # --- map normalized v to parameters ---
        # type_id: 0..4
        type_id = jnp.floor(v[0] * self.n_types).astype(jnp.int32)
        type_id = jnp.clip(type_id, 0, self.n_types - 1)

        # fc: integer in [fc_min, fc_max]
        fc_span = (self.fc_max - self.fc_min + 1)
        fc = self.fc_min + jnp.floor(v[1] * fc_span).astype(jnp.int32)
        fc = jnp.clip(fc, self.fc_min, self.fc_max).astype(jnp.float32)

        # shifts in [0,1]
        sx, sy, sz = v[2], v[3], v[4]

        # tau in [tau_min, tau_max]
        tau = self.tau_min + (self.tau_max - self.tau_min) * v[5]

        # --- mesh coords ---
        Xp = self.fe_mesh.GetNodesX()
        Yp = self.fe_mesh.GetNodesY()
        Zp = self.fe_mesh.GetNodesZ()

        # periodic shift in physical space
        Xs = jnp.mod(Xp + sx * self.L, self.L)
        Ys = jnp.mod(Yp + sy * self.L, self.L)
        Zs = jnp.mod(Zp + sz * self.L, self.L)

        # map to 0..2π*fc
        two_pi = 2.0 * jnp.pi
        X = two_pi * fc * (Xs / self.L)
        Y = two_pi * fc * (Ys / self.L)
        Z = two_pi * fc * (Zs / self.L)

        # --- TPMS definitions ---
        def gyroid(_):
            return jnp.sin(X) * jnp.cos(Y) + jnp.sin(Y) * jnp.cos(Z) + jnp.sin(Z) * jnp.cos(X)

        def schwarz_p(_):
            return jnp.cos(X) + jnp.cos(Y) + jnp.cos(Z)

        def schwarz_d(_):
            return (
                jnp.sin(X) * jnp.sin(Y) * jnp.sin(Z)
                + jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z)
                + jnp.cos(X) * jnp.sin(Y) * jnp.cos(Z)
                + jnp.cos(X) * jnp.cos(Y) * jnp.sin(Z)
            )

        def neovius(_):
            return 3.0 * (jnp.cos(X) + jnp.cos(Y) + jnp.cos(Z)) + 4.0 * jnp.cos(X) * jnp.cos(Y) * jnp.cos(Z)

        def fischer_koch_s(_):
            return jnp.cos(X) * jnp.sin(Y) + jnp.cos(Y) * jnp.sin(Z) + jnp.cos(Z) * jnp.sin(X)

        phi = lax.switch(type_id, [gyroid, schwarz_p, schwarz_d, neovius, fischer_koch_s], operand=None)
        level = phi - tau

        # --- map to [min, max] ---
        if self.binary:
            K = jnp.where(level < 0.0, self.min, self.max)
        else:
            # smooth volume fraction control via tau
            K = (self.max - self.min) * sigmoid(self.beta * (level)) + self.min

        return K

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass