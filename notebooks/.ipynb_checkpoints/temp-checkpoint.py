# %% [markdown]
# # Physics‑Informed Fourier Neural Operator (PI‑FNO) 
# ## Coupled 3D Thermo‑Mechanical Box (Hexahedral FEM)
# 
# This notebook demonstrates **physics‑informed operator learning** for a coupled thermo‑mechanical problem in 3D. The objective is to learn an operator that maps a **parameter field** (here: a spatially varying conductivity‑like field $K$) to the coupled state fields:
# $$
# K \;\mapsto\; \left(T,\; U_x,\; U_y,\; U_z\right)
# $$
# without relying on paired finite‑element solution labels during training.
# 
# ### Workflow (high‑level)
# 1. Construct a structured **3D hexahedral FE mesh** of a unit box.
# 2. Define a **monolithic coupled thermo‑mechanical residual** evaluated at element level.
# 3. Generate randomized material fields $K$ using a **Fourier parameterization**.
# 4. Build a **multi‑output Fourier Neural Operator (FNO)** surrogate for $(T,U_x,U_y,U_z)$.
# 5. Train the surrogate by minimizing the **physics residual loss** (physics‑informed learning).
# 6. Validate on selected samples by comparing against a **nonlinear FE solve**.
# 7. Export fields (state + derived quantities) for downstream visualization.
# 
# 

# %% [markdown]
# ## 1) Directory handling and logging
# 
# We create a clean working directory and redirect `stdout` into a log file, so training progress is saved.
# 

# %%
import os
from fol.tools.usefull_functions import *
# directory & save handling
working_directory_name = 'pi_fno_thermo_mechanical_3d_box'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)

from jax import jit
import os 
import math
import gmsh
import meshio
from jax.experimental.sparse import BCOO
from jax.scipy.sparse.linalg import cg
def GetHeatFluxVector_consistent(loss_function, fe_mesh, conductivity, temperature,
                                   tol=1e-10, maxiter=500):
    """
    Return nodal heat flux q_node of shape (num_nodes, dim)
    by CONSISTENT (non-lumped) L2 projection of Gauss-point fluxes.

    Solve: M q = b,  M_ij = ∫ N_i N_j dΩ,  b_i = ∫ N_i q_gp dΩ
    """
    element_type = loss_function.element_type
    elem_nodes = fe_mesh.GetElementsNodes(element_type)   # (ne, nen)
    dim = loss_function.dim
    XYZ = jnp.asarray(fe_mesh.GetNodesCoordinates())      # (nn, dim)
    nn = fe_mesh.GetNumberOfNodes()

    k = jnp.asarray(conductivity).squeeze()
    T = jnp.asarray(temperature).squeeze()

    gp_points, gp_weights = loss_function.fe_element.GetIntegrationData()
    ne, nen = elem_nodes.shape

    k_is_nodal = (k.ndim == 1 and k.shape[0] == nn)

    # --- element contribution: return (Me, be)
    def element_contrib(nodes_e, e_id):
        xyze = XYZ[nodes_e, :]                 # (nen, dim)
        te = T[nodes_e].reshape(-1, 1)         # (nen, 1)

        if k_is_nodal:
            ke_nodes = k[nodes_e]              # (nen,)
        else:
            ke_const = k[e_id]                 # scalar

        @jit
        def at_gp(gp_point, gp_weight):
            DN_DX = loss_function.fe_element.ShapeFunctionsGlobalGradients(xyze, gp_point)  # (nen, dim)
            N = loss_function.fe_element.ShapeFunctionsValues(gp_point)                      # (nen,)

            J = loss_function.fe_element.Jacobian(xyze, gp_point)
            detJ = jnp.linalg.det(J)
            Jw = jnp.abs(detJ) * gp_weight  # orientation-safe

            gradT = DN_DX.T @ te            # (dim, 1)
            kg = (N @ ke_nodes) if k_is_nodal else ke_const
            qg = (-kg * gradT).squeeze()    # (dim,)

            return N, qg, Jw

        N_g, q_g, Jw_g = jax.vmap(at_gp, in_axes=(0, 0))(gp_points, gp_weights)
        # N_g: (ngp, nen), q_g: (ngp, dim), Jw_g: (ngp,)

        # element mass (consistent): Me = Σ (N^T N) Jw
        # Me_ij = Σ N_i N_j Jw
        # Me = (N_g[:, :, None] * N_g[:, None, :] * Jw_g[:, None, None]).sum(axis=0)  # (nen, nen)
        Me = N_g.T @ (N_g * Jw_g[:, None])    # (nen, nen)

        # element RHS: be_i = Σ N_i q_g Jw
        # be = (N_g[:, :, None] * (q_g * Jw_g[:, None])[:, None, :]).sum(axis=0)      # (nen, dim)
        be = N_g.T @ (q_g * Jw_g[:, None])    # (nen, dim)

        return Me, be

    Me, be = jax.vmap(element_contrib, in_axes=(0, 0))(elem_nodes, jnp.arange(ne))
    # Me: (ne, nen, nen), be: (ne, nen, dim)

    # --- assemble sparse global M and dense rhs b
    # Build COO indices for all element (i,j) pairs
    # rows/cols: (ne, nen, nen)
    rows = jnp.broadcast_to(elem_nodes[:, :, None], (ne, nen, nen))
    cols = jnp.broadcast_to(elem_nodes[:, None, :], (ne, nen, nen))

    idx = jnp.stack([rows.reshape(-1), cols.reshape(-1)], axis=1)    # (ne*nen*nen, 2)
    data = Me.reshape(-1)                                            # (ne*nen*nen,)

    M = BCOO((data, idx), shape=(nn, nn))  # duplicates are summed automatically

    # assemble b: scatter-add (nn, dim)
    b = jnp.zeros((nn, dim))
    b = b.at[elem_nodes].add(be)

    # --- solve M q = b for each component
    # CG wants rhs shape (nn,), so solve per component.
    def solve_one(rhs):
        x, info = cg(M, rhs, tol=tol, maxiter=maxiter)
        # info==0 means converged. (You can return info if you want.)
        return x
    
    if dim == 2:
        qx = solve_one(b[:, 0])
        qy = solve_one(b[:, 1])
        q_node = jnp.stack([qx, qy], axis=1)
    elif dim == 3:
        qx = solve_one(b[:, 0])
        qy = solve_one(b[:, 1])
        qz = solve_one(b[:, 2])
        q_node = jnp.stack([qx, qy, qz], axis=1)

    return q_node

def GetStressVector_consistent(loss_function, fe_mesh, DeT, UVWT, TeT, Te_initT,
                                tol=1e-10, maxiter=500):
    """
    Consistent L2 projection of Gauss-point stresses to global nodal stresses.
    Returns nodal_stress (nn, 3) for [sxx, syy, sxy].
    """
    UVW = jnp.asarray(UVWT)
    De = jnp.asarray(DeT).squeeze()
    Te = jnp.asarray(TeT).squeeze()
    Te_init = jnp.asarray(Te_initT).squeeze()
    dim = loss_function.dim

    element_type = loss_function.element_type
    elem_nodes = jnp.asarray(fe_mesh.GetElementsNodes(element_type))  # (ne, nen)
    XYZ = jnp.asarray(fe_mesh.GetNodesCoordinates())                  # (nn, 2)
    nn = fe_mesh.GetNumberOfNodes()

    e0 = loss_function.loss_settings["material_dict"]["young_modulus"]
    v  = loss_function.loss_settings["material_dict"]["poisson_ratio"]

    gp_points, gp_weights = loss_function.fe_element.GetIntegrationData()
    ne, nen = elem_nodes.shape

    # --- element routine: returns (Me, be) where
    # Me: (nen, nen), be: (nen, 3)
    def element_Me_be(nodes_e, e_id):
        xyze = XYZ[nodes_e, :]  # (nen, 2)

        # collect element dofs for displacement (assumes 2 dof per node)
        dofs = jnp.zeros((dim * nen,), dtype=jnp.int32)
        for i in range(dim):
            dofs = dofs.at[i::dim].set(dim * nodes_e + i)

        se = UVW[dofs].reshape(-1, 1)  # (2*nen, 1)

        te  = jax.lax.stop_gradient(Te[nodes_e].reshape(-1, 1))          # (nen, 1)
        te0 = jax.lax.stop_gradient(Te_init[nodes_e].reshape(-1, 1))     # (nen, 1)

        # De nodal (as in your current code)
        de_nodes = De[nodes_e].reshape(-1, 1)  # (nen, 1)

        @jit
        def at_gp(gp_point, gp_weight):
            N = loss_function.fe_element.ShapeFunctionsValues(gp_point)  # (nen,)
            DN_DX = loss_function.fe_element.ShapeFunctionsGlobalGradients(xyze, gp_point)  # (nen,dim)

            J = loss_function.fe_element.Jacobian(xyze, gp_point)
            detJ = jnp.linalg.det(J)
            Jw = jnp.abs(detJ) * gp_weight

            # --- stress at this gp (same as your ComputeElement2D)
            B = loss_function.CalculateBMatrix(DN_DX)  # (3, dim*nen)
            total_strain = B @ se  # (3, 1)

            temp  = jnp.dot(N, te.squeeze())
            temp0 = jnp.dot(N, te0.squeeze())

            thermal_strain_vec = (
                loss_function.thermal_loss_settings["alpha"]
                * (temp - temp0)
                * loss_function.thermal_st_vec
            )  # (3,1)

            elastic_strain = total_strain - thermal_strain_vec  # (3,1)

            e_at_gp = jnp.dot(N, de_nodes.squeeze())
            if dim==2:
                Dmat = loss_function.CalculateDMatrix2D(e0 * e_at_gp, v)  # (3,3)
            elif dim==3:
                Dmat = loss_function.CalculateDMatrix3D(e0 * e_at_gp, v)  # (6,6)

            sigma_gp = (Dmat @ elastic_strain).squeeze()  # (3,)

            return N, sigma_gp, Jw

        N_g, sigma_g, Jw_g = jax.vmap(at_gp, in_axes=(0, 0))(gp_points, gp_weights)
        # N_g: (ngp, nen), sigma_g: (ngp, 3), Jw_g: (ngp,)

        # consistent element mass matrix
        Me = N_g.T @ (N_g * Jw_g[:, None])                 # (nen, nen)

        # element RHS for stresses
        be = N_g.T @ (sigma_g * Jw_g[:, None])             # (nen, 3)

        return Me, be

    Me, be = jax.vmap(element_Me_be, in_axes=(0, 0))(elem_nodes, jnp.arange(ne))
    # Me: (ne, nen, nen), be: (ne, nen, 3)

    # --- assemble global sparse mass matrix M
    rows = jnp.broadcast_to(elem_nodes[:, :, None], (ne, nen, nen))
    cols = jnp.broadcast_to(elem_nodes[:, None, :], (ne, nen, nen))
    idx = jnp.stack([rows.reshape(-1), cols.reshape(-1)], axis=1)
    data = Me.reshape(-1)

    M = BCOO((data, idx), shape=(nn, nn))  # duplicates summed

    # --- assemble global RHS b (dense)
    if dim == 2:
        b = jnp.zeros((nn, 3))
    elif dim == 3:
        b = jnp.zeros((nn, 6))  # 6 components for 3D stress
    
    b = b.at[elem_nodes].add(be)

    # --- solve M x = b for each stress component
    def solve_one(rhs):
        x, info = cg(M, rhs, tol=tol, maxiter=maxiter)
        return x

    if dim == 2:
        sxx = solve_one(b[:, 0])
        syy = solve_one(b[:, 1])
        sxy = solve_one(b[:, 2])
        nodal_stress = jnp.stack([sxx, syy, sxy], axis=1)

    elif dim == 3:
        sxx = solve_one(b[:, 0])
        syy = solve_one(b[:, 1])
        szz = solve_one(b[:, 2])
        sxy = solve_one(b[:, 3])
        syz = solve_one(b[:, 4])
        sxz = solve_one(b[:, 5])
        nodal_stress = jnp.stack([sxx, syy, szz, sxy, syz, sxz], axis=1)
    
    return nodal_stress


# %% [markdown]
# ## 2) Problem / model settings
# 
# `L` and `N` define the box geometry and mesh resolution.
# 
# Boundary conditions are set for:
# - Temperature `T`
# - Displacements `Ux, Uy, Uz`
# 
# Here, left and right faces are prescribed with different values.
# Then, We build a structured 3D box mesh and initialize it.
# 
# For verification, the mesh is converted to a PyVista object and visualized in the notebook, showing the 3D geometry with element edges and nodes to inspect the discretization.

# %%
model_settings = {
    "L": 1,
    "N": 22,  # number of elements per direction (mesh will have N+1 nodes per axis)

    # Displacement BCs
    "Ux_left": 0.0, "Ux_right": 0.10,
    "Uy_left": 0.0, "Uy_right": 0.10,
    "Uz_left": 0.0, "Uz_right": 0.10,

    # Temperature BCs
    "T_left": 0.5, "T_right": 0.0
}

from fol.tools.usefull_functions import create_3D_box_mesh
# creation of the mesh
fe_mesh = create_3D_box_mesh_structured(
    Nx=model_settings["N"],
    Ny=model_settings["N"],
    Nz=model_settings["N"],
    Lx=model_settings["L"],
    Ly=model_settings["L"],
    Lz=model_settings["L"]
)

fe_mesh.Initialize()

## -----------------------------------------------------------
## Convert FE mesh to PyVista for visualization
## -----------------------------------------------------------

import pyvista as pv
import vtk

# Turn off VTK global warning display (suppresses WARN| ... lines)
vtk.vtkObject.GlobalWarningDisplayOff()

# Force safe rendering (important for headless / SSH / server environments)
pv.OFF_SCREEN = True
pv.set_jupyter_backend("static")

# fe_mesh.mesh_io is assumed to be a meshio.Mesh object
# pv.wrap converts it to a PyVista mesh (UnstructuredGrid)
pv_mesh = pv.wrap(fe_mesh.mesh_io)


## -----------------------------------------------------------
## Visualization: nodes + element edges
## -----------------------------------------------------------

pv_mesh.plot(
    show_edges=True,              # Draw element edges (wireframe overlay)
    edge_color="black",           # Edge color for clear mesh visibility
    render_points_as_spheres=True,# Render nodes as spheres (not pixels)
    point_size=8                  # Size of mesh nodes
)

# %% [markdown]
# ## 3) Define the thermo-mechanical physics loss
# 
# We configure:
# - **Dirichlet boundary conditions** for `T, Ux, Uy, Uz`
# - **Material properties** (elasticity + reference temperature)
# - **Thermal properties** (`alpha`, `beta`, `c`)
# 
# The `ThermoMechanicsLoss3DHexa` object is responsible for evaluating PDE residuals / energy terms for training.
# 

# %% [markdown]
# ## Thermo‑Mechanical Finite‑Element Residual Loss
# 
# ### Scientific description
# The loss evaluates a **monolithically coupled thermo‑mechanical finite‑element residual** in 3D, with unknowns given by the temperature field $T$ and the displacement vector $\mathbf{u}=(U_x,U_y,U_z)$. Training minimizes the violation of the governing PDEs over all elements, enabling **physics‑informed learning** without supervised FE labels.
# 
# The formulation is consistent with:
# 
# **(i) Steady heat conduction**
# $$
# \nabla\cdot\left(k\,\nabla T\right)=0,
# $$
# where $k$ is a spatially varying conductivity‑type coefficient (here parameterized as $K$).
# 
# **(ii) Linear momentum balance**
# $$
# \nabla\cdot\boldsymbol{\sigma}+\mathbf{b}=0,
# $$
# with linear‑elastic constitutive response and thermal expansion coupling:
# $$
# \boldsymbol{\sigma}=\mathbf{D}\left(\boldsymbol{\varepsilon}-\boldsymbol{\varepsilon}_\text{th}\right),
# \qquad
# \boldsymbol{\varepsilon}_\text{th}=\alpha\,(T-T_0)\,\mathbf{s}.
# $$
# Here $\mathbf{D}$ is the elastic constitutive matrix, $\alpha$ is the thermal expansion coefficient, $T_0$ is a reference temperature field, and $\mathbf{s}$ is a dimension‑appropriate selector defining the thermal strain components.
# 
# ### What the loss returns computationally
# At the element level, the loss implementation computes:
# - a **total scalar loss** (weighted-residual objective) for efficient gradient‑based learning,
# - the **coupled residual vector**,
# - and **coupled Jacobian (tangent) matrix** required for Newton‑type FE solves.
# 
# 

# %%
from fol.loss_functions.thermo_mechanics import ThermoMechanicsLoss3DHexa
import numpy as np

# Boundary condition dictionary (Dirichlet-style values on left/right faces)
bc_dict = {
    "T":  {"left": model_settings["T_left"],  "right": model_settings["T_right"]},
    "Ux": {"left": model_settings["Ux_left"], "right": model_settings["Ux_right"]},
    "Uy": {"left": model_settings["Uy_left"], "right": model_settings["Uy_right"]},
    "Uz": {"left": model_settings["Uz_left"], "right": model_settings["Uz_right"]},
}

# Initial temperature field (stored per node)
initial_temp = jnp.full((1, fe_mesh.GetNumberOfNodes()), 1e-4)

# Material parameters
material_dict = {
    "young_modulus": 1.0,
    "poisson_ratio": 0.3,
    "T0": initial_temp.flatten()
}

# Thermal parameters
thermal_dict = {"k1":0.5,"k2":2.0,"k3":20.0,"k4":0.5,
                "e1":1.0,"e2":-0.6}

# Create FE-based thermo-mechanical loss
thermomech_loss_3d = ThermoMechanicsLoss3DHexa(
    "thermomechanical_loss_3d",
    loss_settings={
        "dirichlet_bc_dict": bc_dict,
        "material_dict": material_dict,
        "thermal_dict": thermal_dict
    },
    fe_mesh=fe_mesh
)

thermomech_loss_3d.Initialize()

# %% [markdown]
# ## 5) Control definition
# 
# In the reference implementation this demo uses an **IdentityControl**, meaning:
# - the "control variable" is passed through as-is (no transformation)
# - here it typically means the input parameter field (e.g., `K`) is directly used as the control.
# 
# This is used by the FOL training wrapper.
# 

# %%
from fol.controls.identity_control import IdentityControl
I_control = IdentityControl("I_Control", {}, fe_mesh.GetNumberOfNodes())
I_control.Initialize()

# %% [markdown]
# ## Fourier‑parameterized sampling of the coefficient field $K$
# 
# A compact stochastic parameterization of spatially varying coefficients is obtained by expanding $K(x,y,z)$ in a truncated Fourier basis. For each prescribed set of modal frequencies:
# 1. Sample random Fourier coefficients,
# 2. Map them to a nodal field on the FE mesh,
# 3. Enforce value bounds $[K_{\min}, K_{\max}]$ through the control definition.
# 
# The resulting dataset `K_matrix` has shape:
# $$
# (\text{n\_samples},\; \text{n\_nodes})
# $$
# and is later reshaped to a structured grid for the FNO forward pass.
# 
# For verification, a single sampled conductivity field is attached to the PyVista mesh and visualized to inspect its spatial distribution before further processing.
# 

# %%
import jax
from fol.controls.fourier_control import FourierControl

freq_sets = [
    (np.array([2, 4, 6]), np.array([2, 4, 6]), np.array([2, 4, 6])),
    (np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])),
    (np.array([3, 4, 5]), np.array([3, 4, 5]), np.array([3, 4, 5])),
    (np.array([4, 6, 8]), np.array([4, 6, 8]), np.array([4, 6, 8])),
]

K_matrix_parts = []

# PRNG handling: split keys so each frequency set yields distinct samples
seed = 42
key = jax.random.PRNGKey(seed)

N_samples_per_set = 1500

for idx, (x_freqs, y_freqs, z_freqs) in enumerate(freq_sets):
    key, subkey = jax.random.split(key)

    control_settings = {
        "x_freqs": x_freqs,
        "y_freqs": y_freqs,
        "z_freqs": z_freqs,
        "beta": 10,   # smoothness/sharpness parameter (implementation-dependent)
        "min": 0.1,   # lower bound for K
        "max": 1.0,   # upper bound for K
    }

    fourier_control = FourierControl("K", control_settings, fe_mesh)
    fourier_control.Initialize()

    # Sample random Fourier coefficients (one row per realization)
    coeffs = jax.random.normal(
        subkey, (N_samples_per_set, fourier_control.GetNumberOfVariables())
    )

    # Map coefficients -> bounded spatial field K on FE mesh nodes
    K_realizations = fourier_control.ComputeBatchControlledVariables(coeffs)

    K_matrix_parts.append(K_realizations)

# Concatenate all sets into a single dataset
K_matrix = jnp.concatenate(K_matrix_parts, axis=0)

print(f"{K_matrix.shape[0]} Fourier‑parameterized samples generated.")
print("K_matrix shape:", K_matrix.shape)  # (n_samples, n_nodes)

# Attach one sampled conductivity field (node-wise) to the PyVista mesh
pv_mesh.point_data["Conductivity Sample"] = K_matrix[200].flatten()
# Visualize the selected sample on the 3D mesh
pv_mesh.plot(scalars="Conductivity Sample",
             show_edges=True, 
             edge_color="black",      
             line_width=1,          
             scalar_bar_args=dict(
                 position_x=0.85,
                 position_y=0.1,
                 width=0.08,
                 height=0.8,
                 vertical=True))

# %% [markdown]
# ## Multi‑output Fourier Neural Operator (FNO)
# 
# A Fourier Neural Operator learns mappings between function spaces by combining:
# - pointwise lifting/projection layers, and
# - spectral convolution blocks operating on truncated Fourier modes.
# 
# Here, a **multi‑output** surrogate is constructed by instantiating one scalar‑output FNO per channel and concatenating predictions. The four output channels correspond to:
# $$
# (T,\;U_x,\;U_y,\;U_z).
# $$
# This design is convenient when the underlying FNO implementation is naturally scalar‑output while maintaining a shared input representation.
# 

# %%
from fol.deep_neural_networks.ported_fourier_neural_operator_networks.fno import FNO
from flax import nnx

class MultiFNO(nnx.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = 4,
        hidden_channels: int = 16,
        n_modes=(10, 10, 10),
        n_layers: int = 4,
        rngs: nnx.Rngs,
    ):
        # Create `out_channels` independent FNOs, each predicting 1 channel
        self.fnos = nnx.List([
            FNO(
                in_channels=in_channels,
                out_channels=1,
                hidden_channels=hidden_channels,
                n_modes=n_modes,
                n_layers=n_layers,
                rngs=rngs,  # NNX will draw fresh keys internally
            )
            for _ in range(out_channels)
        ])

    def __call__(self, x: jnp.ndarray):
        # Run each FNO on the same input, concatenate outputs on channel axis
        outs = [fno(x) for fno in self.fnos]  # each (..., 1)
        return jnp.concatenate(outs, axis=-1)  # (..., out_channels)


# %% [markdown]
# ### Instantiate the model and sanity-check shapes
# 
# the input `K_matrix` is stored as flattened node vectors, but FNOs usually expect a **grid**.
# Since the mesh is structured `N x N x N` elements, the node grid is `(N+1)³`.
# 
# So we reshape from:
# - `(B, num_nodes)` → `(B, N+1, N+1, N+1, 1)`
# 

# %%
fno_model = MultiFNO(
    in_channels=1,
    out_channels=4,
    hidden_channels=16,
    n_modes=(10, 10, 10),
    n_layers=4,
    rngs=nnx.Rngs(0),
)

# Count trainable parameters
params = nnx.state(fno_model, nnx.Param)
total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"FNO trainable parameters: {total_params}")

# Sanity-check forward pass with a small batch
B = 8
N = model_settings["N"]
grid_shape = (B, N, N, N, 1)
x0 = K_matrix[0:B].reshape(grid_shape)

y0 = fno_model(x0)
print("Input shape :", x0.shape)
print("Output shape:", y0.shape)  # expected (B, N+1, N+1, N+1, 4)


# %% [markdown]
# ## 8) Training setup (optimizer + schedule)
# 
# We use:
# - a linear learning rate schedule from `1e-2` to `1e-3`
# - Adam optimizer
# 

# %%
import optax
num_epochs = 500

learning_rate_scheduler = optax.linear_schedule(
    init_value=1e-2,
    end_value=1e-3,
    transition_steps=num_epochs
)

optimizer = optax.chain(optax.adam(learning_rate_scheduler))

# %% [markdown]
# ## 9) Physics-Informed Fourier Parametric Operator Learning
# 
# This wrapper:
# - takes the **control** (here identity control)
# - the **physics loss** (thermo-mechanics)
# - the **neural operator** (MultiFNO)
# - and an **Optax optimizer**
# then trains the network by minimizing physics residuals under the sampled `K` fields.
# 
# > In other words: the model learns an **operator** mapping `K → (T, Ux, Uy, Uz)` that satisfies the PDE.
# 

# %%
from fol.deep_neural_networks.fourier_parametric_operator_learning import (
    PhysicsInformedFourierParametricOperatorLearning
)

pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(
    name="pi_fno_pr_learning",
    control=I_control,
    loss_function=thermomech_loss_3d,
    flax_neural_network=fno_model,
    optax_optimizer=optimizer
)

pi_fno_pr_learning.Initialize()

# %% [markdown]
# ### Train / test split
# 
# we use the indices:
# - Train: samples `[0:5000]`
# - Test: samples `[5000:6000]`
# 
# > Make sure `K_matrix` has at least 6000 rows. With 4×1500 = 6000, it matches the reference implementation.
# 

# %%
train_start_id, train_end_id = 0, 500
test_start_id, test_end_id   = 500, 550

assert K_matrix.shape[0] >= test_end_id, "K_matrix doesn't have enough samples."

print("Train samples:", train_end_id - train_start_id)
print("Test samples :", test_end_id - test_start_id)


# %% [markdown]
# ### Physics‑informed training
# 
# Training minimizes the thermo‑mechanical residual loss evaluated over mini‑batches of coefficient fields $K$. The test set is periodically evaluated to monitor generalization across unseen $K$ realizations.
# 
# Key settings:
# - `test_frequency`: test evaluation interval (epochs)
# - `batch_size`: number of coefficient fields per optimization step
# - checkpointing: stores best‑loss state and periodic snapshots
# - plotting: exports diagnostics at a fixed cadence
# 

# %%
# pi_fno_pr_learning.Train(
#     train_set=(K_matrix[train_start_id:train_end_id, :],),
#     test_frequency=100,
#     batch_size=10,
#     convergence_settings={
#         "num_epochs": num_epochs,
#         "relative_error": 1e-100,
#         "absolute_error": 1e-100,
#     },
#     train_checkpoint_settings={"least_loss_checkpointing": True, "frequency": 100},
#     plot_settings={"plot_save_rate": 100},
#     working_directory=case_dir
# )
from IPython.display import Image, display
display(Image(filename=os.path.join(case_dir,f'training_history.png')))

# %% [markdown]
# ### Restore the best saved model state
# 
# the reference implementation restores from `case_dir + "/flax_train_state"`.
# 

# %%
pi_fno_pr_learning.RestoreState(restore_state_directory=case_dir + "/flax_final_state")

# %% [markdown]
# ## Reference nonlinear finite‑element solve (validation)
# 
# To assess surrogate fidelity, the coupled problem is also solved using a **nonlinear residual‑based FE solver**. This provides reference solutions for selected $K$ realizations, enabling qualitative and quantitative comparison of:
# - primary fields $(T,U_x,U_y,U_z)$,
# - derived stress measures,
# - and thermal flux fields.
# 

# %%
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
fe_setting = {
    "linear_solver_settings": {
        "solver": "JAX-bicgstab",
        "tol": 1e-6,
        "atol": 1e-6,
        "maxiter": 1000,
        "pre-conditioner": "ilu",
    },
    "nonlinear_solver_settings": {
        "rel_tol": 1e-7,
        "abs_tol": 1e-7,
        "maxiter": 10,
        "load_incr": 5,
    }
}

nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolver(
    "nonlinear_fe_solver",
    thermomech_loss_3d,
    fe_setting
)
nonlinear_fe_solver.Initialize()

# %% [markdown]
# ## 11) Compare PI-FNO vs FE on selected samples
# 
# We pick two sample indices (as in the reference implementation): `3000` and `5500`.
# 
# For each:
# 1. Store the input `K` field on the mesh
# 2. Predict `T, Ux, Uy, Uz` with PI-FNO
# 3. Compute derived quantities (stress, heat flux) from the predicted fields
# 4. Solve the FE problem and compute FE stress
# 5. Store everything in the mesh fields for export
# 

# %%
# visualization_set_dict = {"train":[100,200],"test":[600,700]}
visualization_set_dict = {"train":[100],"test":[]}
for set_name,ids in visualization_set_dict.items():
    for id in ids:

        # Store K field on the mesh 
        pv_mesh.point_data[f'train conductitivty K {id}'] = np.array(K_matrix[id, :]).reshape(-1, 1)


        # --- PI-FNO prediction ---
        FOL_TUVW = np.array(
            pi_fno_pr_learning.Predict(np.array(K_matrix[id, :]).reshape(1, -1))
        )  # shape (1, num_nodes*4) or similar depending on implementation

        # Store predicted primary fields
        pv_mesh.point_data[f'FOL_T_{id}'] = FOL_TUVW.reshape((-1, 4))[:, 0].flatten()
        pv_mesh.point_data[f'FOL_U_{id}'] = FOL_TUVW.reshape((-1, 4))[:, 1].flatten()
        print(FOL_TUVW.shape)

        # Derived quantities from prediction
        pv_mesh.point_data[f'FOL_Stress_{id}'] = np.array(
            thermomech_loss_3d.ComputeStress(
                np.array(K_matrix[id, :]).flatten(),
                FOL_TUVW.reshape((-1, 4))
            )
        )

        pv_mesh.point_data[f'FOL_Heat_Flux{id}'] = np.array(
            thermomech_loss_3d.ComputeHeatFlux(
                np.array(K_matrix[id, :]).flatten(),
                FOL_TUVW.reshape((-1, 4))[:, 0].flatten()  # temperature channel
            )
        )

        # --- FE reference solve ---
        # Initial guess for (T,Ux,Uy,Uz) stacked vector
        x0 = np.zeros((fe_mesh.GetNumberOfNodes() * 4,))

        FE_TUVW = np.array(
            nonlinear_fe_solver.Solve(np.array(K_matrix[id, :]).flatten(), x0)
        )
        print(FE_TUVW.shape)

        pv_mesh.point_data[f'FE_T_{id}'] = FE_TUVW.reshape((-1, 4))[:, 0].flatten()
        pv_mesh.point_data[f'FE_U_{id}'] = FE_TUVW.reshape((-1, 4))[:, 1].flatten()

        pv_mesh.point_data[f'FE_Stress_{id}'] = np.array(
            thermomech_loss_3d.ComputeStress(
                np.array(K_matrix[id, :]).flatten(),
                FE_TUVW
            )
        )

        pv_mesh.point_data[f'FE_Heat_Flux{id}'] = np.array(
            thermomech_loss_3d.ComputeHeatFlux(
                np.array(K_matrix[id, :]).flatten(),
                FE_TUVW.reshape((-1, 4))[:, 0].flatten()  # temperature channel
            )
        )

        # pv_mesh.point_data[f'FE_Stress_Global_L2_{id}'] = np.array(
        #     thermomech_loss_3d.ComputeStressGlobalL2(
        #         np.array(K_matrix[id, :]).flatten(),
        #         FE_TUVW
        #     )
        # )

        # pv_mesh.point_data[f'FE_Heat_Flux_Global_L2_{id}'] = np.array(
        #     thermomech_loss_3d.ComputeHeatFluxGlobalL2(
        #         np.array(K_matrix[id, :]).flatten(),
        #         FE_TUVW.reshape((-1, 4))[:, 0].flatten()  # temperature channel
        #     )
        # )
        # pv_mesh.point_data[f'FE_diff_Stress_Global_L2_{id}'] = np.abs(
        #     pv_mesh.point_data[f'FE_Stress_Global_L2_{id}'] - pv_mesh.point_data[f'FE_Stress_{id}']
        # )

        # pv_mesh.point_data[f'FE_diff_Heat_Flux_Global_L2_{id}'] = np.abs(
        #     pv_mesh.point_data[f'FE_Heat_Flux_Global_L2_{id}'] - pv_mesh.point_data[f'FE_Heat_Flux{id}']
        # )

# Select sample ID and dataset split
visualization_id = 100
set_name = "train"

# Fields arranged as:
# Row 1 → FOL results (K, T, U, Heat Flux)
# Row 2 → FE results  (K, T, U, Heat Flux)
fields = [
    f'{set_name} conductitivty K {visualization_id}',
    f'FOL_T_{visualization_id}',
    f'FOL_U_{visualization_id}',
    f'FOL_Heat_Flux{visualization_id}',
    f'{set_name} conductitivty K {visualization_id}',
    f'FE_T_{visualization_id}',
    f'FE_U_{visualization_id}',
    f'FE_Heat_Flux{visualization_id}'
]

# Create 2×4 subplot layout
plotter = pv.Plotter(shape=(2, 4), window_size=(1800, 900))

for idx, field in enumerate(fields):
    row = idx // 4
    col = idx % 4
    plotter.subplot(row, col)

    # Plot mesh colored by current field
    plotter.add_mesh(
        pv_mesh.copy(),      # avoid scalar carry-over
        scalars=field,
        show_edges=False,
        scalar_bar_args=dict(
            vertical=True,
            position_x=0.015,
            position_y=0.01,
            height=0.5,
            width=0.1,
            label_font_size=12,
            title_font_size=1  # hide title
        )
    )

    plotter.add_text(field, font_size=8)

# Render figure
plotter.show()

# %% [markdown]
# ## 12) Export results
# 
# Finally, we finalize the mesh and export all stored fields to `case_dir`.
# You can open the exported results in the usual post-processing pipeline.
# 

# %%
pv_mesh.save("out_mesh.vtu", binary=True)

# %%


