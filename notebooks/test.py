import os
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
from fol.tools.usefull_functions import *
# directory & save handling
working_directory_name = 'pi_fno_thermo_mechanical_3d_box'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)

model_settings = {
    "L": 1,
    "N": 22,  # number of nodes per direction (mesh will have N nodes per axis)

    # Displacement BCs
    "Ux_left": 0.0, "Ux_right": 0.10,
    "Uy_left": 0.0, "Uy_right": 0.10,
    "Uz_left": 0.0, "Uz_right": 0.10,

    # Temperature BCs
    "T_left": 0.5, "T_right": 0.0
}

from fol.tools.usefull_functions import create_3D_box_mesh_structured
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
initial_temp = np.full((1, fe_mesh.GetNumberOfNodes()), 1e-4)

# Material parameters
material_dict = {
    "young_modulus": 1.0,
    "poisson_ratio": 0.3,
    "T0": initial_temp.flatten()
}

# Thermal parameters
thermal_dict = {"k1":0.5,"k2":2.0,"k3":20.0,"k4":0.5}
mechanical_dict  = {"e1":1.0,"e2":-0.6}

# Create FE-based thermo-mechanical loss
thermomech_loss_3d = ThermoMechanicsLoss3DHexa(
    "thermomechanical_loss_3d",
    loss_settings={
        "dirichlet_bc_dict": bc_dict,
        "material_dict": material_dict,
        "thermal_dict": thermal_dict,
        "mechanical_dict": mechanical_dict
    },
    fe_mesh=fe_mesh
)

thermomech_loss_3d.Initialize()


train_dr_bc_dict = {}
for dof in thermomech_loss_3d.GetDOFs():
    dof_mask = jnp.zeros(fe_mesh.GetNumberOfNodes())
    dof_mask_values = jnp.zeros(fe_mesh.GetNumberOfNodes())    
    train_dr_bc_dict[dof+"_mask"] = dof_mask
    train_dr_bc_dict[dof+"_mask_value"] = dof_mask_values

for dof,dof_dict in bc_dict.items():
    for bc_name,bc_values in dof_dict.items():
        dr_bc_node_ids = fe_mesh.GetNodeSet(bc_name)
        train_dr_bc_dict[dof+"_mask"] = train_dr_bc_dict[dof+"_mask"].at[dr_bc_node_ids].set(1.0)
        train_dr_bc_dict[dof+"_mask_value"] = train_dr_bc_dict[dof+"_mask_value"].at[dr_bc_node_ids].set(bc_values)


import jax
from fol.controls.fourier_control import FourierControl

fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                            "beta":20,"min":1e-1,"max":1}

K_control = FourierControl("K",fourier_control_settings,fe_mesh)


from fol.controls.identity_control import IdentityControl
T0_control = IdentityControl("T0", {}, fe_mesh.GetNumberOfNodes())

from fol.controls.multi_control import MultiControl

multi_control = MultiControl("first_multi_control",[K_control,T0_control])
multi_control.Initialize()


num_sample = 200

seed = 42
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)

K_coeffs = jax.random.normal(
    key, (num_sample, K_control.GetNumberOfVariables())
)

K_matrix =K_control.ComputeBatchControlledVariables(K_coeffs)


initial_temps = jnp.full((fe_mesh.GetNumberOfNodes()), 1e-4)
initial_temps = jnp.broadcast_to(initial_temps[None, ...], (num_sample,) + initial_temps.shape)

for key in train_dr_bc_dict.keys():
    value = train_dr_bc_dict[key]
    train_dr_bc_dict[key] = np.broadcast_to(value[None, ...], (num_sample,) + value.shape)

from fol.deep_neural_networks.ported_fourier_neural_operator_networks.fno import FNO
from flax import nnx

class MultiFNO(nnx.Module):
    def __init__(
        self,
        *,
        in_channels: dict[str, int],
        out_channels: dict[str, int],
        hidden_channels: int = 16,
        n_modes=(10, 10, 10),
        n_layers: int = 4,
        rngs: nnx.Rngs,
    ):
    
        # Store channel structure
        self.in_spec = in_channels
        self.out_spec = out_channels

        # Total channels for internal FNOs
        total_in = sum(in_channels.values())

        # Create one FNO per output field
        self.fnos = nnx.Dict({
            name: FNO(
                in_channels=total_in,
                out_channels=size,
                hidden_channels=hidden_channels,
                n_modes=n_modes,
                n_layers=n_layers,
                rngs=rngs,
            )
            for name, size in out_channels.items()
        })

    def _pack_input(self, x_dict: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Concatenate named inputs into single tensor."""
        xs = [x_dict[name] for name in self.in_spec]
        return jnp.concatenate(xs, axis=-1)

    def __call__(self, x: dict[str, jnp.ndarray]):

        x_cat = self._pack_input(x)

        # Each FNO predicts its own named output
        outputs = {
            name: fno(x_cat)
            for name, fno in self.fnos.items()
        }

        return outputs


fno_model = MultiFNO(
    in_channels={"K":1,"T0":1},
    out_channels={"T":1,"Ux":1,"Uy":1,"Uz":1},
    hidden_channels=16,
    n_modes=(10, 10, 10),
    n_layers=4,
    rngs=nnx.Rngs(0),
)

# Count trainable parameters
params = nnx.state(fno_model, nnx.Param)
total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"FNO trainable parameters: {total_params}")

import optax
num_epochs = 1000

learning_rate_scheduler = optax.linear_schedule(
    init_value=1e-2,
    end_value=1e-3,
    transition_steps=num_epochs
)

optimizer = optax.chain(optax.adam(learning_rate_scheduler))

from fol.deep_neural_networks.fourier_parametric_operator_learning import (
    PhysicsInformedFourierParametricOperatorLearning
)

pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(
    name="pi_fno_pr_learning",
    control=multi_control,
    loss_function=thermomech_loss_3d,
    flax_neural_network=fno_model,
    optax_optimizer=optimizer
)

pi_fno_pr_learning.Initialize()

pi_fno_pr_learning.Train(
    train_set=({"K":K_coeffs,"T0":initial_temps},train_dr_bc_dict),
    test_frequency=100,
    batch_size=10,
    convergence_settings={
        "num_epochs": num_epochs,
        "relative_error": 1e-100,
        "absolute_error": 1e-100,
    },
    train_checkpoint_settings={"least_loss_checkpointing": True, "frequency": 100},
    plot_settings={"plot_save_rate": 100},
    working_directory=case_dir,
    data_model_sharding_settings={"sharding":False,"num_data_devices":4,"num_nnx_model_devices":2}
)

pi_fno_pr_learning.RestoreState(restore_state_directory=case_dir + "/flax_final_state")


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


FOL_TUVW = (pi_fno_pr_learning.Predict({"K":K_coeffs,"T0":initial_temps,**train_dr_bc_dict}))

# print(FOL_TUVW.keys())

# exit()

visualization_set_dict = {"train":[10,20]}

for set_name,ids in visualization_set_dict.items():
    for id in ids:

        print(f"set_name:{set_name},id:{id}")

        # # Store K field on the mesh 
        fe_mesh[f'K_{id}'] = np.array(K_matrix[id, :]).reshape(-1, 1)


        fe_mesh[f'T_{id}'] = np.array(FOL_TUVW["T"][id, :]).reshape(-1, 1)
        fe_mesh[f'Ux_{id}'] = np.array(FOL_TUVW["Ux"][id, :]).reshape(-1, 1)
        fe_mesh[f'Uy_{id}'] = np.array(FOL_TUVW["Uy"][id, :]).reshape(-1, 1)
        fe_mesh[f'Uz_{id}'] = np.array(FOL_TUVW["Uz"][id, :]).reshape(-1, 1)


        # # --- PI-FNO prediction ---
        # FOL_TUVW = np.array(
        #     pi_fno_pr_learning.Predict(np.array(K_matrix[id, :]).reshape(1, -1))
        # )  # shape (1, num_nodes*4) or similar depending on implementation

        # # Store predicted primary fields
        # pv_mesh.point_data[f'FOL_T_{id}'] = FOL_TUVW.reshape((-1, 4))[:, 0].flatten()
        # pv_mesh.point_data[f'FOL_U_{id}'] = FOL_TUVW.reshape((-1, 4))[:, 1].flatten()

        # # Derived quantities from prediction
        # pv_mesh.point_data[f'FOL_Stress_{id}'] = np.array(
        #     thermomech_loss_3d.ComputeStress(
        #         np.array(K_matrix[id, :]).flatten(),
        #         FOL_TUVW
        #     )
        # )

        # pv_mesh.point_data[f'FOL_Heat_Flux{id}'] = np.array(
        #     thermomech_loss_3d.ComputeHeatFlux(
        #         np.array(K_matrix[id, :]).flatten(),
        #         FOL_TUVW.reshape((-1, 4))[:, 0].flatten()  # temperature channel
        #     )
        # )

        # # --- FE reference solve ---
        # # Initial guess for (T,Ux,Uy,Uz) stacked vector
        # x0 = np.zeros((fe_mesh.GetNumberOfNodes() * 4,))

        # FE_TUVW = np.array(
        #     nonlinear_fe_solver.Solve(np.array(K_matrix[id, :]).flatten(), x0)
        # )

        # pv_mesh.point_data[f'FE_T_{id}'] = FE_TUVW.reshape((-1, 4))[:, 0].flatten()
        # pv_mesh.point_data[f'FE_U_{id}'] = FE_TUVW.reshape((-1, 4))[:, 1].flatten()

        # pv_mesh.point_data[f'FE_Stress_{id}'] = np.array(
        #     thermomech_loss_3d.ComputeStress(
        #         np.array(K_matrix[id, :]).flatten(),
        #         FE_TUVW.reshape((-1, 4))[:, 0].flatten()
        #     )
        # )

        # pv_mesh.point_data[f'FE_Heat_Flux{id}'] = np.array(
        #     thermomech_loss_3d.ComputeHeatFlux(
        #         np.array(K_matrix[id, :]).flatten(),
        #         FE_TUVW.reshape((-1, 4))[:, 0].flatten()  # temperature channel
        #     )
        # )

fe_mesh.Finalize(export_dir=case_dir)
