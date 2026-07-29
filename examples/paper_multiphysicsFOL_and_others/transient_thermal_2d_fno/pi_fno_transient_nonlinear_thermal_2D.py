import sys
import os
import pickle,optax
import numpy as np
from fol.tools.usefull_functions import *
from fol.tools.decoration_functions import *
from fol.tools.logging_functions import Logger
from fol.controls.identity_control import IdentityControl
from fol.controls.multi_control import MultiControl
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.loss_functions.transient_thermal import TransientThermalLoss2DQuad
from thermal_usefull_functions import *
from fol.deep_neural_networks.ported_fourier_neural_operator_networks.fno import FNO
from fol.deep_neural_networks.fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearning

from flax import nnx
import jax
jax.config.update("jax_default_matmul_precision", "highest")

# directory & save handling
working_directory_name = 'pi_fno_transient_nonlinear_thermal_2D_check'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":51,
                "T_left":1.0,"T_right":0.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()

# create some random fields for training
create_random_fields = True
if create_random_fields:
    train_temperature_fields = generate_random_smooth_patterns(model_settings["L"],model_settings["N"],num_samples=9000)
    train_heterogeneity_field = generate_morph_pattern(model_settings["N"]).reshape(1,-1)
    train_data_dict = {"temperatures":train_temperature_fields, "heterogeneity":train_heterogeneity_field}
    with open(f'train_data_dict.pkl', 'wb') as f:
        pickle.dump(train_data_dict,f)
else:
    with open(f'train_data_dict.pkl', 'rb') as f:
        train_data_dict = pickle.load(f)

    train_temperature_fields = train_data_dict["temperatures"]
    train_heterogeneity_field = train_data_dict["heterogeneity"]

    fol_info(f"train temperature field {train_temperature_fields.shape} is imported !")

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]}}
material_dict = {"rho":1.0,"cp":1.0,"k0":train_heterogeneity_field.flatten(),"beta":1.5,"c":1.0}
time_integration_dict = {"method":"implicit-euler","time_step":0.005}

class OperatorCompatibleTransientThermalLoss2DQuad(TransientThermalLoss2DQuad):
    """Support the named-field batch API used by the current PI-FNO trainer."""

    def ComputeElement(self, *args):
        if len(args) != 1 or not isinstance(args[0], dict):
            return super().ComputeElement(*args)

        element_variables = args[0]
        next_temperature = (
            (1.0 - element_variables["T_mask"]) * element_variables["T"]
            + element_variables["T_mask"] * element_variables["T_mask_value"]
        )
        return super().ComputeElement(
            element_variables["XYZ"],
            element_variables["T_current"],
            next_temperature,
            element_variables["K0"],
        )

    def ComputeBatchLoss(self, batch_variables):
        batch_size = next(iter(batch_variables.values())).shape[0]
        connectivity = self.fe_mesh.GetElementsNodes(self.element_type)
        element_coordinates = self.fe_mesh.GetNodesCoordinates()[connectivity, :]
        batch_element_variables = {
            "XYZ": jnp.broadcast_to(
                element_coordinates[None, ...],
                (batch_size,) + element_coordinates.shape,
            )
        }
        for name, values in batch_variables.items():
            nodal_values = jnp.reshape(jnp.atleast_2d(values), (batch_size, -1))
            batch_element_variables[name] = nodal_values[:, connectivity]

        element_losses = jax.vmap(
            jax.vmap(lambda variables: self.ComputeElement(variables)[0], in_axes=0),
            in_axes=0,
        )(batch_element_variables)
        sample_losses = jnp.sum(element_losses.reshape(batch_size, -1), axis=1)
        sample_losses = sample_losses**self.loss_function_exponent
        return jnp.mean(sample_losses), (
            jnp.min(sample_losses),
            jnp.max(sample_losses),
            jnp.mean(sample_losses),
        )


transient_thermal_loss_2d = OperatorCompatibleTransientThermalLoss2DQuad("thermal_transient_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict,
                                                                            "time_integration_dict":time_integration_dict},
                                                                            fe_mesh=fe_mesh)
transient_thermal_loss_2d.Initialize()

temperature_control = IdentityControl("T_current", {}, fe_mesh.GetNumberOfNodes())
multi_control = MultiControl("transient_thermal_controls", [temperature_control])
multi_control.Initialize()

number_of_samples = train_temperature_fields.shape[0]
temperature_mask = jnp.zeros(fe_mesh.GetNumberOfNodes())
temperature_mask_value = jnp.zeros(fe_mesh.GetNumberOfNodes())
for boundary_name, boundary_value in bc_dict["T"].items():
    boundary_node_ids = fe_mesh.GetNodeSet(boundary_name)
    temperature_mask = temperature_mask.at[boundary_node_ids].set(1.0)
    temperature_mask_value = temperature_mask_value.at[boundary_node_ids].set(boundary_value)

training_context = {
    "T_mask":jnp.broadcast_to(
        temperature_mask, (number_of_samples, temperature_mask.size)
    ),
    "T_mask_value":jnp.broadcast_to(
        temperature_mask_value, (number_of_samples, temperature_mask_value.size)
    ),
    "K0":jnp.broadcast_to(
        jnp.asarray(train_heterogeneity_field).reshape(-1),
        (number_of_samples, fe_mesh.GetNumberOfNodes()),
    ),
}

# design synthesizer & modulator NN for hypernetwork

class TransientThermalFNO(nnx.Module):
    """Ported NNX FNO with named transient-temperature input and output."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.fno = FNO(
            n_modes=(24, 24),
            in_channels=1,
            out_channels=1,
            hidden_channels=10,
            n_layers=3,
            projection_channel_ratio=128 / 10,
            rngs=rngs,
        )

    def __call__(self, inputs):
        return {"T":self.fno(inputs["T_current"])}


fno_model = TransientThermalFNO(rngs=nnx.Rngs(0))

# get total number of fno params
params = nnx.state(fno_model, nnx.Param)
total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"total number of fno network param:{total_params}")

num_epochs = 100
learning_rate_scheduler = optax.linear_schedule(init_value=5e-3, end_value=5e-4, transition_steps=num_epochs)
optimizer = optax.chain(optax.adam(learning_rate_scheduler))

# create fol
pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(name="pi_fno_pr_learning",
                                                                        control=multi_control,
                                                                        loss_function=transient_thermal_loss_2d,
                                                                        flax_neural_network=fno_model,
                                                                        optax_optimizer=optimizer)

pi_fno_pr_learning.Initialize()

train_start_id = 0
train_end_id = 6000

pi_fno_pr_learning.Train(train_set=({"T_current":train_temperature_fields[train_start_id:train_end_id,:]},
                                    {key:value[train_start_id:train_end_id,:] for key,value in training_context.items()}),
            batch_size=120,
            convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
            plot_settings={"plot_list":["total_loss"],"save_frequency":100},
            working_directory=case_dir)


# time ineference setup
initial_solution_id = 0
initial_solution = train_temperature_fields[train_end_id + initial_solution_id,:]
num_time_steps = 50

# Predict dynamics with the current named-field FNO interface.
T_ifol_current = jnp.asarray(initial_solution).reshape(-1)
T_ifols = T_ifol_current[None, :]
prediction_context = {
    "T_mask":temperature_mask[None, :],
    "T_mask_value":temperature_mask_value[None, :],
    "K0":jnp.asarray(train_heterogeneity_field).reshape(1, -1),
}
for _ in range(num_time_steps):
    predicted_fields = pi_fno_pr_learning.Predict(
        {"T_current":T_ifol_current[None, :], **prediction_context}
    )
    T_ifol_current = predicted_fields["T"][0].reshape(-1)
    T_ifols = jnp.vstack((T_ifols, T_ifol_current))

# predict dynamics with FE
fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu"},
                "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                            "maxiter":10,"load_incr":1}}
nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",transient_thermal_loss_2d,fe_setting)
nonlin_fe_solver.Initialize()

T_fe_current = initial_solution.flatten()
T_fes = jnp.array(T_fe_current)
for _ in range(0,num_time_steps):
    T_fe_next = np.array(nonlin_fe_solver.Solve(T_fe_current,T_fe_current))
    T_fe_current = T_fe_next
    T_fes = jnp.vstack((T_fes,T_fe_next.flatten()))

# save the solutions
for time_index, (T_ifol,T_fe) in enumerate(zip(T_ifols,T_fes)):
    fe_mesh[f"T_ifol_{time_index}"] = np.array(T_ifol).reshape((fe_mesh.GetNumberOfNodes(), 1))
    fe_mesh[f"T_fe_{time_index}"] = np.array(T_fe).reshape((fe_mesh.GetNumberOfNodes(), 1))

fe_mesh["k0"] = np.array(train_heterogeneity_field.flatten()).reshape((fe_mesh.GetNumberOfNodes(), 1))


absolute_error = np.abs(T_ifols- T_fes)
time_list = [0,1,4,9,19,24,49]

plot_mesh_vec_data_thermal_row(1,[initial_solution],
                   [""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"initial_condition.png"))

plot_mesh_quad(fe_mesh.GetNodesCoordinates()[:,:-1],
               fe_mesh.GetElementsNodes("quad"),
               background=train_heterogeneity_field.reshape((model_settings["N"],model_settings["N"]))[::-1],
               filename=os.path.join(case_dir,"FE_mesh_hetero_info.png"),show=False)

plot_mesh_vec_data_thermal_row(1,[T_ifols[time_list[0],:],T_ifols[time_list[1],:],
                                  T_ifols[time_list[2],:],T_ifols[time_list[3],:],
                                  T_ifols[time_list[4],:],T_ifols[time_list[5],:],
                                  T_ifols[time_list[6],:]],
                   ["","","","","","",""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test_FOL_summary.png"))
plot_mesh_vec_data_thermal_row(1,[T_fes[time_list[0],:],T_fes[time_list[1],:],
                                  T_fes[time_list[2],:],T_fes[time_list[3],:],
                                  T_fes[time_list[4],:],T_fes[time_list[5],:],
                                  T_fes[time_list[6],:]],
                   ["","","","","","",""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test_FE_summary.png"))

plot_mesh_vec_data_thermal_row(1,[absolute_error[time_list[0],:],absolute_error[time_list[1],:],
                                  absolute_error[time_list[2],:],absolute_error[time_list[3],:],
                                  absolute_error[time_list[4],:],absolute_error[time_list[5],:],
                                  absolute_error[time_list[6],:]],
                   ["","","","","","",""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test_Error_summary.png"))


fe_mesh.Finalize(export_dir=case_dir)
