import sys
import os
from fol.controls.voronoi_control2D import VoronoiControl2D
import optax
import numpy as np
from fol.loss_functions.thermo_mechanical_nonlinear import ThermoMechanicalLoss2DQuad
from fol.controls.fourier_control import FourierControl
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.controls.identity_control import IdentityControl
from thermo_mechanical_useful_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import MLP

from fourier_neural_operator_networks import FourierNeuralOperator2D
from fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearning

from fol.tools.decoration_functions import *
from flax.nnx import bridge
import pickle
from flax import nnx
import jax
np.random.seed(42) 
jax.config.update('jax_default_matmul_precision','high')
# jax.config.update('jax_enable_x64', True)

# directory & save handling
working_directory_name = 'pi_fno_thermomech_2D_single_test'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":84,
                  "Ux_left":0.0,"Ux_right":0.10,
                  "Uy_left":0.0,"Uy_right":0.10,
                  "T_left":0.5,"T_right":0.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()
no_control = IdentityControl("No_Control",fe_mesh)
no_control.Initialize()
# create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]},
           "Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
           "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}#

initial_temp = np.full((1,fe_mesh.GetNumberOfNodes()),0.0)
material_dict = {"young_modulus":1.0,"poisson_ratio":0.3,"T0":initial_temp.flatten()}
# freq_sets = [
#     (np.array([2, 4, 6]),    np.array([2, 4, 6]))
# ]
freq_sets = [      
    (np.array([2, 4, 6]),    np.array([2, 4, 6])),
    (np.array([1, 2, 3]),    np.array([1, 2, 3])),
    (np.array([3, 4, 5]),    np.array([3, 4, 5])),
    (np.array([4, 6, 8]),    np.array([4, 6, 8]))
]

number_of_random_samples = 100
all_K_list = []

create_random_coefficients = True
if create_random_coefficients:
    voronoi_control_settings = {"number_of_seeds":64,"E_values":(0.1,1)}
    voronoi_control = VoronoiControl2D("first_voronoi_control",voronoi_control_settings,fe_mesh)
    voronoi_control.Initialize()
    _, K_matrix = create_random_voronoi_samples(voronoi_control,number_of_random_samples)
    np.save("K_matrix_voronoi.npy",K_matrix)
    # np.random.seed(42)  # For reproducibility
    # np.random.shuffle(K_matrix)
    E_matrix = K_matrix

else:
    K_matrix = np.load("K_matrix_voronoi.npy")
    # np.random.seed(42)  # For reproducibility
    # np.random.shuffle(K_matrix)
    E_matrix = K_matrix

loss_settings={"dirichlet_bc_dict":bc_dict,
               "material_dict":material_dict,
               "loss_function_exponent":1.0}
thermomech_loss_2d = ThermoMechanicalLoss2DQuad("thermothermomech_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "material_dict":material_dict, "alpha":1.5,
                                                                            "beta":2.0, 
                                                                            "c":2.0},
                                                                            fe_mesh=fe_mesh)
thermomech_loss_2d.Initialize()

def merge_state(dst: nnx.State, src: nnx.State):
    for k, v in src.items():
        if isinstance(v, nnx.State):
            merge_state(dst[k], v)
        else:
            dst[k] = v

fno_model = bridge.ToNNX(FourierNeuralOperator2D(modes1=12,
                                                modes2=12,
                                                width=34,
                                                depth=4,
                                                channels_last_proj=128,
                                                out_channels=3,
                                                padding=8,
                                                output_scale=0.001),rngs=nnx.Rngs(0)).lazy_init(K_matrix[0:1].reshape(1,model_settings["N"],model_settings["N"],1)) 

# replace RNG key by a dummy to allow checkpoint restoration later
graph_def, state = nnx.split(fno_model)
rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
merge_state(state, rngs_key)
fno_model = nnx.merge(graph_def, state)

# get total number of fno params
params = nnx.state(fno_model, nnx.Param)
total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"total number of fno network param:{total_params}")

num_epochs = 5000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-2, end_value=1e-3, transition_steps=num_epochs)
optimizer = optax.chain(optax.adam(learning_rate_scheduler))

# create fol
pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(name="pi_fno_pr_learning",
                                                                        control=no_control,
                                                                        loss_function=thermomech_loss_2d,
                                                                        flax_neural_network=fno_model,
                                                                        optax_optimizer=optimizer)

pi_fno_pr_learning.Initialize()

train_start_id = 0
train_end_id = 5000
test_start_id = 5000
test_end_id = 6000
# train_start_id = 0
# train_end_id = 1
# test_start_id = 0
# test_end_id = 1
#here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
# pi_fno_pr_learning.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
#                         test_set=(coeffs_matrix[test_start_id:test_end_id,:],),
#                         test_frequency=100,
#                         batch_size=100,
#                         convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
#                         plot_settings={"plot_save_rate":100},
#                         train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
#                         working_directory=case_dir)
# pi_fno_pr_learning.Train(train_set=(K_matrix[train_start_id:train_end_id,:],),
#           test_set=(K_matrix[test_start_id:test_end_id,:],),
#           batch_size=100,
#           convergence_settings={"num_epochs":num_epochs,
#                                 "relative_error":1e-100,
#                                 "absolute_error":1e-100},
#           train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
#           working_directory=case_dir,
#           plot_settings={"plot_list":["total_loss","phy1_loss","phy2_loss"],
#                          "plot_frequency":1,"save_frequency":100,
#                          "save_directory":".","multiphysics":True})
# load teh best model
pi_fno_pr_learning.RestoreState(restore_state_directory="pi_fno_thermomech_2D_single/flax_final_state")



# ---- config ----
start_id = 0
num_samples = 50
# start_id = 0
# num_samples = 1
sample_ids = range(start_id, start_id + num_samples)

n_nodes = fe_mesh.GetNumberOfNodes()
output_size = 3  # T, Ux, Uy

# initialize FE solver once
fe_setting = {"linear_solver_settings":{"solver":"JAX-direct","tol":1e-6,"atol":1e-6,
                                                    "maxiter":1000,"pre-conditioner":"ilu"},
                        "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                                    "maxiter":10,"load_incr":5}}
nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolver(
    "nonlinear_fe_solver", thermomech_loss_2d, fe_setting
)
nonlinear_fe_solver.Initialize()


# base directory to hold all samples (e.g., ".../case_batch")
batch_dir = os.path.join(case_dir, "voronoi_sr")
os.makedirs(batch_dir, exist_ok=True)
import os, gc, time, json
import numpy as np

# -------- helpers for error metrics --------
def _l2(x):        return float(np.sqrt(np.sum(x**2)))
def _linf(x):      return float(np.max(np.abs(x)))
def _rel_l2(err, ref, eps=1e-12):  # ||e||2 / (||ref||2 + eps)
    return _l2(err) / max(_l2(ref), eps)
def _mse(x):     return float(np.mean(x**2))

rows = []  # will collect per-sample metrics rows

# ---------------- your existing loop (kept) ----------------
for sid in sample_ids:
    t0 = time.time()
    print(f"Processing sample {sid}...")
    sample_dir = os.path.join(batch_dir, f"sample_{sid:05d}")
    os.makedirs(sample_dir, exist_ok=True)

    # --- FOL prediction ---
    # (Assumes pi_fno_pr_learning.Predict accepts batch; if not, keep the [None, :] batching)
    K_vec = K_matrix[sid, :]
    plot_mesh_vec_data_hetero(1,[K_vec],["Heterogeneity"],
        cmap="viridis",
        file_name=os.path.join(sample_dir, f"hetero_{sid}.png"),)
    FOL_TUV = np.array(pi_fno_pr_learning.Predict(K_vec[None, :])).reshape(n_nodes, 3)
    fe_mesh['sol_FOL'] = FOL_TUV

    # --- FE solve ---
    FE_TUV_flat = np.array(
        nonlinear_fe_solver.Solve(
            K_vec.flatten(),
            np.zeros(n_nodes * output_size)
        )
    )
    FE_TUV = reshape_T_U_to_nodewise(FE_TUV_flat, n_nodes)
    fe_mesh['sol_FE'] = FE_TUV

    # --- errors (state) ---
    abs_err_state = np.abs(FOL_TUV - FE_TUV)
    fe_mesh['abs_error'] = abs_err_state

    # --- plots (state) ---
    plot_mesh_vec_data_thermal_2Dthermomech(
        1,
        [FOL_TUV[:,0],  FE_TUV[:,0], abs_err_state[:,0],
         FOL_TUV[:,1],  FE_TUV[:,1], abs_err_state[:,1],
         FOL_TUV[:,2],  FE_TUV[:,2], abs_err_state[:,2]],
        ["FNO","FEM","Absolute error"],
        cmap="turbo",
        file_name=os.path.join(sample_dir, f"sol_field_{sid}.png"),
    )

    # --- stress ---
    FOL_UV = FOL_TUV[:, 1:]
    FE_UV  = FE_TUV[:, 1:]
    FOL_T  = FOL_TUV[:, 0]
    FE_T   = FE_TUV[:, 0]

    E_vec = E_matrix[sid, :]

    FOL_stress = GetStressVector2D(
        thermomech_loss_2d, fe_mesh, E_vec.flatten(),
        FOL_UV.flatten(), FOL_T.flatten(), initial_temp.flatten()
    )
    FE_stress = GetStressVector2D(
        thermomech_loss_2d, fe_mesh, E_vec.flatten(),
        FE_UV.flatten(), FE_T.flatten(), initial_temp.flatten()
    )
    abs_err_stress = np.abs(FOL_stress - FE_stress)

    fe_mesh['FOL_stress'] = FOL_stress
    fe_mesh['FE_stress'] = FE_stress
    fe_mesh['abs_error_stress'] = abs_err_stress

    plot_mesh_vec_data_thermal_2Dthermomech(
        1,
        [FOL_stress[:,0],  FE_stress[:,0], abs_err_stress[:,0],
         FOL_stress[:,1],  FE_stress[:,1], abs_err_stress[:,1],
         FOL_stress[:,2],  FE_stress[:,2], abs_err_stress[:,2]],
        ["FNO","FEM","Absolute error"],
        cmap="plasma",
        file_name=os.path.join(sample_dir, f"sol_stress_{sid}.png"),
    )

    # --- heat flux ---
    FOL_q = GetHeatFluxVector2D(thermomech_loss_2d, fe_mesh, K_vec.flatten(), FOL_T.flatten())
    FE_q  = GetHeatFluxVector2D(thermomech_loss_2d, fe_mesh, K_vec.flatten(), FE_T.flatten())
    abs_err_q = np.abs(FOL_q - FE_q)

    fe_mesh['FOL_heat_flux'] = FOL_q
    fe_mesh['FE_heat_flux']  = FE_q
    fe_mesh['abs_error_heat_flux'] = abs_err_q

    plot_mesh_vec_data_thermal_2Dheatflux(
        1,
        [FOL_q[:,0],  FE_q[:,0], abs_err_q[:,0],
         FOL_q[:,1],  FE_q[:,1], abs_err_q[:,1]],
        ["FNO","FEM","Absolute error"],
        file_name=os.path.join(sample_dir, f"sol_heat_flux_{sid}.png"),
    )

    # -------- per-sample metrics (added) --------
    T_err  = FOL_TUV[:,0] - FE_TUV[:,0]
    U_err  = FOL_TUV[:,1:] - FE_TUV[:,1:]
    S_err  = FOL_stress - FE_stress
    q_err  = FOL_q - FE_q

    row = {
        "sid": int(sid),
        # relative L2 (component + vector)
        "T_relL2":   _rel_l2(T_err,  FE_TUV[:,0]),
        "Ux_relL2":  _rel_l2(U_err[:,0], FE_TUV[:,1]),
        "Uy_relL2":  _rel_l2(U_err[:,1], FE_TUV[:,2]),
        "Uvec_relL2": _rel_l2(U_err, FE_TUV[:,1:]),

        "Sxx_relL2": _rel_l2(S_err[:,0], FE_stress[:,0]),
        "Syy_relL2": _rel_l2(S_err[:,1], FE_stress[:,1]),
        "Sxy_relL2": _rel_l2(S_err[:,2], FE_stress[:,2]),
        "Svec_relL2": _rel_l2(S_err, FE_stress),

        "qx_relL2":  _rel_l2(q_err[:,0], FE_q[:,0]),
        "qy_relL2":  _rel_l2(q_err[:,1], FE_q[:,1]),
        "qvec_relL2": _rel_l2(q_err, FE_q),

        # Linf (absolute max error)
        "T_Linf":    _linf(T_err),
        "Uvec_Linf": _linf(U_err),
        "Svec_Linf": _linf(S_err),
        "qvec_Linf": _linf(q_err),

        # MSE
        "T_MSE":     _mse(T_err),
        "Ux_MSE":  _mse(U_err[:,0]),
        "Uy_MSE":  _mse(U_err[:,1]),
        "Uvec_MSE": _mse(U_err),
        "Sxx_MSE":  _mse(S_err[:,0]),
        "Syy_MSE":  _mse(S_err[:,1]), 
        "Sxy_MSE":  _mse(S_err[:,2]),

        "qx_MSE":  _mse(q_err[:,0]),
        "qy_MSE":  _mse(q_err[:,1]),

        "Svec_MSE":  _mse(S_err),
        "qvec_MSE":  _mse(q_err),

        "seconds": float(time.time() - t0),
    }
    rows.append(row)

    # export mesh for this sample
    fe_mesh.Finalize(export_dir=sample_dir)

    # optional: free large arrays to keep memory steady
    del FOL_TUV, FE_TUV, FE_TUV_flat, abs_err_state, FOL_stress, FE_stress, abs_err_stress, FOL_q, FE_q, abs_err_q
    gc.collect()


# -------- after the loop: save aggregated stats --------
per_sample_csv = os.path.join(batch_dir, "error_stats_per_sample.csv")
summary_json   = os.path.join(batch_dir, "error_stats_summary.json")

# CSV (use pandas if available; else csv module)
try:
    import pandas as pd
    pd.DataFrame(rows).to_csv(per_sample_csv, index=False)
except Exception:
    import csv
    keys = sorted({k for r in rows for k in r.keys()})
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

# JSON summary with mean/std/min/max and p50/p90/p95 for numeric fields
def _summarize(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "count": int(a.size),
        "mean":  float(a.mean()),
        "std":   float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min":   float(a.min()),
        "p50":   float(np.quantile(a, 0.50)),
        "p90":   float(np.quantile(a, 0.90)),
        "p95":   float(np.quantile(a, 0.95)),
        "max":   float(a.max()),
    }

numeric_keys = [k for k in rows[0].keys() if k not in ("sid",)]
summary = {k: _summarize([r[k] for r in rows if isinstance(r.get(k),(int,float))]) for k in numeric_keys}
summary["success_count"] = int(len(rows))
summary["seconds_total"] = float(sum(r.get("seconds", 0.0) for r in rows))

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"[stats] Wrote per-sample metrics to: {per_sample_csv}")
print(f"[stats] Wrote summary stats to:      {summary_json}")

print("Voronoi SR Done.")


batch_dir = os.path.join(case_dir, "meta_sr")
os.makedirs(batch_dir, exist_ok=True)
K_matrix = np.load("tpms_2d_thick_viridis_fields_n84.npy")
E_matrix = K_matrix

rows = []  # will collect per-sample metrics rows

# ---------------- your existing loop (kept) ----------------
for sid in sample_ids:
    t0 = time.time()
    print(f"Processing sample {sid}...")
    sample_dir = os.path.join(batch_dir, f"sample_{sid:05d}")
    os.makedirs(sample_dir, exist_ok=True)

    # --- FOL prediction ---
    # (Assumes pi_fno_pr_learning.Predict accepts batch; if not, keep the [None, :] batching)
    K_vec = K_matrix[sid, :]
    plot_mesh_vec_data_hetero(1,[K_vec],["Heterogeneity"],
        cmap="viridis",
        file_name=os.path.join(sample_dir, f"hetero_{sid}.png"),)
    FOL_TUV = np.array(pi_fno_pr_learning.Predict(K_vec[None, :])).reshape(n_nodes, 3)
    fe_mesh['sol_FOL'] = FOL_TUV

    # --- FE solve ---
    FE_TUV_flat = np.array(
        nonlinear_fe_solver.Solve(
            K_vec.flatten(),
            np.zeros(n_nodes * output_size)
        )
    )
    FE_TUV = reshape_T_U_to_nodewise(FE_TUV_flat, n_nodes)
    fe_mesh['sol_FE'] = FE_TUV

    # --- errors (state) ---
    abs_err_state = np.abs(FOL_TUV - FE_TUV)
    fe_mesh['abs_error'] = abs_err_state

    # --- plots (state) ---
    plot_mesh_vec_data_thermal_2Dthermomech(
        1,
        [FOL_TUV[:,0],  FE_TUV[:,0], abs_err_state[:,0],
         FOL_TUV[:,1],  FE_TUV[:,1], abs_err_state[:,1],
         FOL_TUV[:,2],  FE_TUV[:,2], abs_err_state[:,2]],
        ["FNO","FEM","Absolute error"],
        cmap="turbo",
        file_name=os.path.join(sample_dir, f"sol_field_{sid}.png"),
    )

    # --- stress ---
    FOL_UV = FOL_TUV[:, 1:]
    FE_UV  = FE_TUV[:, 1:]
    FOL_T  = FOL_TUV[:, 0]
    FE_T   = FE_TUV[:, 0]

    E_vec = E_matrix[sid, :]

    FOL_stress = GetStressVector2D(
        thermomech_loss_2d, fe_mesh, E_vec.flatten(),
        FOL_UV.flatten(), FOL_T.flatten(), initial_temp.flatten()
    )
    FE_stress = GetStressVector2D(
        thermomech_loss_2d, fe_mesh, E_vec.flatten(),
        FE_UV.flatten(), FE_T.flatten(), initial_temp.flatten()
    )
    abs_err_stress = np.abs(FOL_stress - FE_stress)

    fe_mesh['FOL_stress'] = FOL_stress
    fe_mesh['FE_stress'] = FE_stress
    fe_mesh['abs_error_stress'] = abs_err_stress

    plot_mesh_vec_data_thermal_2Dthermomech(
        1,
        [FOL_stress[:,0],  FE_stress[:,0], abs_err_stress[:,0],
         FOL_stress[:,1],  FE_stress[:,1], abs_err_stress[:,1],
         FOL_stress[:,2],  FE_stress[:,2], abs_err_stress[:,2]],
        ["FNO","FEM","Absolute error"],
        cmap="plasma",
        file_name=os.path.join(sample_dir, f"sol_stress_{sid}.png"),
    )

    # --- heat flux ---
    FOL_q = GetHeatFluxVector2D(thermomech_loss_2d, fe_mesh, K_vec.flatten(), FOL_T.flatten())
    FE_q  = GetHeatFluxVector2D(thermomech_loss_2d, fe_mesh, K_vec.flatten(), FE_T.flatten())
    abs_err_q = np.abs(FOL_q - FE_q)

    fe_mesh['FOL_heat_flux'] = FOL_q
    fe_mesh['FE_heat_flux']  = FE_q
    fe_mesh['abs_error_heat_flux'] = abs_err_q

    plot_mesh_vec_data_thermal_2Dheatflux(
        1,
        [FOL_q[:,0],  FE_q[:,0], abs_err_q[:,0],
         FOL_q[:,1],  FE_q[:,1], abs_err_q[:,1]],
        ["FNO","FEM","Absolute error"],
        file_name=os.path.join(sample_dir, f"sol_heat_flux_{sid}.png"),
    )

    # -------- per-sample metrics (added) --------
    T_err  = FOL_TUV[:,0] - FE_TUV[:,0]
    U_err  = FOL_TUV[:,1:] - FE_TUV[:,1:]
    S_err  = FOL_stress - FE_stress
    q_err  = FOL_q - FE_q

    row = {
        "sid": int(sid),
        # relative L2 (component + vector)
        "T_relL2":   _rel_l2(T_err,  FE_TUV[:,0]),
        "Ux_relL2":  _rel_l2(U_err[:,0], FE_TUV[:,1]),
        "Uy_relL2":  _rel_l2(U_err[:,1], FE_TUV[:,2]),
        "Uvec_relL2": _rel_l2(U_err, FE_TUV[:,1:]),

        "Sxx_relL2": _rel_l2(S_err[:,0], FE_stress[:,0]),
        "Syy_relL2": _rel_l2(S_err[:,1], FE_stress[:,1]),
        "Sxy_relL2": _rel_l2(S_err[:,2], FE_stress[:,2]),
        "Svec_relL2": _rel_l2(S_err, FE_stress),

        "qx_relL2":  _rel_l2(q_err[:,0], FE_q[:,0]),
        "qy_relL2":  _rel_l2(q_err[:,1], FE_q[:,1]),
        "qvec_relL2": _rel_l2(q_err, FE_q),

        # Linf (absolute max error)
        "T_Linf":    _linf(T_err),
        "Uvec_Linf": _linf(U_err),
        "Svec_Linf": _linf(S_err),
        "qvec_Linf": _linf(q_err),

        # MSE
        "T_MSE":     _mse(T_err),
        "Ux_MSE":  _mse(U_err[:,0]),
        "Uy_MSE":  _mse(U_err[:,1]),
        "Uvec_MSE": _mse(U_err),
        "Sxx_MSE":  _mse(S_err[:,0]),
        "Syy_MSE":  _mse(S_err[:,1]), 
        "Sxy_MSE":  _mse(S_err[:,2]),

        "qx_MSE":  _mse(q_err[:,0]),
        "qy_MSE":  _mse(q_err[:,1]),

        "Svec_MSE":  _mse(S_err),
        "qvec_MSE":  _mse(q_err),

        "seconds": float(time.time() - t0),
    }
    rows.append(row)

    # export mesh for this sample
    fe_mesh.Finalize(export_dir=sample_dir)

    # optional: free large arrays to keep memory steady
    del FOL_TUV, FE_TUV, FE_TUV_flat, abs_err_state, FOL_stress, FE_stress, abs_err_stress, FOL_q, FE_q, abs_err_q
    gc.collect()


# -------- after the loop: save aggregated stats --------
per_sample_csv = os.path.join(batch_dir, "error_stats_per_sample.csv")
summary_json   = os.path.join(batch_dir, "error_stats_summary.json")

# CSV (use pandas if available; else csv module)
try:
    import pandas as pd
    pd.DataFrame(rows).to_csv(per_sample_csv, index=False)
except Exception:
    import csv
    keys = sorted({k for r in rows for k in r.keys()})
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

# JSON summary with mean/std/min/max and p50/p90/p95 for numeric fields
def _summarize(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "count": int(a.size),
        "mean":  float(a.mean()),
        "std":   float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min":   float(a.min()),
        "p50":   float(np.quantile(a, 0.50)),
        "p90":   float(np.quantile(a, 0.90)),
        "p95":   float(np.quantile(a, 0.95)),
        "max":   float(a.max()),
    }

numeric_keys = [k for k in rows[0].keys() if k not in ("sid",)]
summary = {k: _summarize([r[k] for r in rows if isinstance(r.get(k),(int,float))]) for k in numeric_keys}
summary["success_count"] = int(len(rows))
summary["seconds_total"] = float(sum(r.get("seconds", 0.0) for r in rows))

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"[stats] Wrote per-sample metrics to: {per_sample_csv}")
print(f"[stats] Wrote summary stats to:      {summary_json}")

print("Metamat Done.")

number_of_random_samples = 100
all_K_list = []
create_random_coefficients = True
if create_random_coefficients:
    voronoi_control_settings = {"number_of_seeds":64,"E_values":[0.1,1]}
    voronoi_control = VoronoiControl2D("first_voronoi_control",voronoi_control_settings,fe_mesh)
    voronoi_control.Initialize()
    _, K_matrix = create_random_voronoi_samples(voronoi_control,number_of_random_samples)
    np.save("K_matrix_voronoi.npy",K_matrix)
    # np.random.seed(42)  # For reproducibility
    # np.random.shuffle(K_matrix)
    E_matrix = K_matrix

else:
    K_matrix = np.load("K_matrix_voronoi.npy")
    # np.random.seed(42)  # For reproducibility
    # np.random.shuffle(K_matrix)
    E_matrix = K_matrix

batch_dir = os.path.join(case_dir, "dualphase_sr")
os.makedirs(batch_dir, exist_ok=True)

rows = []  # will collect per-sample metrics rows

# ---------------- your existing loop (kept) ----------------
for sid in sample_ids:
    t0 = time.time()
    print(f"Processing sample {sid}...")
    sample_dir = os.path.join(batch_dir, f"sample_{sid:05d}")
    os.makedirs(sample_dir, exist_ok=True)

    # --- FOL prediction ---
    # (Assumes pi_fno_pr_learning.Predict accepts batch; if not, keep the [None, :] batching)
    K_vec = K_matrix[sid, :]
    plot_mesh_vec_data_hetero(1,[K_vec],["Heterogeneity"],
        cmap="viridis",
        file_name=os.path.join(sample_dir, f"hetero_{sid}.png"),)
    FOL_TUV = np.array(pi_fno_pr_learning.Predict(K_vec[None, :])).reshape(n_nodes, 3)
    fe_mesh['sol_FOL'] = FOL_TUV

    # --- FE solve ---
    FE_TUV_flat = np.array(
        nonlinear_fe_solver.Solve(
            K_vec.flatten(),
            np.zeros(n_nodes * output_size)
        )
    )
    FE_TUV = reshape_T_U_to_nodewise(FE_TUV_flat, n_nodes)
    fe_mesh['sol_FE'] = FE_TUV

    # --- errors (state) ---
    abs_err_state = np.abs(FOL_TUV - FE_TUV)
    fe_mesh['abs_error'] = abs_err_state

    # --- plots (state) ---
    plot_mesh_vec_data_thermal_2Dthermomech(
        1,
        [FOL_TUV[:,0],  FE_TUV[:,0], abs_err_state[:,0],
         FOL_TUV[:,1],  FE_TUV[:,1], abs_err_state[:,1],
         FOL_TUV[:,2],  FE_TUV[:,2], abs_err_state[:,2]],
        ["FNO","FEM","Absolute error"],
        cmap="turbo",
        file_name=os.path.join(sample_dir, f"sol_field_{sid}.png"),
    )

    # --- stress ---
    FOL_UV = FOL_TUV[:, 1:]
    FE_UV  = FE_TUV[:, 1:]
    FOL_T  = FOL_TUV[:, 0]
    FE_T   = FE_TUV[:, 0]

    E_vec = E_matrix[sid, :]

    FOL_stress = GetStressVector2D(
        thermomech_loss_2d, fe_mesh, E_vec.flatten(),
        FOL_UV.flatten(), FOL_T.flatten(), initial_temp.flatten()
    )
    FE_stress = GetStressVector2D(
        thermomech_loss_2d, fe_mesh, E_vec.flatten(),
        FE_UV.flatten(), FE_T.flatten(), initial_temp.flatten()
    )
    abs_err_stress = np.abs(FOL_stress - FE_stress)

    fe_mesh['FOL_stress'] = FOL_stress
    fe_mesh['FE_stress'] = FE_stress
    fe_mesh['abs_error_stress'] = abs_err_stress

    plot_mesh_vec_data_thermal_2Dthermomech(
        1,
        [FOL_stress[:,0],  FE_stress[:,0], abs_err_stress[:,0],
         FOL_stress[:,1],  FE_stress[:,1], abs_err_stress[:,1],
         FOL_stress[:,2],  FE_stress[:,2], abs_err_stress[:,2]],
        ["FNO","FEM","Absolute error"],
        cmap="plasma",
        file_name=os.path.join(sample_dir, f"sol_stress_{sid}.png"),
    )

    # --- heat flux ---
    FOL_q = GetHeatFluxVector2D(thermomech_loss_2d, fe_mesh, K_vec.flatten(), FOL_T.flatten())
    FE_q  = GetHeatFluxVector2D(thermomech_loss_2d, fe_mesh, K_vec.flatten(), FE_T.flatten())
    abs_err_q = np.abs(FOL_q - FE_q)

    fe_mesh['FOL_heat_flux'] = FOL_q
    fe_mesh['FE_heat_flux']  = FE_q
    fe_mesh['abs_error_heat_flux'] = abs_err_q

    plot_mesh_vec_data_thermal_2Dheatflux(
        1,
        [FOL_q[:,0],  FE_q[:,0], abs_err_q[:,0],
         FOL_q[:,1],  FE_q[:,1], abs_err_q[:,1]],
        ["FNO","FEM","Absolute error"],
        file_name=os.path.join(sample_dir, f"sol_heat_flux_{sid}.png"),
    )

    # -------- per-sample metrics (added) --------
    T_err  = FOL_TUV[:,0] - FE_TUV[:,0]
    U_err  = FOL_TUV[:,1:] - FE_TUV[:,1:]
    S_err  = FOL_stress - FE_stress
    q_err  = FOL_q - FE_q

    row = {
        "sid": int(sid),
        # relative L2 (component + vector)
        "T_relL2":   _rel_l2(T_err,  FE_TUV[:,0]),
        "Ux_relL2":  _rel_l2(U_err[:,0], FE_TUV[:,1]),
        "Uy_relL2":  _rel_l2(U_err[:,1], FE_TUV[:,2]),
        "Uvec_relL2": _rel_l2(U_err, FE_TUV[:,1:]),

        "Sxx_relL2": _rel_l2(S_err[:,0], FE_stress[:,0]),
        "Syy_relL2": _rel_l2(S_err[:,1], FE_stress[:,1]),
        "Sxy_relL2": _rel_l2(S_err[:,2], FE_stress[:,2]),
        "Svec_relL2": _rel_l2(S_err, FE_stress),

        "qx_relL2":  _rel_l2(q_err[:,0], FE_q[:,0]),
        "qy_relL2":  _rel_l2(q_err[:,1], FE_q[:,1]),
        "qvec_relL2": _rel_l2(q_err, FE_q),

        # Linf (absolute max error)
        "T_Linf":    _linf(T_err),
        "Uvec_Linf": _linf(U_err),
        "Svec_Linf": _linf(S_err),
        "qvec_Linf": _linf(q_err),

        # MSE
        "T_MSE":     _mse(T_err),
        "Ux_MSE":  _mse(U_err[:,0]),
        "Uy_MSE":  _mse(U_err[:,1]),
        "Uvec_MSE": _mse(U_err),
        "Sxx_MSE":  _mse(S_err[:,0]),
        "Syy_MSE":  _mse(S_err[:,1]), 
        "Sxy_MSE":  _mse(S_err[:,2]),

        "qx_MSE":  _mse(q_err[:,0]),
        "qy_MSE":  _mse(q_err[:,1]),

        "Svec_MSE":  _mse(S_err),
        "qvec_MSE":  _mse(q_err),

        "seconds": float(time.time() - t0),
    }
    rows.append(row)

    # export mesh for this sample
    fe_mesh.Finalize(export_dir=sample_dir)

    # optional: free large arrays to keep memory steady
    del FOL_TUV, FE_TUV, FE_TUV_flat, abs_err_state, FOL_stress, FE_stress, abs_err_stress, FOL_q, FE_q, abs_err_q
    gc.collect()

# -------- after the loop: save aggregated stats --------
per_sample_csv = os.path.join(batch_dir, "error_stats_per_sample.csv")
summary_json   = os.path.join(batch_dir, "error_stats_summary.json")

# CSV (use pandas if available; else csv module)
try:
    import pandas as pd
    pd.DataFrame(rows).to_csv(per_sample_csv, index=False)
except Exception:
    import csv
    keys = sorted({k for r in rows for k in r.keys()})
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

# JSON summary with mean/std/min/max and p50/p90/p95 for numeric fields
def _summarize(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "count": int(a.size),
        "mean":  float(a.mean()),
        "std":   float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min":   float(a.min()),
        "p50":   float(np.quantile(a, 0.50)),
        "p90":   float(np.quantile(a, 0.90)),
        "p95":   float(np.quantile(a, 0.95)),
        "max":   float(a.max()),
    }

numeric_keys = [k for k in rows[0].keys() if k not in ("sid",)]
summary = {k: _summarize([r[k] for r in rows if isinstance(r.get(k),(int,float))]) for k in numeric_keys}
summary["success_count"] = int(len(rows))
summary["seconds_total"] = float(sum(r.get("seconds", 0.0) for r in rows))

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"[stats] Wrote per-sample metrics to: {per_sample_csv}")
print(f"[stats] Wrote summary stats to:      {summary_json}")

print("Dualphase Done.")
