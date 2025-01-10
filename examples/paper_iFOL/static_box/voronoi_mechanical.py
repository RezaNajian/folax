# import necessaries 
import sys
import os
import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss3DTetra
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.voronoi_control3D import VoronoiControl3D
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle,optax

# directory & save handling
working_directory_name = "voronoi_box_3D_tetra"
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# create mesh
fe_mesh = Mesh("box_3D","box_3D_coarse.med",'../../meshes/')

# create fe-based loss function
bc_dict = {"Ux":{"left":0.0},
           "Uy":{"left":0.0,"right":-0.05},
           "Uz":{"left":0.0,"right":-0.05}}
material_dict = {"young_modulus":1,"poisson_ratio":0.3}
mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "material_dict":material_dict,
                                                                                "body_foce":jnp.zeros((3,1))},
                                                                                fe_mesh=fe_mesh)

# voronoi control
voronoi_control_settings = {"number_of_seeds":16,"E_values":(0.1,1)}
voronoi_control = VoronoiControl3D("voronoi_control",voronoi_control_settings,fe_mesh)

fe_mesh.Initialize()
mechanical_loss_3d.Initialize()
voronoi_control.Initialize()

# create/load some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_samples = 200
    coeffs_matrix,E_matrix = create_random_voronoi_samples(voronoi_control,number_of_samples,dim=3)
    export_dict = {}
    export_dict["coeffs_matrix"] = coeffs_matrix
    with open(f'voronoi_3D_control_dict.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'voronoi_3D_control_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]

E_matrix = voronoi_control.ComputeBatchControlledVariables(coeffs_matrix)

# now save K matrix 
export_Ks = False
if export_Ks:
    for i in range(E_matrix.shape[0]):
        solution_file = os.path.join(case_dir, f"K_{i}.vtu")
        fe_mesh[f'K_{i}'] = np.array(E_matrix[i,:])
    fe_mesh.Finalize(export_dir=case_dir)

# now create implicit parametric deep learning
# design synthesizer & modulator NN for hypernetwork
characteristic_length = 64
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=3,
                     output_size=3,
                     hidden_layers=[characteristic_length] * 6,
                     activation_settings={"type":"sin",
                                          "prediction_gain":30,
                                          "initialization_gain":1.0})

latent_size = 2 * characteristic_length
modulator_nn = MLP(name="modulator_nn",
                   input_size=latent_size,
                   use_bias=False) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 2000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",
                                             control=voronoi_control,
                                             loss_function=mechanical_loss_3d,
                                             flax_neural_network=hyper_network,
                                             main_loop_optax_optimizer=main_loop_transform,
                                             latent_step_size=1e-2,
                                             num_latent_iterations=3,
                                             checkpoint_settings={"restore_state":False,
                                             "state_directory":case_dir+"/flax_state"},
                                             working_directory=case_dir)
fol.Initialize()

# now train for the defined train and test samples
train_start_id = 0
train_end_id = 20
test_start_id = 20
test_end_id = 40
fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
          test_set=(coeffs_matrix[test_start_id:test_end_id,:],),
          test_settings={"test_frequency":10},batch_size=1,
            convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
            plot_settings={"plot_save_rate":100},
            save_settings={"save_nn_model":True,
                           "best_model_checkpointing":True,
                           "best_model_checkpointing_frequency":100})

# load the best model
fol.RestoreCheckPoint(fol.checkpoint_settings)

for test in range(train_start_id,test_end_id):
    eval_id = test
    fe_mesh[f'E_{eval_id}'] = E_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(), 1))
    FOL_UVW = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
    fe_mesh[f'U_FOL_{eval_id}'] = FOL_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    # solve FE here
    fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu"},
                    "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                "maxiter":10,"load_incr":5}}
    linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_3d,fe_setting)
    linear_fe_solver.Initialize()
    FE_UVW = np.array(linear_fe_solver.Solve(E_matrix[eval_id],np.zeros(3*fe_mesh.GetNumberOfNodes())))  
    fe_mesh[f'U_FE_{eval_id}'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    absolute_error = abs(FOL_UVW.reshape(-1,1)- FE_UVW.reshape(-1,1))
    fe_mesh[f'abs_error_{eval_id}'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 3))

fe_mesh.Finalize(export_dir=case_dir)