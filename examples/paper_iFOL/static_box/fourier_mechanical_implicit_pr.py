# import necessaries 
import sys
import os

import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss3DTetra
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle,optax
from fol.tools.decoration_functions import *

# directory & save handling
working_directory_name = "fourier_box_3D_tetra"
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# create mesh
fe_mesh = Mesh("box_3D","box_3D_coarse.med",'../../meshes/')

# create fe-based loss function
bc_dict = {"Ux":{"left":0.0,"right":0.00},
           "Uy":{"left":0.0,"right":-0.05},
           "Uz":{"left":0.0,"right":-0.05}}
material_dict = {"young_modulus":1,"poisson_ratio":0.3}
mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "material_dict":material_dict,
                                                                                "body_foce":jnp.zeros((3,1))},
                                                                                fe_mesh=fe_mesh)

# fourier control
fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                            "beta":20,"min":1e-2,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

fe_mesh.Initialize()
mechanical_loss_3d.Initialize()
fourier_control.Initialize()

# create/load some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_samples = 1000
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_samples)
    export_dict = {}
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = fourier_control.x_freqs
    export_dict["y_freqs"] = fourier_control.y_freqs
    export_dict["z_freqs"] = fourier_control.z_freqs
    with open(f'/workspace/fourier_3D_control_dict_1K.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'fourier_3D_control_dict_1K.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]

K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

# now save K matrix 
export_Ks = False
if export_Ks:
    for i in range(K_matrix.shape[0]):
        solution_file = os.path.join(case_dir, f"K_{i}.vtu")
        fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
    fe_mesh.Finalize(export_dir=case_dir)
    exit()

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

latent_size = characteristic_length
modulator_nn = MLP(name="modulator_nn",
                   input_size=latent_size,
                   use_bias=False) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 2000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.adam(1e-5))
latent_step_optimizer = optax.chain(optax.adam(1e-1))
# create fol
fol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_ol",
                                                        control=fourier_control,
                                                        loss_function=mechanical_loss_3d,
                                                        flax_neural_network=hyper_network,
                                                        main_loop_optax_optimizer=main_loop_transform,
                                                        latent_step_optax_optimizer=latent_step_optimizer,
                                                        latent_step_size=10.0)

fol.Initialize()


# now train for the defined train and test samples
train_OTF_id = 0
train_start_id = 0
train_end_id = 10
fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
          test_set=(coeffs_matrix[train_start_id:train_start_id+1,:],),
          test_frequency=10,
          batch_size=3,
          convergence_settings={"num_epochs":2000,"relative_error":1e-100,"absolute_error":1e-100},
          plot_settings={"save_frequency":1},
          train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10,"state_directory":case_dir+"/flax_train_state"},
          test_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10,"state_directory":case_dir+"/flax_test_state"},
          restore_nnx_state_settings={"restore":False,"state_directory":case_dir+"/flax_final_state"},
          working_directory=case_dir)

# load the best model
fol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")


for id in range(train_start_id,train_end_id):
    train_OTF_id = id
    fe_mesh[f'K_{train_OTF_id}'] = K_matrix[train_OTF_id,:].reshape((fe_mesh.GetNumberOfNodes(), 1))
    FOL_UVW = np.array(fol.Predict(coeffs_matrix[train_OTF_id,:].reshape(-1,1).T)).reshape(-1)
    fe_mesh[f'U_FOL_{train_OTF_id}'] = FOL_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))


    # solve FE here
    fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu"},
                    "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                "maxiter":10,"load_incr":5}}
    linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_3d,fe_setting)
    linear_fe_solver.Initialize()
    FE_UVW = np.array(linear_fe_solver.Solve(K_matrix[train_OTF_id],np.zeros(3*fe_mesh.GetNumberOfNodes())))  
    fe_mesh[f'U_FE_{train_OTF_id}'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    absolute_error = abs(FOL_UVW.reshape(-1,1)- FE_UVW.reshape(-1,1))
    fe_mesh[f'abs_error_{train_OTF_id}'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 3))

    # compute and export the residuals 
    _,r_FEM = mechanical_loss_3d.ComputeJacobianMatrixAndResidualVector(K_matrix[train_OTF_id],FE_UVW)
    _,r_FOL = mechanical_loss_3d.ComputeJacobianMatrixAndResidualVector(K_matrix[train_OTF_id],FOL_UVW)

    fe_mesh[f'res_FOL_{train_OTF_id}'] = r_FOL.reshape((fe_mesh.GetNumberOfNodes(), 3))
    fe_mesh[f'res_FE_{train_OTF_id}'] = r_FEM.reshape((fe_mesh.GetNumberOfNodes(), 3))


    # compute energies
    FOL_loss = np.sqrt(mechanical_loss_3d.ComputeSingleLoss(K_matrix[train_OTF_id],FOL_UVW.flatten()[mechanical_loss_3d.non_dirichlet_indices])[0])
    FE_loss = np.sqrt(mechanical_loss_3d.ComputeSingleLoss(K_matrix[train_OTF_id],FE_UVW.flatten()[mechanical_loss_3d.non_dirichlet_indices])[0])

    fol_info(f"FOL_loss:{FOL_loss}, FE_loss:{FE_loss}")
fe_mesh.Finalize(export_dir=case_dir)
