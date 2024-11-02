import sys
import os
import optax
import numpy as np
from fol.loss_functions.mechanical_3D_fe_tetra import MechanicalLoss3DTetra
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.implicit_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from siren_nn import Siren
import pickle

# directory & save handling
working_directory_name = 'siren_implicit_mechanical_3D'
case_dir = os.path.join('.', working_directory_name)
# create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
fe_mesh = Mesh("fol_io","box_3D.med",'../meshes/')


# create fe-based loss function
bc_dict = {"Ux":{"left":0.0},
            "Uy":{"left":0.0,"right":-0.05},
            "Uz":{"left":0.0,"right":-0.05}}
material_dict = {"young_modulus":1,"poisson_ratio":0.3}
mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "material_dict":material_dict},
                                                                                fe_mesh=fe_mesh)

fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                            "beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

fe_mesh.Initialize()
mechanical_loss_3d.Initialize()
fourier_control.Initialize()

# create some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 20
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = {}
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = fourier_control.x_freqs
    export_dict["y_freqs"] = fourier_control.y_freqs
    export_dict["z_freqs"] = fourier_control.z_freqs
    with open(f'fourier_control_dict_box_3d.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'fourier_control_dict_box_3d.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]

K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

export_Ks = False
if export_Ks:
    for i in range(K_matrix.shape[0]):
        fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
    fe_mesh.Finalize(export_dir=case_dir)
    exit()

# specify id of the K of interest
eval_id = 10


# design siren NN for learning
siren_NN = Siren(31,3,[50,50,50])


# create fol optax-based optimizer
# chained_transform = optax.chain(optax.normalize_by_update_norm(),
#                                 optax.adam(1e-5))

chained_transform = optax.chain(optax.normalize_by_update_norm(),optax.scale_by_lbfgs(),optax.scale(-1e-4))

# create fol
fol = ImplicitParametricOperatorLearning(name="dis_fol",control=fourier_control,
                                        loss_function=mechanical_loss_3d,
                                        flax_neural_network=siren_NN,
                                        optax_optimizer=chained_transform,
                                        checkpoint_settings={"restore_state":True,
                                        "state_directory":case_dir+"/flax_state"},
                                        working_directory=case_dir)

fol.Initialize()

# here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
fol.Train(train_set=(coeffs_matrix[eval_id,:].reshape(-1,1).T,),batch_size=100,
            convergence_settings={"num_epochs":2000,"relative_error":1e-100},
            plot_settings={"plot_save_rate":1000},
            save_settings={"save_nn_model":True})


FOL_UV = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
fe_mesh['U_FOL'] = FOL_UV.reshape((fe_mesh.GetNumberOfNodes(), 3))

# solve FE here
fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl","tol":1e-6,"atol":1e-6,
                                            "maxiter":1000,"pre-conditioner":"ilu"},
                "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                            "maxiter":10,"load_incr":5}}
linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_3d,fe_setting)
linear_fe_solver.Initialize()
FE_UV = np.array(linear_fe_solver.Solve(K_matrix[eval_id],np.zeros(3*fe_mesh.GetNumberOfNodes())))  
fe_mesh['U_FE'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 3))

absolute_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
fe_mesh['abs_error'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 3))

fe_mesh.Finalize(export_dir=case_dir)
