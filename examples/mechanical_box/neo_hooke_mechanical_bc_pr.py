import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DTetra
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.controls.dirichlet_control import DirichletControl
from fol.deep_neural_networks.explicit_parametric_operator_learning import ExplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle
import optax
from flax import nnx
import jax

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "box_3D_tetra_nonlin_bc"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # create mesh_io
    fe_mesh = Mesh("fol_io","box_3D_coarse.med",'../meshes/')
    fe_mesh.Initialize()

    # creation of fe model and loss function
    bc_dict = {"Ux":{"left":0.0,"right":0.5},
                "Uy":{"left":0.0,"right":0.15},
                "Uz":{"left":0.0,"right":0.15}}
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    loss_settings = {"dirichlet_bc_dict":bc_dict,"parametric_boundary_learning":True,"material_dict":material_dict}
    mechanical_loss_3d = NeoHookeMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings=loss_settings,
                                                                                   fe_mesh=fe_mesh)
    mechanical_loss_3d.Initialize()

    # dirichlet control
    dirichlet_control_settings = {}
    dirichlet_control = DirichletControl(control_name='dirichlet_control',control_settings=dirichlet_control_settings, 
                                         fe_mesh= fe_mesh,fe_loss=mechanical_loss_3d)
    dirichlet_control.Initialize()

    # create some random coefficients & K for training
    mean, std, n_samples = 0.2, 0.05, 100
    coeffs_matrix = np.random.normal(loc=mean, scale=std, size=(n_samples,3))
    np.random.seed(42)
    ux_comp = np.random.normal(loc=0.4, scale=0.05, size=n_samples).reshape(-1,1)
    uy_comp = np.random.normal(loc=0.1, scale=0.05, size=n_samples).reshape(-1,1)
    uz_comp = np.random.normal(loc=0.1, scale=0.05, size=n_samples).reshape(-1,1)
    coeffs_matrix = np.concatenate((np.concatenate((ux_comp,uy_comp),axis=1),uz_comp),axis=1)
    coeffs_matrix = np.round(coeffs_matrix, 4)

    K_matrix = dirichlet_control.ComputeBatchControlledVariables(coeffs_matrix)

    eval_id = 0
    print("coeff matrix: ", coeffs_matrix[eval_id])

    # now we need to create, initialize and train fol
    # design NN for learning
    class MLP(nnx.Module):
        def __init__(self, in_features: int, dmid: int, out_features: int, *, rngs: nnx.Rngs):
            self.dense1 = nnx.Linear(in_features, dmid, rngs=rngs,kernel_init=nnx.initializers.normal(0.01),bias_init=nnx.initializers.normal(0.01))
            self.dense2 = nnx.Linear(dmid, out_features, rngs=rngs,kernel_init=nnx.initializers.normal(0.01),bias_init=nnx.initializers.normal(0.01))
            self.in_features = in_features
            self.out_features = out_features

        def __call__(self, x: jax.Array) -> jax.Array:
            x = self.dense1(x)
            x = jax.nn.tanh(x)
            x = self.dense2(x)
            return x

    fol_net = MLP(dirichlet_control.GetNumberOfVariables(),10, 
                  mechanical_loss_3d.GetNumberOfUnknowns(), 
                  rngs=nnx.Rngs(0))

    # create fol optax-based optimizer
    chained_transform = optax.chain(optax.normalize_by_update_norm(), 
                                    optax.adam(1e-3))
    
    fol = ExplicitParametricOperatorLearning(name="dis_fol",control=dirichlet_control,
                                             loss_function=mechanical_loss_3d,
                                             flax_neural_network=fol_net,
                                             optax_optimizer=chained_transform)
    fol.Initialize()

    # single sample training for eval_id
    fol.Train(train_set=(coeffs_matrix[eval_id].reshape(-1,1).T,),
              convergence_settings={"num_epochs":fol_num_epochs,
                                    "relative_error":1e-10},
              save_nnx_state_settings={"least_loss_checkpointing":True,"frequency":100,
                                       "state_directory":case_dir,
                                       "save_final_state": False,
                                       "final_state_directory":case_dir},
              working_directory=case_dir)

    FOL_UVW = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T)).reshape(-1)
    print("coeff matrix in fol.Predict: ", coeffs_matrix[eval_id])
    fe_mesh['U_FOL'] = FOL_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))
    fe_mesh['K'] = np.ones((fe_mesh.GetNumberOfNodes(),1))

    # solve FE here
    if solve_FE:
        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"},
                      "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                                    "maxiter":5,"load_incr":10}}
        updated_bc = bc_dict.copy()
        updated_bc.update({"Ux":{"left":0.,"right":coeffs_matrix[eval_id,0]},
                            "Uy":{"left":0.,"right":coeffs_matrix[eval_id,1]},
                            "Uz":{"left":0.,"right":coeffs_matrix[eval_id,2]}})
        print("updated bc: ",updated_bc)
        updated_loss_setting = loss_settings.copy()
        updated_loss_setting.update({"dirichlet_bc_dict":updated_bc})
        mechanical_loss_3d_updated = NeoHookeMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings=updated_loss_setting,
                                                                                   fe_mesh=fe_mesh)
        mechanical_loss_3d_updated.Initialize()
        nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_3d_updated,fe_setting)
        nonlin_fe_solver.Initialize()
        FE_UVW = np.array(nonlin_fe_solver.Solve(np.ones(fe_mesh.GetNumberOfNodes()),np.zeros(3*fe_mesh.GetNumberOfNodes())))  
        fe_mesh['U_FE'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    fe_mesh.Finalize(export_dir=case_dir,export_format='vtu')

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    fol_num_epochs = 2000
    solve_FE = True
    clean_dir = False

    # Parse the command-line arguments
    args = sys.argv[1:]

    # Process the arguments if provided
    for arg in args:
        if arg.startswith("fol_num_epochs="):
            try:
                fol_num_epochs = int(arg.split("=")[1])
            except ValueError:
                print("fol_num_epochs should be an integer.")
                sys.exit(1)
        elif arg.startswith("solve_FE="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                solve_FE = value.lower() == 'true'
            else:
                print("solve_FE should be True or False.")
                sys.exit(1)
        elif arg.startswith("clean_dir="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                clean_dir = value.lower() == 'true'
            else:
                print("clean_dir should be True or False.")
                sys.exit(1)
        else:
            print("Usage: python thermal_fol.py fol_num_epochs=10 solve_FE=False clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(fol_num_epochs, solve_FE,clean_dir)