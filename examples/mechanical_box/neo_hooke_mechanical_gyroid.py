import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DTetra
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.explicit_parametric_operator_learning import ExplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle
import optax
from flax import nnx
import jax

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "box_3D_tetra_nonlin_gyroid"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # create mesh_io
    fe_mesh = Mesh("fol_io","box_3D_coarse.med",'../meshes/')

    # creation of fe model and loss function
    bc_dict = {"Ux":{"left":0.0,"right":0.5},
                "Uy":{"left":0.0},
                "Uz":{"left":0.0}}
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}

    mechanical_loss_3d = NeoHookeMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                   "material_dict":material_dict},
                                                                                   fe_mesh=fe_mesh)

    fe_mesh.Initialize()
    mechanical_loss_3d.Initialize()

    identity_control = IdentityControl('identity_control', num_vars=fe_mesh.GetNumberOfNodes())
    tpms_settings = {"phi_x": 0, "phi_y": 0, "phi_z": 0,
                     "constant": 0, "threshold": 0.5, "coefficients":(2,2,2)}
    K_matrix = create_gyroid(fe_mesh=fe_mesh, tpms_settings=tpms_settings)


    # now we need to create, initialize and train fol
    # design NN for learning
    class MLP(nnx.Module):
        def __init__(self, in_features: int, dmid: int, out_features: int, *, rngs: nnx.Rngs):
            self.dense1 = nnx.Linear(in_features, dmid, rngs=rngs,kernel_init=nnx.initializers.zeros,bias_init=nnx.initializers.zeros)
            self.dense2 = nnx.Linear(dmid, out_features, rngs=rngs,kernel_init=nnx.initializers.zeros,bias_init=nnx.initializers.zeros)
            self.in_features = in_features
            self.out_features = out_features

        def __call__(self, x: jax.Array) -> jax.Array:
            x = self.dense1(x)
            x = jax.nn.tanh(x)
            x = self.dense2(x)
            return x

    fol_net = MLP(identity_control.GetNumberOfVariables(),1, 
                  mechanical_loss_3d.GetNumberOfUnknowns(), 
                  rngs=nnx.Rngs(0))

    # create fol optax-based optimizer
    chained_transform = optax.chain(optax.normalize_by_update_norm(), 
                                    optax.adam(1e-3))
    
    fol = ExplicitParametricOperatorLearning(name="dis_fol",control=identity_control,
                                             loss_function=mechanical_loss_3d,
                                             flax_neural_network=fol_net,
                                             optax_optimizer=chained_transform)
    fol.Initialize()

    # single sample training for eval_id
    fol.Train(train_set=(K_matrix.reshape(-1,1).T,),
              convergence_settings={"num_epochs":fol_num_epochs,
                                    "relative_error":1e-10},
              working_directory=case_dir)

    FOL_UVW = np.array(fol.Predict(K_matrix.reshape(-1,1).T)).reshape(-1)
    fe_mesh['U_FOL'] = FOL_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))
    fe_mesh["K"] = K_matrix.reshape((fe_mesh.GetNumberOfNodes(),1))

    # solve FE here
    if solve_FE:
        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"},
                      "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                                    "maxiter":5,"load_incr":10}}
        nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_3d,fe_setting)
        nonlin_fe_solver.Initialize()
        FE_UVW = np.array(nonlin_fe_solver.Solve(K_matrix,np.zeros(3*fe_mesh.GetNumberOfNodes())))  
        fe_mesh['U_FE'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    fe_mesh.Finalize(export_dir=case_dir)

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