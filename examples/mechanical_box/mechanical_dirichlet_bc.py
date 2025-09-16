import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss3DHexa
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from dirichlet_control import DirichletControl3D
from fol.deep_neural_networks.explicit_parametric_operator_learning import ExplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle
import optax
from flax import nnx
import jax

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    
    # directory & save handling
    working_directory_name = "box_3D_hexa"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # create mesh_io
    fe_mesh = create_3D_box_mesh(Nx=11,Ny=11,Nz=11,Lx=1.,Ly=1.,Lz=1.,case_dir=case_dir)
    fe_mesh.Initialize()

    # create fe-based loss function
    bc_dict = {"Ux":{"left":0.0, "right":0.02},
                "Uy":{"left":0.0,"right":-0.04},
                "Uz":{"left":0.0,"right":-0.06}}
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    loss_settings = {"dirichlet_bc_dict":bc_dict,"parametric_boundary_learning":True,"material_dict":material_dict}
    mechanical_loss_3d = MechanicalLoss3DHexa("mechanical_loss_3d",loss_settings=loss_settings,
                                                                                   fe_mesh=fe_mesh)
    mechanical_loss_3d.Initialize()

    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

    dirichlet_control_settings = {}
    dirichlet_control = DirichletControl(control_name='dirichlet_control',control_settings=dirichlet_control_settings, 
                                         fe_mesh= fe_mesh,fe_loss=mechanical_loss_3d)

    
    fourier_control.Initialize()
    dirichlet_control.Initialize()
    
    # # create some random coefficients & K for training
    mean, std, n_samples = 0.1, 0.05, 100
    np.random.seed(42)
    coeffs_matrix = np.random.normal(loc=mean, scale=std, size=(n_samples,3))
    K_matrix = dirichlet_control.ComputeBatchControlledVariables(coeffs_matrix)

    eval_id = -1

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

    fol_net = MLP(dirichlet_control.GetNumberOfVariables(),1, 
                  mechanical_loss_3d.GetNumberOfUnknowns(), 
                  rngs=nnx.Rngs(0))

    # create fol optax-based optimizer
    scheduler = optax.exponential_decay(
        init_value=1e-3,
        transition_steps=10,
        decay_rate=0.99)
    chained_transform = optax.chain(optax.normalize_by_update_norm(), 
                                    optax.scale_by_adam(),
                                    optax.scale_by_schedule(scheduler),
                                    optax.scale(-1.0))

    # create fol
    fol = ExplicitParametricOperatorLearning(name="dis_fol",control=dirichlet_control,
                                             loss_function=mechanical_loss_3d,
                                             flax_neural_network=fol_net,
                                             optax_optimizer=chained_transform)

    fol.Initialize()

    # single sample training for eval_id
    fol.Train(train_set=(coeffs_matrix[eval_id].reshape(-1,1).T,),
              convergence_settings={"num_epochs":fol_num_epochs},
              save_nnx_state_settings={"least_loss_checkpointing":True,"frequency":100,
                                       "state_directory":case_dir,
                                       "save_final_state": False,
                                       "final_state_directory":case_dir},
              working_directory=case_dir)

    # predict for all samples 
    FOL_UVWs = fol.Predict(coeffs_matrix)

    # assign the prediction for  eval_id
    fe_mesh['U_FOL'] = FOL_UVWs[eval_id].reshape((fe_mesh.GetNumberOfNodes(), 3))
    fe_mesh['K'] = np.ones((fe_mesh.GetNumberOfNodes(),1))

    # solve FE here
    if solve_FE:
        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"}}
        
        updated_bc = bc_dict.copy()
        updated_bc.update({"Ux":{"left":0.,"right":coeffs_matrix[eval_id,0]},
                            "Uy":{"left":0.,"right":coeffs_matrix[eval_id,1]},
                            "Uz":{"left":0.,"right":coeffs_matrix[eval_id,2]}})
        updated_loss_setting = loss_settings.copy()
        updated_loss_setting.update({"dirichlet_bc_dict":updated_bc})
        print("loss settings: ",loss_settings)
        print("updated loss settings: ",updated_loss_setting)
        mechanical_loss_3d_updated = MechanicalLoss3DHexa("mechanical_loss_3d",loss_settings=updated_loss_setting,
                                                                                   fe_mesh=fe_mesh)
        mechanical_loss_3d_updated.Initialize()
        first_fe_solver = FiniteElementLinearResidualBasedSolver("first_fe_solver",mechanical_loss_3d_updated,fe_setting)
        first_fe_solver.Initialize()
        FE_UVW = np.array(first_fe_solver.Solve(np.ones(fe_mesh.GetNumberOfNodes()),jnp.ones(3*fe_mesh.GetNumberOfNodes())))  
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