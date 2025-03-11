
import numpy as np
import optax
from flax import nnx
import jax
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.mechanical import MechanicalLoss3DTetra
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.explicit_parametric_operator_learning import ExplicitParametricOperatorLearning
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle
from fol.deep_neural_networks.nns import HyperNetwork,MLP

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "implicit_hook_3D"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # create mesh_io
    fe_mesh = Mesh("fol_io","hook.mdpa")

    # create fe-based loss function
    bc_dict = {"Ux":{"support_horizontal_1":0.0,"tip_1":-0.1},
               "Uy":{"support_horizontal_1":0.0},
               "Uz":{"support_horizontal_1":0.0,"tip_1":-0.1}}

    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                   "material_dict":material_dict},
                                                                                   fe_mesh=fe_mesh)

    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)


    fe_mesh.Initialize()
    min_coords = np.min(fe_mesh.GetNodesCoordinates())
    max_coords = np.max(fe_mesh.GetNodesCoordinates())
    fe_mesh.nodes_coordinates = jnp.array((fe_mesh.nodes_coordinates-min_coords)/(max_coords-min_coords))
    fe_mesh.mesh_io = meshio.Mesh(fe_mesh.nodes_coordinates,fe_mesh.elements_nodes)
    mechanical_loss_3d.Initialize()
    fourier_control.Initialize()

    with open(f'fourier_control_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]

    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    eval_id = -1
    fe_mesh['K'] = np.array(K_matrix[eval_id,:])


    # design synthesizer & modulator NN for hypernetwork
    characteristic_length = 64
    synthesizer_nn = MLP(name="synthesizer_nn",
                        input_size=3,
                        output_size=3,
                        hidden_layers=[characteristic_length] * 6,
                        activation_settings={"type":"sin",
                                            "prediction_gain":60,
                                            "initialization_gain":1.0},
                        skip_connections_settings={"active":False,"frequency":1})

    latent_size = 2 * characteristic_length
    modulator_nn = MLP(name="modulator_nn",
                    input_size=latent_size,
                    use_bias=False) 

    hyper_network = HyperNetwork(name="hyper_nn",
                                modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                                coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

    # create fol optax-based optimizer
    num_epochs = fol_num_epochs
    learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
    main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))

    # create fol
    fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=fourier_control,
                                                loss_function=mechanical_loss_3d,
                                                flax_neural_network=hyper_network,
                                                main_loop_optax_optimizer=main_loop_transform,
                                                latent_step_size=1e-2,
                                                num_latent_iterations=3)
    fol.Initialize()

    fol.Train(train_set=(coeffs_matrix[eval_id].reshape(-1,1).T,),batch_size=1,
                convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
                working_directory=case_dir)

    # load teh best model
    fol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

    FOL_UVW = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T)).reshape(-1)
    fe_mesh['U_FOL'] = FOL_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    # solve FE here
    solve_FE = True
    if solve_FE:
        fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                      "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                            "maxiter":10,"load_incr":10}}
        first_fe_solver = FiniteElementLinearResidualBasedSolver("first_fe_solver",mechanical_loss_3d,fe_setting)
        first_fe_solver.Initialize()
        FE_UVW = np.array(first_fe_solver.Solve(K_matrix[eval_id],jnp.ones(3*fe_mesh.GetNumberOfNodes())))  
        fe_mesh['U_FE'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    fol_num_epochs = 2000
    solve_FE = False
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