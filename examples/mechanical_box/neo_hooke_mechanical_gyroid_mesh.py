import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DTetra
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
import optax
from flax import nnx
import jax

def main(ifol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "box_3D_tetra_nonlin"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # create mesh_io
    fe_mesh = Mesh("fol_io","cube_schwarzP_coarse.med",'../meshes/')

    # creation of fe model and loss function
    bc_dict = {"Ux":{"left":0.0,"right":0.05},
                "Uy":{"left":0.0,"right":0.05},
                "Uz":{"left":0.0,"right":0.05}}
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}

    mechanical_loss_3d = NeoHookeMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                   "material_dict":material_dict},
                                                                                   fe_mesh=fe_mesh)

    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

    fe_mesh.Initialize()
    mechanical_loss_3d.Initialize()
    fourier_control.Initialize()

    # create some random coefficients & K for training
    create_random_coefficients = False
    if create_random_coefficients:
        number_of_random_samples = 200
        coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
        export_dict = {}
        export_dict["coeffs_matrix"] = coeffs_matrix
        export_dict["x_freqs"] = fourier_control.x_freqs
        export_dict["y_freqs"] = fourier_control.y_freqs
        export_dict["z_freqs"] = fourier_control.z_freqs
        with open(f'fourier_control_dict.pkl', 'wb') as f:
            pickle.dump(export_dict,f)
    else:
        with open(f'fourier_control_dict.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        
        coeffs_matrix = loaded_dict["coeffs_matrix"]

    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    # now save K matrix 
    export_Ks = False
    if export_Ks:
        for i in range(K_matrix.shape[0]):
            fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
        fe_mesh.Finalize(export_dir=case_dir)

    eval_id = -1
    fe_mesh['K'] = np.array(K_matrix[eval_id,:])

    # design synthesizer & modulator NN for hypernetwork
    characteristic_length = 64
    synthesizer_nn = MLP(name="synthesizer_nn",
                        input_size=3,
                        output_size=3,
                        hidden_layers=[characteristic_length] * 2,
                        activation_settings={"type":"sin",
                                            "prediction_gain":95,
                                            "initialization_gain":1.0},
                        skip_connections_settings={"active":False,"frequency":1})

    latent_size = 16
    modulator_nn = MLP(name="modulator_nn",
                    input_size=latent_size,
                    use_bias=False) 

    hyper_network = HyperNetwork(name="hyper_nn",
                                modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                                coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

    # create fol optax-based optimizer
    num_epochs = ifol_num_epochs
    # learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
    # main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))
    main_loop_transform = optax.chain(optax.adam(1e-5))
    latent_step_optimizer = optax.chain(optax.adam(1e-4))

    # create fol
    fol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_fol",control=fourier_control,
                                                            loss_function=mechanical_loss_3d,
                                                            flax_neural_network=hyper_network,
                                                            main_loop_optax_optimizer=main_loop_transform,
                                                            latent_step_optax_optimizer=latent_step_optimizer,
                                                            latent_step_size=1e-2,
                                                            num_latent_iterations=3)
    fol.Initialize()

    train_start_id = 0
    train_end_id = 20
    test_start_id = 3*train_end_id
    test_end_id = 3*train_end_id + 2
    train_otf_index = -1
    # here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
    fol.Train(train_set=(coeffs_matrix[train_otf_index,:].reshape(-1,1).T,),
            test_set=(coeffs_matrix[test_start_id:test_end_id,:],),
            test_frequency=10,
            batch_size=1,
            convergence_settings={"num_epochs":num_epochs,
                                    "relative_error":1e-100,
                                    "absolute_error":1e-100},
            working_directory=case_dir)

    # load teh best model
    fol.RestoreState(restore_state_directory=case_dir+"/flax_final_state")

    for test in [train_otf_index]:
        eval_id = test
        FOL_UV = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
        fe_mesh['U_FOL'] = FOL_UV.reshape((fe_mesh.GetNumberOfNodes(), 3))

        # solve FE here
        fe_setting = {"linear_solver_settings":{"solver":"JAX-direct","tol":1e-6,"atol":1e-6,
                                                    "maxiter":1000,"pre-conditioner":"ilu"},
                        "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                    "maxiter":10,"load_incr":2}}
        nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlinear_fe_solver",mechanical_loss_3d,fe_setting)
        nonlinear_fe_solver.Initialize()
        FE_UV = np.array(nonlinear_fe_solver.Solve(K_matrix[eval_id],np.zeros(3*fe_mesh.GetNumberOfNodes())))  
        fe_mesh['U_FE'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 3))

        absolute_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
        fe_mesh['abs_error'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 3))

    fe_mesh.Finalize(export_dir=case_dir, export_format="vtu")

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 2000
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
    main(ifol_num_epochs, solve_FE,clean_dir)