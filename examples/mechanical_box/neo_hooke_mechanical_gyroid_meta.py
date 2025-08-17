import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DTetra
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
import optax
from flax import nnx
from examples.mechanical_box.mechanical3d_utilities import *

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "box_3D_tetra_nonlin_gyroid"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # create mesh_io
    fe_mesh = Mesh("fol_io","box_3D_coarse.med",'../meshes/')

    # creation of fe model and loss function
    bc_dict = {"Ux":{"left":0.0,"right":0.05},
                "Uy":{"left":0.0},
                "Uz":{"left":0.0}}
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}

    mechanical_loss_3d = NeoHookeMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                   "material_dict":material_dict},
                                                                                   fe_mesh=fe_mesh)

    fe_mesh.Initialize()
    mechanical_loss_3d.Initialize()

    identity_control = IdentityControl('identity_control', num_vars=fe_mesh.GetNumberOfNodes())
    tpms_settings = {"phi_x": 0., "phi_y": 0., "phi_z": 0., "max": 1., "min": 0.1,
                      "fiber_length": 0.4, "fiber_radius": 0.05,
                     "threshold": 0.5, "coefficients":(2.,2.,2.)}
    

    K_matrix = create_gyroid(fe_mesh=fe_mesh, tpms_settings=tpms_settings)

    # fe_mesh["K"] = K_matrix.reshape((fe_mesh.GetNumberOfNodes(),1))
    # fe_mesh.Finalize(export_dir=case_dir, export_format="vtu")
   
    # now we need to create, initialize and train fol
    # design synthesizer & modulator NN for hypernetwork
    # characteristic_length = model_settings["N"]
    characteristic_length = 64
    synthesizer_nn = MLP(name="synthesizer_nn",
                        input_size=3,
                        output_size=3,
                        hidden_layers=[characteristic_length] * 2,
                        activation_settings={"type":"sin",
                                            "prediction_gain":30,
                                            "initialization_gain":1.0},
                        skip_connections_settings={"active":False,"frequency":1})

    latent_size = 1 * characteristic_length
    modulator_nn = MLP(name="modulator_nn",
                    input_size=latent_size,
                    use_bias=False) 

    hyper_network = HyperNetwork(name="hyper_nn",
                                modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                                coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

    # create fol optax-based optimizer
    #learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
    main_loop_transform = optax.chain(optax.adam(1e-5))
    latent_step_optimizer = optax.chain(optax.adam(1e-5))

    # create fol
    ifol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_fol",control=identity_control,
                                                            loss_function=mechanical_loss_3d,
                                                            flax_neural_network=hyper_network,
                                                            main_loop_optax_optimizer=main_loop_transform,
                                                            latent_step_optax_optimizer=latent_step_optimizer,
                                                            latent_step_size=1e-2,
                                                            num_latent_iterations=3)
    ifol.Initialize()

    otf_id = 0
    train_set_otf = K_matrix.reshape(-1,1).T     # for On The Fly training

    train_start_id = 0
    train_end_id = 8000
    train_set_pr = K_matrix     # for parametric training

    test_start_id = 8000
    test_end_id = 10000

    train_set = train_set_otf   # OTF or Parametric 
    #here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
    ifol.Train(train_set=(K_matrix,),
                test_set=(K_matrix,),
                test_frequency=100,
                batch_size=350,
                convergence_settings={"num_epochs":ifol_num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                plot_settings={"plot_save_rate":100},
                train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
                working_directory=case_dir)


    # load teh best model
    ifol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

    FOL_UVW = np.array(ifol.Predict(K_matrix.reshape(-1,1).T)).reshape(-1)
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