import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DTetra
from fol.loss_functions.mechanical_mooney_rivlin_voigt import MooneyRivlinMechanicalLoss3DTetra
from fol.loss_functions.mechanical import MechanicalLoss3DTetra
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.dirichlet_control import DirichletControl3D
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
import optax
from flax import nnx
from examples.mechanical_box.mechanical3d_utilities import *

def main(ifol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "3D_tetra_nonlin_meta_alpha_pr"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    #call the function to create the mesh
    fe_mesh = create_cube_with_spheres_mesh(
        num_spheres=9,
        Lx=1.0, Ly=1.0, Lz=1.0,
        case_dir=f"{case_dir}/shwartz",
        min_radius=0.35, max_radius=0.45,
        mesh_size_min=0.04,   # finer elements near features
        mesh_size_max=0.08    # coarser elements elsewhere
    )
    fe_mesh.Initialize()

    # creation of fe model and loss function
    bc_dict = {"Ux":{"left":0.0,"right":0.5},
                "Uy":{"left":0.0,"right":0.15},
                "Uz":{"left":0.0,"right":0.15}}
    material_dict = {"young_modulus":1,"poisson_ratio":0.3,"c10":0.2588,"c01":-0.0449}
    loss_settings = {"dirichlet_bc_dict":bc_dict,"parametric_boundary_learning":True,"material_dict":material_dict}
    mechanical_loss_3d = MooneyRivlinMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings=loss_settings,
                                                                                   fe_mesh=fe_mesh)

    # fe_mesh.Initialize()
    mechanical_loss_3d.Initialize()

    # identity_control = IdentityControl('identity_control', control_settings={}, fe_mesh=fe_mesh)
    tpms_settings = {"phi_x": 0., "phi_y": 0., "phi_z": 0., "max": 1., "min": 0.1,
                      "fiber_length": 0.4, "fiber_radius": 0.05,
                     "threshold": 0.5, "coefficients":(2.,2.,2.)}
    
     # dirichlet control
    dirichlet_control_settings = {}
    dirichlet_control = DirichletControl3D(control_name='dirichlet_control',control_settings=dirichlet_control_settings, 
                                         fe_mesh= fe_mesh,fe_loss=mechanical_loss_3d)
    dirichlet_control.Initialize()

    # create some random coefficients & K for training
    mean, std, n_samples = 0.2, 0.05, 200
    #coeffs_matrix = np.random.normal(loc=mean, scale=std, size=(n_samples,3))
    np.random.seed(42)
    ux_comp = np.random.normal(loc=0.25, scale=0.05, size=n_samples).reshape(-1,1)
    uy_comp = np.random.normal(loc=0.05, scale=0.02, size=n_samples).reshape(-1,1)
    uz_comp = np.random.normal(loc=0.05, scale=0.02, size=n_samples).reshape(-1,1)
    coeffs_matrix = np.concatenate((np.concatenate((ux_comp,uy_comp),axis=1),uz_comp),axis=1)

    K_matrix = dirichlet_control.ComputeBatchControlledVariables(coeffs_matrix)
    
   
    # now we need to create, initialize and train fol
    ifol_settings_dict = {
        "characteristic_length": 1*64,
        "synthesizer_depth": 4,
        "activation_settings":{"type":"sin",
                                "prediction_gain":130.0,
                                "initialization_gain":1.0},
        "skip_connections_settings": {"active":False,"frequency":1},
        "latent_size":  1*64,
        "modulator_bias": False,
        "main_loop_transform": 1e-5,
        "latent_step_optimizer": 1e-4,
        "ifol_nn_latent_step_size": 1e-4
    }
    
    characteristic_length = ifol_settings_dict["characteristic_length"]
    synthesizer_nn = MLP(name="synthesizer_nn",
                        input_size=3,
                        output_size=3,
                        hidden_layers=[characteristic_length] * ifol_settings_dict["synthesizer_depth"],
                        activation_settings=ifol_settings_dict["activation_settings"],
                        skip_connections_settings=ifol_settings_dict["skip_connections_settings"])

    latent_size = ifol_settings_dict["latent_size"]
    modulator_nn = MLP(name="modulator_nn",
                    input_size=latent_size,
                    use_bias=ifol_settings_dict["modulator_bias"]) 

    hyper_network = HyperNetwork(name="hyper_nn",
                                modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                                coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

    # create fol optax-based optimizer
    #learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
    main_loop_transform = optax.chain(optax.adam(ifol_settings_dict["main_loop_transform"]))
    latent_step_optimizer = optax.chain(optax.adam(ifol_settings_dict["latent_step_optimizer"]))

     # create fol
    meta_alpha = True
    if meta_alpha:
        ifol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_fol",control=dirichlet_control,
                                                            loss_function=mechanical_loss_3d,
                                                            flax_neural_network=hyper_network,
                                                            main_loop_optax_optimizer=main_loop_transform,
                                                            latent_step_optax_optimizer=latent_step_optimizer,
                                                            latent_step_size=ifol_settings_dict["ifol_nn_latent_step_size"],
                                                            num_latent_iterations=3)
    else:
        ifol = MetaImplicitParametricOperatorLearning(name="meta_implicit_fol",control=dirichlet_control,
                                                            loss_function=mechanical_loss_3d,
                                                            flax_neural_network=hyper_network,
                                                            main_loop_optax_optimizer=main_loop_transform,
                                                            latent_step_size=ifol_settings_dict["ifol_nn_latent_step_size"],
                                                            num_latent_iterations=3)
    ifol.Initialize()

    eval_id = 13
    train_set_otf = coeffs_matrix[eval_id,:].reshape(-1,1).T     # for On The Fly training
    

    train_start_id = 0
    train_end_id = 100
    # train_set_pr = K_matrix[train_start_id,train_end_id]     # for parametric training
    train_set_pr = coeffs_matrix[train_start_id:train_end_id,:]

    test_start_id = 160
    test_end_id = 199
    # test_set_pr = K_matrix[test_start_id,test_end_id]
    test_set_pr = coeffs_matrix[test_start_id:test_end_id,:]
    
    train_settings_dict = {"batch_size": 20,
                            "num_epoch":ifol_num_epochs,
                            "parametric_learning": False,
                            "OTF_id": eval_id,
                            "train_start_id": train_start_id,
                            "train_end_id": train_end_id,
                            "test_start_id": test_start_id,
                            "test_end_id": test_end_id}

    # OTF or Parametric 
    if train_settings_dict["parametric_learning"]:
        train_set = train_set_pr
        test_set = test_set_pr
    else:
        train_set = train_set_otf   
        test_set = train_set
    #here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
    print(f"\ncheck...\tBCs for sample no. {eval_id}: {coeffs_matrix[eval_id,:]}")
    print(f"\ncheck...\tParametric learning: {train_settings_dict['parametric_learning']}")
    print(f"\ncheck...\ttraining sample ids: {train_start_id} -> {train_end_id}\n")

    # ifol.Train(train_set=(train_set,),
    #             test_set=(test_set,),
    #             test_frequency=100,
    #             batch_size=train_settings_dict["batch_size"],
    #             convergence_settings={"num_epochs":train_settings_dict["num_epoch"],"relative_error":1e-100,"absolute_error":1e-100},
    #             plot_settings={"plot_save_rate":100},
    #             train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
    #             working_directory=case_dir)


    # # load teh best model
    # ifol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

    if train_settings_dict["parametric_learning"]:
        FOL_UVW = np.array(ifol.Predict(test_set))
    else:    
        FOL_UVW = np.array(ifol.Predict(train_set)).reshape(-1)
        fe_mesh['U_FOL'] = FOL_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))
        # fe_mesh["K"] = train_set.reshape((fe_mesh.GetNumberOfNodes(),1))

        
    for eval_id in range(5):
        # FOL_UVW = np.array(ifol.Predict(train_set[eval_id,:].reshape(-1,1).T)).reshape(-1)
        # fe_mesh[f'U_FOL_{eval_id}'] = FOL_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))
        # solve FE here
        updated_bc = bc_dict.copy()
        updated_bc.update({"Ux":{"left":0.,"right":coeffs_matrix[eval_id,0]},
                            "Uy":{"left":0.,"right":coeffs_matrix[eval_id,1]},
                            "Uz":{"left":0.,"right":coeffs_matrix[eval_id,2]}})

        updated_loss_setting = loss_settings.copy()
        updated_loss_setting.update({"dirichlet_bc_dict":updated_bc})
        mechanical_loss_3d_updated = MooneyRivlinMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings=updated_loss_setting,
                                                                                    fe_mesh=fe_mesh)
        mechanical_loss_3d_updated.Initialize()
        # try:
        hfe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                                "maxiter":8,"load_incr":20}}
        nonlin_hfe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_3d_updated,hfe_setting)
        nonlin_hfe_solver.Initialize()
        # HFE_UVW = np.array(nonlin_hfe_solver.Solve(np.ones(fe_mesh.GetNumberOfNodes()),FOL_UVW.reshape(3*fe_mesh.GetNumberOfNodes())))
        HFE_UVW = np.array(nonlin_hfe_solver.Solve(np.ones(fe_mesh.GetNumberOfNodes()),np.zeros(3*fe_mesh.GetNumberOfNodes())))
        # except:
        #     ValueError('res_norm contains nan values!')
        #     HFE_UVW = np.zeros(3*fe_mesh.GetNumberOfNodes())
        fe_mesh[f'U_HFE_{eval_id}'] = HFE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))
        # abs_err = abs(HFE_UVW.reshape(-1,1) - FOL_UVW.reshape(-1,1))
        # fe_mesh[f"abs_error_{eval_id}"] = abs_err.reshape((fe_mesh.GetNumberOfNodes(), 3))

    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 5000
    solve_FE = True
    clean_dir = False

    # Parse the command-line arguments
    args = sys.argv[1:]

    # Process the arguments if provided
    for arg in args:
        if arg.startswith("ifol_num_epochs="):
            try:
                ifol_num_epochs = int(arg.split("=")[1])
            except ValueError:
                print("ifol_num_epochs should be an integer.")
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
            print("Usage: python thermal_fol.py ifol_num_epochs=10 solve_FE=False clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(ifol_num_epochs, solve_FE,clean_dir)
