import sys
import os
import optax
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import fol
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DTetra
from fol.mesh_input_output.mesh import Mesh
from fol.controls.dirichlet_control import DirichletControl
from fol.controls.voronoi_control3D import VoronoiControl3D
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.tools.decoration_functions import *
import pickle

def main(ifol_num_epochs=10,clean_dir=False):

    if ifol_num_epochs<5000:
        fol_warning(f"ifol_num_epochs is set to {ifol_num_epochs}, recommended value for good results is 5000 !")
    # directory & save handling
    working_directory_name = 'meta_implicit_tpms_gyroid_nonlin'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # p# create fe-based loss function
    bc_dict = {"Ux":{"left":0.0,"right":0.00},
                    "Uy":{"left":0.0,"right":0.25},
                    "Uz":{"left":0.0,"right":0.25}}

    # creation of the model
    fe_mesh = Mesh("fol_io","tpms_gyroid_coarse.med",'../meshes')
    fe_mesh.Initialize()

    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    loss_settings={"dirichlet_bc_dict":bc_dict,
                "material_dict":material_dict,
                "parametric_boundary_learning":True}
    mechanical_loss_3d = NeoHookeMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings=loss_settings,fe_mesh=fe_mesh)

    # dirichlet boundary control
    displ_control_settings = {"parametric_boundary_learning": {"Ux":["right"],
                            "Uy":["right"],
                            "Uz":["right"]},
                            "dirichlet_bc_dict":bc_dict}

    displ_control = DirichletControl("displ_control",displ_control_settings,fe_mesh)


    mechanical_loss_3d.Initialize()
    displ_control.Initialize()

    # create some random bcs 
    create_random_coefficients = True
    if create_random_coefficients:
        number_of_random_samples = 200
        bc_matrix,bc_nodal_value_matrix = create_normal_dist_bc_samples(displ_control,
                                                                        numberof_sample=number_of_random_samples,
                                                                        center=0.2,standard_dev=0.1)
        export_dict = {}
        export_dict["bc_matrix"] = bc_matrix
        export_dict["point_bc_settings"] = bc_dict
        export_dict["displ_control_settings"] = displ_control_settings
        with open(f'bc_control_dict.pkl', 'wb') as f:
            pickle.dump(export_dict,f)
    else:
        with open(f'bc_control_dict.pkl', 'rb') as f:
            loaded_control_dict = pickle.load(f)
        
        bc_matrix = loaded_control_dict["bc_matrix"]

    # add intended BC to the end of samples
    wanted_bc = np.array([0.00,0.25,0.25])
    bc_matrix = np.vstack((bc_matrix,wanted_bc))
    print(bc_matrix.shape, "is bc matrix shape")
    bc_nodal_value_matrix = displ_control.ComputeBatchControlledVariables(bc_matrix)

    characteristic_length = 64
    depth = 2
    latent_size_factor = 4

    print(f"characteristic lenght: {characteristic_length} \n depth: {depth} \n latent size: {latent_size_factor*characteristic_length}")
    # design synthesizer & modulator NN for hypernetwork
    synthesizer_nn = MLP(name="synthesizer_nn",
                        input_size=3,
                        output_size=3,
                        hidden_layers=[characteristic_length] * depth,
                        activation_settings={"type":"sin",
                                            "prediction_gain":30,
                                            "initialization_gain":1.0},
                        skip_connections_settings={"active":False,"frequency":1})

    latent_size = latent_size_factor * characteristic_length
    modulator_nn = MLP(name="modulator_nn",
                    input_size=latent_size,
                    use_bias=False) 

    hyper_network = HyperNetwork(name="hyper_nn",
                                modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                                coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

    # create fol optax-based optimizer
    num_epochs = ifol_num_epochs
    learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
    main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))
    latent_step_optimizer = optax.chain(optax.adam(1e-4))

    # create fol
    fol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_fol",control=displ_control,
                                                            loss_function=mechanical_loss_3d,
                                                            flax_neural_network=hyper_network,
                                                            main_loop_optax_optimizer=main_loop_transform,
                                                            latent_step_optax_optimizer=latent_step_optimizer,
                                                            latent_step_size=1e-2,
                                                            num_latent_iterations=3)
    fol.Initialize()


    otf_id = -1
    train_set_otf = bc_matrix[otf_id,:].reshape(-1,1).T

    train_start_id = 0
    train_end_id = 4
    train_set_pr = bc_matrix[train_start_id:train_end_id,:]

    test_start_id = 3*train_end_id
    test_end_id = 4*train_end_id

    train_set = train_set_otf    # OTF or Parametric 

    # here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
    fol.Train(train_set=(train_set,),
                test_set=(bc_matrix[test_start_id:test_end_id,:],),
                test_frequency=100,
                batch_size=1,
                convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                plot_settings={"plot_save_rate":100},
                train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
                working_directory=case_dir)


    # load teh best model
    fol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

    # setting FE parameters here
    fe_setting = {"linear_solver_settings":{"solver":"JAX-direct","tol":1e-6,"atol":1e-6,
                                            "maxiter":1000,"pre-conditioner":"ilu"},
                "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                            "maxiter":20,"load_incr":1}}


    # test_index = np.arange(test_start_id,test_start_id+4)

    for eval_id in [otf_id]:
        FOL_UVW = np.array(fol.Predict(bc_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
        fe_mesh[f'U_FOL_{eval_id}'] = FOL_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))
        fe_mesh[f'K_{eval_id}'] = mechanical_loss_3d.K_matrix.reshape((fe_mesh.GetNumberOfNodes(), 1))
    
        # Update bc dict for FE
        learning_bc = bc_matrix[eval_id,:]
        updated_bc_dict = {"Ux":{"left":0.0,"right":learning_bc[0]},
                    "Uy":{"left":0.0,"right":learning_bc[1]},
                    "Uz":{"left":0.0,"right":learning_bc[2]}}
        bc_dict.update(updated_bc_dict)
        loss_settings.update(bc_dict)


        mechanical_loss_3d = NeoHookeMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings=loss_settings,fe_mesh=fe_mesh)
        mechanical_loss_3d.Initialize()

        linear_fe_solver = FiniteElementNonLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_3d,fe_setting)
        linear_fe_solver.Initialize()

        FE_UVW = np.array(linear_fe_solver.Solve(bc_nodal_value_matrix[eval_id],np.zeros(3*fe_mesh.GetNumberOfNodes())))  
        fe_mesh[f'U_FE_{eval_id}'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

        absolute_error = abs(FOL_UVW.reshape(-1,1)- FE_UVW.reshape(-1,1))
        fe_mesh[f'abs_error_{eval_id}'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 3))

    fe_mesh.Finalize(export_dir=case_dir, export_format="vtu")

    for eval_id in [otf_id]:
        fe_mesh_name=fe_mesh.file_name.split('.')[0]
        plot3D(fe_mesh_name, mesh_file_path=case_dir, fe_vec=f'U_FE_{eval_id}', fol_vec=f'U_FOL_{eval_id}',
            plot_name=f"deformed_mesh_comparison_{eval_id}.png", scale_factor=2.0)

if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 500
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
        elif arg.startswith("clean_dir="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                clean_dir = value.lower() == 'true'
            else:
                print("clean_dir should be True or False.")
                sys.exit(1)
        else:
            print("Usage: python script.py ifol_num_epochs=10 clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(ifol_num_epochs, clean_dir)