import sys
import os

import optax
import numpy as np

from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from mechanical2D_usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.tools.decoration_functions import *
import pickle
import jax

jax.config.update('jax_default_matmul_precision','high')
# jax.config.update('jax_enable_x64', True)
def main(ifol_num_epochs=10,clean_dir=False):

    if ifol_num_epochs<5000:
        fol_warning(f"ifol_num_epochs is set to {ifol_num_epochs}, recommended value for good results is 5000 !")

    # directory & save handling
    working_directory_name = 'meta_implicit_mechanical_2D'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,"N":41,
                        "Ux_left":0.0,"Ux_right":0.5,
                        "Uy_left":0.0,"Uy_right":0.5}

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
    fe_mesh.Initialize()

    # create fe-based loss function
    bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
            "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}

    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)
    fourier_control.Initialize()

    # create some random coefficients & K for training
    create_random_coefficients = False
    if create_random_coefficients:
        number_of_random_samples = 10
        coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
        export_dict = model_settings.copy()
        export_dict["coeffs_matrix"] = coeffs_matrix
        export_dict["x_freqs"] = fourier_control.x_freqs
        export_dict["y_freqs"] = fourier_control.y_freqs
        export_dict["z_freqs"] = fourier_control.z_freqs
        with open(f'fourier_control_dict_N_21.pkl', 'wb') as f:
            pickle.dump(export_dict,f)
    else:
        with open(f'fourier_control_dict_N_21.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        
        coeffs_matrix = loaded_dict["coeffs_matrix"]

    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    export_Ks = False
    if export_Ks:
        for i in range(K_matrix.shape[0]):
            fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
        fe_mesh.Finalize(export_dir=case_dir)
        exit()

    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    loss_settings={"dirichlet_bc_dict":bc_dict,
                "material_dict":material_dict}
    mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad("mechanical_loss_2d",loss_settings=loss_settings,
                                                                                fe_mesh=fe_mesh)
    mechanical_loss_2d.Initialize()

    # design synthesizer & modulator NN for hypernetwork
    # characteristic_length = model_settings["N"]
    characteristic_length = 64
    synthesizer_nn = MLP(name="synthesizer_nn",
                        input_size=3,
                        output_size=2,
                        hidden_layers=[characteristic_length] * 4,
                        activation_settings={"type":"sin",
                                            "prediction_gain":30,
                                            "initialization_gain":1.0},
                        skip_connections_settings={"active":False,"frequency":1})

    latent_size = 8 * characteristic_length
    modulator_nn = MLP(name="modulator_nn",
                    input_size=latent_size,
                    use_bias=False) 

    hyper_network = HyperNetwork(name="hyper_nn",
                                modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                                coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

    # create fol optax-based optimizer
    #learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
    main_loop_transform = optax.chain(optax.adam(1e-6))
    latent_step_optimizer = optax.chain(optax.adam(1e-5))

    # create fol
    fol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_fol",control=fourier_control,
                                                            loss_function=mechanical_loss_2d,
                                                            flax_neural_network=hyper_network,
                                                            main_loop_optax_optimizer=main_loop_transform,
                                                            latent_step_optax_optimizer=latent_step_optimizer,
                                                            latent_step_size=1e-2,
                                                            num_latent_iterations=3)
    fol.Initialize()

    otf_id = 0
    train_set_otf = coeffs_matrix[otf_id,:].reshape(-1,1).T     # for On The Fly training

    train_start_id = 0
    train_end_id = 8000
    train_set_pr = coeffs_matrix[train_start_id:train_end_id,:]     # for parametric training

    test_start_id = 8000
    test_end_id = 10000

    train_set = train_set_otf   # OTF or Parametric 
    #here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
    fol.Train(train_set=(train_set,),
                test_set=(coeffs_matrix[test_start_id:test_end_id,:],),
                test_frequency=100,
                batch_size=350,
                convergence_settings={"num_epochs":ifol_num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                plot_settings={"plot_save_rate":100},
                train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
                working_directory=case_dir)


    # load teh best model
    fol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

    fe_setting = {"linear_solver_settings":{"solver":"JAX-direct","tol":1e-6,"atol":1e-6,
                                                    "maxiter":1000,"pre-conditioner":"ilu"},
                        "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                                    "maxiter":20,"load_incr":40}}
    
    # train_test = [0,100,220,330,440,550,660,770,880,900,920,990]
    # for test in train_test:
    for test in [otf_id]:
        eval_id = test
        FOL_UV = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
        fe_mesh[f'U_FOL_{eval_id}'] = FOL_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))
        fe_mesh[f'K_{eval_id}'] = K_matrix[eval_id].reshape((fe_mesh.GetNumberOfNodes(), 1))


        ## solve FE here
        linear_fe_solver = FiniteElementNonLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_2d,fe_setting)
        linear_fe_solver.Initialize()
        FE_UV = np.array(linear_fe_solver.Solve(K_matrix[eval_id],jnp.zeros(2*fe_mesh.GetNumberOfNodes())))
        fe_mesh[f'U_FE_{eval_id}'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

        absolute_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
        fe_mesh['abs_error'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 2))


        # plot U
        vectors_list = [K_matrix[eval_id,:],FOL_UV[::2],FE_UV[::2]]
        plot_mesh_res(vectors_list, file_name=f'{case_dir}\plot_U_{eval_id}.png',dir="U")
	
        # plot V
        vectors_list = [K_matrix[eval_id,:],FOL_UV[1::2],FE_UV[1::2]]
        plot_mesh_res(vectors_list, file_name=f'{case_dir}\plot_V_{eval_id}.png',dir="V")


    fe_mesh.Finalize(export_dir=case_dir, export_format='vtk')

    if clean_dir:
        shutil.rmtree(case_dir)   


if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 10000
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
