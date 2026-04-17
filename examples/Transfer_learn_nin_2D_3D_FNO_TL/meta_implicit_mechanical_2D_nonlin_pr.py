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
from fol.tools.newton_residual_tracker import custom_newton_solve
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
    working_directory_name = '81_meta_implicit_mechanical_2D_new'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,"N":81,
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
        with open(f'fourier_control_dict_ifol.pkl', 'wb') as f:
            pickle.dump(export_dict,f)
    else:
        pkl_path = os.path.join(os.path.dirname(__file__), 'fourier_control_dict_ifol.pkl')
        with open(pkl_path, 'rb') as f:
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
    main_loop_transform = optax.chain(optax.adam(1e-5))
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

    # ---------------------------------------------------------------
    # Load from paper model instead of training
    # Set load_paper_model=True to skip training and use pre-trained weights
    # ---------------------------------------------------------------
    load_paper_model = True
    paper_model_state_dir = os.path.join(
        os.path.dirname(__file__),
        "iFOL_NiN_PAPER_MODEL",
        "iFOL_PARAMETRIC_train_fourier_control",
        "flax_train_state",
    )

    if not load_paper_model:
        train_set = train_set_otf   # OTF or Parametric
        fol.Train(train_set=(train_set,),
                    test_set=(coeffs_matrix[test_start_id:test_end_id,:],),
                    test_frequency=100,
                    batch_size=350,
                    convergence_settings={"num_epochs":ifol_num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                    plot_settings={"plot_save_rate":100},
                    train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
                    working_directory=case_dir)
        fol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")
    else:
        print(f"[INFO] Skipping training. Loading paper model from:\n  {paper_model_state_dir}")
        fol.RestoreState(restore_state_directory=paper_model_state_dir)

    fe_setting = {"linear_solver_settings":{"solver":"JAX-direct","tol":1e-6,"atol":1e-6,
                                                    "maxiter":1000,"pre-conditioner":"ilu"},
                        "nonlinear_solver_settings":{"rel_tol":1e-7,"abs_tol":5e-8,
                                                    "maxiter":20,"load_incr":40}}

    # Warmstart: pass iFOL prediction as FE initial DOFs.
    # Requires load_incr=1 (single full-load Newton solve) because the load-stepper
    # rescales BCs at each step — incompatible with full-load initial DOFs.
    use_ifol_warmstart = True

    import matplotlib.pyplot as plt

    # snapshot defaults so the loop can restore them each iteration
    _default_load_incr = fe_setting["nonlinear_solver_settings"]["load_incr"]
    _default_maxiter   = fe_setting["nonlinear_solver_settings"]["maxiter"]

    train_test = [
        0, 4, 5, 7, 11, 12, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74,
        75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
        94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
        125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
        140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155,
        156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171,
        172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
        187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201,
        202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
    ]
    for eval_id in train_test:
        # per-sample subfolder
        sample_dir = os.path.join(case_dir, f"sample_{eval_id}")
        os.makedirs(sample_dir, exist_ok=True)

        FOL_UV = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
        fe_mesh[f'U_FOL_{eval_id}'] = FOL_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))
        fe_mesh[f'K_{eval_id}'] = K_matrix[eval_id].reshape((fe_mesh.GetNumberOfNodes(), 1))

        ## solve FE here
        # restore defaults in case a previous iteration changed them (e.g. warmstart)
        fe_setting["nonlinear_solver_settings"]["load_incr"] = _default_load_incr
        fe_setting["nonlinear_solver_settings"]["maxiter"]   = _default_maxiter
        if use_ifol_warmstart:
            fe_setting["nonlinear_solver_settings"]["load_incr"] = 1  # single solve at full load
            initial_dofs = FOL_UV
        else:
            initial_dofs = np.zeros(2 * fe_mesh.GetNumberOfNodes())

        nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver", mechanical_loss_2d, fe_setting)
        nonlin_fe_solver.Initialize()
        FE_UV_jax, residuals, total_iters = custom_newton_solve(
            fe_solver=nonlin_fe_solver,
            control_vars=K_matrix[eval_id],
            initial_dofs=initial_dofs,
            case_dir=sample_dir,
            sample_tag=f"sample_{eval_id}",
            stop_only_at_full_load=True,
        )
        FE_UV = np.array(FE_UV_jax).reshape(-1)
        fe_mesh[f'U_FE_{eval_id}'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

        abs_err_uv = np.abs(FOL_UV - FE_UV)
        abs_err_ux = abs_err_uv[0::2]
        abs_err_uy = abs_err_uv[1::2]
        fe_mesh['abs_error'] = abs_err_uv.reshape((fe_mesh.GetNumberOfNodes(), 2))

        L = model_settings["L"]

        # plot Ux: heterogeneity | iFOL_Ux | FE_Ux | abs_error_Ux
        plot_mesh_vec_data(
            L,
            [K_matrix[eval_id, :], FOL_UV[0::2], FE_UV[0::2], abs_err_ux],
            subplot_titles=["Heterogeneity", "iFOL_Ux", "FE_Ux", "absolute_error_Ux"],
            cmap="viridis",
            colour_bar=True,
            show=False,
            file_name=os.path.join(sample_dir, f"plot_Ux_{eval_id}.png"),
        )

        # plot Uy: heterogeneity | iFOL_Uy | FE_Uy | abs_error_Uy
        plot_mesh_vec_data(
            L,
            [K_matrix[eval_id, :], FOL_UV[1::2], FE_UV[1::2], abs_err_uy],
            subplot_titles=["Heterogeneity", "iFOL_Uy", "FE_Uy", "absolute_error_Uy"],
            cmap="viridis",
            colour_bar=True,
            show=False,
            file_name=os.path.join(sample_dir, f"plot_Uy_{eval_id}.png"),
        )

        # plot Newton residuals
        iters = np.arange(1, len(residuals) + 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(iters, residuals, marker="o", markersize=3)
        ax.set_xlabel("Global Newton iteration")
        ax.set_ylabel(r"RMS residual $\|r\|_{\mathrm{rms}}$")
        ax.set_title(f"Newton convergence – sample {eval_id}")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, f"newton_residuals_sample_{eval_id}.png"), dpi=150)
        plt.close()

        # export per-sample VTK
        fe_mesh.Finalize(export_dir=sample_dir, export_format='vtk')

        print(f"[eval_id={eval_id}]  Newton iters={total_iters}  final_residual={residuals[-1]:.3e}")
        print(f"  Ux RMS error={float(np.sqrt(np.mean(abs_err_ux**2))):.4e}  Uy RMS error={float(np.sqrt(np.mean(abs_err_uy**2))):.4e}")
        print(f"  Results saved to: {sample_dir}")

    if clean_dir:
        shutil.rmtree(case_dir)   


if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 1000
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
