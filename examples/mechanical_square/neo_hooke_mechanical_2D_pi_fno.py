import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..',)))
import optax
from flax import nnx
import jax
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearning
from fol.deep_neural_networks.fourier_neural_operator_networks import FourierNeuralOperator2D
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.tools.newton_residual_tracker import custom_newton_solve
import pickle, time
from flax.nnx import bridge

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False,use_nin=False):
    # directory & save handling
    working_directory_name = 'nn_output_mechanical_2D_neohooke_pi_fno'
    case_dir = os.path.join('.', working_directory_name)
    
    # If NIN is enabled, don't clean the directory (we need to load the model)
    if not use_nin:
        create_clean_directory(working_directory_name)
    else:
        # Create directory if it doesn't exist, but don't clean it
        os.makedirs(case_dir, exist_ok=True)
    
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))
    print(f"Running with use_nin={use_nin}")

    # problem setup
    model_settings = {"L":1,"N":42,
                    "Ux_left":0.0,"Ux_right":0.1,
                    "Uy_left":0.0,"Uy_right":0.1}

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

    # create fe-based loss function
    bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
               "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}
    
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                              "num_gp":2,
                                                                              "material_dict":material_dict},
                                                                              fe_mesh=fe_mesh)

    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

    fe_mesh.Initialize()
    mechanical_loss_2d.Initialize()
    fourier_control.Initialize()

    # create some random coefficients & K for training
    create_random_coefficients = False
    if create_random_coefficients:
        number_of_random_samples = 200
        coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
        export_dict = model_settings.copy()
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
    
    n_samples = coeffs_matrix.shape[0]
    print(f"Total samples available: {n_samples}")

    dofs = mechanical_loss_2d.GetDOFs()
    def merge_state(dst: nnx.State, src: nnx.State):
        for k, v in src.items():
            if isinstance(v, nnx.State):
                merge_state(dst[k], v)
            else:
                dst[k] = v

    fno_model = bridge.ToNNX(FourierNeuralOperator2D(modes1=6,
                                                    modes2=6,
                                                    width=8,
                                                    depth=4,
                                                    channels_last_proj=32,
                                                    out_channels=2,
                                                    output_scale=0.1),rngs=nnx.Rngs(0)).lazy_init(K_matrix[0:1].reshape(1,model_settings["N"],model_settings["N"],1)) 

    # replace RNG key by a dummy to allow checkpoint restoration later
    graph_def, state = nnx.split(fno_model)
    rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
    merge_state(state, rngs_key)
    fno_model = nnx.merge(graph_def, state)

    # get total number of fno params
    params = nnx.state(fno_model, nnx.Param)
    total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print(f"total number of fno network param:{total_params}")

    num_epochs = 1000
    learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-5, transition_steps=num_epochs)
    optimizer = optax.chain(optax.adam(1e-3))

    # create fol
    pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(name="pi_fno_pr_learning",
                                                                            control=fourier_control,
                                                                            loss_function=mechanical_loss_2d,
                                                                            flax_neural_network=fno_model,
                                                                            optax_optimizer=optimizer)

    pi_fno_pr_learning.Initialize()

    # Adjust indices based on available data
    n_samples = coeffs_matrix.shape[0]
    
    otf_id = 0
    train_start_id = 0
    train_end_id = min(10, n_samples)  # Use first 10 samples or less if not available
    test_start_id = max(0, n_samples - 2)  # Use last 2 samples
    test_end_id = n_samples
    
    print(f"Train range: [{train_start_id}:{train_end_id}], Test range: [{test_start_id}:{test_end_id}]")
    
    # Parametric learning or On The Fly learning
    parametric_learning = True
    if parametric_learning:
        train_set = coeffs_matrix[train_start_id:train_end_id,:]
        test_set = coeffs_matrix[test_start_id:test_end_id,:]
        eval_cases = range(test_start_id,test_end_id)
        batch_size = 5
    else:
        train_set = coeffs_matrix[otf_id,:].reshape(-1,1).T
        test_set = coeffs_matrix[otf_id,:].reshape(-1,1).T
        eval_cases = [otf_id]
        batch_size = 1

    # Train only if NIN is not enabled (first run)
    if not use_nin:
        print("[Training Phase] Training PI-FNO model...")
        pi_fno_pr_learning.Train(train_set=(train_set,),
                                test_set=(test_set,),
                                test_frequency=100,
                                batch_size=batch_size,
                                convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                                plot_settings={"plot_save_rate":100},
                                train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
                                working_directory=case_dir)
        print("[Training Phase] Training complete.")
    else:
        print("[NIN Mode] Skipping training, will load pre-trained model for warm-start...")

    # Load the best model (either just trained or from previous run)
    model_path = case_dir+"/flax_train_state"
    if os.path.exists(model_path):
        print(f"[Loading] Restoring model from {model_path}")
        pi_fno_pr_learning.RestoreState(restore_state_directory=model_path)
        print("[Loading] Model successfully restored.")
        
        # Verify model works by testing prediction on first sample
        if use_nin:
            test_pred = pi_fno_pr_learning.Predict(coeffs_matrix[0,:].reshape(-1,1).T)
            print(f"[Verification] Model prediction test passed. Output shape: {np.array(test_pred).shape}")
    else:
        print(f"[Warning] Model path {model_path} not found!")
        if use_nin:
            print("[Error] NIN mode requires a pre-trained model. Please run without use_nin first.")
            return


    for eval_id in eval_cases:
        print(f"\n{'='*60}")
        print(f"Processing sample {eval_id}")
        print(f"{'='*60}")
        
        # Validate control variables
        K_current = K_matrix[eval_id]
        if not np.all(np.isfinite(K_current)):
            print(f"[Error-{eval_id}] K_matrix contains NaN/Inf values. Skipping sample.")
            continue
        
        print(f"[Info-{eval_id}] K_matrix stats: min={np.min(K_current):.3e}, max={np.max(K_current):.3e}, mean={np.mean(K_current):.3e}")
        
        FNO_UV = np.array(pi_fno_pr_learning.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
        
        # Validate FNO prediction
        if not np.all(np.isfinite(FNO_UV)):
            print(f"[Warning-{eval_id}] FNO prediction contains NaN/Inf values. Using zero initial guess.")
            FNO_UV = np.zeros_like(FNO_UV)
        
        fe_mesh[f'U_FNO_{eval_id}'] = FNO_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

        # solve FE here - Use custom_newton_solve for robustness in both cases
        fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                                    "maxiter":1000,"pre-conditioner":"ilu"},
                      "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                    "maxiter":8,"load_incr":21}}
        nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_2d,fe_setting)
        nonlin_fe_solver.Initialize()
        
        if use_nin:
            # NIN mode: Use FNO prediction as warm-start for Newton solver
            print(f"[NIN-{eval_id}] Using FNO prediction as initial guess for Newton solver...")
            initial_dofs = jax.numpy.array(FNO_UV)
            sample_tag = f"nin_sample_{eval_id}"
        else:
            # Baseline mode: Use zero initial guess with custom solver for robustness
            print(f"[Baseline-{eval_id}] Solving from zero initial guess with custom Newton solver...")
            initial_dofs = jax.numpy.zeros(2*fe_mesh.GetNumberOfNodes())
            sample_tag = f"baseline_sample_{eval_id}"
        
        try:
            FE_UV, residuals_rms, total_iters = custom_newton_solve(
                fe_solver=nonlin_fe_solver,
                control_vars=K_matrix[eval_id],
                initial_dofs=initial_dofs,
                case_dir=case_dir,
                sample_tag=sample_tag,
                target_best=1e-6,
                growth_tol=50.0,
                rmsmean_window=5,
                target_rmsmean=1e-6,
                stop_only_at_full_load=True,
                use_line_search=True,
                guard_return_mode="best",
                return_run_info=False
            )
            FE_UV = np.array(FE_UV)
            print(f"[Success-{eval_id}] Newton solver completed: {total_iters} iterations, final_rms={residuals_rms[-1]:.3e}")
            
            # Note: Plot is already generated inside custom_newton_solve, no need to plot again
            
        except Exception as e:
            print(f"[Error-{eval_id}] Newton solver failed: {e}")
            print(f"[Error-{eval_id}] Skipping this sample.")
            continue  

        fe_mesh[f'U_FE_{eval_id}'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

        absolute_error = abs(FNO_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
        fe_mesh[f'abs_error_{eval_id}'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 2))
        
        plot_mesh_vec_data(model_settings["L"], [K_matrix[eval_id,:],FNO_UV[::2],FE_UV[::2],absolute_error[::2]], 
                        subplot_titles= ['Heterogeneity', 'FNO_U', 'FE_U', "absolute_error"], fig_title=None, cmap='viridis',
                            block_bool=True, colour_bar=True, colour_bar_name=None,
                            X_axis_name=None, Y_axis_name=None, show=False, file_name=os.path.join(case_dir,f'plot_results_{eval_id}.png'))
    
    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    fol_num_epochs = 2000
    solve_FE = False
    clean_dir = False
    use_nin = True

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
        elif arg.startswith("use_nin="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                use_nin = value.lower() == 'true'
            else:
                print("use_nin should be True or False.")
                sys.exit(1)
        else:
            print("Usage: python neo_hooke_mechanical_2D_pi_fno.py fol_num_epochs=10 solve_FE=False clean_dir=False use_nin=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(fol_num_epochs, solve_FE, clean_dir, use_nin)