import sys
import os
import optax
import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss2DQuad
from fol.controls.fourier_control import FourierControl
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.responses.fe_response import FiniteElementResponse
from fol.solvers.adjoint_fe_solver import AdjointFiniteElementSolver
from fol.tools.decoration_functions import *
import pickle
import jax
import jax.numpy as jnp

def main(mesh_size=11,clean_dir=False):
    # directory & save handling
    working_directory_name = str(os.path.basename(__file__).split(".")[0])
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,"N":mesh_size,
                        "Ux_left":0.0,"Ux_right":0.05,
                        "Uy_left":0.0,"Uy_right":0.05}

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

    # create fe-based loss function
    bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
            "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}

    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    mechanical_loss_2d = MechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "num_gp":2,
                                                                                "material_dict":material_dict},
                                                                                fe_mesh=fe_mesh)

    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("E",fourier_control_settings,fe_mesh)


    myresponse = FiniteElementResponse("my_response",response_formula="(E**2)*U[0]",fe_loss=mechanical_loss_2d,control=fourier_control)

    fe_mesh.Initialize()
    mechanical_loss_2d.Initialize()
    fourier_control.Initialize()
    myresponse.Initialize()


    with open(f'fourier_control_dict_2D.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    coeffs_matrix = loaded_dict["coeffs_matrix"]

    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)


    eval_id = 0
    # solve FE here
    fe_setting = {"linear_solver_settings":{"solver":"JAX-direct","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu"},
                    "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                "maxiter":10,"load_incr":5}}
    linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_2d,fe_setting)
    linear_fe_solver.Initialize()
    FE_UV = np.array(linear_fe_solver.Solve(K_matrix[eval_id],np.zeros(2*fe_mesh.GetNumberOfNodes())))  
    fe_mesh['U_FE'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

    fol_info(f"myresponse.ComputeValue:{myresponse.ComputeValue(K_matrix[eval_id],FE_UV)}")

    plot_mesh_vec_data(1,[K_matrix[eval_id,:],FE_UV[0::2],FE_UV[1::2]],
                    ["K","U","V"],
                    fig_title="conductivity and FEM solution",
                    file_name=os.path.join(case_dir,f"FEM-KUV-dist_test_{eval_id}.png"))


    adj_fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"}}
    first_adj_fe_solver = AdjointFiniteElementSolver("first_adj_fe_solver",myresponse,adj_fe_setting)
    first_adj_fe_solver.Initialize()

    FE_adj_UV = first_adj_fe_solver.Solve(K_matrix[eval_id],
                            FE_UV,
                            jnp.ones(2*fe_mesh.GetNumberOfNodes()))
    
    plot_mesh_vec_data(1,[FE_UV[0::2],FE_UV[1::2],
                        FE_adj_UV[0::2],FE_adj_UV[1::2]],
                    ["U","V","adj-U","adj-V"],
                    fig_title=" FEM and adj FEM solution",
                    file_name=os.path.join(case_dir,f"FEM-adj-UV-dist_test_{eval_id}.png"))

    control_derivatives = myresponse.ComputeAdjointNodalControlDerivatives(K_matrix[eval_id],FE_UV,FE_adj_UV)
    shape_derivatives = myresponse.ComputeAdjointNodalShapeDerivatives(K_matrix[eval_id],FE_UV,FE_adj_UV)

    # now compute FD-based sensitivities
    FD_control_sens = myresponse.ComputeFDNodalControlDerivatives(K_matrix[eval_id],linear_fe_solver,
                                                        fd_step_size=1e-5,fd_mode="CD")

    plot_mesh_vec_data(1,[control_derivatives,FD_control_sens],
                        ["df/dE","FD-df/dE"],
                        fig_title="Control Derivatives Verification",
                        file_name=os.path.join(case_dir,f"control_derivatives_verification_{eval_id}.png"))
    
    FD_shape_sens = myresponse.ComputeFDNodalShapeDerivatives(K_matrix[eval_id],linear_fe_solver,
                                                            fd_step_size=1e-5,fd_mode="CD")

    plot_mesh_vec_data(1,[shape_derivatives[0::3],shape_derivatives[1::3],
                        FD_shape_sens[0::3],FD_shape_sens[1::3]],
                        ["df/dx","df/dy","FD-df/dx","FD-df/dy"],
                        fig_title="Shape Derivatives Verification",
                        file_name=os.path.join(case_dir,f"shape_derivatives_verification_{eval_id}.png"))

    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)


if __name__ == "__main__":
    # Initialize default values
    mesh_size = 11
    clean_dir = False

    # Parse the command-line arguments
    args = sys.argv[1:]

    # Process the arguments if provided
    for arg in args:
        if arg.startswith("mesh_size="):
            try:
                mesh_size = int(arg.split("=")[1])
            except ValueError:
                print("mesh_size should be an integer.")
                sys.exit(1)
        elif arg.startswith("clean_dir="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                clean_dir = value.lower() == 'true'
            else:
                print("clean_dir should be True or False.")
                sys.exit(1)
        else:
            print("Usage: python script.py mesh_size=11 clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(mesh_size, clean_dir)