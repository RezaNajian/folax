import sys
import os

import numpy as np
from fol.loss_functions.mechanical_2D_fe_quad import MechanicalLoss2D
from fol.mesh_input_output.mesh import Mesh
from fol.controls.voronoi_control2D import VoronoiControl2D
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = 'mechanical_2D_poly_lin'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))
    
    # problem setup
    model_settings = {"L":1,"N":10,
                    "Ux_left":0.0,"Ux_right":0.05,
                    "Uy_left":0.0,"Uy_right":0.05}

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=1,N=10)

    # create fe-based loss function
    bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
               "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}
    
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    mechanical_loss_2d = MechanicalLoss2D("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                              "num_gp":2,
                                                                              "material_dict":material_dict},
                                                                              fe_mesh=fe_mesh)

    # k_rangeof_values in the following could be a certain amount of values from a list instead of a tuple
    # voronoi_control_settings = {"numberof_seeds":5,"k_rangeof_values":[10,20,30,40,50,60,70,80,90,100]}
    voronoi_control_settings = {"number_of_seeds":5,"E_values":(0.1,1)}
    voronoi_control = VoronoiControl2D("first_voronoi_control",voronoi_control_settings,fe_mesh)

    fe_mesh.Initialize()
    mechanical_loss_2d.Initialize()
    voronoi_control.Initialize()

    # create some random coefficients & K for training
    number_of_random_samples = 100
    coeffs_matrix,K_matrix = create_random_voronoi_samples(voronoi_control,number_of_random_samples)

    # now save K matrix 
    solution_file = os.path.join(case_dir, "K_matrix.txt")
    np.savetxt(solution_file,K_matrix)

    # specify id of the K of interest
    eval_id = 25

    # now we need to create, initialize and train fol
    fol = FiniteElementOperatorLearning("first_fol",voronoi_control,[mechanical_loss_2d],[1],
                                        "swish",load_NN_params=False,working_directory=working_directory_name)
    fol.Initialize()

    fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[eval_id].reshape(-1,1).T,batch_size=1,num_epochs=fol_num_epochs,
                learning_rate=0.0001,optimizer="adam",convergence_criterion="total_loss",
                relative_error=1e-12,absolute_error=1e-12,NN_params_save_file_name="NN_params_"+working_directory_name)

    FOL_UV = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T))
    fe_mesh['U_FOL'] = FOL_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

    # solve FE here
    if solve_FE:
        fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                                    "maxiter":1000,"pre-conditioner":"ilu"},
                      "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                    "maxiter":10,"load_incr":5}}
        linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_2d,fe_setting)
        linear_fe_solver.Initialize()
        FE_UV = np.array(linear_fe_solver.Solve(K_matrix[eval_id],np.zeros(2*fe_mesh.GetNumberOfNodes())))  
        fe_mesh['U_FE'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

        absolute_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
        fe_mesh['abs_error'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 2))

        vectors_list = [K_matrix[eval_id],FE_UV[::2],FOL_UV[::2]]
        plot_mesh_res(vectors_list, file_name=os.path.join(case_dir,'plot_U.png'),dir="U")
        plot_mesh_grad_res_mechanics(vectors_list, file_name=os.path.join(case_dir,'plot_stress_U.png'), loss_settings=material_dict)
        
        vectors_list = [K_matrix[eval_id],FE_UV[1::2],FOL_UV[1::2]]
        plot_mesh_res(vectors_list, file_name=os.path.join(case_dir,'plot_V.png'),dir="V")
        plot_mesh_grad_res_mechanics(vectors_list, file_name=os.path.join(case_dir,'plot_stress_V.png'), loss_settings=material_dict)
        
    
    fe_mesh.Finalize(export_dir=case_dir)

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
            print("Usage: python mechanical_2D_poly.py fol_num_epochs=10 solve_FE=False clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(fol_num_epochs, solve_FE,clean_dir)