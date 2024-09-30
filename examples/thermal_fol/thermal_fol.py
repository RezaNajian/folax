import os,time,sys
import numpy as np
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.thermal_3D_fe_tetra import ThermalLoss3DTetra
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # cleaning & logging
    working_directory_name = 'thermal_fol'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,"fol_thermal_3D.log"))

    # create mesh_io
    fe_mesh = Mesh("fol_io","fol_3D_tet_mesh_coarse.med",'../meshes/')

    # create fe-based loss function
    bc_dict = {"T":{"left_fol":1,"right_fol":0.1}}

    thermal_loss_3d = ThermalLoss3DTetra("thermal_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                          "beta":2,"c":4},
                                                                            fe_mesh=fe_mesh)

    # create Fourier parametrization/control
    x_freqs = np.array([1,2,3])
    y_freqs = np.array([1,2,3])
    z_freqs = np.array([0])
    fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":5,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

    fe_mesh.Initialize()
    thermal_loss_3d.Initialize()
    fourier_control.Initialize()

    # create some random coefficients & K for training
    create_random_coefficients = False
    if create_random_coefficients:
        number_of_random_samples = 200
        coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
        export_dict = {}
        export_dict["coeffs_matrix"] = coeffs_matrix
        export_dict["x_freqs"] = x_freqs
        export_dict["y_freqs"] = y_freqs
        export_dict["z_freqs"] = z_freqs
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

    eval_id = 5
    fe_mesh['K'] = np.array(K_matrix[eval_id,:])

    # now we need to create, initialize and train fol
    fol = FiniteElementOperatorLearning("first_fol",fourier_control,[thermal_loss_3d],[1],
                                        "swish",load_NN_params=False,working_directory=working_directory_name)
    fol.Initialize()
    fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[eval_id].reshape(-1,1).T,batch_size=1,num_epochs=fol_num_epochs,
                learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
                NN_params_save_file_name="NN_params_"+working_directory_name)

    FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T))
    fe_mesh['T_FOL'] = FOL_T.reshape((fe_mesh.GetNumberOfNodes(), 1))

    # solve FE here
    if solve_FE: 
        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"},
                      "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                    "maxiter":20,"load_incr":1}}
        nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",thermal_loss_3d,fe_setting)
        nonlin_fe_solver.Initialize()
        FE_T = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id],np.zeros(fe_mesh.GetNumberOfNodes())))  
        fe_mesh['T_FE'] = FE_T.reshape((fe_mesh.GetNumberOfNodes(), 1))

        relative_error = 100 * (abs(FOL_T.reshape(-1,1)-FE_T.reshape(-1,1)))/abs(FE_T.reshape(-1,1))
        fe_mesh['relative_error'] = relative_error.reshape((fe_mesh.GetNumberOfNodes(), 1))

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