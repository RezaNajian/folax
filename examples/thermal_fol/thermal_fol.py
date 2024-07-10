import os,time,sys
import numpy as np
from fol.IO.mesh_io import MeshIO
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.thermal_3D_fe_tetra import ThermalLoss3DTetra
from fol.solvers.nonlinear_solver import NonLinearSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # cleaning & logging
    working_directory_name = 'results'
    case_dir = os.path.join('.', working_directory_name)
    clean_dir = True
    if clean_dir:
        create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,"fol_thermal_3D.log"))

    # importing mesh & creating model info
    point_bc_settings = {"T":{"left_fol":1,"right_fol":0.1}}
    io = MeshIO("fol_io",'../meshes/',"fol_3D_tet_mesh_coarse.med",point_bc_settings)
    # io = MeshIO("fol_io",'.',"fol_3D_tet_mesh.med",point_bc_settings)
    model_info = io.Import()

    # create FE model
    fe_model = FiniteElementModel("FE_model",model_info)

    # create thermal loss
    thermal_loss_3d = ThermalLoss3DTetra("thermal_loss_3d",fe_model,{"beta":2,"c":4})

    # create Fourier parametrization/control
    x_freqs = np.array([1,2,3])
    y_freqs = np.array([1,2,3])
    z_freqs = np.array([0])
    fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":5,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_model)

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

    # export random Fourier-based K fields 
    export_Ks = False
    if export_Ks:
        for i in range(K_matrix.shape[0]):
            solution_file = os.path.join(case_dir, f"K_{i}.vtu")
            io.mesh_io.point_data['K'] = np.array(K_matrix[i,:])
            io.mesh_io.write(solution_file)

    # specify id of the K of interest
    eval_id = 5
    io.mesh_io.point_data['K'] = np.array(K_matrix[eval_id,:])

    # now we need to create, initialize and train fol
    fol = FiniteElementOperatorLearning("first_fol",fourier_control,[thermal_loss_3d],[1],
                                        "swish",load_NN_params=False,working_directory=working_directory_name)
    fol.Initialize()
    fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[eval_id].reshape(-1,1).T,batch_size=1,num_epochs=fol_num_epochs,
                learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
                NN_params_save_file_name="NN_params_"+working_directory_name)

    solution_file = os.path.join(case_dir, f"K_{eval_id}_comp.vtu")
    FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T))
    io.mesh_io.point_data['T_FOL'] = FOL_T.reshape((fe_model.GetNumberOfNodes(), 1))

    # solve FE here
    if solve_FE: 
        first_fe_solver = NonLinearSolver("first_fe_solver",thermal_loss_3d,relative_error=1e-5,max_num_itr=20)
        start_time = time.process_time()
        FE_T = np.array(first_fe_solver.SingleSolve(K_matrix[eval_id],np.zeros(fe_model.GetNumberOfNodes())))  
        print(f"\n############### FE solve took: {time.process_time() - start_time} s ###############\n")
        io.mesh_io.point_data['T_FE'] = FE_T.reshape((fe_model.GetNumberOfNodes(), 1))

        relative_error = 100 * (abs(FOL_T.reshape(-1,1)-FE_T.reshape(-1,1)))/abs(FE_T.reshape(-1,1))
        io.mesh_io.point_data['relative_error'] = relative_error.reshape((fe_model.GetNumberOfNodes(), 1))

    io.mesh_io.write(solution_file)

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
    main(fol_num_epochs, solve_FE)