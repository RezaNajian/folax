
import numpy as np
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.mechanical_3D_fe_tetra import MechanicalLoss3DTetra
from fol.IO.mdpa_io import MdpaIO
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle

def main(fol_num_epochs=10,clean_dir=False):
    # directory & save handling
    working_directory_name = "results"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    point_bc_settings = {"Ux":{"support_horizontal_1":0.0,"tip_1":-1.0},
                        "Uy":{"support_horizontal_1":0.0},
                        "Uz":{"support_horizontal_1":0.0,"tip_1":-1.0}}
    mdpa_io = MdpaIO("mdpa_io","hook.mdpa",bc_settings=point_bc_settings,scale_factor=1.0)
    model_info = mdpa_io.Import()

    # creation of fe model and loss function
    fe_model = FiniteElementModel("FE_model",model_info)
    mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",fe_model,{"young_modulus":1,"poisson_ratio":0.3})

    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_model)

    # create some random coefficients & K for training
    create_random_coefficients = False
    if create_random_coefficients:
        number_of_random_samples = 200
        coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
        export_dict = {}
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

    eval_id = -1
    mdpa_io['K'] = np.array(K_matrix[eval_id,:])

    # now we need to create, initialize and train fol
    fol = FiniteElementOperatorLearning("first_fol",fourier_control,[mechanical_loss_3d],[1],
                                        "tanh",load_NN_params=False,working_directory=working_directory_name)
    fol.Initialize()

    fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[eval_id].reshape(-1,1).T,batch_size=1,num_epochs=fol_num_epochs,
                learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
                relative_error=1e-10,NN_params_save_file_name="NN_params_"+working_directory_name)

    FOL_UVW = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T))
    mdpa_io['U_FOL'] = FOL_UVW.reshape((fe_model.GetNumberOfNodes(), 3))

    mdpa_io.Export(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    fol_num_epochs = 2000
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
        elif arg.startswith("clean_dir="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                clean_dir = value.lower() == 'true'
            else:
                print("clean_dir should be True or False.")
                sys.exit(1)
        else:
            print("Usage: python thermal_fol.py fol_num_epochs=10 clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(fol_num_epochs, clean_dir)