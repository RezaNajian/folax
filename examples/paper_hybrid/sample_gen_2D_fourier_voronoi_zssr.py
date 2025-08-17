import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.controls.fourier_control import FourierControl
from fol.controls.voronoi_control2D import VoronoiControl2D
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle
from mechanical2d_utilities import *
from fol.tools.decoration_functions import *

### Script's goal:
####### the following script is to 
####### create fourier and voronoi samples and save the pkl file as well as txt file.
####### one can plot some topology using plot_random_K_matrix


def main(solve_FE=False,ifol_num_epochs=10,clean_dir=False):

    # directory & save handling
    working_directory_name = "sample_gen_2D_fourier_voronoi_res_141_for_multi"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,"N":141,
                    "Ux_left":0.0,"Ux_right":0.1,
                    "Uy_left":0.0,"Uy_right":0.1}

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
    fe_mesh.Initialize()

    # create fe-based loss function
    bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
               "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}
    
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}

    mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                        "num_gp":2,
                                                                                        "material_dict":material_dict},
                                                        fe_mesh=fe_mesh)

    mechanical_loss_2d.Initialize()
    
    # define settings to create K_matrix samples
    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)
    fourier_control.Initialize()

    even_I = np.array([2,4,6])
    even_II = np.array([4,6,8])
    even_III = np.array([6,8,10])
    odd_I = np.array([1,3,5])
    odd_II = np.array([3,5,7])
    odd_III = np.array([5,7,9])

    freq_list = [even_I,even_II,even_III,odd_I,odd_II,odd_III]
    # create a dictionary to save different topologies and the corresponding settings
    # fourier samples
    
    mixed_fourier_sample_dict_new = {}
    file_path_base = f"ifol_fourier_test_samples_res_81.pkl"
    with open(file_path_base, 'rb') as f:
            mixed_fourier_sample_dict_base = pickle.load(f)
    file_path = os.path.join(case_dir,f"ifol_fourier_test_samples_res_{model_settings['N']}.pkl")

    # create new samples?
    confirm = "y"
    if confirm == "y":
        with open (file_path, 'wb') as f:
            i = 1
            for _ in list(mixed_fourier_sample_dict_base.keys()):
                mixed_fourier_sample_dict_new[f"series_{i}"] = {}
                fourier_control_settings_copy = mixed_fourier_sample_dict_base[f"series_{i}"]["settings"]
                coeffs_matrix = mixed_fourier_sample_dict_base[f"series_{i}"]["coeffs_matrix"]
                fourier_control = FourierControl("fourier_control",fourier_control_settings_copy,fe_mesh)
                fourier_control.Initialize()
                K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)
                mixed_fourier_sample_dict_new[f"series_{i}"]["K_matrix"] = K_matrix
                mixed_fourier_sample_dict_new[f"series_{i}"]["coeffs_matrix"] = coeffs_matrix
                mixed_fourier_sample_dict_new[f"series_{i}"]["settings"] = fourier_control_settings_copy
                i += 1
            pickle.dump(mixed_fourier_sample_dict_new,f)
            fol_info(f"{file_path} saved successfully!")
    elif confirm == "n":
        with open(file_path, 'rb') as f:
            mixed_fourier_sample_dict_new = pickle.load(f)
        fol_info(f"The file {file_path} loaded!")
    else:
        fol_info("Invalid input. Please enter Y or N.")
    

    # voronoi samples
    voronoi_control_settings = {"number_of_seeds":25,"E_values":[0.1,1.]}
    voronoi_control = VoronoiControl2D("first_voronoi_control",voronoi_control_settings,fe_mesh)
    voronoi_control.Initialize()

    seed_list = [16,32,64,128]
    file_path_base = f"ifol_voronoi_test_samples_res_41.pkl"
    with open(file_path_base, 'rb') as f:
            voronoi_sample_dict_base = pickle.load(f)

    file_path = os.path.join(case_dir,f"ifol_voronoi_test_samples_res_{model_settings['N']}.pkl")
    voronoi_sample_dict_new = {}

    confirm = "y"
    if confirm == "y":
        with open (file_path, 'wb') as f:
            i = 1
            for _ in list(voronoi_sample_dict_base.keys()):
                voronoi_sample_dict_new[f"series_{i}"] = {}
                voronoi_control_settings_copy = voronoi_sample_dict_base[f"series_{i}"]["settings"]
                coeffs_matrix = voronoi_sample_dict_base[f"series_{i}"]["coeffs_matrix"]
                voronoi_control = VoronoiControl2D("first_voronoi_control",voronoi_control_settings_copy,fe_mesh)
                voronoi_control.Initialize()
                K_matrix = voronoi_control.ComputeBatchControlledVariables(coeffs_matrix)
                coeffs_matrix = voronoi_sample_dict_new[f"series_{i}"]["K_matrix"] = K_matrix
                coeffs_matrix = voronoi_sample_dict_new[f"series_{i}"]["coeffs_matrix"] = coeffs_matrix
                coeffs_matrix = voronoi_sample_dict_new[f"series_{i}"]["settings"] = voronoi_control_settings_copy
                i += 1
            pickle.dump(voronoi_sample_dict_new,f)
            fol_info(f"{file_path} saved successfully!")
    elif confirm == "n":
        with open(file_path, 'rb') as f:
            voronoi_sample_dict_new = pickle.load(f)
        fol_info(f"The file {file_path} loaded!")
    else:
        fol_info("Invalid input. Please enter Y or N.")
    
    # voronoi multi phase samples
    voronoi_multi_control_settings = {"number_of_seeds":25,"E_values":(0.1,1.)}
    voronoi_multi_control = VoronoiControl2D("first_voronoi_control",voronoi_multi_control_settings,fe_mesh)
    voronoi_multi_control.Initialize()

    seed_list = [16,32,64,128]
    file_path_base_multi = f"ifol_voronoi_multi_test_samples_res_21.pkl"
    with open(file_path_base_multi, 'rb') as f:
            voronoi_multi_sample_dict_base = pickle.load(f)

    file_path = os.path.join(case_dir,f"ifol_voronoi_multi_test_samples_res_{model_settings['N']}.pkl")
    voronoi_multi_sample_dict_new = {}

    confirm = "y"
    if confirm == "y":
        with open (file_path, 'wb') as f:
            i = 1
            for _ in list(voronoi_multi_sample_dict_base.keys()):
                voronoi_multi_sample_dict_new[f"series_{i}"] = {}
                voronoi_multi_control_settings_copy = voronoi_multi_sample_dict_base[f"series_{i}"]["settings"]
                coeffs_matrix = voronoi_multi_sample_dict_base[f"series_{i}"]["coeffs_matrix"]
                voronoi_multi_control = VoronoiControl2D("first_voronoi_control",voronoi_multi_control_settings_copy,fe_mesh)
                voronoi_multi_control.Initialize()
                K_matrix = voronoi_multi_control.ComputeBatchControlledVariables(coeffs_matrix)
                coeffs_matrix = voronoi_multi_sample_dict_new[f"series_{i}"]["K_matrix"] = K_matrix
                coeffs_matrix = voronoi_multi_sample_dict_new[f"series_{i}"]["coeffs_matrix"] = coeffs_matrix
                coeffs_matrix = voronoi_multi_sample_dict_new[f"series_{i}"]["settings"] = voronoi_multi_control_settings_copy
                i += 1
            pickle.dump(voronoi_multi_sample_dict_new,f)
            fol_info(f"{file_path} saved successfully!")
    elif confirm == "n":
        with open(file_path, 'rb') as f:
            voronoi_multi_sample_dict_new = pickle.load(f)
        fol_info(f"The file {file_path} loaded!")
    else:
        fol_info("Invalid input. Please enter Y or N.")
    
    index = 1
    N = model_settings["N"]

    def plot_random_K_matrix(sample_dict,file_name):
        for series in sample_dict.keys():
            K_matrix_samples = sample_dict[series]["K_matrix"]

            fig, axes = plt.subplots(1, 6, figsize=(24, 4))
            if sample_dict[series]['settings'].get("x_freqs") is not None:
                fig.suptitle(
                f"x frequencies: {sample_dict[series]['settings']['x_freqs']} "
                f"and y frequencies: {sample_dict[series]['settings']['y_freqs']}",
                fontsize=18
                )
            elif sample_dict[series]['settings'].get("number_of_seeds") is not None:
                fig.suptitle(
                f"number of seeds: {sample_dict[series]['settings']['number_of_seeds']} ",
                fontsize=18
                )
            for i in range(6):
                ax = axes[i]
                ax.imshow(K_matrix_samples[i, :].reshape(N, N), cmap="viridis")
                ax.set_xticks([])
                ax.set_yticks([])
                

            plt.tight_layout()
            plt.savefig(f"{file_name}_{series}_res_{model_settings['N']}.png", dpi=300)
            plt.close(fig)

    plot_random_K_matrix(sample_dict=mixed_fourier_sample_dict_new,file_name=os.path.join(case_dir,"plot_fourier"))
    plot_random_K_matrix(sample_dict=voronoi_sample_dict_new,file_name=os.path.join(case_dir,"plot_voronoi"))
    plot_random_K_matrix(sample_dict=voronoi_multi_sample_dict_new,file_name=os.path.join(case_dir,"plot_voronoi_multi"))


    fourier_coeffs_matrix, fourier_K_matrix = fourier_to_K_matrix(mixed_fourier_sample_dict_new)
    fourier_file = os.path.join(case_dir,f"ifol_fourier_test_samples_K_matrix_res_{model_settings['N']}.txt")
    fourier_file_coeffs = os.path.join(case_dir,f"ifol_fourier_test_samples_coeffs_matrix_res_{model_settings['N']}.txt")
    np.savetxt(fourier_file, fourier_K_matrix)
    np.savetxt(fourier_file_coeffs, fourier_coeffs_matrix)

    voronoi_coeffs_matrix, voronoi_K_matrix = voronoi_to_K_matrix(voronoi_sample_dict_new)
    voronoi_file = os.path.join(case_dir,f"ifol_voronoi_test_samples_K_matrix_res_{model_settings['N']}.txt")
    voronoi_file_coeffs = os.path.join(case_dir,f"ifol_voronoi_test_samples_coeffs_matrix_res_{model_settings['N']}.txt")
    np.savetxt(voronoi_file, voronoi_K_matrix)
    np.savetxt(voronoi_file_coeffs, voronoi_coeffs_matrix)

    voronoi_multi_coeffs_matrix, voronoi_multi_K_matrix = voronoi_to_K_matrix(voronoi_multi_sample_dict_new)
    voronoi_multi_file = os.path.join(case_dir,f"ifol_voronoi_multi_test_samples_K_matrix_res_{model_settings['N']}.txt")
    voronoi_multi_file_coeffs = os.path.join(case_dir,f"ifol_voronoi_multi_test_samples_coeffs_matrix_res_{model_settings['N']}.txt")
    np.savetxt(voronoi_multi_file, voronoi_multi_K_matrix)
    np.savetxt(voronoi_multi_file_coeffs, voronoi_multi_coeffs_matrix)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 2000
    solve_FE = True
    clean_dir=False
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
    main(ifol_num_epochs, solve_FE,clean_dir)