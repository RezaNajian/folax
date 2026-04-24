import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import jax
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.identity_control import IdentityControl
from fol.controls.fourier_control import FourierControl
from fol.controls.voronoi_control2D import VoronoiControl2D
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
import optax
from mechanical2d_utilities import *
from fol.tools.decoration_functions import *
import requests
import zipfile

# def prepare_net_params(case_dir):
#         """
#         Extract only the contents of 'folder_in_zip' from the ZIP archive
#         and place them into 'extract_to'.
#         """
#         extract_to = case_dir
#         folder_in_zip = "2d_hyperelastic/"  # ensure correct format

#         url = "https://zenodo.org/records/17752752/files/NiN.zip?download=1"
#         filename = "NiN.zip"

#         fol_info(f"⬇ Downloading '{filename}' from Zenodo...")

#         response = requests.get(url, stream=True)
#         response.raise_for_status()  # raises if e.g. 404, 403, etc.

#         with open(filename, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 if chunk:  # filter out keep-alive chunks
#                     f.write(chunk)

#         with zipfile.ZipFile(filename, "r") as z:
#             # Filter the files that start with the folder path
#             members = [m for m in z.namelist() if m.startswith(folder_in_zip)]

#             if not members:
#                 raise ValueError(f"Folder '{folder_in_zip}' not found inside ZIP.")

#             fol_info(f"📦 Extracting {len(members)} files from '{folder_in_zip}'...")

#             for member in members:
#                 # Compute final extraction path
#                 destination = os.path.join(extract_to, os.path.relpath(member, folder_in_zip))

#                 # Create directories if needed
#                 if member.endswith("/"):
#                     os.makedirs(destination, exist_ok=True)
#                 else:
#                     # Ensure parent directory exists
#                     os.makedirs(os.path.dirname(destination), exist_ok=True)
#                     with z.open(member, "r") as src, open(destination, "wb") as dest:
#                         dest.write(src.read())

#             fol_info(f"✔ Extracted to: {os.path.abspath(extract_to)}")

# directory & save handling
working_directory_name = "2d_hyperelastic_sample_gen"   # should be the same dir that contains network parameters
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))
confirm = input("Do you want generate new samples? (Y/N): ").strip().lower()
regenerate_samples = True if confirm=='y' else False

# problem setup
model_settings = {"L":1,"N":82,
                "Ux_left":0.0,"Ux_right":0.5,
                "Uy_left":0.0,"Uy_right":0.5,
                "max_phase": 1., "min_phase":0.1}

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

# define a control class which reconstrcut the input space from a reduced space
# identity control maps X: -> X
identity_control = IdentityControl('identity_control', control_settings={},num_vars=fe_mesh.GetNumberOfNodes())
identity_control.Initialize()

##### Fourier Samples #####
# define fourier control to create synthethic microstructures
fourier_control_settings = {"x_freqs":np.array([6,8,10]),"y_freqs":np.array([6,8,10]),"z_freqs":np.array([0]),
                            "beta":20,"min":model_settings["min_phase"],"max":model_settings["max_phase"]}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)
fourier_control.Initialize()
filename_fourier = os.path.join(working_directory_name,f'tests_fourier_control_dict_res_{model_settings["N"]}.pkl')
save_fourier = regenerate_samples
number_of_random_samples=50
if save_fourier:
    fourier_dict = {}
    coeffs_matrix,K_matrix_fourier = create_random_fourier_samples(fourier_control,number_of_random_samples)
    fourier_dict["coeffs_matrix"] = coeffs_matrix
    fourier_dict["K_matrix"] = K_matrix_fourier
    fourier_dict["fourier_settings"] = fourier_control_settings
    # save
    with open(f'{filename_fourier}', 'wb') as f:
        pickle.dump(fourier_dict,f)
else:
    # load fourier coefficients and compute K
    loaded_dict = {}
    with open(f'{filename_fourier}', 'rb') as f:
        loaded_dict = pickle.load(f)
    coeffs_matrix = loaded_dict["coeffs_matrix"]
    K_matrix_fourier = loaded_dict["K_matrix"]


##### Voronoi Samples #####
seeds_nums = [5,10,15,20]
num_samples = 5
for seeds_num in seeds_nums:
    filename_dual = os.path.join(working_directory_name,f'tests_voronoi_dual_control_dict_seeds_num_{seeds_num}_res_{model_settings["N"]}.pkl')
    filename_multi = os.path.join(working_directory_name,f'tests_voronoi_multi_control_dict_seeds_num_{seeds_num}_res_{model_settings["N"]}.pkl')
    voronoi_control_settings_multi = {"number_of_seeds":seeds_num,"E_values":(model_settings["min_phase"],model_settings["max_phase"])}
    voronoi_control_settings_dual = {"number_of_seeds":seeds_num,"E_values":[model_settings["min_phase"],model_settings["max_phase"]]}
    voronoi_control_multi = VoronoiControl2D("first_voronoi_control",voronoi_control_settings_multi,fe_mesh)
    voronoi_control_multi.Initialize()
    voronoi_control_dual = VoronoiControl2D("first_voronoi_control",voronoi_control_settings_dual,fe_mesh)
    voronoi_control_dual.Initialize()


    save_voronoi = regenerate_samples
    if save_voronoi:
        voronoi_dict_multi = {}
        coeffs_matrix_multi,K_matrix_multi = create_random_voronoi_samples(voronoi_control_multi,num_samples)
        voronoi_dict_multi["coeffs_matrix"] = coeffs_matrix_multi
        voronoi_dict_multi["K_matrix"] = K_matrix_multi
        voronoi_dict_multi["voronoi_settings"] = voronoi_control_settings_multi
        voronoi_dict_dual = {}
        coeffs_matrix_dual,K_matrix_dual = create_random_voronoi_samples(voronoi_control_dual,num_samples)
        voronoi_dict_dual["coeffs_matrix"] = coeffs_matrix_dual
        voronoi_dict_dual["K_matrix"] = K_matrix_dual
        voronoi_dict_dual["voronoi_settings"] = voronoi_control_settings_dual
        
        # save
        with open(f'{filename_multi}', 'wb') as f:
            pickle.dump(voronoi_dict_multi,f)
        with open(f'{filename_dual}', 'wb') as f:
            pickle.dump(voronoi_dict_dual,f)
    else:
        # load voronoi coefficients and compute K
        loaded_dict = {}
        with open(f'{filename_multi}', 'rb') as f:
            loaded_dict = pickle.load(f)
        coeffs_matrix_multi = loaded_dict["coeffs_matrix"]
        K_matrix_voronoi_multi = loaded_dict["K_matrix"]
        loaded_dict = {}
        with open(f'{filename_dual}', 'rb') as f:
            loaded_dict = pickle.load(f)
        coeffs_matrix_dual = loaded_dict["coeffs_matrix"]
        K_matrix_voronoi_dual = loaded_dict["K_matrix"]



##### TPMS Samples #####
section_axis_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for section_axis_value in section_axis_values:
    filename_tpms = os.path.join(working_directory_name,f'tests_tpms_control_dict_axis_{section_axis_value}_res_{model_settings["N"]}.pkl')

    tpms_settings = {"phi_x": 0., "phi_y": 0., "phi_z": 0., "max": 1., "min": 0.1, "section_axis_value": section_axis_value,
                        "constant": 0., "threshold": 0.5, "coefficients":(2.,2.,2.)}
    tpms_dict = {}
    tpms_K_matrix = []
    save_tpms = regenerate_samples

    if save_tpms:
        K_matrix_gyroid = create_tpms_gyroid(fe_mesh=fe_mesh, tpms_settings=tpms_settings)
        K_matrix_schwarz_P = create_tpms_schwarz_P(fe_mesh=fe_mesh, tpms_settings=tpms_settings)
        K_matrix_diamond = create_tpms_diamond(fe_mesh=fe_mesh, tpms_settings=tpms_settings)
        K_matrix_lidinoid = create_tpms_lidinoid(fe_mesh=fe_mesh, tpms_settings=tpms_settings)
        K_matrix_split_p = create_tpms_split_p(fe_mesh=fe_mesh, tpms_settings=tpms_settings)
        
        tpms_dict["tpms_settings"] = tpms_settings

        tpms_K_matrix.append(K_matrix_gyroid)
        tpms_K_matrix.append(K_matrix_schwarz_P)
        tpms_K_matrix.append(K_matrix_diamond)
        tpms_K_matrix.append(K_matrix_lidinoid)
        tpms_K_matrix.append(K_matrix_split_p)
        tpms_dict["K_matrix"] = np.asarray(tpms_K_matrix)

        # save
        with open(f'{filename_tpms}', 'wb') as f:
            pickle.dump(tpms_dict,f)
    else:
        # load voronoi coefficients and compute K
        loaded_dict = {}
        with open(f'{filename_tpms}', 'rb') as f:
            loaded_dict = pickle.load(f)

            K_matrix_gyroid = loaded_dict["K_matrix_gyroid"]
            K_matrix_schwarz_P = loaded_dict["K_matrix_schwarz_P"]
            K_matrix_diamond = loaded_dict["K_matrix_diamond"]
            K_matrix_lidinoid = loaded_dict["K_matrix_lidinoid"]
            K_matrix_split_p = loaded_dict["K_matrix_split_p"]


##### Circular Inclusion Samples #####
section_axis_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for section_axis_value in section_axis_values:
    filename_sphere = os.path.join(working_directory_name,f'tests_sphere_control_dict_axis_{section_axis_value}_res_{model_settings["N"]}.pkl')
    sphere_settings = {"sphere_diameter": 0.2, "fiber_length": 0.4, "fiber_radius": 0.05,
                            "max": 1., "min": 0.1, "num_spheres": 40}

    sphere_dict = {}
    sphere_K_matrix = []

    save_sphere = regenerate_samples

    if save_sphere:
        K_matrix_sphere_lattice = create_sphere_lattice(fe_mesh=fe_mesh, tpms_settings=sphere_settings)
        K_matrix_periodic_sphere_field = create_random_periodic_sphere_field(fe_mesh=fe_mesh, tpms_settings=sphere_settings)
        K_matrix_fiber_field = create_random_fiber_field(fe_mesh=fe_mesh, tpms_settings=sphere_settings)
        
        sphere_dict["sphere_settings"] = sphere_settings

        sphere_K_matrix.append(K_matrix_sphere_lattice)
        sphere_K_matrix.append(K_matrix_periodic_sphere_field)
        sphere_K_matrix.append(K_matrix_fiber_field)
        sphere_dict["K_matrix"] = np.asarray(sphere_K_matrix)

        # save
        with open(f'{filename_sphere}', 'wb') as f:
            pickle.dump(tpms_dict,f)
    else:
        # load voronoi coefficients and compute K
        loaded_dict = {}
        with open(f'{filename_sphere}', 'rb') as f:
            loaded_dict = pickle.load(f)

            K_matrix_sphere_lattice = loaded_dict["K_matrix_sphere_lattice"]
            K_matrix_periodic_sphere_field = loaded_dict["K_matrix_periodic_sphere_field"]
            K_matrix_fiber_field = loaded_dict["K_matrix_fiber_field"]