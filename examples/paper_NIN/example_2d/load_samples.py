import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import jax
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.mesh_input_output.mesh import Mesh
from fol.controls.identity_control import IdentityControl
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle
from mechanical2d_utilities import *
from fol.tools.decoration_functions import *



# directory & save handling
working_directory_name = "2d_hyperelastic_sample_load"   # should be the same dir that contains network parameters
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":82,
                "Ux_left":0.0,"Ux_right":0.5,
                "Uy_left":0.0,"Uy_right":0.5,
                "max_phase": 1., "min_phase":0.1}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()

folder = "2d_hyperelastic_sample_gen"


# load K_matrix from pickles file
K_matrix_total = []

for fname in os.listdir(folder):
    if fname.endswith(".pkl"):
        path = os.path.join(folder, fname)

        with open(path, "rb") as f:
            data = pickle.load(f)

        if "K_matrix" in data:
            print(f"[INFO]: the shape of K_matrix is: {data['K_matrix'].shape} from {path}")
            K_matrix_total.append(data["K_matrix"])
        else:
            print(f"Warning: 'K_matrix' not found in {fname}")

print(f"Loaded {len(K_matrix_total)} K_matrices")
K_matrix_total = np.concatenate(K_matrix_total, axis=0)
print(f"K_matrix shape: {K_matrix_total.shape}")
np.savetxt('new_test_sample.txt',K_matrix_total)

export_dict = {}
export_dict["K_matrix"] = K_matrix_total
with open(f'new_test_sample.pkl', 'wb') as f:
            pickle.dump(export_dict,f)
with open(f'new_test_sample.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
print(loaded_dict.keys())
print(loaded_dict["K_matrix"].shape)

plot=False
if plot:
    for id in range(150,181):
        plt.imshow(K_matrix_total[id,:].reshape((model_settings['N'],model_settings['N'])))
        plt.savefig(os.path.join(working_directory_name,f'sample_{id}'))