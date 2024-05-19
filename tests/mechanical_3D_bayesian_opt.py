import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
from computational_models import FiniteElementModel
from loss_functions import MechanicalLoss3D
from solvers import FiniteElementSolver
from controls import FourierControl
from deep_neural_networks import FiniteElementOperatorLearning
from tools import *
import pickle
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from tools import *

def Bayesian_Opt_FOL(num_elem,learning_rate, num_layers, num_neurons, batch_size, activation_fn):
    
    # problem setup
    model_settings = {"Lx":1,"Ly":1,"Lz":1,
                    "Nx":num_elem,"Ny":num_elem,"Nz":num_elem,
                    "Ux_left":0.0,"Ux_right":"",
                    "Uy_left":0.0,"Uy_right":-0.05,
                    "Uz_left":0.0,"Uz_right":-0.05}

    # fourier freqs
    x_freqs = np.array([2,4,6])
    y_freqs = np.array([2,4,6])
    z_freqs = np.array([2,4,6])

    # directory & save handling
    working_directory_name = f'mechanical_3D_Nx_{model_settings["Nx"]}_Ny_{model_settings["Ny"]}_Nz_{model_settings["Nz"]}'
    case_dir = os.path.join('.', working_directory_name)
    clean_dir = True
    if clean_dir:
        create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # creation of the model
    model_info,model_io = create_3D_box_model_info_mechanical(model_settings,case_dir)

    # creation of the objects
    fe_model = FiniteElementModel("FE_model",model_info)
    mechanical_loss_3d = MechanicalLoss3D("mechanical_loss_3d",fe_model)
    fe_solver = FiniteElementSolver("fe_solver",mechanical_loss_3d)
    fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":20,"min":1e-2,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_model)

    with open(f'fourier_control_dict_Nx_{model_settings["Nx"]}_Ny_{model_settings["Ny"]}_Nz_{model_settings["Nz"]}.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    coeffs_matrix = loaded_dict["coeffs_matrix"]

    # now we need to create, initialize and train fol
    fol = FiniteElementOperatorLearning("first_fol",fourier_control,[mechanical_loss_3d],[num_neurons] * num_layers,
                                        activation_fn,load_NN_params=False,working_directory=working_directory_name)
    fol.Initialize()
    fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix,batch_size=batch_size,num_epochs=2000,
                learning_rate=learning_rate,optimizer="adam",convergence_criterion="total_loss",
                relative_error=1e-10,NN_params_save_file_name="NN_params_"+working_directory_name)

    return fol.train_history_dict["total_loss"][-1]

sys.stdout = Logger("bayesian_opt.log")

for num_elem in [10,20,50]:

    # Define the search space for hyperparameters
    dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform', name='learning_rate')
    dim_num_layers = Integer(low=1, high=5, name='num_layers')
    dim_num_neurons = Integer(low=10, high=500, name='num_neurons')
    dim_batch_size = Integer(low=1, high=20, name='batch_size')
    dim_activation = Categorical(categories=["relu","swish","tanh"], name='activation_fn')

    dimensions = [dim_learning_rate, dim_num_layers, dim_num_neurons, dim_batch_size, dim_activation]

    # Define the objective function for Bayesian optimization
    @use_named_args(dimensions=dimensions)
    def objective(learning_rate, num_layers, num_neurons, batch_size, activation_fn):
        return float(Bayesian_Opt_FOL(num_elem,learning_rate, num_layers, num_neurons, batch_size, activation_fn))

    # Run Bayesian optimization
    res = gp_minimize(objective, dimensions=dimensions, n_calls=50, random_state=0)

    print(f"num_elem:{num_elem}, Best parameters: {res.x}")
    print(f"num_elem:{num_elem}, Best validation loss: {res.fun}")
