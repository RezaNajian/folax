import sys
import os

import numpy as np
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.mechanical_2D_fe_quad import MechanicalLoss2D
from fol.solvers.fe_solver import FiniteElementSolver
from fol.controls.fourier_control import FourierControl
from fol.controls.voronoi_control import VoronoiControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
import pickle, time

# problem setup
model_settings = {"L":1,
                  "N":25,
                  "Ux_left":0.0,"Ux_right":0.05,
                  "Uy_left":0.0,"Uy_right":-0.05}

# directory & save handling
working_directory_name = f'mechanical_2D_N_{model_settings["N"]}'
case_dir = os.path.join('.', working_directory_name)
clean_dir = True
if clean_dir:
    create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# creation of the model
model_info = create_2D_square_model_info_mechanical(**model_settings)

# creation of the objects
fe_model = FiniteElementModel("FE_model",model_info)
mechanical_loss_2d = MechanicalLoss2D("mechanical_loss_2d",fe_model)
fe_solver = FiniteElementSolver("fe_solver",mechanical_loss_2d)

# k_rangeof_values in the following could be a certain amount of values from a list instead of a tuple
voronoi_control_settings = {"numberof_seeds":10,"k_rangeof_values":[1,2,3,4,5,6]}
# voronoi_control_settings = {"numberof_seeds":10,"k_rangeof_values":(1,6)}
voronoi_control = VoronoiControl("first_voronoi_control",voronoi_control_settings,fe_model)

# create some random coefficients & K for training
number_of_random_samples = 100
coeffs_matrix,K_matrix = create_random_voronoi_samples(voronoi_control,number_of_random_samples)

# now save K matrix 
solution_file = os.path.join(case_dir, "K_matrix.txt")
np.savetxt(solution_file,K_matrix)

# specify id of the K of interest
eval_id = 25

# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",voronoi_control,[mechanical_loss_2d],[50,50],
                                    "swish",load_NN_params=False,working_directory=working_directory_name)
fol.Initialize()

start_time = time.process_time()
fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[eval_id].reshape(-1,1).T,batch_size=5,num_epochs=100,
            learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
            relative_error=1e-10,NN_params_save_file_name="NN_params_"+working_directory_name)

print(f"\n############### FOL train took: {time.process_time() - start_time} s ###############\n")

FOL_UV = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T))

# solve FE here
solve_FE = True
if solve_FE:
    first_fe_solver = FiniteElementSolver("first_fe_solver",mechanical_loss_2d)
    start_time = time.process_time()
    FE_UV = np.array(first_fe_solver.SingleSolve(K_matrix[eval_id],np.zeros(2*fe_model.GetNumberOfNodes())))  
    print(f"\n############### FE solve took: {time.process_time() - start_time} s ###############\n")

    relative_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
    
plot_mesh_vec_data(model_settings["L"], [K_matrix[eval_id],FOL_UV[::2],FE_UV[::2],relative_error[::2]], 
                   subplot_titles= ['Heterogeneity', 'FOL_U', 'FE_U', "absolute error"], fig_title=None, cmap='viridis',
                    block_bool=False, colour_bar=True, colour_bar_name=None,
                    X_axis_name='X', Y_axis_name='Y', show=False, file_name='plot_U_error.png')

plot_mesh_vec_data(model_settings["L"], [K_matrix[eval_id],FOL_UV[1::2],FE_UV[1::2],relative_error[1::2]], 
                   subplot_titles= ['Heterogeneity', 'FOL_V', 'FE_V', "absolute error"], fig_title=None, cmap='viridis',
                    block_bool=False, colour_bar=True, colour_bar_name=None,
                    X_axis_name='X', Y_axis_name='Y', show=False, file_name='plot_V_error.png')