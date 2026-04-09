import sys
import os
import optax
import numpy as np
from fol.loss_functions.thermal import ThermalLoss2DQuad
from fol.controls.fourier_control import FourierControl
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.data_generators.fem_data_generator import FemDataGenerator
from fol.data_input_output.zarr_io import ZarrIO
import pickle

# problem setup
model_settings = {"L":1,"N":11,
                  "T_left":0.1,"T_right":0.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]}}

thermal_loss_2d = ThermalLoss2DQuad("thermal_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                          "beta":2,"c":4},
                                                                            fe_mesh=fe_mesh)
# create FE solver 
fe_setting = {"linear_solver_settings":{"solver":"JAX-direct","tol":1e-6,"atol":1e-6,
                                            "maxiter":1000,"pre-conditioner":"ilu"},
                "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                            "maxiter":10,"load_incr":1}}
nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlinear_fe_solver",thermal_loss_2d,fe_setting)

# create control
fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                            "beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

# create fem data generator 
fem_generator = FemDataGenerator("linear_fem_generator",nonlinear_fe_solver)

# initialize all
fe_mesh.Initialize()
thermal_loss_2d.Initialize()
nonlinear_fe_solver.Initialize()
fourier_control.Initialize()
fem_generator.Initialize()

with open(f'fourier_control_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

coeffs_matrix = loaded_dict["coeffs_matrix"]
K_matrix = np.array(fourier_control.ComputeBatchControlledVariables(coeffs_matrix))

fem_data = fem_generator.Generate(K_matrix)

ZarrIO("zarr_io").Export(data_dict={"K":K_matrix,
                                    "T_FEM":fem_data[:,:]},
                        file_name="data_sets")
