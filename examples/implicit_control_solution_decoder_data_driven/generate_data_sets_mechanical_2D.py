import sys
import os
import optax
import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss2DQuad
from fol.controls.fourier_control import FourierControl
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.data_generators.fem_data_generator import FemDataGenerator
from fol.data_input_output.zarr_io import ZarrIO
import pickle

# problem setup
model_settings = {"L":1,"N":11,
                  "Ux_left":0.0,"Ux_right":0.05,
                  "Uy_left":0.0,"Uy_right":0.05}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create fe-based loss function
bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
           "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}

material_dict = {"young_modulus":1,"poisson_ratio":0.3}
mechanical_loss_2d = MechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)
# create FE solver 
fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl","tol":1e-6,"atol":1e-6,
                                            "maxiter":1000,"pre-conditioner":"ilu"},
                "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                            "maxiter":10,"load_incr":5}}
linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_2d,fe_setting)

# create control
fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                            "beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

# create fem data generator 
fem_generator = FemDataGenerator("linear_fem_generator",linear_fe_solver)

# initialize all
fe_mesh.Initialize()
mechanical_loss_2d.Initialize()
linear_fe_solver.Initialize()
fourier_control.Initialize()
fem_generator.Initialize()

with open(f'fourier_control_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

coeffs_matrix = loaded_dict["coeffs_matrix"]
K_matrix = np.array(fourier_control.ComputeBatchControlledVariables(coeffs_matrix))

fem_data = fem_generator.Generate(K_matrix)

ZarrIO("zarr_io").Export(data_dict={"fourier_conductivity":K_matrix,
                                    "U_FEM":fem_data[:,0::2],
                                    "V_FEM":fem_data[:,1::2]},
                        file_name="data_sets")
