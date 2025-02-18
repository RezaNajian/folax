import sys
import os
import optax
import numpy as np
from fol.loss_functions.phasefield import PhaseFieldLoss2DTri
from fol.mesh_input_output.mesh import Mesh
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning_pf import MetaImplicitParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver_phasefield import FiniteElementNonLinearResidualBasedSolverPhasefield
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from jax import config
from fe_plotter import FEPlotter
config.update("jax_default_matmul_precision", "float32")

# creation of the model
mesh_res_rate = 1
fe_mesh = Mesh("fol_io","Li_battery_particle_fine_scaled.med",'../meshes/')
fe_mesh.Initialize()

FEPlotter.plot_mesh(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),filename='FE_mesh_particle.png')