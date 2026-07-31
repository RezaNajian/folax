import sys
import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import optax
import numpy as np
from meta_implicit_parametric_operator_learning_no_ad import (
    CastingThermoMechanicsLoss3DTetra as ThermoMechanicsLoss3DTetra,
)
# from fol.loss_functions.thermal import ThermalLoss3DTetra
from fol.mesh_input_output.mesh import Mesh
from fol.controls.identity_control import IdentityControl
from fol.controls.fourier_control import FourierControl
from fol.controls.dirichlet_control import DirichletControl
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from thermo_mechanical_useful_functions import *
from jax import config
import jax 
jax.config.update("jax_enable_x64", False)
config.update("jax_default_matmul_precision", "highest")
# directory & save handling
working_directory_name = 'NFEM_thermo_mechanical'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

mesh_res_rate = 1
fe_mesh = Mesh("fol_io","casting_base.med")
fe_mesh.Initialize()
# # create fe-based loss function

bc_dict = {"T":{"Inlet_top":1.0, "F_bottom":0.001,"Surroundings":0.001},#
           "Ux":{"Inlet_top":0.0,"Inlet_bottom":0.0,"F_bottom":0.0},#"F_right1":model_settings["Ux_right"],"F_right1":model_settings["Ux_right"]
           "Uy":{"Inlet_top":0.0,"Inlet_bottom":0.0,"F_bottom":0.0},
           "Uz":{"Inlet_top":0.0,"F_bottom":0.0}}#,"F_rear":model_settings["Uz_back"]

Dirichlet_BCs = False
material_dict = {"young_modulus":1.0,"poisson_ratio":0.3,"T0":jnp.full((fe_mesh.GetNumberOfNodes(),),1e-4),}
# Thermal parameters
thermal_dict = {"k1":0.5,"k2":2.0,"k3":20.0,"k4":0.5}
mechanical_dict  = {"e1":1.0,"e2":-0.6}

thermomech_loss_3d = ThermoMechanicsLoss3DTetra("thermomechanical_loss_3d",
    loss_settings={
        "dirichlet_bc_dict": bc_dict,
        "material_dict": material_dict,
        "thermal_dict": thermal_dict,
        "mechanical_dict": mechanical_dict
    },
                                                                            fe_mesh=fe_mesh)
no_control = IdentityControl("No_Control",fe_mesh,{})
thermomech_loss_3d.Initialize()
no_control.Initialize()
displ_control_settings = {"learning_boundary": {"T":{"Inlet_top"}}}

displ_control = DirichletControl("displ_control",displ_control_settings,fe_mesh,thermomech_loss_3d)
displ_control.Initialize()

K_matrix = np.full((1,fe_mesh.GetNumberOfNodes()),1.0)
test_E_coeff = K_matrix[0].reshape(1,-1)
test_K_coeff = K_matrix[0].reshape(1,-1)
initial_temp = np.full((1,fe_mesh.GetNumberOfNodes()),1e-4)
E_matrix = test_E_coeff.flatten()#fourier_control.ComputeBatchControlledVariables(coeffs_matrix)
K_matrix = test_K_coeff.flatten()#fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

wanted_bc = 0.5
# solve FE here
# Initialize the fe loss function with the desired boundary conditions
bc_dict = {"T":{"Inlet_top":wanted_bc,"F_bottom":0.001,"Surroundings":0.001},#
           "Ux":{"Inlet_top":0.0,"Inlet_bottom":0.0,"F_bottom":0.0},#"F_right1":model_settings["Ux_right"],"F_right1":model_settings["Ux_right"]
           "Uy":{"Inlet_top":0.0,"Inlet_bottom":0.0,"F_bottom":0.0},
           "Uz":{"Inlet_top":0.0,"F_bottom":0.0}}#,"F_rear":model_settings["Uz_back"]

Dirichlet_BCs = False
material_dict = {"young_modulus":1.0,"poisson_ratio":0.3,"T0":jnp.full((fe_mesh.GetNumberOfNodes(),),1e-4),}
thermal_dict = {"k1":0.5,"k2":2.0,"k3":20.0,"k4":0.5}
mechanical_dict  = {"e1":1.0,"e2":-0.6}

thermomech_loss_3d = ThermoMechanicsLoss3DTetra("thermomechanical_loss_3d",
                                                    loss_settings={
        "dirichlet_bc_dict": bc_dict,
        "material_dict": material_dict,
        "thermal_dict": thermal_dict,
        "mechanical_dict": mechanical_dict
    },
                                                                            fe_mesh=fe_mesh)

# Thermal parameters

thermomech_loss_3d.Initialize()
displ_control_settings = {"learning_boundary": {"T":{"Inlet_top"}}}

displ_control = DirichletControl("displ_control",displ_control_settings,fe_mesh,thermomech_loss_3d)
displ_control.Initialize()
fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                            "maxiter":1000,"pre-conditioner":"ilu","Dirichlet_BCs":Dirichlet_BCs},
                "nonlinear_solver_settings":{"rel_tol":1e-4,"abs_tol":1e-4,
                                            "maxiter":10,"load_incr":1}}
nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlinear_fe_solver",thermomech_loss_3d,fe_setting)
nonlinear_fe_solver.Initialize()

FE_TUVW = np.array(nonlinear_fe_solver.Solve(test_K_coeff.flatten(),np.zeros((fe_mesh.GetNumberOfNodes()*4))))  #
FE_TUVW = reshape_T_U_to_nodewise3D(FE_TUVW, fe_mesh.GetNumberOfNodes())
fe_mesh['sol_FE'] = FE_TUVW

# # Stress and heat flux
FE_UVW = FE_TUVW[:,1:]
FE_T = FE_TUVW[:,0]

FE_stress_at_nodes = GetStressVector3D(thermomech_loss_3d,fe_mesh, test_E_coeff.flatten(),
                                       FE_UVW.flatten(),FE_T.flatten(),initial_temp.flatten())
fe_mesh['FE_stress'] = FE_stress_at_nodes   

FE_heat_flux_at_nodes = GetHeatFluxVector3D(thermomech_loss_3d,fe_mesh,test_K_coeff.flatten(),FE_T.flatten())
fe_mesh['FE_heat_flux'] = FE_heat_flux_at_nodes
fe_mesh.Finalize(export_dir=case_dir)
