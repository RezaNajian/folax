import sys
import os
import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss3DHexa
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import timeit
import statistics

# directory & save handling
working_directory_name = "results"
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1.0,"N":20,
                "Ux_left":0.0,"Ux_right":0.1,
                "Uy_left":0.0,"Uy_right":0.1,
                "Uz_left":0.0,"Uz_right":0.1}

fe_mesh = create_3D_box_mesh(Nx=model_settings["N"],
                             Ny=model_settings["N"],
                             Nz=model_settings["N"],
                             Lx=model_settings["L"],
                             Ly=model_settings["L"],
                             Lz=model_settings["L"],
                             case_dir=case_dir)

bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
           "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]},
           "Uz":{"left":model_settings["Uz_left"],"right":model_settings["Uz_right"]}}

material_dict = {"young_modulus":1,"poisson_ratio":0.3}
mechanical_loss_3d = MechanicalLoss3DHexa("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)
    
linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_3d)

fe_mesh.Initialize()
mechanical_loss_3d.Initialize()
linear_fe_solver.Initialize()

test_control_vars = np.ones(fe_mesh.GetNumberOfNodes())
test_initial_solution = np.zeros(3*fe_mesh.GetNumberOfNodes())

def glob_jac_res(control_vars,initial_solution):
    BC_applied_dofs = mechanical_loss_3d.ApplyDirichletBCOnDofVector(initial_solution)
    global_jac,global_r = mechanical_loss_3d.ComputeJacobianMatrixAndResidualVector(
                                            control_vars,BC_applied_dofs)
    return global_jac,global_r

jit_time = timeit.repeat(lambda: glob_jac_res(np.copy(test_control_vars),np.copy(test_initial_solution)), 
                         repeat=1, number=1)

# Benchmark
n_repeat = 20   # How many times to repeat the timing
n_number = 20    # How many times to run the function in each repeat

times = timeit.repeat(lambda: glob_jac_res(np.copy(test_control_vars),np.copy(test_initial_solution)), 
                      repeat=n_repeat, number=n_number)
normalized_times = np.array(times) / n_number

print(f"jit time: {jit_time[0]:.6f} sec")
print(f"Per-run mean time: {statistics.mean(normalized_times):.6f} sec")
print(f"Per-run std dev  : {statistics.stdev(normalized_times):.6f} sec")