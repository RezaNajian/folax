import pytest
import unittest
import os
import numpy as np
from fol.loss_functions.mechanical_3D_fe_tetra import MechanicalLoss3DTetra
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.controls.voronoi_control3D import VoronoiControl3D
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *

class TestMechanical3D(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        test_name = 'test_mechanical_3D_tetra_poly_lin'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)
        self.fe_mesh = Mesh("box_io","box_3D_coarse.med",os.path.join(os.path.dirname(os.path.abspath(__file__)),"../meshes"))
        dirichlet_bc_dict = {"Ux":{"left":0.0},
                "Uy":{"left":0.0,"right":-0.05},
                "Uz":{"left":0.0,"right":-0.05}}
        self.mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":dirichlet_bc_dict,
                                                                                            "material_dict":{"young_modulus":1,"poisson_ratio":0.3}},
                                                                                            fe_mesh=self.fe_mesh)
        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"},
                      "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                    "maxiter":5,"load_incr":5}}
        self.linear_fe_solver = FiniteElementLinearResidualBasedSolver("lin_fe_solver",self.mechanical_loss_3d,fe_setting)

        voronoi_control_settings = {"number_of_seeds":16,"E_values":[0.1,1]}
        self.voronoi_control = VoronoiControl3D("voronoi_control",voronoi_control_settings,self.fe_mesh)
        self.fol = FiniteElementOperatorLearning("fol_thermal_nonlin",self.voronoi_control,[self.mechanical_loss_3d],[],
                                                "swish",working_directory=self.test_directory)

        self.fe_mesh.Initialize()
        self.mechanical_loss_3d.Initialize()
        self.voronoi_control.Initialize()
        self.fol.Initialize()
        self.linear_fe_solver.Initialize()        
        self.coeffs_matrix,self.K_matrix = create_random_voronoi_samples(self.voronoi_control,1,dim=3)

    def test_compute(self):
        self.fol.Train(loss_functions_weights=[1],X_train=self.coeffs_matrix.reshape(-1,1).T,batch_size=1,num_epochs=200,
                       learning_rate=0.0001,optimizer="adam",convergence_criterion="total_loss",relative_error=1e-8,absolute_error=1e-8)
        T_FOL = np.array(self.fol.Predict(self.coeffs_matrix.reshape(-1,1).T))
        T_FEM = np.array(self.linear_fe_solver.Solve(self.K_matrix,np.zeros(T_FOL.shape)))
        l2_error = 100 * np.linalg.norm(T_FOL-T_FEM,ord=2)/ np.linalg.norm(T_FEM,ord=2)
        self.assertLessEqual(l2_error, 1)

        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)
        else:
            self.fe_mesh['K'] = np.array(self.K_matrix)
            self.fe_mesh['T_FOL'] = np.array(T_FOL)
            self.fe_mesh['T_FEM'] = np.array(T_FEM)
            self.fe_mesh.Finalize(export_dir=self.test_directory,export_format='vtu')

if __name__ == '__main__':
    unittest.main()