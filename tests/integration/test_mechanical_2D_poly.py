import pytest
import unittest
import sys
import os
import numpy as np
from fol.loss_functions.mechanical_2D_fe_quad_neohooke import MechanicalLoss2D
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.voronoi_control import VoronoiControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *

class TestMechanicalPoly2D(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        test_name = 'test_mechanical_poly_2D'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)
        self.fe_mesh = create_2D_square_mesh(L=1,N=11)
        bc_dict = {"Ux":{"left":0.0,"right":0.1},
                   "Uy":{"left":0.0,"right":0.1}}
        
        material_dict = {"young_modulus":1,"poisson_ratio":0.3}
        self.mechanical_loss = MechanicalLoss2D("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "num_gp":2,
                                                                                "material_dict":material_dict},
                                                                                fe_mesh=self.fe_mesh)
        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"},
                      "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                    "maxiter":5,"load_incr":5}}
        self.fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",self.mechanical_loss,fe_setting)

        voronoi_control_settings = {"numberof_seeds":5,"k_rangeof_values":[10,10]}
        self.voronoi_control = VoronoiControl("fourier_control",voronoi_control_settings,self.fe_mesh)
        self.fol = FiniteElementOperatorLearning("fol_mechanical_loss_2d",self.voronoi_control,[self.mechanical_loss],[1],
                                                "swish",working_directory=self.test_directory)
        self.fe_mesh.Initialize()
        self.mechanical_loss.Initialize()
        self.voronoi_control.Initialize()
        self.fol.Initialize()
        self.fe_solver.Initialize()

        self.coeffs_matrix,self.K_matrix = create_random_voronoi_samples(self.voronoi_control,1)

    def test_compute(self):

        self.fol.Train(loss_functions_weights=[1],X_train=self.coeffs_matrix[-1,:].reshape(-1,1).T,batch_size=1,num_epochs=200,
                       learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",relative_error=1e-6)
        UV_FOL = np.array(self.fol.Predict(self.coeffs_matrix[-1,:].reshape(-1,1).T))
        UV_FEM = np.array(self.fe_solver.Solve(self.K_matrix[-1].reshape(-1,1),np.zeros(UV_FOL.shape)))
        l2_error = 100 * np.linalg.norm(UV_FOL-UV_FEM,ord=2)/ np.linalg.norm(UV_FEM,ord=2)
        self.assertLessEqual(l2_error, 10)
        
        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)
        else:
            plot_mesh_vec_data(1,[self.K_matrix[-1,:],UV_FOL[0::2],UV_FOL[1::2]],["K","U","V"],file_name=os.path.join(self.test_directory,"FOL-KUV-dist.png"))
            plot_mesh_vec_data(1,[self.K_matrix[-1,:],UV_FEM[0::2],UV_FEM[1::2]],["K","U","V"],file_name=os.path.join(self.test_directory,"FEM-KUV-dist.png"))
            self.fe_mesh['K'] = np.array(self.K_matrix[-1,:])
            self.fe_mesh['UV_FOL'] = np.array(UV_FOL)
            self.fe_mesh['UV_FEM'] = np.array(UV_FEM)
            self.fe_mesh.Finalize(export_dir=self.test_directory)
            
if __name__ == '__main__':
    unittest.main()