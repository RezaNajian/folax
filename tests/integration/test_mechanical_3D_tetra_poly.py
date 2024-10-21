import pytest
import unittest
import optax
from flax import nnx     
import jax 
import os
import numpy as np
from fol.loss_functions.mechanical_3D_fe_tetra import MechanicalLoss3DTetra
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.controls.voronoi_control3D import VoronoiControl3D
from fol.deep_neural_networks.explicit_parametric_operator_learning import ExplicitParametricOperatorLearning
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

        self.fe_mesh.Initialize()
        self.mechanical_loss_3d.Initialize()
        self.voronoi_control.Initialize()

       # design NN for learning
        class MLP(nnx.Module):
            def __init__(self, in_features: int, dmid: int, out_features: int, *, rngs: nnx.Rngs):
                self.dense1 = nnx.Linear(in_features, dmid, rngs=rngs,kernel_init=nnx.initializers.zeros,bias_init=nnx.initializers.zeros)
                self.dense2 = nnx.Linear(dmid, out_features, rngs=rngs,kernel_init=nnx.initializers.zeros,bias_init=nnx.initializers.zeros)
                self.in_features = in_features
                self.out_features = out_features

            def __call__(self, x: jax.Array) -> jax.Array:
                x = self.dense1(x)
                x = jax.nn.swish(x)
                x = self.dense2(x)
                return x

        fol_net = MLP(self.voronoi_control.GetNumberOfVariables(),1, 
                        self.mechanical_loss_3d.GetNumberOfUnknowns(), 
                        rngs=nnx.Rngs(0))

        # create fol optax-based optimizer
        chained_transform = optax.chain(optax.normalize_by_update_norm(), 
                                        optax.adam(1e-4))

        # create fol
        self.fol = ExplicitParametricOperatorLearning(name="dis_fol",control=self.voronoi_control,
                                                        loss_function=self.mechanical_loss_3d,
                                                        flax_neural_network=fol_net,
                                                        optax_optimizer=chained_transform,
                                                        checkpoint_settings={"restore_state":False,
                                                        "state_directory":self.test_directory+"/flax_state"},
                                                        working_directory=self.test_directory)

        self.fol.Initialize()
        self.linear_fe_solver.Initialize()        
        self.coeffs_matrix,self.K_matrix = create_random_voronoi_samples(self.voronoi_control,1,dim=3)

    def test_compute(self):
        self.fol.Train(train_set=(self.coeffs_matrix[-1].reshape(-1,1).T,),
                       convergence_settings={"num_epochs":1000})
        T_FOL = np.array(self.fol.Predict(self.coeffs_matrix.reshape(-1,1).T)).reshape(-1)
        T_FEM = np.array(self.linear_fe_solver.Solve(self.K_matrix,np.zeros(T_FOL.shape)))
        l2_error = 100 * np.linalg.norm(T_FOL-T_FEM,ord=2)/ np.linalg.norm(T_FEM,ord=2)
        self.assertLessEqual(l2_error, 10)

        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)
        else:
            self.fe_mesh['K'] = np.array(self.K_matrix)
            self.fe_mesh['T_FOL'] = np.array(T_FOL)
            self.fe_mesh['T_FEM'] = np.array(T_FEM)
            self.fe_mesh.Finalize(export_dir=self.test_directory,export_format='vtu')

if __name__ == '__main__':
    unittest.main()