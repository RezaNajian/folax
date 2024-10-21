import pytest
import unittest
import optax
from flax import nnx     
import jax 
import os
import numpy as np
from fol.loss_functions.thermal_3D_fe_tetra import ThermalLoss3DTetra
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.explicit_parametric_operator_learning import ExplicitParametricOperatorLearning
from fol.tools.usefull_functions import *

class TestMechanical3D(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        test_name = 'test_thermal_3D_nonlin'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)
        self.fe_mesh = Mesh("box_io","box_3D_coarse.med",os.path.join(os.path.dirname(os.path.abspath(__file__)),"../meshes"))
        dirichlet_bc_dict = {"T":{"left":1,"right":0.1}}
        self.thermal_loss = ThermalLoss3DTetra("thermal",loss_settings={"dirichlet_bc_dict":dirichlet_bc_dict,
                                                                        "beta":0,"c":4},
                                                                        fe_mesh=self.fe_mesh)  
        self.nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",self.thermal_loss,
                                                                          {"nonlinear_solver_settings":{"maxiter":3}})

        fourier_control_settings = {"x_freqs":np.array([0]),"y_freqs":np.array([0]),"z_freqs":np.array([0]),"beta":2}
        self.fourier_control = FourierControl("fourier_control",fourier_control_settings,self.fe_mesh)

        self.fe_mesh.Initialize()
        self.thermal_loss.Initialize()
        self.fourier_control.Initialize()        

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

        fol_net = MLP(self.fourier_control.GetNumberOfVariables(),1, 
                        self.thermal_loss.GetNumberOfUnknowns(), 
                        rngs=nnx.Rngs(0))

        # create fol optax-based optimizer
        chained_transform = optax.chain(optax.normalize_by_update_norm(), 
                                        optax.adam(1e-3))

        # create fol
        self.fol = ExplicitParametricOperatorLearning(name="dis_fol",control=self.fourier_control,
                                                        loss_function=self.thermal_loss,
                                                        flax_neural_network=fol_net,
                                                        optax_optimizer=chained_transform,
                                                        checkpoint_settings={"restore_state":False,
                                                        "state_directory":self.test_directory+"/flax_state"},
                                                        working_directory=self.test_directory)

        self.fol.Initialize()
        self.nonlin_fe_solver.Initialize()        
        self.coeffs_matrix,self.K_matrix = create_random_fourier_samples(self.fourier_control,0)

    def test_compute(self):
        self.fol.Train(train_set=(self.coeffs_matrix[-1].reshape(-1,1).T,),
                       convergence_settings={"num_epochs":1000})
        T_FOL = np.array(self.fol.Predict(self.coeffs_matrix[-1,:].reshape(-1,1).T)).reshape(-1)
        T_FEM = np.array(self.nonlin_fe_solver.Solve(self.K_matrix[-1,:],np.zeros(T_FOL.shape)))
        l2_error = 100 * np.linalg.norm(T_FOL-T_FEM,ord=2)/ np.linalg.norm(T_FEM,ord=2)
        self.assertLessEqual(l2_error, 1)

        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)
        else:
            self.fe_mesh['K'] = np.array(self.K_matrix[-1,:])
            self.fe_mesh['T_FOL'] = np.array(T_FOL)
            self.fe_mesh['T_FEM'] = np.array(T_FEM)
            self.fe_mesh.Finalize(export_dir=self.test_directory)

if __name__ == '__main__':
    unittest.main()