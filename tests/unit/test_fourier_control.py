import pytest
import unittest
import os
import numpy as np
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.tools.usefull_functions import *
import jax.random as random

class TestFourierControl(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        test_name = 'test_fourier_control'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)
        self.fe_mesh = Mesh("med_io",file_name="box_3D_coarse.med",
                         case_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"../meshes"))
        fourier_control_settings = {"x_freqs":np.array([1,2,3]),"y_freqs":np.array([4,5]),"z_freqs":np.array([6]),"beta":2}
        self.fourier_control = FourierControl("fourier_control",fourier_control_settings,self.fe_mesh)

        self.fe_mesh.Initialize()
        self.fourier_control.Initialize()

    def test_io(self):

        self.assertEqual(self.fourier_control.GetNumberOfControlledVariables(),1213)
        self.assertEqual(self.fourier_control.GetNumberOfVariables(),7)
        # Generate random coefficients
        batch_coefficients = random.uniform(random.PRNGKey(42), shape=(10,self.fourier_control.GetNumberOfVariables()))
        batch_controlled_variables = self.fourier_control.ComputeBatchControlledVariables(batch_coefficients)
        assert abs(jnp.min(batch_controlled_variables) - 0.00030614433) < 1e-6
        assert abs(jnp.max(batch_controlled_variables) - 0.99903476) < 1e-6
        assert abs(jnp.mean(batch_controlled_variables) - 0.37126848) < 1e-6

        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)

if __name__ == '__main__':
    unittest.main()