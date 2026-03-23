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
        batch_coefficients = jnp.array([[0.19043303, 0.53117931, 0.59302986, 0.11387968, 0.52422845, 0.57513821,0.51164913],
                                        [0.43865383, 0.83727634, 0.7856009 , 0.44929075, 0.51185298, 0.40663743,0.6444757 ],
                                        [0.41020179, 0.75160241, 0.5089581 , 0.18264079, 0.92751539, 0.38282084,0.99222636],
                                        [0.40227389, 0.18236911, 0.26072943, 0.87362838, 0.65606976, 0.47843194,0.54831207],
                                        [0.08547151, 0.2305634 , 0.75642991, 0.24703646, 0.36943841, 0.04529929,0.16996026],
                                        [0.42944622, 0.46134901, 0.83165467, 0.17770588, 0.25554228, 0.52125704,0.25937998],
                                        [0.53061974, 0.39223528, 0.51197338, 0.12757349, 0.97692645, 0.00321233,0.62650323],
                                        [0.42386425, 0.43606746, 0.89602172, 0.64852309, 0.38202906, 0.70676994,0.68973041],
                                        [0.39307904, 0.4418    , 0.32222474, 0.03009605, 0.36899745, 0.6727109 ,0.83522081],
                                        [0.48610044, 0.62330449, 0.90642619, 0.01321077, 0.6145705 , 0.05081677,0.47763526]])
        batch_controlled_variables = self.fourier_control.ComputeBatchControlledVariables(batch_coefficients)
        assert abs(jnp.min(batch_controlled_variables) - 0.00030614433) < 1e-6
        assert abs(jnp.max(batch_controlled_variables) - 0.99903476) < 1e-6
        assert abs(jnp.mean(batch_controlled_variables) - 0.37126848) < 1e-6

        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)

if __name__ == '__main__':
    unittest.main()