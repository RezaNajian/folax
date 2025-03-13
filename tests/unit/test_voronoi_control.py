import pytest
import unittest
import os
import numpy as np
from fol.mesh_input_output.mesh import Mesh
from fol.controls.voronoi_control3D import VoronoiControl3D
from fol.controls.voronoi_control2D import VoronoiControl2D
from fol.tools.usefull_functions import *
import jax.random as random

class TestVoronoiControl(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        test_name = 'test_voronoi_control'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)
        self.fe_mesh3D = Mesh("med_io",file_name="box_3D_coarse.med",
                         case_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"../meshes"))
        self.fe_mesh2D = create_2D_square_mesh(L=1,N=10)
        voronoi_control2D_settings = {"number_of_seeds":5,"E_values":(0.1,1)}
        voronoi_control3D_settings = {"number_of_seeds":16,"E_values":(0.1,1)}
        self.voronoi_control2D = VoronoiControl2D("voronoi_control",voronoi_control2D_settings,self.fe_mesh2D)
        self.voronoi_control3D = VoronoiControl3D("voronoi_control",voronoi_control3D_settings,self.fe_mesh3D)

        self.fe_mesh2D.Initialize()
        self.fe_mesh3D.Initialize()
        self.voronoi_control2D.Initialize()
        self.voronoi_control3D.Initialize()

    def test_io(self):

        # 3D model
        self.assertEqual(self.voronoi_control3D.GetNumberOfControlledVariables(),1213)
        self.assertEqual(self.voronoi_control3D.GetNumberOfVariables(),64)

        # 2D model
        self.assertEqual(self.voronoi_control2D.GetNumberOfControlledVariables(),100)
        self.assertEqual(self.voronoi_control2D.GetNumberOfVariables(),15)

        # Generate random coefficients for 2D model
        batch_coefficients = random.uniform(random.PRNGKey(42), shape=(10,self.voronoi_control2D.GetNumberOfVariables()))
        batch_coefficients = batch_coefficients.at[:,:10].set(random.uniform(random.PRNGKey(42), shape=(10,10)))
        batch_coefficients = batch_coefficients.at[:,10:].set(random.uniform(random.PRNGKey(42), shape=(10,5),minval=0.1,maxval=1))
        batch_controlled_variables = self.voronoi_control2D.ComputeBatchControlledVariables(batch_coefficients)
        assert jnp.min(batch_controlled_variables) > 0.1
        assert jnp.max(batch_controlled_variables) < 1
        assert abs(jnp.mean(batch_controlled_variables) - 0.5) < 1e-2

        # Generate random coefficients for 3D model
        batch_coefficients = random.uniform(random.PRNGKey(42), shape=(10,self.voronoi_control3D.GetNumberOfVariables()))
        batch_coefficients = batch_coefficients.at[:,:48].set(random.uniform(random.PRNGKey(42), shape=(10,48)))
        batch_coefficients = batch_coefficients.at[:,48:].set(random.uniform(random.PRNGKey(42), shape=(10,16),minval=0.1,maxval=1))
        batch_controlled_variables = self.voronoi_control3D.ComputeBatchControlledVariables(batch_coefficients)
        assert jnp.min(batch_controlled_variables) > 1e-1
        assert jnp.max(batch_controlled_variables) < 1
        assert abs(jnp.mean(batch_controlled_variables) - 0.5) < 1e-1


        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)

if __name__ == '__main__':
    unittest.main()