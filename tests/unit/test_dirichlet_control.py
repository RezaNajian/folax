import pytest
import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))
import numpy as np
from fol.mesh_input_output.mesh import Mesh
from fol.controls.dirichlet_control import DirichletControl
from fol.tools.usefull_functions import *
import jax.random as random

class TestDirichletControl(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        test_name = 'test_dirichlet_control'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)
        self.fe_mesh3D = Mesh("med_io",file_name="box_3D_coarse.med",
                         case_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"../meshes"))
        self.number_of_nodes_at_right_bc = 145
        self.number_of_nodes_at_left_bc = 127
        self.number_of_nodes_at_bc = self.number_of_nodes_at_right_bc + self.number_of_nodes_at_left_bc
        bc_dict = {"Ux":{"left":0.0,"right":0.02},"Uy":{"left":0.0,"right":-0.04},"Uz":{"left":0.0,"right":-0.06}}
        dirichlet_control_settings = {"parametric_boundary_learning": {"Ux":["right"],
                                        "Uy":["right"],
                                        "Uz":["right"]},
                                        "dirichlet_bc_dict":bc_dict}
        self.dirichlet_control = DirichletControl("dirichlet_control",dirichlet_control_settings,self.fe_mesh3D)

        self.fe_mesh3D.Initialize()
        self.dirichlet_control.Initialize()

    def test_io(self):

        # 3D model
        self.assertEqual(self.dirichlet_control.GetNumberOfControlledVariables(),1213)
        self.assertEqual(self.dirichlet_control.GetNumberOfVariables(),3)
        
        self.assertEqual(self.dirichlet_control.control_dirichlet_dict['Ux']['left'].shape[0],127)
        self.assertEqual(self.dirichlet_control.control_dirichlet_dict['Uy']['left'].shape[0],127)
        self.assertEqual(self.dirichlet_control.control_dirichlet_dict['Uz']['left'].shape[0],127)
        self.assertEqual(self.dirichlet_control.control_dirichlet_dict['Ux']['right'].shape[0],145)
        self.assertEqual(self.dirichlet_control.control_dirichlet_dict['Uy']['right'].shape[0],145)
        self.assertEqual(self.dirichlet_control.control_dirichlet_dict['Uz']['right'].shape[0],145)

        self.assertEqual(self.dirichlet_control.control_dirichlet_dict['Ux']['left'].all(),0)
        self.assertEqual(self.dirichlet_control.control_dirichlet_dict['Uy']['left'].all(),0)
        self.assertEqual(self.dirichlet_control.control_dirichlet_dict['Uz']['left'].all(),0)
        self.assertEqual(self.dirichlet_control.control_dirichlet_dict['Ux']['right'].all(),1)
        self.assertEqual(self.dirichlet_control.control_dirichlet_dict['Uy']['right'].all(),1)
        self.assertEqual(self.dirichlet_control.control_dirichlet_dict['Uz']['right'].all(),1)

        # Generate random coefficients for 3D model
        coefficients = [0.02, 0.04, 0.06]
        controlled_varibales = self.dirichlet_control.ComputeControlledVariables(coefficients)
        
        assert jnp.all(controlled_varibales[:127] == 0)
        assert jnp.all(controlled_varibales[127:272] == 0.02)
        assert jnp.all(controlled_varibales[272:399] == 0)
        assert jnp.all(controlled_varibales[399:544] == 0.04)
        assert jnp.all(controlled_varibales[544:671] == 0)
        assert jnp.all(controlled_varibales[671:816] == 0.06)
        

        batch_coefficients = random.uniform(random.PRNGKey(42), shape=(10,self.dirichlet_control.GetNumberOfVariables()))
        batch_controlled_variables = self.dirichlet_control.ComputeBatchControlledVariables(batch_coefficients)
        
        assert batch_controlled_variables.shape == (10,3*self.number_of_nodes_at_bc)


        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)

if __name__ == '__main__':
    unittest.main()