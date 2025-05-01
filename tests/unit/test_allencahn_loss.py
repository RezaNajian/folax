import pytest
import unittest 
import os
import numpy as np
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
from fol.loss_functions.phase_field import AllenCahnLoss2DTri
from fol.loss_functions.phase_field import AllenCahnLoss2DQuad
from fol.loss_functions.phase_field import AllenCahnLoss3DHexa
from fol.tools.usefull_functions import *


class TestAllenCahn(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        self.fe_mesh = Mesh("box_io","box_3D_coarse.med",os.path.join(os.path.dirname(os.path.abspath(__file__)),"../meshes"))
        self.dirichlet_bc_dict = {"Phi":{"left":1.0,"right":-1.0}}

        self.fe_mesh.Initialize()

    def test_hexa(self):
        hex_points_coordinates = jnp.array([[0.24900,  0.34200,  0.19200],
                                            [0.32000,  0.18600,  0.64300],
                                            [0.16500,  0.74500,  0.70200],
                                            [0.27300,  0.75000,  0.23000],
                                            [0.00000,  0.00000,  0.00000],
                                            [0.00000,  0.00000,  1.00000],
                                            [0.00000,  1.00000,  1.00000],
                                            [0.00000,  1.00000,  0.00000]])
        
        allencahn_loss_3d = AllenCahnLoss3DHexa("allencahn_loss_3d",loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                  "material_dict":{"rho":1.0,"cp":1.0,"dt":0.001,"epsilon":0.2}},
                                                   fe_mesh=self.fe_mesh)
        allencahn_loss_3d.Initialize()

        en, residuals, stiffness = allencahn_loss_3d.ComputeElement(hex_points_coordinates,
                                                                        jnp.ones((8)),
                                                                        jnp.ones((8,1)))
        
        np.testing.assert_allclose(stiffness.flatten(),jnp.array([3.517133416607975960e-03,1.707774004898965359e-03,6.962693878449499607e-04,1.727647962979972363e-03,2.662429353222250938e-03,1.216818694956600666e-03,5.191651289351284504e-04,1.303078955970704556e-03,1.707774004898965359e-03,4.163531120866537094e-03,1.688555232249200344e-03,7.436659070663154125e-04,1.232243143022060394e-03,2.972587943077087402e-03,1.268008840270340443e-03,5.534706288017332554e-04,6.962693878449499607e-04,1.688555232249200344e-03,2.864593872800469398e-03,1.257676631212234497e-03,5.412970785982906818e-04,1.280233031138777733e-03,2.257707761600613594e-03,1.000181655399501324e-03,1.727647962979972363e-03,7.436659070663154125e-04,1.257676631212234497e-03,3.478546859696507454e-03,1.308233127929270267e-03,5.400278023444116116e-04,9.444564348086714745e-04,2.598859136924147606e-03,2.662429353222250938e-03,1.232243143022060394e-03,5.412970785982906818e-04,1.308233127929270267e-03,7.310334593057632446e-03,3.295850008726119995e-03,1.472316915169358253e-03,3.584268735721707344e-03,1.216818694956600666e-03,2.972587943077087402e-03,1.280233031138777733e-03,5.400278023444116116e-04,3.295850008726119995e-03,7.939203642308712006e-03,3.487393492832779884e-03,1.490992726758122444e-03,5.191651289351284504e-04,1.268008840270340443e-03,2.257707761600613594e-03,9.444564348086714745e-04,1.472316915169358253e-03,3.487393492832779884e-03,6.263631861656904221e-03,2.716715680435299873e-03,1.303078955970704556e-03,5.534706288017332554e-04,1.000181655399501324e-03,2.598859136924147606e-03,3.584268735721707344e-03,1.490992726758122444e-03,2.716715680435299873e-03,7.104270160198211670e-03]),
                                   rtol=1e-5, atol=1e-5)
        
        np.testing.assert_allclose(residuals.flatten(),jnp.array([-9.313225746154785156e-10,-9.313225746154785156e-10,1.862645149230957031e-09,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,-1.862645149230957031e-09]),
                                   rtol=1e-5, atol=1e-5)
        
    def test_tri(self):
                
        quad_points_coordinates = jnp.array([[1.00,0.00,0.00],
                                            [0.50,0.50,0.00],
                                            [0.00,0.00,0.00]])

        self.dirichlet_bc_dict = {"Phi":{"left":1.0,"right":-1.0}}

        allencahn_loss_3d = AllenCahnLoss2DTri("allencahn_loss_2d",
                                                  loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                                 "material_dict":{"rho":1.0,"cp":1.0,"dt":0.001,"epsilon":0.2}},
                                                                 fe_mesh=self.fe_mesh)
        allencahn_loss_3d.Initialize()

        en, residuals, stiffness = allencahn_loss_3d.ComputeElement(quad_points_coordinates,
                                                                     jnp.ones((3)),
                                                                     jnp.ones((3,1)))
        
        np.testing.assert_allclose(stiffness.flatten(),jnp.array([2.838888019323348999e-02,2.588888630270957947e-02,2.488888613879680634e-02,2.588888630270957947e-02,2.663889154791831970e-02,2.663889154791831970e-02,2.488888613879680634e-02,2.663889154791831970e-02,2.763889171183109283e-02]),
                                   rtol=1e-5, atol=1e-5)
        
        np.testing.assert_allclose(residuals.flatten(),jnp.array([0.000000000000000000e+00,7.450580596923828125e-09,7.450580596923828125e-09]),
                                   rtol=1e-5, atol=1e-5)

        
    def test_quad(self):
                
        quad_points_coordinates = jnp.array([[1.00,0.00,0.00],
                                            [1.00,1.00,0.00],
                                            [0.00,1.00,0.00],
                                            [0.00,0.00,0.00]])

        self.dirichlet_bc_dict = {"Phi":{"left":1.0,"right":-1.0}}

        allencahn_loss_3d = AllenCahnLoss2DQuad("allencahn_loss_2d",
                                                  loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                                 "material_dict":{"rho":1.0,"cp":1.0,"dt":0.001,"epsilon":0.2}},
                                                                 fe_mesh=self.fe_mesh)
        allencahn_loss_3d.Initialize()

        en, residuals, stiffness = allencahn_loss_3d.ComputeElement(quad_points_coordinates,
                                                                     jnp.ones((4)),
                                                                     jnp.ones((4,1)))

        np.testing.assert_allclose(stiffness.flatten(),jnp.array([1.062222048640251160e-01,5.261111259460449219e-02,2.605555765330791473e-02,5.261111259460449219e-02,5.261111259460449219e-02,1.062222048640251160e-01,5.261111259460449219e-02,2.605555765330791473e-02,2.605555765330791473e-02,5.261111259460449219e-02,1.062222048640251160e-01,5.261111259460449219e-02,5.261111259460449219e-02,2.605555765330791473e-02,5.261111259460449219e-02,1.062222048640251160e-01]),
                                   rtol=1e-5, atol=1e-5)
        
        np.testing.assert_allclose(residuals.flatten(),jnp.array([1.490116119384765625e-08,1.490116119384765625e-08,1.490116119384765625e-08,-2.980232238769531250e-09]),
                                   rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()