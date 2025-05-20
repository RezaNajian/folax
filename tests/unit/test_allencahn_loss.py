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
                                                                        jnp.zeros((8,1)))
        
        np.testing.assert_allclose(stiffness.flatten(),jnp.array([3.777465783059597015e-03,1.851124223321676254e-03,7.607493898831307888e-04,1.856124494224786758e-03,2.864157548174262047e-03,1.323940115980803967e-03,5.684257484972476959e-04,1.402301364578306675e-03,1.851124223321676254e-03,4.476600326597690582e-03,1.817999058403074741e-03,8.081459091044962406e-04,1.339364564046263695e-03,3.199345665052533150e-03,1.365829026326537132e-03,6.027312483638525009e-04,7.607493898831307888e-04,1.817999058403074741e-03,3.069299971684813499e-03,1.372246770188212395e-03,5.905576981604099274e-04,1.378053217194974422e-03,2.422231016680598259e-03,1.090102712623775005e-03,1.856124494224786758e-03,8.081459091044962406e-04,1.372246770188212395e-03,3.732121316716074944e-03,1.407455536536872387e-03,5.892884219065308571e-04,1.034377492032945156e-03,2.794020343571901321e-03,2.864157548174262047e-03,1.339364564046263695e-03,5.905576981604099274e-04,1.407455536536872387e-03,7.856915704905986786e-03,3.580985590815544128e-03,1.604879391379654408e-03,3.852681489661335945e-03,1.323940115980803967e-03,3.199345665052533150e-03,1.378053217194974422e-03,5.892884219065308571e-04,3.580985590815544128e-03,8.533164858818054199e-03,3.749230410903692245e-03,1.623555202968418598e-03,5.684257484972476959e-04,1.365829026326537132e-03,2.422231016680598259e-03,1.034377492032945156e-03,1.604879391379654408e-03,3.749230410903692245e-03,6.717018783092498779e-03,2.961829770356416702e-03,1.402301364578306675e-03,6.027312483638525009e-04,1.090102712623775005e-03,2.794020343571901321e-03,3.852681489661335945e-03,1.623555202968418598e-03,2.961829770356416702e-03,7.631339598447084427e-03]),
                                   rtol=1e-5, atol=1e-5)
        
        np.testing.assert_allclose(residuals.flatten(),jnp.array([-1.405296474695205688e-02,-1.508403755724430084e-02,-1.219633128494024277e-02,-1.326222531497478485e-02,-2.253365516662597656e-02,-2.339274436235427856e-02,-1.992568001151084900e-02,-2.142298594117164612e-02]),
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
                                                                     jnp.zeros((3,1)))
        
        np.testing.assert_allclose(stiffness.flatten(),jnp.array([3.047221340239048004e-02,2.797221951186656952e-02,2.697221934795379639e-02,2.797221951186656952e-02,2.872222475707530975e-02,2.872222475707530975e-02,2.697221934795379639e-02,2.872222475707530975e-02,2.972222492098808289e-02]),
                                   rtol=1e-5, atol=1e-5)
        
        np.testing.assert_allclose(residuals.flatten(),jnp.array([-8.333332091569900513e-02,-8.333332836627960205e-02,-8.333332836627960205e-02]),
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
                                                                     jnp.zeros((4,1)))

        np.testing.assert_allclose(stiffness.flatten(),jnp.array([1.145555377006530762e-01,5.677777901291847229e-02,2.813889086246490479e-02,5.677777901291847229e-02,5.677777901291847229e-02,1.145555377006530762e-01,5.677777901291847229e-02,2.813889086246490479e-02,2.813889086246490479e-02,5.677777901291847229e-02,1.145555377006530762e-01,5.677777901291847229e-02,5.677777901291847229e-02,2.813889086246490479e-02,5.677777901291847229e-02,1.145555377006530762e-01]),
                                   rtol=1e-5, atol=1e-5)
        
        np.testing.assert_allclose(residuals.flatten(),jnp.array([-2.499999850988388062e-01,-2.499999850988388062e-01,-2.499999850988388062e-01,-2.500000000000000000e-01]),
                                   rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()