import pytest
import unittest 
import os
import numpy as np
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
from fol.loss_functions.transient_thermal import TransientThermalLoss2DQuad
from fol.loss_functions.transient_thermal import TransientThermalLoss2DTri
from fol.loss_functions.transient_thermal import TransientThermalLoss3DHexa
from fol.loss_functions.transient_thermal import TransientThermalLoss3DTetra
from fol.tools.usefull_functions import *


class TestNonlinearTransientThermal(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        self.fe_mesh = Mesh("box_io","box_3D_coarse.med",os.path.join(os.path.dirname(os.path.abspath(__file__)),"../meshes"))
        self.dirichlet_bc_dict = {"T":{"left":1.0,"right":0.0}}
        self.material_dict = {"rho":1.0,"cp":1.0,"k0":np.ones(1213),"beta":1.5,"c":1.0}
        self.time_integration_dict = {"method":"implicit-euler","time_step":0.005}

        self.fe_mesh.Initialize()

    def test_tetra(self):
        thermal_loss_3d = TransientThermalLoss3DTetra("transient_thermal_loss_3d",loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                                            "material_dict":self.material_dict,
                                                                            "time_integration_dict":self.time_integration_dict},
                                                                            fe_mesh=self.fe_mesh)
        thermal_loss_3d.Initialize()
        
        tet_points_coordinates = jnp.array([[0.1, 0.1, 0.1],
                                            [0.28739360416666665, 0.27808503701741405, 0.05672979583333333],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 1.0, 0.1]])
        en, residuals, stiffness = thermal_loss_3d.ComputeElement(tet_points_coordinates,
                                                                        jnp.ones((4)),
                                                                        jnp.ones((4,1)),
                                                                        jnp.ones((4,1)))

        np.testing.assert_allclose(stiffness,jnp.array([[1.315292250365018845e-02,-2.355600008741021156e-03,-6.230268045328557491e-04,-9.397367946803569794e-03],
                                                        [-2.355600008741021156e-03,1.318590482696890831e-03,3.680836525745689869e-04,1.445854199118912220e-03],
                                                        [-6.230268627405166626e-04,3.680836525745689869e-04,2.669433015398681164e-04,7.649281178601086140e-04],
                                                        [-9.397367946803569794e-03,1.445854199118912220e-03,7.649280014447867870e-04,7.963513955473899841e-03]])
                                                        ,rtol=1e-5, atol=1e-10)

        np.testing.assert_allclose(residuals.flatten(),jnp.array([-1.164153218269348145e-09,-1.746229827404022217e-10,0.000000000000000000e+00,-2.328306436538696289e-10])
                                   ,rtol=1e-4, atol=1e-10)

    def test_hexa(self):
        hex_points_coordinates = jnp.array([[0.24900,  0.34200,  0.19200],
                                            [0.32000,  0.18600,  0.64300],
                                            [0.16500,  0.74500,  0.70200],
                                            [0.27300,  0.75000,  0.23000],
                                            [0.00000,  0.00000,  0.00000],
                                            [0.00000,  0.00000,  1.00000],
                                            [0.00000,  1.00000,  1.00000],
                                            [0.00000,  1.00000,  0.00000]])
        
        transient_thermal_3d = TransientThermalLoss3DHexa("transient_thermal_loss_3d",loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                                            "material_dict":self.material_dict,
                                                                            "time_integration_dict":self.time_integration_dict},
                                                                            fe_mesh=self.fe_mesh)
        transient_thermal_3d.Initialize()

        en, residuals, stiffness = transient_thermal_3d.ComputeElement(hex_points_coordinates,
                                                                        jnp.ones((8)),
                                                                        jnp.ones((8,1)),
                                                                        jnp.ones((8,1)))
          
        np.testing.assert_allclose(stiffness,jnp.array([[6.215974688529968262e-03,5.613754037767648697e-04,-6.462504970841109753e-04,2.966466359794139862e-03,4.029757808893918991e-03,-3.223773965146392584e-04,-6.532279658131301403e-04,1.901246607303619385e-03],
                                                        [5.613754037767648697e-04,6.649139802902936935e-03,2.337607555091381073e-03,-5.379342474043369293e-05,-1.295738475164398551e-04,4.277510568499565125e-03,1.666182768531143665e-03,-2.244097995571792126e-04],
                                                        [-6.462506134994328022e-04,2.337607555091381073e-03,6.125086918473243713e-03,-8.917090017348527908e-04,-3.765789733733981848e-04,1.818985329009592533e-03,4.365476779639720917e-03,-5.362870288081467152e-04],
                                                        [2.966466359794139862e-03,-5.379342474043369293e-05,-8.917090017348527908e-04,6.713502109050750732e-03,1.965672476217150688e-03,-3.924447519239038229e-04,-1.232851296663284302e-03,4.187382292002439499e-03],
                                                        [4.029757808893918991e-03,-1.295738475164398551e-04,-3.765789733733981848e-04,1.965672476217150688e-03,1.212495844811201096e-02,-1.465172681491822004e-04,-8.175904513336718082e-04,5.883527453988790512e-03],
                                                        [-3.223776875529438257e-04,4.277510568499565125e-03,1.818985329009592533e-03,-3.924446646124124527e-04,-1.465175737394019961e-04,1.311568729579448700e-02,5.626044236123561859e-03,-5.841427482664585114e-04],
                                                        [-6.532280822284519672e-04,1.666182768531143665e-03,4.365476779639720917e-03,-1.232851296663284302e-03,-8.175907423719763756e-04,5.626044236123561859e-03,1.255425903946161270e-02,-1.582612399943172932e-03],
                                                        [1.901246607303619385e-03,-2.244097995571792126e-04,-5.362869123928248882e-04,4.187382292002439499e-03,5.883527453988790512e-03,-5.841425736434757710e-04,-1.582612749189138412e-03,1.237828284502029419e-02]]),
                                   rtol=1e-5, atol=1e-10)
        
        np.testing.assert_allclose(residuals.flatten(),jnp.array([0.000000000000000000e+00,9.313225746154785156e-10,0.000000000000000000e+00,0.000000000000000000e+00,
                                                                  1.862645149230957031e-09,1.862645149230957031e-09,0.000000000000000000e+00,1.862645149230957031e-09]),
                                   rtol=1e-5, atol=1e-10)
        
    def test_tri(self):
                
        quad_points_coordinates = jnp.array([[1.00,0.00,0.00],
                                            [0.50,0.50,0.00],
                                            [0.00,0.00,0.00]])

        transient_thermal_3d = TransientThermalLoss2DTri("transient_thermal_2d",
                                                  loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                                            "material_dict":self.material_dict,
                                                                            "time_integration_dict":self.time_integration_dict},
                                                                 fe_mesh=self.fe_mesh)
        transient_thermal_3d.Initialize()

        en, residuals, stiffness = transient_thermal_3d.ComputeElement(quad_points_coordinates,
                                                                     jnp.ones((3)),
                                                                     jnp.ones((3,1)),
                                                                     jnp.ones((3,1)))

        np.testing.assert_allclose(stiffness,jnp.array([[5.277776718139648438e-02,2.152777463197708130e-02,9.027775377035140991e-03],
                                                                  [2.152777463197708130e-02,3.090278059244155884e-02,3.090278059244155884e-02],
                                                                  [9.027775377035140991e-03,3.090278059244155884e-02,4.340277984738349915e-02]]),
                                   rtol=1e-5, atol=1e-10)
        
        np.testing.assert_allclose(residuals.flatten(),jnp.array([-7.450580596923828125e-09,7.450580596923828125e-09,7.450580596923828125e-09]),
                                   rtol=1e-5, atol=1e-10)
        
    def test_quad(self):
                
        quad_points_coordinates = jnp.array([[1.00,0.00,0.00],
                                            [1.00,1.00,0.00],
                                            [0.00,1.00,0.00],
                                            [0.00,0.00,0.00]])

        transient_thermal_2d = TransientThermalLoss2DQuad("transient_thermal_2d",
                                                  loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                                            "material_dict":self.material_dict,
                                                                            "time_integration_dict":self.time_integration_dict},
                                                                 fe_mesh=self.fe_mesh)
        transient_thermal_2d.Initialize()
                
        en, residuals, stiffness = transient_thermal_2d.ComputeElement(quad_points_coordinates,
                                                                     jnp.ones((4)),
                                                                     jnp.ones((4,1)),
                                                                     jnp.ones((4,1)))

        np.testing.assert_allclose(stiffness,jnp.array([[1.194444298744201660e-01,5.347222462296485901e-02,2.361111342906951904e-02,5.347222089767456055e-02],
                                                        [5.347222462296485901e-02,1.194444298744201660e-01,5.347222089767456055e-02,2.361111342906951904e-02],
                                                        [2.361111342906951904e-02,5.347222089767456055e-02,1.194444298744201660e-01,5.347222462296485901e-02],
                                                        [5.347222089767456055e-02,2.361111342906951904e-02,5.347222462296485901e-02,1.194444298744201660e-01]])
                                                        ,rtol=1e-5, atol=1e-10)

        np.testing.assert_allclose(residuals.flatten(),jnp.array([0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,-1.490116119384765625e-08])
                                   ,rtol=1e-5, atol=1e-10)


if __name__ == '__main__':
    unittest.main()