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
        self.dirichlet_bc_dict = {"T":{}}
        self.material_dict = {"rho":1.0,"cp":1.0,"beta":1.5,"c":1.0}
        self.time_integration_dict = {"method":"implicit-euler","time_step":0.005}

    def test_tetra(self):

        tet_points_coordinates = jnp.array([[0.1, 0.1, 0.1],
                                            [0.28739360416666665, 0.27808503701741405, 0.05672979583333333],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 1.0, 0.1]])
        
        fe_mesh = Mesh("",".")
        fe_mesh.node_ids = jnp.arange(len(tet_points_coordinates))
        fe_mesh.nodes_coordinates = tet_points_coordinates
        fe_mesh.elements_nodes = {"tetra":fe_mesh.node_ids.reshape(1,-1)}      

        thermal_loss_3d = TransientThermalLoss3DTetra("transient_thermal_loss_3d",loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                                            "material_dict":self.material_dict,
                                                                            "time_integration_dict":self.time_integration_dict},
                                                                            fe_mesh=fe_mesh)
        thermal_loss_3d.Initialize()
        

        en, residuals, stiffness = thermal_loss_3d.ComputeElement(tet_points_coordinates,
                                                                        jnp.ones((4)),
                                                                        jnp.zeros((4,1)),
                                                                        jnp.ones((4,1)))

        np.testing.assert_allclose(stiffness,jnp.array([[5.377708002924919128e-03,-8.257009321823716164e-04,-1.326714846072718501e-04,-3.642407711595296860e-03],
                                                        [-8.257009321823716164e-04,6.439753924496471882e-04,2.637726720422506332e-04,6.948809023015201092e-04],
                                                        [-1.326714846072718501e-04,2.637726720422506332e-04,2.233165432699024677e-04,4.225104348734021187e-04],
                                                        [-3.642407711595296860e-03,6.948809023015201092e-04,4.225104348734021187e-04,3.301944816485047340e-03]])
                                                        ,rtol=1e-5, atol=1e-10)

        np.testing.assert_allclose(residuals.flatten(),jnp.array([-7.769281510263681412e-04,-7.769281510263681412e-04,-7.769281510263681412e-04,-7.769281510263681412e-04])
                                   ,rtol=1e-5, atol=1e-10)

    def test_hexa(self):
        hex_points_coordinates = jnp.array([[0.24900,  0.34200,  0.19200],
                                            [0.32000,  0.18600,  0.64300],
                                            [0.16500,  0.74500,  0.70200],
                                            [0.27300,  0.75000,  0.23000],
                                            [0.00000,  0.00000,  0.00000],
                                            [0.00000,  0.00000,  1.00000],
                                            [0.00000,  1.00000,  1.00000],
                                            [0.00000,  1.00000,  0.00000]])

        fe_mesh = Mesh("",".")
        fe_mesh.node_ids = jnp.arange(len(hex_points_coordinates))
        fe_mesh.nodes_coordinates = hex_points_coordinates
        fe_mesh.elements_nodes = {"hexahedron":fe_mesh.node_ids.reshape(1,-1)}  

        transient_thermal_3d = TransientThermalLoss3DHexa("transient_thermal_loss_3d",loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                                            "material_dict":self.material_dict,
                                                                            "time_integration_dict":self.time_integration_dict},
                                                                            fe_mesh=fe_mesh)
        transient_thermal_3d.Initialize()

        en, residuals, stiffness = transient_thermal_3d.ComputeElement(hex_points_coordinates,
                                                                        jnp.ones((8)),
                                                                        jnp.zeros((8,1)),
                                                                        jnp.ones((8,1)))

        np.testing.assert_allclose(stiffness,jnp.array([[4.569048993289470673e-03,1.371352816931903362e-03,2.573407255113124847e-04,2.214400796219706535e-03,3.225729567930102348e-03,7.280206191353499889e-04,1.327938662143424153e-04,1.554277492687106133e-03],
                                                        [1.371352816931903362e-03,5.164207424968481064e-03,1.970592420548200607e-03,4.943234962411224842e-04,8.051422191783785820e-04,3.525064326822757721e-03,1.449034665711224079e-03,3.043211472686380148e-04],
                                                        [2.573407255113124847e-04,1.970592420548200607e-03,4.087680950760841370e-03,5.598774878308176994e-04,2.434534981148317456e-04,1.510155620053410530e-03,3.062376752495765686e-03,5.048538441769778728e-04],
                                                        [2.214400796219706535e-03,4.943234962411224842e-04,5.598774878308176994e-04,4.713998176157474518e-03,1.580047886818647385e-03,2.371072041569277644e-04,2.262281632283702493e-04,3.236241638660430908e-03],
                                                        [3.225729567930102348e-03,8.051422191783785820e-04,2.434534981148317456e-04,1.580047886818647385e-03,9.222630411386489868e-03,2.222476759925484657e-03,7.334632100537419319e-04,4.500711802393198013e-03],
                                                        [7.280206191353499889e-04,3.525064326822757721e-03,1.510155620053410530e-03,2.371072041569277644e-04,2.222476759925484657e-03,9.997963905334472656e-03,4.345113877207040787e-03,8.268423844128847122e-04],
                                                        [1.327938662143424153e-04,1.449034665711224079e-03,3.062376752495765686e-03,2.262281632283702493e-04,7.334632100537419319e-04,4.345113877207040787e-03,8.648800663650035858e-03,1.327868434600532055e-03],
                                                        [1.554277492687106133e-03,3.043211472686380148e-04,5.048538441769778728e-04,3.236241638660430908e-03,4.500711802393198013e-03,8.268423844128847122e-04,1.327868434600532055e-03,9.167869575321674347e-03]]),
                                   rtol=1e-5, atol=1e-10)
        
        np.testing.assert_allclose(residuals.flatten(),jnp.array([-1.405296474695205688e-02,-1.508403755724430084e-02,-1.219633128494024277e-02,-1.326222531497478485e-02,-2.253365516662597656e-02,-2.339274436235427856e-02,-1.992568001151084900e-02,-2.142298594117164612e-02]),
                                   rtol=1e-5, atol=1e-10)
        
    def test_tri(self):
                
        tri_points_coordinates = jnp.array([[1.00,0.00,0.00],
                                            [0.50,0.50,0.00],
                                            [0.00,0.00,0.00]])
        
        fe_mesh = Mesh("",".")
        fe_mesh.node_ids = jnp.arange(len(tri_points_coordinates))
        fe_mesh.nodes_coordinates = tri_points_coordinates
        fe_mesh.elements_nodes = {"triangle":fe_mesh.node_ids.reshape(1,-1)}  

        transient_thermal_3d = TransientThermalLoss2DTri("transient_thermal_2d",
                                                  loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                                 "material_dict":self.material_dict,
                                                                 "time_integration_dict":self.time_integration_dict},
                                                                  fe_mesh=fe_mesh)
        transient_thermal_3d.Initialize()

        en, residuals, stiffness = transient_thermal_3d.ComputeElement(tri_points_coordinates,
                                                                     jnp.ones((3)),
                                                                     jnp.zeros((3,1)),
                                                                     jnp.ones((3,1)))

        np.testing.assert_allclose(stiffness,jnp.array([[3.777776658535003662e-02,2.527777478098869324e-02,2.027777396142482758e-02],
                                                        [2.527777478098869324e-02,2.902778051793575287e-02,2.902778051793575287e-02],
                                                        [2.027777396142482758e-02,2.902778051793575287e-02,3.402777761220932007e-02]]),
                                   rtol=1e-5, atol=1e-10)
        
        np.testing.assert_allclose(residuals.flatten(),jnp.array([-8.333332091569900513e-02,-8.333332836627960205e-02,-8.333332836627960205e-02]),
                                   rtol=1e-5, atol=1e-10)
        
    def test_quad(self):
                
        quad_points_coordinates = jnp.array([[1.00,0.00,0.00],
                                            [1.00,1.00,0.00],
                                            [0.00,1.00,0.00],
                                            [0.00,0.00,0.00]])
        
        fe_mesh = Mesh("",".")
        fe_mesh.node_ids = jnp.arange(len(quad_points_coordinates))
        fe_mesh.nodes_coordinates = quad_points_coordinates
        fe_mesh.elements_nodes = {"quad":fe_mesh.node_ids.reshape(1,-1)}  


        transient_thermal_2d = TransientThermalLoss2DQuad("transient_thermal_2d",
                                                  loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                                 "material_dict":self.material_dict,
                                                                 "time_integration_dict":self.time_integration_dict},
                                                                 fe_mesh=fe_mesh)
        transient_thermal_2d.Initialize()
                
        en, residuals, stiffness = transient_thermal_2d.ComputeElement(quad_points_coordinates,
                                                                     jnp.ones((4)),
                                                                     jnp.zeros((4,1)),
                                                                     jnp.ones((4,1)))

        np.testing.assert_allclose(stiffness,jnp.array([[1.144444271922111511e-01,5.472222343087196350e-02,2.611111290752887726e-02,5.472222343087196350e-02],
                                                        [5.472222343087196350e-02,1.144444271922111511e-01,5.472222343087196350e-02,2.611111290752887726e-02],
                                                        [2.611111290752887726e-02,5.472222343087196350e-02,1.144444271922111511e-01,5.472222343087196350e-02],
                                                        [5.472222343087196350e-02,2.611111290752887726e-02,5.472222343087196350e-02,1.144444271922111511e-01]])
                                                        ,rtol=1e-5, atol=1e-10)

        np.testing.assert_allclose(residuals.flatten(),jnp.array([-2.499999850988388062e-01,-2.499999850988388062e-01,-2.499999850988388062e-01,-2.500000000000000000e-01])
                                   ,rtol=1e-5, atol=1e-10)


if __name__ == '__main__':
    unittest.main()