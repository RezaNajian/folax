"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2025
 License: FOL/LICENSE
"""
from  .thermal import ThermalLoss
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *

class TransientThermalLoss(ThermalLoss):

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:  
        if self.initialized and not reinitialize:
            return
        super().Initialize() 

        self.default_material_settings = {"rho":1.0,"cp":1.0,"k0":np.ones((self.fe_mesh.GetNumberOfNodes())),"beta":0.0,"c":1.0}
        self.default_time_integration_settings = {"method":"implicit-euler","time_step":None}

        if "material_dict" in self.loss_settings.keys():
            self.material_settings = UpdateDefaultDict(self.default_material_settings,self.loss_settings["material_dict"])
            if "k0" in self.loss_settings["material_dict"].keys():
                input_k0_shape = self.loss_settings["material_dict"]["k0"].shape
                if  input_k0_shape != self.default_material_settings["k0"].shape:
                    fol_error(f"provided k0({input_k0_shape}) in the material_dict does not match the mesh with {self.fe_mesh.GetNumberOfNodes()} nodes !")
                else:
                    self.material_settings["k0"] = jnp.array(self.material_settings["k0"])
        else:
            self.material_settings = self.default_material_settings

        self.time_integration_settings = UpdateDefaultDict(self.default_time_integration_settings,self.loss_settings["time_integration_dict"])
        if self.time_integration_settings["time_step"] == None:
            fol_error("time step should be provided in the time_integration_dict ")

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,te,body_force=0):
        @jit
        def compute_at_gauss_point(gp_point,gp_weight,te):
            te = jax.lax.stop_gradient(te)
            N = self.fe_element.ShapeFunctionsValues(gp_point)
            conductivity_at_gauss = jnp.dot(N, de.squeeze()) * (1 + 
                                    self.material_settings["beta"]*(jnp.dot(N,te.squeeze()))**self.material_settings["c"])
            DN_DX = self.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            gp_stiffness = conductivity_at_gauss * (DN_DX @ DN_DX.T) * detJ * gp_weight
            gp_f = gp_weight * detJ * body_force *  N.reshape(-1,1) 
            return gp_stiffness,gp_f
        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        k_gps,f_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0,None))(gp_points,gp_weights,te)
        Se = jnp.sum(k_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        def compute_elem_res(Se,te ,Fe):
            te = jax.lax.stop_gradient(te)
            return (Se @ te - Fe)
        element_residuals = compute_elem_res(Se,te ,Fe)
        return  ((te.T @ element_residuals)[0,0]), (Se @ te - Fe), Se

class TransientThermalLoss3DTetra(TransientThermalLoss):
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T"],  
                               "element_type":"tetra"},fe_mesh)
        
class TransientThermalLoss3DHexa(TransientThermalLoss):
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T"],  
                               "element_type":"hexahedron"},fe_mesh)

class TransientThermalLoss2DQuad(TransientThermalLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T"],  
                               "element_type":"quad"},fe_mesh)
        
class TransientThermalLoss2DTri(TransientThermalLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T"],  
                               "element_type":"triangle"},fe_mesh)