"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/LICENSE
"""
from  .fe_loss_multiphysics import FiniteElementLoss
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *

class ThermoMechanicalNonlinearLoss(FiniteElementLoss):

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:  
        if self.initialized and not reinitialize:
            return
        super().Initialize() 
        self.thermal_loss_settings = {"beta":1,"c":1,"alpha":1,"T0":jnp.zeros((self.fe_mesh.GetNumberOfNodes()))}
        if "beta" in self.loss_settings.keys():
            self.thermal_loss_settings["beta"] = self.loss_settings["beta"]
        if "c" in self.loss_settings.keys():
            self.thermal_loss_settings["c"] = self.loss_settings["c"]
        if "alpha" in self.loss_settings.keys():
            self.thermal_loss_settings["alpha"] = self.loss_settings["alpha"]
        
        if "T0" in self.loss_settings["material_dict"].keys():
            self.thermal_loss_settings["T0"] = jnp.array(self.loss_settings["material_dict"]["T0"])
        else:
            self.thermal_loss_settings["T0"] = jnp.zeros((self.fe_mesh.GetNumberOfNodes(),))

        if self.dim == 2:
            self.CalculateNMatrix = self.CalculateNMatrix2D
            self.CalculateBMatrix = self.CalculateBMatrix2D
            self.D = self.CalculateDMatrix2D(self.loss_settings["material_dict"]["young_modulus"],
                                            self.loss_settings["material_dict"]["poisson_ratio"])
            self.body_force = jnp.zeros((2,1))
            self.thermal_st_vec = jnp.array([[1.0], [1.0], [0.0]])
            if "body_force" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_force"])
        else:
            self.CalculateNMatrix = self.CalculateNMatrix3D
            self.CalculateBMatrix = self.CalculateBMatrix3D
            self.D = self.CalculateDMatrix3D(self.loss_settings["material_dict"]["young_modulus"],
                                            self.loss_settings["material_dict"]["poisson_ratio"])
            self.body_force = jnp.zeros((3,1))
            self.thermal_st_vec = jnp.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
            if "body_force" in self.loss_settings:
                self.body_force = jnp.array(self.loss_settings["body_force"])

    @partial(jit, static_argnums=(0,))
    def CalculateBMatrix2D(self,DN_DX:jnp.array) -> jnp.array:
        B = jnp.zeros((3, 2 * DN_DX.shape[0]))
        indices = jnp.arange(DN_DX.shape[0])
        B = B.at[0, 2 * indices].set(DN_DX[indices,0])
        B = B.at[1, 2 * indices + 1].set(DN_DX[indices,1])
        B = B.at[2, 2 * indices].set(DN_DX[indices,1])
        B = B.at[2, 2 * indices + 1].set(DN_DX[indices,0])  
        return B

    @partial(jit, static_argnums=(0,))
    def CalculateBMatrix3D(self,DN_DX:jnp.array) -> jnp.array:
        B = jnp.zeros((6,3*DN_DX.shape[0]))
        index = jnp.arange(DN_DX.shape[0]) * 3
        B = B.at[0, index + 0].set(DN_DX[:,0])
        B = B.at[1, index + 1].set(DN_DX[:,1])
        B = B.at[2, index + 2].set(DN_DX[:,2])
        B = B.at[3, index + 0].set(DN_DX[:,1])
        B = B.at[3, index + 1].set(DN_DX[:,0])
        B = B.at[4, index + 1].set(DN_DX[:,2])
        B = B.at[4, index + 2].set(DN_DX[:,1])
        B = B.at[5, index + 0].set(DN_DX[:,2])
        B = B.at[5, index + 2].set(DN_DX[:,0])
        return B

    def CalculateDMatrix2D(self,young_modulus:float,poisson_ratio:float) -> jnp.array:
        return jnp.array([[1,poisson_ratio,0],[poisson_ratio,1,0],[0,0,(1-poisson_ratio)/2]]) * (young_modulus/(1-poisson_ratio**2))

    def CalculateDMatrix3D(self,young_modulus:float,poisson_ratio:float) -> jnp.array:
            # construction of the constitutive matrix
            c1 = young_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
            c2 = c1 * (1.0 - poisson_ratio)
            c3 = c1 * poisson_ratio
            c4 = c1 * 0.5 * (1.0 - 2.0 * poisson_ratio)
            D = jnp.zeros((6,6))
            D = D.at[0,0].set(c2)
            D = D.at[0,1].set(c3)
            D = D.at[0,2].set(c3)
            D = D.at[1,0].set(c3)
            D = D.at[1,1].set(c2)
            D = D.at[1,2].set(c3)
            D = D.at[2,0].set(c3)
            D = D.at[2,1].set(c3)
            D = D.at[2,2].set(c2)
            D = D.at[3,3].set(c4)
            D = D.at[4,4].set(c4)
            D = D.at[5,5].set(c4)
            return D
    
    @partial(jit, static_argnums=(0,))
    def CalculateNMatrix2D(self,N_vec:jnp.array) -> jnp.array:
        N_mat = jnp.zeros((2, 2 * N_vec.size))
        indices = jnp.arange(N_vec.size)   
        N_mat = N_mat.at[0, 2 * indices].set(N_vec)
        N_mat = N_mat.at[1, 2 * indices + 1].set(N_vec)    
        return N_mat
    
    @partial(jit, static_argnums=(0,))
    def CalculateNMatrix3D(self,N_vec:jnp.array) -> jnp.array:
        N_mat = jnp.zeros((3,3*N_vec.size))
        N_mat = N_mat.at[0,0::3].set(N_vec)
        N_mat = N_mat.at[1,1::3].set(N_vec)
        N_mat = N_mat.at[2,2::3].set(N_vec)
        return N_mat

    @partial(jit, static_argnums=(0,))
    def ComputeElement_1(self,xyze,de,te,se,te_init,body_force=0):
        # Thermal loss
        # de: conductivity/stiffness
        # te: temperature
        # ke: stiffness
        # se: displacement
        # te_init: initial temperature
        de = jax.lax.stop_gradient(de.reshape(-1,1))
        te = te.reshape(-1,1)
        # se = jax.lax.stop_gradient(se.reshape(-1,1))
        @jit
        def compute_at_gauss_point(gp_point,gp_weight):
            # Thermal part
            N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
            conductivity_at_gauss = jnp.dot(N_vec.reshape(1,-1), de) \
            * (1 + self.thermal_loss_settings["beta"]*(jnp.dot(N_vec,te.squeeze())**self.thermal_loss_settings["c"]))
            DN_DX = self.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            grad_T = DN_DX.T @ te
            gp_loss_t = 0.5*(grad_T.T@grad_T) * conductivity_at_gauss * detJ *gp_weight
            dk_dT = jnp.dot(N_vec.reshape(1,-1), de) * self.thermal_loss_settings["beta"] * \
                           self.thermal_loss_settings["c"] *((jnp.dot(N_vec,te.squeeze()))** (self.thermal_loss_settings["c"] - 1))
            gp_stiffness = conductivity_at_gauss * (DN_DX @ DN_DX.T) * detJ * gp_weight 
            gp_stiffness_thermal = dk_dT* ((DN_DX@grad_T)@N_vec.reshape(1,-1)) * detJ * gp_weight 
            gp_f = gp_weight * detJ * body_force *  N_vec.reshape(-1,1) 
            return gp_loss_t, gp_stiffness, gp_stiffness_thermal, gp_f
        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        loss_t_gps, s_gps, st_gps, f_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        Loss_t_e = jnp.sum(loss_t_gps)
        Se = jnp.sum(s_gps, axis=0)
        Se_t = jnp.sum(st_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        def compute_elem_res(Se,te,Fe):
            te = jax.lax.stop_gradient(te)
            return (Se @ te - Fe)
        element_residuals = compute_elem_res(Se,te ,Fe)
        # jax.debug.print("elem_res_thermal = {}", element_residuals.shape)
        Se_s = jnp.zeros((te.shape[0],se.shape[0]))
        # Se_t = jax.jacrev(compute_elem_res, argnums=1)(Se,te,Fe)
        return  te.T@jax.lax.stop_gradient(element_residuals), element_residuals, Se + Se_t, Se_s
    
    @partial(jit, static_argnums=(0,))
    def ComputeElement_2(self,xyze,de,te,se,te_init,body_force=0):
        # Mechanics loss
        # de: conductivity/stiffness
        # te: temperature
        # ke: stiffness
        # se: displacement
        # te_init: initial temperature
        # te = te.reshape(-1,1)
        ke = de.reshape(-1,1)
        se = se.reshape(-1,1)
        te = jax.lax.stop_gradient(te.reshape(-1,1))
        te_init = jax.lax.stop_gradient(te_init.reshape(-1,1))
        @jit
        def compute_at_gauss_point(gp_point,gp_weight):
            # Mechanical part
            N_vec = self.fe_element.ShapeFunctionsValues(gp_point)
            N_mat = self.CalculateNMatrix(N_vec)
            DN_DX = self.fe_element.ShapeFunctionsGlobalGradients(xyze,gp_point)
            B_mat = self.CalculateBMatrix(DN_DX)
            J = self.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)
            temp_at_gauss = jnp.dot(N_vec,te.squeeze())
            total_strain_at_gauss = B_mat@se
            thermal_strain_vec = self.thermal_loss_settings["alpha"] \
                * (temp_at_gauss - jnp.dot(N_vec, te_init.squeeze())) * self.thermal_st_vec
            elastic_strain = total_strain_at_gauss - thermal_strain_vec
            e_at_gauss = jnp.dot(N_vec, ke.squeeze())* (1 + self.thermal_loss_settings["beta"]*(temp_at_gauss)**self.thermal_loss_settings["c"])
            gp_loss_m = 0.5 * elastic_strain.T @ self.D @ elastic_strain *e_at_gauss * detJ * gp_weight
            gp_stiffness = gp_weight * detJ * e_at_gauss * (B_mat.T @ (self.D @ B_mat))
            gp_lhs = gp_weight * detJ * e_at_gauss * B_mat.T @ self.D @ elastic_strain
            # gp_f_thermal = gp_weight * detJ * e_at_gauss * B_mat.T @ self.D @ thermal_strain_vec
            gp_f = gp_weight * detJ * (N_mat.T @ self.body_force)
            de_dT = jnp.dot(N_vec, ke.squeeze())*self.thermal_loss_settings["beta"]*self.thermal_loss_settings["c"]*((temp_at_gauss)**(self.thermal_loss_settings["c"]-1))
            gp_stiffness_thermal = gp_weight * detJ * B_mat.T @ self.D @\
                (elastic_strain*de_dT - e_at_gauss*self.thermal_loss_settings["alpha"]*self.thermal_st_vec).reshape(-1,1) @ N_vec.reshape(1,-1)
            return gp_loss_m, gp_stiffness, gp_f, gp_stiffness_thermal,gp_lhs
        gp_points,gp_weights = self.fe_element.GetIntegrationData()
        loss_m_gps,s_gps, f_gps, st_gps, lhs_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        Loss_m_e = jnp.sum(loss_m_gps)
        Se = jnp.sum(s_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        # Fe_thermal = jnp.sum(ft_gps)
        Se_t = jnp.sum(st_gps, axis=0)
        LHSe = jnp.sum(lhs_gps,axis=0)
        def compute_elem_res(Se,se ,Fe, Fe_thermal):
            # se = jax.lax.stop_gradient(se)
            return (Se @ se - Fe - Fe_thermal)
        # element_residuals = compute_elem_res(Se,se ,Fe, Fe_t)
        element_residuals = LHSe - Fe
        # Se_t = jax.jacrev(compute_elem_res, argnums=1)(Se,se ,Fe, Fe_t)
        # jax.debug.print("elem_res_mech = {}", element_residuals.shape)
        return  se.T@jax.lax.stop_gradient(LHSe - Fe), element_residuals, Se, Se_t

    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergy(self,
                             elem_xyz:jnp.array,
                             elem_controls:jnp.array,
                             elem_dofs_1:jnp.array,
                             elem_dofs_2:jnp.array,
                             elem_features:jnp.array) -> float:
        return  (self.ComputeElement_1(elem_xyz,elem_controls,elem_dofs_1,elem_dofs_2,elem_features)[0],self.ComputeElement_2(elem_xyz,elem_controls,elem_dofs_1,elem_dofs_2,elem_features)[0])
    
    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergyVmapCompatible(self,
                                           element_id:jnp.integer,
                                           elements_nodes:jnp.array,
                                           xyz:jnp.array,
                                           full_control_vector:jnp.array,
                                           full_dof_vector_1:jnp.array,
                                           full_dof_vector_2:jnp.array,
                                           elem_features:jnp.array                                           
                                           ):
        return self.ComputeElementEnergy(xyz[elements_nodes[element_id],:],
                                         full_control_vector[elements_nodes[element_id]],
                                         full_dof_vector_1[((self.number_first_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                         jnp.arange(self.number_first_dofs_per_node))].reshape(-1,1),
                                         full_dof_vector_2[((self.number_second_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                         jnp.arange(self.number_second_dofs_per_node))].reshape(-1,1),  
                                         elem_features[elements_nodes[element_id]],
                                         )

    @partial(jit, static_argnums=(0,))
    def ComputeElementsEnergies(self,total_control_vars:jnp.array,total_primal_vars_1:jnp.array,total_primal_vars_2:jnp.array):
        # parallel calculation of energies
        return jax.vmap(self.ComputeElementEnergyVmapCompatible,(0,None,None,None,None,None,None)) \
                        (self.fe_mesh.GetElementsIds(self.element_type),
                        self.fe_mesh.GetElementsNodes(self.element_type),
                        self.fe_mesh.GetNodesCoordinates(),
                        total_control_vars,
                        total_primal_vars_1,
                        total_primal_vars_2,
                        self.thermal_loss_settings["T0"]                  
                        )
    
    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualAndJacobian(self,
                                          elem_xyz:jnp.array,
                                          elem_controls:jnp.array,
                                          elem_dofs_1:jnp.array,
                                          elem_dofs_2:jnp.array,
                                          elem_features:jnp.array,
                                          elem_BC_1:jnp.array,
                                          elem_BC_2:jnp.array,
                                          elem_mask_BC_1:jnp.array,
                                          elem_mask_BC_2:jnp.array,
                                          transpose_jac:bool):
        _,re_1,ke_11,ke_12 = self.ComputeElement_1(elem_xyz,elem_controls,elem_dofs_1,elem_dofs_2,elem_features)
        _,re_2,ke_22,ke_21 = self.ComputeElement_2(elem_xyz,elem_controls,elem_dofs_1,elem_dofs_2,elem_features)

       # Convert transpose_jac (bool) to an integer index (0 = False, 1 = True)
        index = jnp.asarray(transpose_jac, dtype=jnp.int32)

        re = jnp.concatenate([re_1, re_2], axis=0) 
        ke = jnp.block([[ke_11, ke_12], [ke_21, ke_22]])

        # Define the two branches for switch
        branches = [
            lambda _: ke,                  # Case 0: No transpose
            lambda _: jnp.transpose(ke)    # Case 1: Transpose ke
        ]

        # Apply the switch operation
        ke = jax.lax.switch(index, branches, None)
        elem_BC = jnp.concatenate([elem_BC_1, elem_BC_2], axis=0)
        elem_mask_BC = jnp.concatenate([elem_mask_BC_1, elem_mask_BC_2], axis=0)

        return self.ApplyDirichletBCOnElementResidualAndJacobian(re,ke,elem_BC,elem_mask_BC)

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualAndJacobianVmapCompatible(self,element_id:jnp.integer,
                                                        elements_nodes:jnp.array,
                                                        xyz:jnp.array,
                                                        full_control_vector:jnp.array,
                                                        full_dof_vector_1:jnp.array,
                                                        full_dof_vector_2:jnp.array,
                                                        full_dirichlet_BC_vec_first_dofs:jnp.array,
                                                        full_dirichlet_BC_vec_second_dofs:jnp.array,
                                                        full_mask_dirichlet_BC_vec_first_dofs:jnp.array,
                                                        full_mask_dirichlet_BC_vec_second_dofs:jnp.array,
                                                        transpose_jac:bool):
        return self.ComputeElementResidualAndJacobian(xyz[elements_nodes[element_id],:],
                                                      full_control_vector[elements_nodes[element_id]],
                                                      full_dof_vector_1[((self.number_first_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_first_dofs_per_node))].reshape(-1,1),
                                                      full_dof_vector_2[((self.number_second_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_second_dofs_per_node))].reshape(-1,1),
                                                      self.thermal_loss_settings["T0"][elements_nodes[element_id]],
                                                      full_dirichlet_BC_vec_first_dofs[((self.number_first_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_first_dofs_per_node))].reshape(-1,1),
                                                      full_dirichlet_BC_vec_second_dofs[((self.number_second_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_second_dofs_per_node))].reshape(-1,1),
                                                      full_mask_dirichlet_BC_vec_first_dofs[((self.number_first_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_first_dofs_per_node))].reshape(-1,1),
                                                      full_mask_dirichlet_BC_vec_second_dofs[((self.number_second_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.number_second_dofs_per_node))].reshape(-1,1),                                                      
                                                      transpose_jac)
    
class ThermoMechanicalLoss3DTetra(ThermoMechanicalNonlinearLoss):
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T", "Ux","Uy","Uz"],  
                               "first_dofs": ["T"], 
                               "second_dofs": ["Ux","Uy","Uz"],                                           
                               "element_type":"tetra"},fe_mesh)
        
class ThermoMechanicalLoss3DHexa(ThermoMechanicalNonlinearLoss):
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":3,
                               "ordered_dofs": ["T", "Ux","Uy","Uz"], 
                               "first_dofs": ["T"], 
                               "second_dofs": ["Ux","Uy","Uz"],                               
                               "element_type":"hexahedron"},fe_mesh)

class ThermoMechanicalLoss2DQuad(ThermoMechanicalNonlinearLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        if not "num_gp" in loss_settings.keys():
            loss_settings["num_gp"] = 2
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T", "Ux","Uy"], 
                               "first_dofs": ["T"], 
                               "second_dofs": ["Ux","Uy"],  
                               "element_type":"quad"},fe_mesh)
        
class ThermoMechanicalLoss2DTri(ThermoMechanicalNonlinearLoss):
    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh):
        super().__init__(name,{**loss_settings,"compute_dims":2,
                               "ordered_dofs": ["T", "Ux","Uy"],
                               "first_dofs": ["T"], 
                               "second_dofs": ["Ux","Uy"],  
                               "element_type":"triangle"},fe_mesh)
