"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: January, 2025
 License: FOL/LICENSE
"""
from  .response import Response
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.loss_functions.fe_loss import FiniteElementLoss
from fol.controls.control import Control
import jax
import jax.numpy as jnp
from sympy import symbols, Matrix
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify


class FiniteElementResponse(Response):

    def __init__(self, name: str, response_formula: str, fe_loss: FiniteElementLoss, control: Control):
        super().__init__(name)
        self.response_formula = response_formula
        self.fe_loss = fe_loss
        self.control = control

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:

        if self.initialized and not reinitialize:
            return
        
        self.fe_loss.Initialize()
        self.control.Initialize()

        variables_list=[self.control.GetName(),self.fe_loss.dofs[0][0]]
        func_str = f"lambda {', '.join(variables_list)}: {self.response_formula}"
        self.jit_response_function = jax.jit(eval(func_str, {"jnp": jnp}))

        if self.fe_loss.dim == 2:
            self.CalculateNMatrix = self.CalculateNMatrix2D    
        else:
            self.CalculateNMatrix = self.CalculateNMatrix3D

        self.initialized = True

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
    def ComputeResponseElementValue(self,xyze,de,uvwe):
        @jit
        def compute_at_gauss_point(gp_point,gp_weight):
            N_vec = self.fe_loss.fe_element.ShapeFunctionsValues(gp_point)
            N_mat = self.CalculateNMatrix(N_vec)
            gp_dofs = (N_mat @ uvwe).flatten()
            gp_d = jnp.dot(N_vec, de.squeeze())
            J = self.fe_loss.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)            
            return gp_weight * detJ * self.jit_response_function(gp_d,gp_dofs)

        gp_points,gp_weights = self.fe_loss.fe_element.GetIntegrationData()
        v_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        return  jnp.sum(v_gps)
    
    @partial(jit, static_argnums=(0,))
    def ComputeResponseElementValueStateGrad(self,xyze,de,uvwe):
        return jax.grad(self.ComputeResponseElementValue,argnums=2)(xyze,de,uvwe)
    
    @partial(jit, static_argnums=(0,))
    def ComputeResponseElementValueControlGrad(self,xyze,de,uvwe):
        return jax.grad(self.ComputeResponseElementValue,argnums=1)(xyze,de,uvwe)
    
    @partial(jit, static_argnums=(0,))
    def ComputeResponseElementValueShapeGrad(self,xyze,de,uvwe):
        return jax.grad(self.ComputeResponseElementValue,argnums=0)(xyze,de,uvwe).flatten()

    @partial(jit, static_argnums=(0,))
    def ComputeResponseElementValueVmapCompatible(self,
                                           element_id:jnp.integer,
                                           elements_nodes:jnp.array,
                                           xyz:jnp.array,
                                           full_control_vector:jnp.array,
                                           full_dof_vector:jnp.array):
        return self.ComputeResponseElementValue(xyz[elements_nodes[element_id],:],
                                         full_control_vector[elements_nodes[element_id]],
                                         full_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                         jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeResponseElementsValues(self,total_control_vars:jnp.array,total_primal_vars:jnp.array):
        # parallel calculation of element values
        return jax.vmap(self.ComputeResponseElementValueVmapCompatible,(0,None,None,None,None)) \
                        (self.fe_loss.fe_mesh.GetElementsIds(self.fe_loss.element_type),
                        self.fe_loss.fe_mesh.GetElementsNodes(self.fe_loss.element_type),
                        self.fe_loss.fe_mesh.GetNodesCoordinates(),
                        total_control_vars,
                        total_primal_vars)

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def ComputeValue(self,nodal_control_values:jnp.array,nodal_dof_values:jnp.array):
        return jnp.sum(self.ComputeResponseElementsValues(nodal_control_values,nodal_dof_values))

    @partial(jit, static_argnums=(0,))
    def ComputeElementRHS(self,
                          elem_xyz:jnp.array,
                          elem_controls:jnp.array,
                          elem_dofs:jnp.array):
        return self.ComputeResponseElementValueStateGrad(elem_xyz,elem_controls,elem_dofs)

    @partial(jit, static_argnums=(0,))
    def ComputeElementRHSVmapCompatible(self,element_id:jnp.integer,
                                            elements_nodes:jnp.array,
                                            xyz:jnp.array,
                                            full_control_vector:jnp.array,
                                            full_dof_vector:jnp.array):
        return self.ComputeElementRHS(xyz[elements_nodes[element_id],:],
                                      full_control_vector[elements_nodes[element_id]],
                                      full_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1))


    @print_with_timestamp_and_execution_time
    def ComputeAdjointJacobianMatrixAndRHSVector(self,nodal_control_values:jnp.array,nodal_dof_values:jnp.array):
        
        elements_rhs = jax.vmap(self.ComputeElementRHSVmapCompatible,(0,None,None,None,None)) \
                                                            (self.fe_loss.fe_mesh.GetElementsIds(self.fe_loss.element_type),
                                                             self.fe_loss.fe_mesh.GetElementsNodes(self.fe_loss.element_type),
                                                             self.fe_loss.fe_mesh.GetNodesCoordinates(),
                                                             nodal_control_values,
                                                             nodal_dof_values)
        
        # first compute the global rhs vector
        rhs_vector = jnp.zeros((self.fe_loss.total_number_of_dofs))
        for dof_idx in range(self.fe_loss.number_dofs_per_node):
            rhs_vector = rhs_vector.at[self.fe_loss.number_dofs_per_node*self.fe_loss.fe_mesh.GetElementsNodes(self.fe_loss.element_type)+dof_idx].add(jnp.squeeze(elements_rhs[:,dof_idx::self.fe_loss.number_dofs_per_node]))

        # apply dirichlet bcs
        rhs_vector = rhs_vector.at[self.fe_loss.dirichlet_indices].set(0.0)

        # multiple by -1 
        rhs_vector *= -1

        # get the jacobian of the loss
        sparse_jacobian,_ = self.fe_loss.ComputeJacobianMatrixAndResidualVector(nodal_control_values,nodal_dof_values,False)

        return sparse_jacobian,rhs_vector

    @partial(jit, static_argnums=(0,))
    def ComputeResponseLocalNodalShapeDerivativesVmapCompatible(self,element_id:jnp.integer,
                                            elements_nodes:jnp.array,
                                            xyz:jnp.array,
                                            full_control_vector:jnp.array,
                                            full_dof_vector:jnp.array):
        return self.ComputeResponseElementValueShapeGrad(xyz[elements_nodes[element_id],:],
                                                    full_control_vector[elements_nodes[element_id]],
                                                    full_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                    jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeLossElementShapeGrad(self,xyze,de,uvwe,adj_uvwe):
        jacobian_fn = jax.jacrev(lambda *args: self.fe_loss.ComputeElement(*args)[1], argnums=0)
        res_shape_grads = jnp.squeeze(jacobian_fn(xyze, de, uvwe))
        res_shape_grads = res_shape_grads.reshape(*res_shape_grads.shape[:-2], -1)
        return (adj_uvwe.T @ res_shape_grads).flatten()

    @partial(jit, static_argnums=(0,))
    def ComputeAdjointLossElementShapeDerivativesVmapCompatible(self,element_id:jnp.integer,
                                            elements_nodes:jnp.array,
                                            xyz:jnp.array,
                                            full_control_vector:jnp.array,
                                            full_dof_vector:jnp.array,
                                            full_adj_dof_vector:jnp.array):
        return self.ComputeLossElementShapeGrad(xyz[elements_nodes[element_id],:],
                                                    full_control_vector[elements_nodes[element_id]],
                                                    full_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                    jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1),
                                                    full_adj_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                    jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1)                                                                    
                                                                    )

    @print_with_timestamp_and_execution_time
    def ComputeAdjointNodalShapeDerivatives(self,nodal_control_values:jnp.array,
                                                 nodal_dof_values:jnp.array,
                                                 nodal_adj_dof_values:jnp.array):
        response_elements_local_shape_derv = jax.vmap(self.ComputeResponseLocalNodalShapeDerivativesVmapCompatible,(0,None,None,None,None)) \
                                                            (self.fe_loss.fe_mesh.GetElementsIds(self.fe_loss.element_type),
                                                             self.fe_loss.fe_mesh.GetElementsNodes(self.fe_loss.element_type),
                                                             self.fe_loss.fe_mesh.GetNodesCoordinates(),
                                                             nodal_control_values,
                                                             nodal_dof_values)

        elements_residuals_adj_shape_derv = jax.vmap(self.ComputeAdjointLossElementShapeDerivativesVmapCompatible,(0,None,None,None,None,None)) \
                                                            (self.fe_loss.fe_mesh.GetElementsIds(self.fe_loss.element_type),
                                                             self.fe_loss.fe_mesh.GetElementsNodes(self.fe_loss.element_type),
                                                             self.fe_loss.fe_mesh.GetNodesCoordinates(),
                                                             nodal_control_values,
                                                             nodal_dof_values,
                                                             nodal_adj_dof_values)

        total_elem_shape_grads = response_elements_local_shape_derv + elements_residuals_adj_shape_derv
        # compute the global derivative vector
        grad_vector = jnp.zeros((3*self.fe_loss.fe_mesh.GetNumberOfNodes()))
        number_controls_per_node = 3
        for control_idx in range(number_controls_per_node):
            grad_vector = grad_vector.at[number_controls_per_node*self.fe_loss.fe_mesh.GetElementsNodes(self.fe_loss.element_type)+control_idx].add(jnp.squeeze(total_elem_shape_grads[:,control_idx::number_controls_per_node]))

        return grad_vector

    @partial(jit, static_argnums=(0,))
    def ComputeResponseLocalNodalControlDerivativesVmapCompatible(self,element_id:jnp.integer,
                                            elements_nodes:jnp.array,
                                            xyz:jnp.array,
                                            full_control_vector:jnp.array,
                                            full_dof_vector:jnp.array):
        return self.ComputeResponseElementValueControlGrad(xyz[elements_nodes[element_id],:],
                                                    full_control_vector[elements_nodes[element_id]],
                                                    full_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                    jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeLossElementControlGrad(self,xyze,de,uvwe,adj_uvwe):
        jacobian_fn = jax.jacrev(lambda *args: self.fe_loss.ComputeElement(*args)[1], argnums=1)
        res_control_grads = jnp.squeeze(jacobian_fn(xyze, de, uvwe))
        return (adj_uvwe.T @ res_control_grads).flatten()

    @partial(jit, static_argnums=(0,))
    def ComputeAdjointLossElementControlDerivativesVmapCompatible(self,element_id:jnp.integer,
                                            elements_nodes:jnp.array,
                                            xyz:jnp.array,
                                            full_control_vector:jnp.array,
                                            full_dof_vector:jnp.array,
                                            full_adj_dof_vector:jnp.array):
        return self.ComputeLossElementControlGrad(xyz[elements_nodes[element_id],:],
                                                    full_control_vector[elements_nodes[element_id]],
                                                    full_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                    jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1),
                                                    full_adj_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                    jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1)                                                                    
                                                                    )

    @print_with_timestamp_and_execution_time
    def ComputeAdjointNodalControlDerivatives(self,nodal_control_values:jnp.array,
                                                   nodal_dof_values:jnp.array,
                                                   nodal_adj_dof_values:jnp.array):
        response_elements_local_control_derv = jax.vmap(self.ComputeResponseLocalNodalControlDerivativesVmapCompatible,(0,None,None,None,None)) \
                                                            (self.fe_loss.fe_mesh.GetElementsIds(self.fe_loss.element_type),
                                                             self.fe_loss.fe_mesh.GetElementsNodes(self.fe_loss.element_type),
                                                             self.fe_loss.fe_mesh.GetNodesCoordinates(),
                                                             nodal_control_values,
                                                             nodal_dof_values)

        elements_residuals_adj_control_derv = jax.vmap(self.ComputeAdjointLossElementControlDerivativesVmapCompatible,(0,None,None,None,None,None)) \
                                                            (self.fe_loss.fe_mesh.GetElementsIds(self.fe_loss.element_type),
                                                             self.fe_loss.fe_mesh.GetElementsNodes(self.fe_loss.element_type),
                                                             self.fe_loss.fe_mesh.GetNodesCoordinates(),
                                                             nodal_control_values,
                                                             nodal_dof_values,
                                                             nodal_adj_dof_values)
        
        total_elem_control_grads = response_elements_local_control_derv + elements_residuals_adj_control_derv
        # compute the global derivative vector
        grad_vector = jnp.zeros((self.control.num_controlled_vars))
        number_controls_per_node = number_controls_per_node = int(self.control.num_controlled_vars / self.fe_loss.fe_mesh.GetNumberOfNodes())
        for control_idx in range(number_controls_per_node):
            grad_vector = grad_vector.at[number_controls_per_node*self.fe_loss.fe_mesh.GetElementsNodes(self.fe_loss.element_type)+control_idx].add(jnp.squeeze(total_elem_control_grads[:,control_idx::number_controls_per_node]))

        return grad_vector

    def Finalize(self) -> None:
        pass

