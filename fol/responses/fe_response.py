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
from fol.solvers.fe_solver import FiniteElementSolver
import jax
from tqdm import trange
import jax.numpy as jnp
import numpy as np

class FiniteElementResponse(Response):
    """
    A derived class that represents a finite element response in numerical optimization.

    This class extends the `Response` base class and provides functionalities for handling
    finite element loss computations, control parameters, and response evaluations using JAX.

    Attributes:
        response_formula (str): The formula used to compute the response.
        fe_loss (FiniteElementLoss): The finite element loss object containing DOFs and problem settings.
        control (Control): The control object representing optimization parameters.
    """

    def __init__(self, name: str, response_formula: str, fe_loss: FiniteElementLoss, control: Control):
        """
        Initializes the `FiniteElementResponse` object.

        Args:
            name (str): The name of the response.
            response_formula (str): A string representation of the response formula.
            fe_loss (FiniteElementLoss): A finite element loss object containing DOFs and configurations.
            control (Control): A control object representing optimization parameters.
        """

        super().__init__(name)
        self.response_formula = response_formula
        self.fe_loss = fe_loss
        self.control = control

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initializes the finite element response by setting up necessary computations.

        If the response is already initialized, it will not be reinitialized unless
        explicitly requested.

        Args:
            reinitialize (bool, optional): If True, forces reinitialization. Defaults to False.
        """

        if self.initialized and not reinitialize:
            return
        
        self.fe_loss.Initialize()
        self.control.Initialize()

        variables_list=[self.control.GetName(),self.fe_loss.dofs[0][0]]
        func_str = f"lambda {', '.join(variables_list)}: {self.response_formula}"
        self.jit_response_function = jax.jit(eval(func_str, {"jnp": jnp}))

        self.initialized = True

    @partial(jit, static_argnums=(0,))
    def CalculateNMatrix(self,N_vec:jnp.array) -> jnp.array:
        """
        Computes the shape function matrix (N) for finite elements.

        This function generates a num_dofsx(num_dofs*N) shape function matrix, where N is the number of shape functions.

        Args:
            N_vec (jnp.array): The vector of shape function values.

        Returns:
            jnp.array: The computed shape function matrix.
        """

        num_dofs = self.fe_loss.number_dofs_per_node
        N_mat = jnp.zeros((num_dofs,  num_dofs * N_vec.size))
        indices = jnp.arange(num_dofs)[:, None] 
        cols = jnp.arange(N_vec.size) * num_dofs 
        N_mat = N_mat.at[indices, cols + indices].set(N_vec)
        return N_mat
    
    @partial(jit, static_argnums=(0,))
    def ComputeResponseElementValue(self,xyze,de,uvwe):
        """
        Computes the response value for a single finite element.

        This method calculates the response contribution from a single element by integrating 
        over the element's Gauss points.

        Args:
            xyze (jnp.array): The nodal coordinates of the element.
            de (jnp.array): The control variables associated with the element.
            uvwe (jnp.array): The state variables (displacements) associated with the element.

        Returns:
            jnp.array: The computed response value for the element.
        """

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
        """
        Computes the gradient of the response's element with respect to the state variables.

        Args:
            xyze (jnp.array): The nodal coordinates of the element.
            de (jnp.array): The control variables associated with the element.
            uvwe (jnp.array): The state variables (displacements) associated with the element.

        Returns:
            jnp.array: The gradient of the response with respect to the state variables.
        """

        return jax.grad(self.ComputeResponseElementValue,argnums=2)(xyze,de,uvwe)
    
    @partial(jit, static_argnums=(0,))
    def ComputeResponseElementValueControlGrad(self,xyze,de,uvwe):
        """
        Computes the gradient of the response's element with respect to the control variables.

        Args:
            xyze (jnp.array): The nodal coordinates of the element.
            de (jnp.array): The control variables associated with the element.
            uvwe (jnp.array): The state variables (displacements) associated with the element.

        Returns:
            jnp.array: The gradient of the response with respect to the control variables.
        """

        return jax.grad(self.ComputeResponseElementValue,argnums=1)(xyze,de,uvwe)
    
    @partial(jit, static_argnums=(0,))
    def ComputeResponseElementValueShapeGrad(self,xyze,de,uvwe):
        """
        Computes the gradient of the response's element with respect to the shape (nodal coordinates).

        Args:
            xyze (jnp.array): The nodal coordinates of the element.
            de (jnp.array): The control variables associated with the element.
            uvwe (jnp.array): The state variables (displacements) associated with the element.

        Returns:
            jnp.array: The gradient of the response with respect to the nodal coordinates, flattened.
        """

        return jax.grad(self.ComputeResponseElementValue,argnums=0)(xyze,de,uvwe).flatten()

    @partial(jit, static_argnums=(0,))
    def ComputeResponseElementValueVmapCompatible(self,
                                           element_id:jnp.integer,
                                           elements_nodes:jnp.array,
                                           xyz:jnp.array,
                                           full_control_vector:jnp.array,
                                           full_dof_vector:jnp.array):
        """
        Computes the response value for a single element in a vectorized-compatible manner.

        Args:
            element_id (jnp.integer): The ID of the element.
            elements_nodes (jnp.array): The connectivity matrix of elements to nodes.
            xyz (jnp.array): The coordinates of all nodes.
            full_control_vector (jnp.array): The global control variable vector.
            full_dof_vector (jnp.array): The global state variable vector.

        Returns:
            jnp.array: The computed response value for the given element.
        """
        return self.ComputeResponseElementValue(xyz[elements_nodes[element_id],:],
                                         full_control_vector[elements_nodes[element_id]],
                                         full_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                         jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1))

    @print_with_timestamp_and_execution_time
    def ComputeValue(self,nodal_control_values:jnp.array,nodal_dof_values:jnp.array):
        """
        Computes the total response value by summing the contributions from all elements.

        Args:
            nodal_control_values (jnp.array): The global nodal control variable vector.
            nodal_dof_values (jnp.array): The global nodal state variable vector.

        Returns:
            jnp.array: The total computed response value.
        """
        return jnp.sum(jax.vmap(self.ComputeResponseElementValueVmapCompatible,(0,None,None,None,None)) \
                        (self.fe_loss.fe_mesh.GetElementsIds(self.fe_loss.element_type),
                        self.fe_loss.fe_mesh.GetElementsNodes(self.fe_loss.element_type),
                        self.fe_loss.fe_mesh.GetNodesCoordinates(),
                        nodal_control_values,
                        nodal_dof_values))

    @partial(jit, static_argnums=(0,))
    def ComputeElementRHSVmapCompatible(self,element_id:jnp.integer,
                                            elements_nodes:jnp.array,
                                            xyz:jnp.array,
                                            full_control_vector:jnp.array,
                                            full_dof_vector:jnp.array):
        """
        Computes the RHS vector for a single element in a vectorized-compatible manner.

        The element RHS vector is obtained as the gradient of the response with respect to 
        the element's state variables.

        Args:
            element_id (jnp.integer): The ID of the element.
            elements_nodes (jnp.array): The connectivity matrix of elements to nodes.
            xyz (jnp.array): The coordinates of all nodes.
            full_control_vector (jnp.array): The global control variable vector.
            full_dof_vector (jnp.array): The global state variable vector.

        Returns:
            jnp.array: The computed RHS vector for the given element.
        """

        return self.ComputeResponseElementValueStateGrad(xyz[elements_nodes[element_id],:],
                                      full_control_vector[elements_nodes[element_id]],
                                      full_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                      jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1))


    @print_with_timestamp_and_execution_time
    def ComputeAdjointJacobianMatrixAndRHSVector(self,nodal_control_values:jnp.array,nodal_dof_values:jnp.array):
        """
        Computes the adjoint Jacobian matrix and RHS vector for the finite element system.

        The RHS vector is computed by summing element-wise contributions, applying Dirichlet 
        boundary conditions, and scaling appropriately. The adjoint Jacobian matrix is obtained from 
        the finite element loss function, which is transpose of the state Jacobian matrix.

        Args:
            nodal_control_values (jnp.array): The global nodal control variable vector.
            nodal_dof_values (jnp.array): The global nodal state variable vector.

        Returns:
            Tuple[jnp.array, jnp.array]: A tuple containing:
                - sparse_jacobian (jnp.array): The computed adjoint Jacobian matrix.
                - rhs_vector (jnp.array): The computed RHS vector for the system.
        """

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

        # get the jacobian of the loss with transpose flag 
        sparse_jacobian,_ = self.fe_loss.ComputeJacobianMatrixAndResidualVector(nodal_control_values,nodal_dof_values,True)

        return sparse_jacobian,rhs_vector

    @partial(jit, static_argnums=(0,))
    def ComputeResponseLocalNodalShapeDerivativesVmapCompatible(self,element_id:jnp.integer,
                                            elements_nodes:jnp.array,
                                            xyz:jnp.array,
                                            full_control_vector:jnp.array,
                                            full_dof_vector:jnp.array):
        """
        Computes the local nodal shape derivatives of the response function for a given element 
        in a vectorized-compatible manner.

        Args:
            element_id (jnp.integer): The ID of the element.
            elements_nodes (jnp.array): The connectivity matrix of elements to nodes.
            xyz (jnp.array): The coordinates of all nodes.
            full_control_vector (jnp.array): The global control variable vector.
            full_dof_vector (jnp.array): The global state variable vector.

        Returns:
            jnp.array: The computed shape derivatives for the given element.
        """

        return self.ComputeResponseElementValueShapeGrad(xyz[elements_nodes[element_id],:],
                                                    full_control_vector[elements_nodes[element_id]],
                                                    full_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                    jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeLossElementShapeGrad(self,xyze,de,uvwe,adj_uvwe):
        """
        Computes the adjoint-based shape gradient of the loss function for a given finite element.

        This function calculates the sensitivity of the loss function with respect to 
        nodal coordinates using automatic differentiation (jacobian of the residual) and adjoint vars.

        Args:
            xyze (jnp.array): The nodal coordinates of the element.
            de (jnp.array): The control variables associated with the element.
            uvwe (jnp.array): The state variables (displacements) associated with the element.
            adj_uvwe (jnp.array): The adjoint state variables.

        Returns:
            jnp.array: The shape gradient of the loss function for the element.
        """

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
        """
        Computes the shape derivatives of the loss function for an element in a vectorized-compatible manner.

        Args:
            element_id (jnp.integer): The ID of the element.
            elements_nodes (jnp.array): The connectivity matrix of elements to nodes.
            xyz (jnp.array): The coordinates of all nodes.
            full_control_vector (jnp.array): The global control variable vector.
            full_dof_vector (jnp.array): The global state variable vector.
            full_adj_dof_vector (jnp.array): The global adjoint state variable vector.

        Returns:
            jnp.array: The computed shape derivatives for the given element.
        """

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
        """
        Computes the adjoint-based nodal shape derivatives for the entire finite element mesh.

        This function calculates local shape derivatives for each element using automatic differentiation,
        then assembles the global derivative vector.

        Args:
            nodal_control_values (jnp.array): The global nodal control variable vector.
            nodal_dof_values (jnp.array): The global nodal state variable vector.
            nodal_adj_dof_values (jnp.array): The global adjoint state variable vector.

        Returns:
            jnp.array: The assembled global shape derivative vector.
        """        

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
        """
        Computes the local nodal control derivatives of the response function for a given element 
        in a vectorized-compatible manner.

        Args:
            element_id (jnp.integer): The ID of the element.
            elements_nodes (jnp.array): The connectivity matrix of elements to nodes.
            xyz (jnp.array): The coordinates of all nodes.
            full_control_vector (jnp.array): The global control variable vector.
            full_dof_vector (jnp.array): The global state variable vector.

        Returns:
            jnp.array: The computed control derivatives for the given element.
        """
        
        return self.ComputeResponseElementValueControlGrad(xyz[elements_nodes[element_id],:],
                                                    full_control_vector[elements_nodes[element_id]],
                                                    full_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                    jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeLossElementControlGrad(self,xyze,de,uvwe,adj_uvwe):
        """
        Computes the adjoint-based control gradient of the loss function for a given finite element.

        This function calculates the sensitivity of the loss function with respect to control variables
        using automatic differentiation (jacobian of the residual) and element adjoint vector.

        Args:
            xyze (jnp.array): The nodal coordinates of the element.
            de (jnp.array): The control variables associated with the element.
            uvwe (jnp.array): The state variables (displacements) associated with the element.
            adj_uvwe (jnp.array): The adjoint state variables.

        Returns:
            jnp.array: The control gradient of the loss function for the element.
        """

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
        """
        Computes the control derivatives of the loss function for an element in a vectorized-compatible manner.

        Args:
            element_id (jnp.integer): The ID of the element.
            elements_nodes (jnp.array): The connectivity matrix of elements to nodes.
            xyz (jnp.array): The coordinates of all nodes.
            full_control_vector (jnp.array): The global control variable vector.
            full_dof_vector (jnp.array): The global state variable vector.
            full_adj_dof_vector (jnp.array): The global adjoint state variable vector.

        Returns:
            jnp.array: The computed control derivatives for the given element.
        """

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
        """
        Computes the adjoint-based nodal control derivatives for the entire finite element mesh.

        This function calculates local control derivatives for each element using automatic differentiation,
        then assembles the global derivative vector.

        Args:
            nodal_control_values (jnp.array): The global nodal control variable vector.
            nodal_dof_values (jnp.array): The global nodal state variable vector.
            nodal_adj_dof_values (jnp.array): The global adjoint state variable vector.

        Returns:
            jnp.array: The assembled global control derivative vector.
        """

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
        number_controls_per_node = int(self.control.num_controlled_vars / self.fe_loss.fe_mesh.GetNumberOfNodes())
        for control_idx in range(number_controls_per_node):
            grad_vector = grad_vector.at[number_controls_per_node*self.fe_loss.fe_mesh.GetElementsNodes(self.fe_loss.element_type)+control_idx].add(jnp.squeeze(total_elem_control_grads[:,control_idx::number_controls_per_node]))

        return grad_vector
    
    @print_with_timestamp_and_execution_time
    def ComputeFDNodalControlDerivatives(self,nodal_control_values:jnp.array,
                                              fe_solver:FiniteElementSolver,
                                              fd_step_size:float=1e-4,
                                              fd_mode="FWD"):
        # solve for the unperturbed controls
        unpert_dofs = fe_solver.Solve(nodal_control_values,jnp.zeros(self.fe_loss.number_dofs_per_node*self.fe_loss.fe_mesh.GetNumberOfNodes()))

        if fd_mode=="FWD" or fd_mode=="CD":
            unpert_res_val = self.ComputeValue(nodal_control_values,unpert_dofs)
        else:
            fol_error("only Forward (FWD), Central Difference (CD) methods are implemented !")

        pbar = trange(nodal_control_values.shape[0])

        FD_grad_vector = jnp.zeros((nodal_control_values.shape[0]))
        for control_idx in pbar:
            # per forward
            nodal_control_values = nodal_control_values.at[control_idx].add(fd_step_size)
            # calculate fw
            fw_dofs = fe_solver.Solve(nodal_control_values,jnp.zeros(self.fe_loss.number_dofs_per_node*self.fe_loss.fe_mesh.GetNumberOfNodes()))
            fw_res_val = self.ComputeValue(nodal_control_values,fw_dofs)
            if fd_mode=="FWD":
                FD_sens = (fw_res_val-unpert_res_val)/fd_step_size
            
            # remove pert
            nodal_control_values = nodal_control_values.at[control_idx].add(-fd_step_size)

            # now backward if CD
            if fd_mode=="CD":
                nodal_control_values = nodal_control_values.at[control_idx].add(-fd_step_size)
                bw_dofs = fe_solver.Solve(nodal_control_values,jnp.zeros(self.fe_loss.number_dofs_per_node*self.fe_loss.fe_mesh.GetNumberOfNodes()))
                bw_res_val = self.ComputeValue(nodal_control_values,bw_dofs)
                FD_sens = (fw_res_val-bw_res_val)/(2*fd_step_size)
                # remove pert
                nodal_control_values = nodal_control_values.at[control_idx].add(fd_step_size)

            pbar.set_postfix({f"control:":control_idx,f"{fd_mode} sensitivity:":FD_sens})

            FD_grad_vector = FD_grad_vector.at[control_idx].set(FD_sens)

        return FD_grad_vector

    @print_with_timestamp_and_execution_time
    def ComputeFDNodalShapeDerivatives(self,nodal_control_values:jnp.array,
                                            fe_solver:FiniteElementSolver,
                                            fd_step_size:float=1e-4,
                                            fd_mode="FWD"):
        # solve for the unperturbed controls
        unpert_dofs = fe_solver.Solve(nodal_control_values,jnp.zeros(self.fe_loss.number_dofs_per_node*self.fe_loss.fe_mesh.GetNumberOfNodes()))

        if fd_mode=="FWD" or fd_mode=="CD":
            unpert_res_val = self.ComputeValue(nodal_control_values,unpert_dofs)
        else:
            fol_error("only Forward (FWD), Central Difference (CD) methods are implemented !")

        pbar = trange(self.fe_loss.fe_mesh.GetNumberOfNodes())

        def pert_and_compute(node_idx,component):
            self.fe_loss.fe_mesh.nodes_coordinates = self.fe_loss.fe_mesh.nodes_coordinates.at[node_idx,component].add(fd_step_size)
            fw_pert_dofs = fe_solver.Solve(nodal_control_values,jnp.zeros(self.fe_loss.number_dofs_per_node*self.fe_loss.fe_mesh.GetNumberOfNodes()))
            fw_pert_res_val = self.ComputeValue(nodal_control_values,fw_pert_dofs)
            if fd_mode=="FWD":
                self.fe_loss.fe_mesh.nodes_coordinates = self.fe_loss.fe_mesh.nodes_coordinates.at[node_idx,component].add(-fd_step_size)
                return (fw_pert_res_val-unpert_res_val)/fd_step_size
            elif fd_mode=="CD":
                self.fe_loss.fe_mesh.nodes_coordinates = self.fe_loss.fe_mesh.nodes_coordinates.at[node_idx,component].add(-2*fd_step_size)
                bw_pert_dofs = fe_solver.Solve(nodal_control_values,jnp.zeros(self.fe_loss.number_dofs_per_node*self.fe_loss.fe_mesh.GetNumberOfNodes()))
                bw_pert_res_val = self.ComputeValue(nodal_control_values,bw_pert_dofs)
                self.fe_loss.fe_mesh.nodes_coordinates = self.fe_loss.fe_mesh.nodes_coordinates.at[node_idx,component].add(fd_step_size)
                return (fw_pert_res_val-bw_pert_res_val)/(2*fd_step_size)

        FD_grad_vector = jnp.zeros((self.fe_loss.fe_mesh.GetNumberOfNodes(),3))
        for node_idx in pbar:
            FD_sens = jnp.zeros((3))
            for component in range(self.fe_loss.dim):
                FD_sens = FD_sens.at[component].set(pert_and_compute(node_idx,component))
            
            FD_grad_vector = FD_grad_vector.at[node_idx].set(FD_sens)

            pbar.set_postfix({f"Node:":node_idx,f"{fd_mode} shape sensitivity:":FD_sens})

        return FD_grad_vector.flatten()

    def Finalize(self) -> None:
        pass

