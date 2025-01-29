"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: January, 2025
 License: FOL/LICENSE
"""
from  .response import Response
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.loss_functions.fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from sympy import symbols, Matrix
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify


class FiniteElementResponse(Response):

    def __init__(self, name: str, response_formula: str, fe_loss: FiniteElementLoss):
        super().__init__(name)
        self.response_formula = response_formula
        self.fe_loss = fe_loss

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:

        if self.initialized and not reinitialize:
            return
        
        self.fe_loss.Initialize()


        # Dynamically generate symbolic variables
        self.symbols_dict = {name: symbols(name) for name in self.fe_loss.dofs}
        self.symbols_list = list(self.symbols_dict.values())  # Preserve order

        # Define the vector U symbolically
        U = Matrix(self.symbols_list)

        # Parse the response formula using the generated symbols
        formula_expr = parse_expr(self.response_formula, local_dict={"U": U, "Matrix": Matrix})

        # Convert the expression to a JAX-compatible function
        self.jit_function = jax.jit(
            lambda U: lambdify(self.symbols_list, formula_expr, 'jax')(*U)
        )

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
    def ComputeElementValue(self,xyze,de,uvwe):
        @jit
        def compute_at_gauss_point(gp_point,gp_weight):
            N_vec = self.fe_loss.fe_element.ShapeFunctionsValues(gp_point)
            N_mat = self.CalculateNMatrix(N_vec)
            gp_dofs = N_mat @ uvwe
            J = self.fe_loss.fe_element.Jacobian(xyze,gp_point)
            detJ = jnp.linalg.det(J)            
            return gp_weight * detJ * self.jit_function(gp_dofs)

        gp_points,gp_weights = self.fe_loss.fe_element.GetIntegrationData()
        v_gps = jax.vmap(compute_at_gauss_point,in_axes=(0,0))(gp_points,gp_weights)
        return  jnp.sum(v_gps)
    
    @partial(jit, static_argnums=(0,))
    def ComputeElementStateGrad(self,xyze,de,uvwe):
        return jax.grad(self.ComputeElementValue,argnums=2)(xyze,de,uvwe)

    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergyVmapCompatible(self,
                                           element_id:jnp.integer,
                                           elements_nodes:jnp.array,
                                           xyz:jnp.array,
                                           full_control_vector:jnp.array,
                                           full_dof_vector:jnp.array):
        return self.ComputeElementValue(xyz[elements_nodes[element_id],:],
                                         full_control_vector[elements_nodes[element_id]],
                                         full_dof_vector[((self.fe_loss.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                         jnp.arange(self.fe_loss.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeElementsValues(self,total_control_vars:jnp.array,total_primal_vars:jnp.array):
        # parallel calculation of element values
        return jax.vmap(self.ComputeElementEnergyVmapCompatible,(0,None,None,None,None)) \
                        (self.fe_loss.fe_mesh.GetElementsIds(self.fe_loss.element_type),
                        self.fe_loss.fe_mesh.GetElementsNodes(self.fe_loss.element_type),
                        self.fe_loss.fe_mesh.GetNodesCoordinates(),
                        total_control_vars,
                        total_primal_vars)

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def ComputeValue(self,nodal_control_values:jnp.array,nodal_dof_values:jnp.array):
        return jnp.sum(self.ComputeElementsValues(nodal_control_values,nodal_dof_values))

    @partial(jit, static_argnums=(0,))
    def ComputeElementRHS(self,
                          elem_xyz:jnp.array,
                          elem_controls:jnp.array,
                          elem_dofs:jnp.array):
        return self.ComputeElementStateGrad(elem_xyz,elem_controls,elem_dofs)

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

        rhs_vector = rhs_vector.at[self.fe_loss.dirichlet_indices].set(0.0)

        # multiple by -1 
        rhs_vector *= -1

        # get the jacobian of the loss
        sparse_jacobian,_ = self.fe_loss.ComputeJacobianMatrixAndResidualVector(nodal_control_values,nodal_dof_values)

        return sparse_jacobian,rhs_vector


    def Finalize(self) -> None:
        pass

