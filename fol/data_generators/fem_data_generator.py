"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: December, 2024
 License: FOL/LICENSE
"""
from fol.tools.decoration_functions import *
from .data_generator import DataGenerator
from fol.solvers.fe_solver import FiniteElementSolver
import jax.numpy as jnp
import numpy as np

class FemDataGenerator(DataGenerator):
    """
    A class to generate finite element method (FEM) data using a specified finite element solver.

    Inherits from:
        DataGenerator: Base class for data generation tasks.

    Attributes:
        fem_solver (FiniteElementSolver): An instance of a finite element solver used to solve FEM problems.
    """    
    def __init__(self, generator_name: str, fem_solver:FiniteElementSolver) -> None:
        """
        Initializes the FemDataGenerator with a name and a finite element solver.

        Args:
            generator_name (str): The name of the data generator.
            fem_solver (FiniteElementSolver): The finite element solver used for generating data.
        """
        super().__init__(generator_name)
        self.fem_solver = fem_solver
    
    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:
        """
        Initializes the data generator if it is not already initialized.

        This method sets the internal state to `initialized` to ensure subsequent operations can be performed.
        """
        if not self.initialized:
            self.fem_solver.Initialize()
            self.initialized = True

    @print_with_timestamp_and_execution_time        
    def Generate(self,control_matrix_data:jnp.array) -> np.array:
        """
        Generates FEM solutions for a given control matrix data.

        Args:
            control_matrix_data (jnp.array): A JAX array where each row represents control inputs for generating FEM solutions.

        Returns:
            np.array: A NumPy array where each row contains the FEM solution corresponding to a control input.

        Notes:
            - The size of each solution is determined by the number of degrees of freedom (DOFs) and the number of nodes in the FEM mesh.
            - Solutions are computed using the provided finite element solver.
        """
        num_samples = control_matrix_data.shape[0]
        solution_size = len(self.fem_solver.fe_loss_function.dofs) * \
                                self.fem_solver.fe_loss_function.fe_mesh.GetNumberOfNodes()
        self.generated_data = np.empty((0,solution_size))
        for data_index in range(num_samples):
            self.generated_data = np.vstack((self.generated_data,np.array(self.fem_solver.Solve(control_matrix_data[data_index],jnp.zeros(solution_size)))))
            fol_info(f"generated fem solution for sample {data_index+1}")
        
        return self.generated_data
    
    def Finalize(self) -> None:
        pass