"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: January, 2025
 License: FOL/LICENSE
"""
from abc import ABC, abstractmethod

class Response(ABC):
    """
    Abstract base class for defining a response in numerical optimization problems.
    """

    def __init__(self, response_name: str) -> None:
        """
        Initializes the Response object.

        Args:
            response_name (str): The name of the response.
        """
        self.__name = response_name
        self.initialized = False

    def GetName(self) -> str:
        """
        Returns the name of the response.

        Returns:
            str: The name of the response.
        """
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        """
        Initializes the response. 

        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def ComputeValue(self):
        """
        Computes the response value.

        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def ComputeAdjointJacobianMatrixAndRHSVector(self):
        """
        Computes the adjoint Jacobian matrix and the right-hand side (RHS) vector.

        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def ComputeAdjointNodalControlDerivatives(self):
        """
        Computes the adjoint-based nodal control derivatives.

        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def ComputeAdjointNodalShapeDerivatives(self):
        """
        Computes the adjoint-based nodal shape derivatives.

        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def ComputeFDNodalControlDerivatives(self):
        """
        Computes the finite difference (FD)-based nodal control derivatives.

        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def ComputeFDNodalShapeDerivatives(self):
        """
        Computes the finite difference (FD)-based nodal shape derivatives.

        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def Finalize(self) -> None:
        """
        Finalizes the response computation.

        This method must be implemented by subclasses.
        """
        pass



