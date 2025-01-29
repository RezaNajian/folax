"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: January, 2025
 License: FOL/LICENSE
"""
from abc import ABC, abstractmethod

class Response(ABC):

    def __init__(self, response_name: str) -> None:
        self.__name = response_name
        self.initialized = False

    def GetName(self) -> str:
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        pass

    @abstractmethod
    def ComputeValue(self):
        pass

    @abstractmethod
    def ComputeAdjointJacobianMatrixAndRHSVector(self):
        pass

    @abstractmethod
    def Finalize(self) -> None:
        pass



