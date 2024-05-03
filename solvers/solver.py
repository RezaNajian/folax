"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/License.txt
"""
from abc import ABC, abstractmethod

class Solver(ABC):
    """Base abstract solver class.

    The base abstract control class has the following responsibilities.
        1. Initalizes and finalizes the solver.

    """
    def __init__(self, solver_name: str) -> None:
        self.__name = solver_name

    def GetName(self) -> str:
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        """Initializes the loss.

        This method initializes the loss. This is only called once in the whole training process.

        """
        pass

    @abstractmethod
    def Solve(self) -> None:
        """Solves for unknows.

        This method solves for the unknowns.

        """
        pass

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the solver.

        This method finalizes the solver.
        """
        pass



