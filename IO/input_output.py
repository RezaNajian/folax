"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/License.txt
"""
from abc import ABC, abstractmethod

class InputOutput(ABC):
    """Base abstract input-output class.

    The base abstract InputOutput class has the following responsibilities.
        1. Initalizes and finalizes the IO.

    """
    def __init__(self, io_name: str) -> None:
        self.__name = io_name

    def GetName(self) -> str:
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        """Initializes the io.

        This method initializes the io.

        """
        pass

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the io.

        This method finalizes the io. 

        """
        pass



