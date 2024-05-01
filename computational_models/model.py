"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from abc import ABC, abstractmethod

class Model(ABC):
    """Base abstract model class.

    The base abstract control class has the following responsibilities.
        1. Initalizes and finalizes the model.

    """
    def __init__(self, model_name: str) -> None:
        self.__name = model_name

    def GetName(self) -> str:
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        """Initializes the model.

        This method initializes the model. This is only called once in the whole training process.

        """
        pass

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the model.

        This method finalizes the model. This is only called once in the whole training process.

        """
        pass



