from abc import ABC, abstractmethod

class Loss(ABC):
    """Base abstract loss class.

    The base abstract control class has the following responsibilities.
        1. Initalizes and finalizes the loss.

    """
    def __init__(self, loss_name: str) -> None:
        self.__name = loss_name

    def GetName(self) -> str:
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        """Initializes the loss.

        This method initializes the loss. This is only called once in the whole training process.

        """
        pass

    @abstractmethod
    def ComputeSingleLoss(self) -> None:
        """Computes the single loss.

        This method initializes the loss. This is only called once in the whole training process.

        """
        pass

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the loss.

        This method finalizes the loss. This is only called once in the whole training process.

        """
        pass



