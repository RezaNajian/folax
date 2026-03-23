"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: December, 2024
 License: FOL/LICENSE
"""
from abc import ABC, abstractmethod
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *

class DataGenerator(ABC):
    """
    Abstract base class for creating data generators. This class provides a 
    standard interface for initializing, generating, and finalizing data generation processes.

    Attributes:
        generator_name (str): The name of the data generator.
        initialized (bool): Indicates whether the data generator has been initialized.
    """
    def __init__(self, generator_name: str) -> None:
        """
        Initializes the DataGenerator with a name.

        Args:
            generator_name (str): The name of the data generator.
        """
        self.__name = generator_name
        self.initialized = False

    def GetName(self) -> str:
        """
        Retrieves the name of the data generator.

        Returns:
            str: The name of the data generator.
        """
        return self.__name
    
    @abstractmethod
    def Initialize(self) -> None:
        """
        Abstract method to initialize the data generator.
        Implement this method to set up any resources or configurations needed for data generation.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass

    @abstractmethod
    def Generate(self):
        """
        Abstract method to generate data.
        Implement this method to define the data generation logic.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass

    @abstractmethod
    def Finalize(self) -> None:
        """
        Abstract method to finalize and clean up the data generator.
        Implement this method to release resources or perform any required cleanup.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass