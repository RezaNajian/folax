"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: December, 2024
 License: FOL/LICENSE
"""
from abc import ABC, abstractmethod
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *

class DataIO(ABC):
    """
    DataIO: Abstract Base Class for Data Input/Output Operations

    The `DataIO` class serves as a blueprint for creating input/output (I/O) handlers 
    in data-processing pipelines. It provides a unified interface and enforces the implementation 
    of `Import` and `Export` methods in derived classes. 

    This class is designed to be extended by subclasses that implement specific I/O behaviors 
    (e.g., reading from or writing to a database, file system, or API).

    Attributes:
        io_name (str): A name identifying the I/O instance. This name helps in distinguishing 
                    between different I/O operations or contexts.

    Methods:
        __init__(io_name: str) -> None:
            Initializes the instance with a given name.
            
        GetName() -> str:
            Retrieves the name of the I/O instance.
            
        Import() -> None:
            Abstract method that must be implemented in subclasses to handle data import functionality.

        Export() -> None:
            Abstract method that must be implemented in subclasses to handle data export functionality.

    """
    def __init__(self, io_name: str) -> None:
        """
        Initializes a DataIO instance.

        Args:
            io_name (str): The name of the I/O instance. This name is used to
            uniquely identify the I/O operation or its associated context.
        """
        self.__name = io_name

    def GetName(self) -> str:
        """
        Retrieves the name of the I/O instance.

        Returns:
            str: The name of the I/O instance.
        """
        return self.__name
    
    @abstractmethod
    def Import(self) -> None:
        """
        Abstract method for importing data.

        This method must be implemented in a derived class to define
        how data is imported in the context of the specific I/O operation.
        """
        pass

    @abstractmethod
    def Export(self) -> None:
        """
        Abstract method for exporting data.

        This method must be implemented in a derived class to define
        how data is exported in the context of the specific I/O operation.
        """
        pass