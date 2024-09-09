"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/License.txt
"""
from abc import ABC, abstractmethod
import jax.numpy as jnp

class InputOutput(ABC):
    """Base abstract input-output class.

    The base abstract InputOutput class has the following responsibilities.
        1. Initalizes and finalizes the IO.

    """
    def __init__(self, io_name: str) -> None:
        self.__name = io_name
        self.node_ids = jnp.array([])
        self.nodes_coordinates = jnp.array([])
        self.elements_nodes = {}
        self.point_sets = {}
        self.element_sets = {}
        self.mesh_io = None
        self.is_initialized = False

    def GetName(self) -> str:
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        """Initializes the io.

        This method initializes the io.

        """
        pass

    def GetNodesIds(self) -> jnp.array:
        return self.node_ids
    
    def GetNumberOfNodes(self) -> int:
        return len(self.node_ids)

    def GetNodesCoordinates(self) -> jnp.array:
        return self.nodes_coordinates
    
    def GetNodesX(self) -> jnp.array:
        return self.nodes_coordinates[:,0]
    
    def GetNodesY(self) -> jnp.array:
        return self.nodes_coordinates[:,1]
    
    def GetNodesZ(self) -> jnp.array:
        return self.nodes_coordinates[:,2]
    
    def GetElementsIds(self,element_type) -> jnp.array:
        return jnp.arange(len(self.elements_nodes[element_type]))

    def GetElementsNodes(self,element_type) -> jnp.array:
        return self.elements_nodes[element_type]

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the io.

        This method finalizes the io. 

        """
        pass



