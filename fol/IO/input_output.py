"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/License.txt
"""
from abc import ABC, abstractmethod
import jax.numpy as jnp
import numpy as np
import os
from fol.tools.decoration_functions import *

class InputOutput(ABC):
    """Base abstract input-output class.

    The base abstract InputOutput class has the following responsibilities.
        1. Initalizes and finalizes the IO.

    """
    def __init__(self, io_name: str, file_name:str, case_dir:str=".", scale_factor:float=1) -> None:
        self.__name = io_name
        self.file_name = file_name
        self.case_dir = case_dir
        self.scale_factor = scale_factor
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
    
    def __getitem__(self, key):
        return self.mesh_io.point_data[key]
    
    def __setitem__(self, key, value):
        self.mesh_io.point_data[key] = np.array(value)

    @print_with_timestamp_and_execution_time
    def Finalize(self,export_dir:str=".",export_format:str="vtk") -> None:
        file_name=self.file_name.split('.')[0]+"."+export_format
        self.mesh_io.write(os.path.join(export_dir, file_name),file_format=export_format)



