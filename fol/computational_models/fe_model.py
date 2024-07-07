"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from  .model import Model
from fol.tools.decoration_functions import *

class FiniteElementModel(Model):
    """Base abstract model class.

    The base abstract control class has the following responsibilities.
        1. Initalizes and finalizes the model.

    """
    @print_with_timestamp_and_execution_time
    def __init__(self, model_name: str, model_info) -> None:
        super().__init__(model_name)

        self.nodes_dict = model_info["nodes_dict"]
        self.total_number_nodes = self.nodes_dict["nodes_ids"].shape[-1] 
        self.elements_dict = model_info["elements_dict"]
        self.total_number_elements = self.elements_dict["elements_ids"].shape[-1]
        self.dofs_dict = model_info["dofs_dict"]  

    def Initialize(self) -> None:
        pass

    def GetNumberOfNodes(self):
        return self.total_number_nodes

    def GetNumberOfElements(self):
        return self.total_number_elements

    def GetNodesIds(self):
        return self.nodes_dict["nodes_ids"]    

    def GetElementsIds(self):
        return self.elements_dict["elements_ids"]
    
    def GetElementsNodes(self):
        return self.elements_dict["elements_nodes"]
    
    def GetNodesCoordinates(self):
        return self.nodes_dict["X"],self.nodes_dict["Y"],self.nodes_dict["Z"]
    
    def GetNodesX(self):
        return self.nodes_dict["X"]
    
    def GetNodesY(self):
        return self.nodes_dict["Y"]
    
    def GetNodesZ(self):
        return self.nodes_dict["Z"]

    def GetDofsDict(self):
        return self.dofs_dict
    
    def Finalize(self) -> None:
        pass



