from  .model import Model

class FiniteElementModel(Model):
    """Base abstract model class.

    The base abstract control class has the following responsibilities.
        1. Initalizes and finalizes the model.

    """
    def __init__(self, model_name: str, model_info) -> None:
        super().__init__(model_name)
        self.elements_ids = model_info["element_ids"]
        self.elements_nodes = model_info["elements_nodes"]
        self.total_number_elements = self.elements_nodes.shape[0]
        self.X = model_info["X"]
        self.Y = model_info["Y"]
        self.Z = model_info["Z"]
        self.boundary_nodes = model_info["boundary_nodes"]
        self.boundary_values = model_info["boundary_values"]
        self.non_boundary_nodes = model_info["non_boundary_nodes"]
        self.total_number_nodes = self.non_boundary_nodes.shape[0] + self.boundary_nodes.shape[0]

    def Initialize(self) -> None:
        pass

    def GetNumberOfNodes(self):
        return self.total_number_nodes

    def GetNumberOfElements(self):
        return self.total_number_elements
    
    def GetElementsIds(self):
        return self.elements_ids
    
    def GetElementsNodes(self):
        return self.elements_nodes
    
    def GetNodesCoordinates(self):
        return self.X,self.Y,self.Z
    
    def GetNodesX(self):
        return self.X
    
    def GetNodesY(self):
        return self.Y
    
    def GetNodesZ(self):
        return self.Z
    
    def GetBoundaryNodesIds(self):
        return self.boundary_nodes
    
    def GetBoundaryNodesValues(self):
        return self.boundary_values
    
    def GetNumberOfBoundaryNodes(self):
        return self.boundary_nodes.shape[0]

    def GetNoneBoundaryNodesIds(self):
        return self.non_boundary_nodes
    
    def GetNumberOfNoneBoundaryNodes(self):
        return self.non_boundary_nodes.shape[0]

    def Finalize(self) -> None:
        pass



