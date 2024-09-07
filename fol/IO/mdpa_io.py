"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: July, 2024
 License: FOL/License.txt
"""
import os
import numpy as np
import jax.numpy as jnp
from  .input_output import InputOutput
from fol.tools.decoration_functions import *
from meshio import Mesh

_mdpa_to_meshio_type = {
    "Line2D2": "line",
    "Line3D2": "line",
    "Triangle2D3": "triangle",
    "Triangle3D3": "triangle",
    "Quadrilateral2D4": "quad",
    "Quadrilateral3D4": "quad",
    "Tetrahedra3D4": "tetra",
    "Hexahedra3D8": "hexahedron",
    "Prism3D6": "wedge",
    "Line2D3": "line3",
    "Triangle2D6": "triangle6",
    "Triangle3D6": "triangle6",
    "Quadrilateral2D9": "quad9",
    "Quadrilateral3D9": "quad9",
    "Tetrahedra3D10": "tetra10",
    "Hexahedra3D27": "hexahedron27",
    "Point2D": "vertex",
    "Point3D": "vertex",
    "Quadrilateral2D8": "quad8",
    "Quadrilateral3D8": "quad8",
    "Hexahedra3D20": "hexahedron20",
}

class MdpaIO(InputOutput):
    """MdpaIO class.

    The MdpaIO class has the following responsibilities.
        1. Initalizes and finalizes the meshio.

    """

    def __init__(self,io_name:str,file_name:str,case_dir:str='.',scale_factor:float=1,bc_settings:dict={}) -> None:
        super().__init__(io_name)
        self.file_name = file_name
        self.case_dir = case_dir
        self.bc_settings = bc_settings
        self.scale_factor = scale_factor

    def Initialize(self) -> None:
        pass

    @print_with_timestamp_and_execution_time
    def Import(self) -> None:
        self.__ReadMesh()

        self.dofs_dict = {}
        for dof_name,dof_settings in self.bc_settings.items():
            fol_dof_settings = {"non_dirichlet_nodes_ids":[],
                                "dirichlet_nodes_ids":[],
                                "dirichlet_nodes_dof_value":[]}
            for boundary_name,boundary_value in dof_settings.items():
                boundary_node_ids = self.point_sets[boundary_name]
                boundary_node_values = [boundary_value] * len(boundary_node_ids)
                fol_dof_settings["dirichlet_nodes_ids"].extend(boundary_node_ids)
                fol_dof_settings["dirichlet_nodes_dof_value"].extend(boundary_node_values)

            for point_index in range(len(self.points)):
                if not point_index in fol_dof_settings["dirichlet_nodes_ids"]:
                    fol_dof_settings["non_dirichlet_nodes_ids"].append(point_index)

            fol_dof_settings["dirichlet_nodes_dof_value"] = np.array(fol_dof_settings["dirichlet_nodes_dof_value"])
            fol_dof_settings["non_dirichlet_nodes_ids"] = np.array(fol_dof_settings["non_dirichlet_nodes_ids"])
            fol_dof_settings["dirichlet_nodes_ids"] = np.array(fol_dof_settings["dirichlet_nodes_ids"])

            self.dofs_dict[dof_name] = fol_dof_settings

        self.nodes_dict = {"nodes_ids":jnp.arange(len(self.points)),"X":self.points[:,0],
                           "Y":self.points[:,1],"Z":self.points[:,2]}
        self.elements_dict = {"elements_ids":jnp.arange(len(self.cells[-1][1])),
                        "elements_nodes":jnp.array(self.cells[-1][1])}

        return {"nodes_dict":self.nodes_dict,"elements_dict":self.elements_dict,"dofs_dict":self.dofs_dict}

    def __ReadMesh(self):
        self.points = []
        self.cells = []
        self.point_sets = {}
        with open(os.path.join(self.case_dir, self.file_name), "rb") as f:
            # Read mesh
            while True:
                line = f.readline().decode()
                if not line:
                    break
                environ = line.strip()
                if environ.startswith("Begin Nodes"):
                    self.__ReadNodes(f)
                elif environ.startswith("Begin Elements"):
                    self.__ReadCells(f, environ)
                elif environ.startswith("Begin SubModelPart "):
                    self.__ReadSubModelPart(f, environ)
        
        self.io = Mesh(self.points, self.cells)

    def __ReadNodes(self, f):
        pos = f.tell()
        num_nodes = 0
        while True:
            line = f.readline().decode()
            if "End Nodes" in line:
                break
            num_nodes += 1
        f.seek(pos)

        self.points = np.fromfile(f, count=num_nodes * 4, sep=" ").reshape((num_nodes, 4))
        # The first number is the index
        self.points = self.points[:, 1:] * self.scale_factor
        self.total_number_nodes = self.points.shape[0]
        fol_info(f"{self.total_number_nodes} points read ")

    def __ReadCells(self, f, environ=None):
        t = None
        if environ is not None:
            if environ.startswith("Begin Elements "):
                entity_name = environ[15:]
                for key in _mdpa_to_meshio_type:
                    if key in entity_name:
                        t = _mdpa_to_meshio_type[key]
                        break
        while True:
            line = f.readline().decode()
            if line.startswith("End Elements"):
                break
            data = [int(k) for k in filter(None, line.split())]
            num_nodes_per_elem = len(data) - 2

            if len(self.cells) == 0 or t != self.cells[-1][0]:
                self.cells.append((t, []))
            # Subtract one to account for the fact that python indices are 0-based.
            self.cells[-1][1].append(np.array(data[-num_nodes_per_elem:]) - 1)

        self.total_number_elements = len(self.cells[-1][1])
        fol_info(f"{self.total_number_elements} cells read ")

    def __ReadSubModelPart(self, f, environ=None):
        if environ is not None:
            model_part_name = environ[19:]
        else:
            return 
        node_ids = []
        line = f.readline().decode()
        if line.strip().startswith("Begin SubModelPartNodes"):
            while True:
                line = f.readline().decode()
                if line.strip().startswith("End SubModelPartNodes"):
                    break
                node_ids.append(int(line.strip())-1)

            self.point_sets[model_part_name] = node_ids
            fol_info(f"({model_part_name},{len(node_ids)} nodes) read ")

    def __getitem__(self, key):
        return self.io.point_data[key]
    
    def __setitem__(self, key, value):
        self.io.point_data[key] = value

    def GetNumberOfNodes(self):
        return self.total_number_nodes
    
    def GetSetNumberOfNodes(self,point_set_name):
        if point_set_name in self.point_sets.keys():
            return len(self.point_sets[point_set_name])
        else:
            raise ValueError(f"Set {point_set_name} does not exist ! ")

    def GetSetNodesIds(self,point_set_name):
        if point_set_name in self.point_sets.keys():
            return self.point_sets[point_set_name]
        else:
            raise ValueError(f"Set {point_set_name} does not exist ! ")

    def GetNumberOfElements(self):
        return self.total_number_elements

    def Export(self,export_dir:str=".",format:str="vtk"):
        file_name=self.file_name.split('.')[0]+"."+format
        self.io.write(os.path.join(export_dir, file_name),file_format=format)

    def Finalize(self) -> None:
        pass



