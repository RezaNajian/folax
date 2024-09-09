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

    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:
        self.cells = []
        self.node_sets = {}
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
                    self.__ReadElements(f, environ)
                elif environ.startswith("Begin SubModelPart "):
                    self.__ReadSubModelPart(f, environ)

        self.mesh_io = Mesh(self.nodes_coordinates,self.elements_nodes)

    def __ReadNodes(self, f):
        pos = f.tell()
        num_nodes = 0
        while True:
            line = f.readline().decode()
            if "End Nodes" in line:
                break
            num_nodes += 1
        f.seek(pos)

        nodes_data = np.fromfile(f, count=num_nodes * 4, sep=" ").reshape((num_nodes, 4))
        self.nodes_coordinates = nodes_data[:, 1:] * self.scale_factor
        self.node_ids = jnp.arange(len(self.nodes_coordinates))
        fol_info(f"{len(self.node_ids)} points read ")

    def __ReadElements(self, f, environ=None):
        mesh_io_element_type = None
        if environ is not None:
            if environ.startswith("Begin Elements "):
                entity_name = environ[15:]
                for key in _mdpa_to_meshio_type:
                    if key in entity_name:
                        mesh_io_element_type = _mdpa_to_meshio_type[key]
                        break
        kr_element_nodes = []          
        while True:
            line = f.readline().decode()
            if line.startswith("End Elements"):
                break
            data = [int(k) for k in filter(None, line.split())]
            num_nodes_per_elem = len(data) - 2

            # Subtract one to account for the fact that python indices are 0-based.
            kr_element_nodes.append(np.array(data[-num_nodes_per_elem:]) - 1)

        if mesh_io_element_type not in self.elements_nodes.keys():
            self.elements_nodes[mesh_io_element_type] = jnp.array(kr_element_nodes)
        else:
            self.elements_nodes[mesh_io_element_type] = jnp.vstack((self.elements_nodes[mesh_io_element_type],
                                                                    jnp.array(kr_element_nodes)))

        fol_info(f"{len(kr_element_nodes)} {mesh_io_element_type} elements read ")

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

            self.node_sets[model_part_name] = jnp.array(node_ids)
            fol_info(f"({model_part_name},{len(node_ids)} nodes) read ")

