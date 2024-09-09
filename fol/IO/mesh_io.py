"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/License.txt
"""
import os
import jax.numpy as jnp
from  .input_output import InputOutput
from fol.tools.decoration_functions import *
import meshio

class MeshIO(InputOutput):
    """MeshIO class.

    The MeshIO class has the following responsibilities.
        1. Initalizes and finalizes the meshio.

    """

    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:
        self.mesh_io = meshio.read(os.path.join(self.case_dir, self.file_name))
        self.mesh_io.point_data_to_sets('point_tags')
        self.mesh_io.cell_data_to_sets('cell_tags')
        self.node_ids = jnp.arange(len(self.mesh_io.points))
        self.nodes_coordinates = self.scale_factor * jnp.array(self.mesh_io.points)
        
        #create elemnt nodes dict based on element types
        self.elements_nodes = {}
        for elements_info in self.mesh_io.cells:
            self.elements_nodes[elements_info.type] = jnp.array(elements_info.data)
        
        # create node sets
        self.point_sets = {}
        for tag,tag_info_list in self.mesh_io.point_tags.items():
            if len(tag_info_list)>1:
                point_set_name = tag_info_list[1]
                self.point_sets[point_set_name] = jnp.array(self.mesh_io.point_sets[f"set-key-{tag}"])
                
        # TODO: create element sets 
        self.element_sets = {}

        self.is_initialized = True



