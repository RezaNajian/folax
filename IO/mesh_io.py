"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/License.txt
"""
from  .input_output import InputOutput
from tools import *
import meshio

class MeshIO(InputOutput):
    """MeshIO class.

    The MeshIO class has the following responsibilities.
        1. Initalizes and finalizes the meshio.

    """
    @print_with_timestamp_and_execution_time
    def __init__(self,io_name:str,case_dir:str,file_name:str,bc_settings:dict) -> None:
        super().__init__(io_name)
        self.file_name = file_name
        self.case_dir = case_dir
        self.bc_settings = bc_settings

    def Initialize(self) -> None:
        pass

    @print_with_timestamp_and_execution_time
    def Import(self) -> None:
        self.mesh_io = meshio.read(os.path.join(self.case_dir, self.file_name))
        self.mesh_io.point_data_to_sets('point_tags')
        self.mesh_io.cell_data_to_sets('cell_tags')
        # points = mesh.points 
        # cells = mesh.cells_dict['tetra']

        self.dofs_dict = {}
        for dof_name,dof_settings in self.bc_settings.items():
            fol_dof_settings = {"non_dirichlet_nodes_ids":[],
                                "dirichlet_nodes_ids":[],
                                "dirichlet_nodes_dof_value":[]}
            for boundary_name,boundary_value in dof_settings.items():
                boundary_tag = None
                for tag_value,tage_names in self.mesh_io.point_tags.items():
                    if boundary_name in tage_names:
                        boundary_tag = tag_value
                if boundary_tag == None:
                    raise ValueError(f"boundary {boundary_name} does not exist !")
                else:
                    boundary_node_ids = self.mesh_io.point_sets[f"set-key-{boundary_tag}"]
                    boundary_node_values = [boundary_value] * len(boundary_node_ids)
                    fol_dof_settings["dirichlet_nodes_ids"].extend(boundary_node_ids)
                    fol_dof_settings["dirichlet_nodes_dof_value"].extend(boundary_node_values)

            for point_index in range(len(self.mesh_io.points)):
                if not point_index in fol_dof_settings["dirichlet_nodes_ids"]:
                    fol_dof_settings["non_dirichlet_nodes_ids"].append(point_index)

            fol_dof_settings["dirichlet_nodes_dof_value"] = np.array(fol_dof_settings["dirichlet_nodes_dof_value"])
            fol_dof_settings["non_dirichlet_nodes_ids"] = np.array(fol_dof_settings["non_dirichlet_nodes_ids"])
            fol_dof_settings["dirichlet_nodes_ids"] = np.array(fol_dof_settings["dirichlet_nodes_ids"])

            self.dofs_dict[dof_name] = fol_dof_settings

        self.nodes_dict = {"nodes_ids":jnp.arange(len(self.mesh_io.points)),"X":self.mesh_io.points[:,0],
                           "Y":self.mesh_io.points[:,1],"Z":self.mesh_io.points[:,2]}
        self.elements_dict = {"elements_ids":jnp.arange(len(self.mesh_io.cells_dict['tetra'])),
                        "elements_nodes":jnp.array(self.mesh_io.cells_dict['tetra'])}

    
        return {"nodes_dict":self.nodes_dict,"elements_dict":self.elements_dict,"dofs_dict":self.dofs_dict}

    def Finalize(self) -> None:
        pass



