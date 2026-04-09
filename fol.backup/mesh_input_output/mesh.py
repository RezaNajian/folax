"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/LICENSE
"""
from abc import ABC
import jax.numpy as jnp
import jax
import numpy as np
import os
import meshio
from fol.tools.decoration_functions import *
from fol.geometries import fe_element_dict

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

class Mesh(ABC):
    """
    Base mesh input/output class for finite element simulations.

    This class provides a unified interface for reading, storing, querying, and
    exporting finite element meshes used throughout FoLax. It supports both
    general-purpose mesh formats via the ``meshio`` Python library and Kratos
    Multiphysics ``.mdpa`` mesh files via a custom parser.

    When the input file format is not ``.mdpa``, the mesh is read using
    ``meshio.read``. In this case, ``meshio`` is responsible for parsing node
    coordinates, element connectivities, and mesh metadata. Node sets are
    constructed from ``meshio`` point tags, and element connectivities are
    stored by element type in a dictionary. This allows FoLax to support a wide
    range of mesh formats such as VTK, Gmsh, Salome med, and others supported by
    ``meshio``.

    When the input file format is ``.mdpa``, the mesh is read using a custom
    parser implemented in this class. Nodes, elements, and sub-model parts are
    extracted directly from the file. The resulting data are then wrapped in a
    ``meshio.Mesh`` object so that downstream operations (such as export) can
    use the same interface as meshes read via ``meshio``.

    After reading the mesh, element orientations are checked using the
    reference finite element definitions in ``fe_element_dict``. Elements with
    negative Jacobian determinants at the reference integration point are
    automatically re-oriented to ensure consistent element orientation.

    The mesh object stores:
    - node identifiers and coordinates,
    - element connectivity grouped by element type,
    - node sets defined by boundary or sub-model-part names,
    - optional point data accessible through ``mesh_io``.

    Args:
        io_name (str):
            Name identifier for the mesh object.
        file_name (str):
            Name of the mesh file to be read. The file extension determines
            whether ``meshio`` or the Kratos ``.mdpa`` parser is used.
        case_dir (str, optional):
            Directory containing the mesh file. Default is the current
            directory.
        scale_factor (float, optional):
            Scaling factor applied to node coordinates after reading the mesh.
            Default is ``1``.

    Notes:
        - The ``meshio`` library is used as a backend for reading and writing
          standard mesh formats and for managing point data. FoLax relies on
          ``meshio`` to provide a consistent representation of nodes and
          elements across different file formats.
        - The ``Finalize`` method exports the mesh using ``meshio.write`` and
          can be used to write results (including point data added during a
          simulation) to formats such as VTK.
        - This class does not perform any FE assembly itself; it only provides
          mesh topology and geometry to losses, solvers, and controls.
    """
    def __init__(self, io_name: str, file_name:str, case_dir:str=".", scale_factor:float=1) -> None:
        self.__name = io_name
        self.file_name = file_name
        self.mesh_format = file_name.split(sep=".")[1]
        self.case_dir = case_dir
        self.scale_factor = scale_factor
        self.node_ids = jnp.array([])
        self.nodes_coordinates = jnp.array([])
        self.elements_nodes = {}
        self.node_sets = {}
        self.element_sets = {}
        self.mesh_io = None
        self.is_initialized = False

    def GetName(self) -> str:
        return self.__name

    def Initialize(self) -> None:
        """
        Read and initialize the mesh from disk.

        If the mesh has already been initialized, this method returns
        immediately without modifying the internal state.

        The behavior depends on the mesh file extension. For files that are not
        in ``mdpa`` format, the mesh is read using the ``meshio`` library. Node
        coordinates, element connectivity, and tag-based node sets are extracted
        from the ``meshio`` object and stored internally.

        For files in ``mdpa`` format, a custom parser is used to read nodes,
        elements, and sub-model parts directly from the file. The parsed data
        are then wrapped into a ``meshio.Mesh`` object so that export and point
        data handling use the same interface as meshes read via ``meshio``.

        After reading the mesh, element orientations are checked and corrected
        when possible to ensure consistent element Jacobians.
        """

        if self.is_initialized:
            return

        if self.mesh_format != "mdpa":
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
            self.node_sets = {}
            for tag,tag_info_list in self.mesh_io.point_tags.items():
                filtered_tag_info_list = [item for item in tag_info_list if 'Group_Of_All_Nodes' not in item]
                if len(filtered_tag_info_list)>1:
                    fol_error(f" the input mesh is not valid ! point set {filtered_tag_info_list} is not unique !")
                elif len(filtered_tag_info_list)==1:
                    point_set_name = filtered_tag_info_list[0]
                    self.node_sets[point_set_name] = jnp.array(self.mesh_io.point_sets[f"set-key-{tag}"])

            # TODO: create element sets
            self.element_sets = {}

        else:
            with open(os.path.join(self.case_dir, self.file_name), "rb") as f:
                    # Read mesh
                    while True:
                        line = f.readline().decode()
                        if not line:
                            break
                        environ = line.strip()
                        if environ.startswith("Begin Nodes"):
                            self.__ReadKratosNodes(f)
                        elif environ.startswith("Begin Elements"):
                            self.__ReadKratosElements(f, environ)
                        elif environ.startswith("Begin SubModelPart "):
                            self.__ReadKratosSubModelPart(f, environ)

            self.mesh_io = meshio.Mesh(self.nodes_coordinates,self.elements_nodes)

        fol_info(f"{len(self.node_ids)} points read ")
        for element_type,element_nodes in self.elements_nodes.items():
            fol_info(f"{len(element_nodes)} {element_type} elements read ")
        for node_set_name,node_ids in self.node_sets.items():
            fol_info(f"({node_set_name},{len(node_ids)} nodes) read ")

        self.CheckAndOrientElements()

        self.is_initialized = True

    def CheckAndOrientElements(self):
        """
        Check element orientation and swap nodes for inverted elements.

        For each element type present in ``self.elements_nodes`` that also exists
        in ``fe_element_dict``, this method computes the Jacobian determinant at
        the reference integration point. If the determinant is negative, the
        element is considered inverted and its first two node indices are swapped
        (a minimal correction strategy).

        This method updates ``self.elements_nodes`` in place for affected element
        types.

        Notes:
            - Only element types recognized by ``fe_element_dict`` are processed.
            - If swapping does not fix some elements, a warning is emitted.
        """
        jax_nodes_coords = jnp.array(self.nodes_coordinates)
        for element_type,elements_nodes in self.elements_nodes.items():
            if element_type in fe_element_dict.keys():
                fol_element = fe_element_dict[element_type]
                gp_point,_= fol_element.GetIntegrationData()
                @jax.jit
                def negative_det(elem_nodes):
                    elem_nodes_coordinates = jax_nodes_coords[elem_nodes]
                    det = jnp.linalg.det(fol_element.Jacobian(elem_nodes_coordinates,gp_point))
                    return jnp.where(det >= 0, 0, 1),jnp.where(det >= 0, elem_nodes, elem_nodes.at[0].set(elem_nodes[1]).at[1].set(elem_nodes[0]))
                elem_state,swap_elems_nodes = jax.vmap(negative_det)(elements_nodes)
                num_neg_jac_elems = jnp.sum(elem_state)
                if num_neg_jac_elems>0:
                    fol_warning(f"nodes of {num_neg_jac_elems} {element_type} elements with negative jacobian are swapped !")
                    self.elements_nodes[element_type] = swap_elems_nodes
                    new_elem_state,_ = jax.vmap(negative_det)(self.elements_nodes[element_type])
                    num_neg_jac_elems = jnp.sum(new_elem_state)
                    if num_neg_jac_elems>0:
                        fol_warning(f"although nodes are swapped, {num_neg_jac_elems} {element_type} elements still have negative jacobians or inverted !")

    def GetNodesIds(self) -> jnp.array:
        """
        Return the node id array.

        Returns:
            Array of node ids with shape ``(num_nodes,)``.
        """
        return self.node_ids

    def GetNumberOfNodes(self) -> int:
        """
        Return the number of nodes in the mesh.

        Returns:
            Total number of mesh nodes.
        """
        return len(self.node_ids)

    def GetNodesCoordinates(self) -> jnp.array:
        """
        Return nodal coordinates.

        Returns:
            Array of nodal coordinates with shape ``(num_nodes, dim)``.
        """
        return self.nodes_coordinates

    def GetNodesX(self) -> jnp.array:
        """
        Return x-coordinates of all nodes.

        Returns:
            Array of x-coordinates with shape ``(num_nodes,)``.
        """
        return self.nodes_coordinates[:,0]

    def GetNodesY(self) -> jnp.array:
        """
        Return y-coordinates of all nodes.

        Returns:
            Array of y-coordinates with shape ``(num_nodes,)``.
        """
        return self.nodes_coordinates[:,1]

    def GetNodesZ(self) -> jnp.array:
        """
        Return z-coordinates of all nodes.

        This assumes the mesh is 3D or that the third coordinate exists.

        Returns:
            Array of z-coordinates with shape ``(num_nodes,)``.
        """
        return self.nodes_coordinates[:,2]

    def GetElementsIds(self,element_type) -> jnp.array:
        """
        Return element ids for a given element type.

        Args:
            element_type:
                Element type key used in ``self.elements_nodes`` (meshio type).

        Returns:
            Array of element ids with shape ``(num_elements,)``.

        Raises:
            KeyError:
                If ``element_type`` is not present in the mesh.
        """
        return jnp.arange(len(self.elements_nodes[element_type]))

    def GetNumberOfElements(self,element_type) -> jnp.array:
        """
        Return number of elements for a given element type.

        Args:
            element_type:
                Element type key used in ``self.elements_nodes``.

        Returns:
            Number of elements of that type.

        Raises:
            KeyError:
                If ``element_type`` is not present in the mesh.
        """
        return len(self.elements_nodes[element_type])

    def GetElementsNodes(self,element_type) -> jnp.array:
        """
        Return element connectivity for a given element type.

        Args:
            element_type:
                Element type key used in ``self.elements_nodes``.

        Returns:
            Connectivity array of shape ``(num_elements, num_nodes_per_element)``.

        Raises:
            KeyError:
                If ``element_type`` is not present in the mesh.
        """
        return self.elements_nodes[element_type]

    def GetNodeSet(self,set_name) -> jnp.array:
        """
        Return node ids belonging to a named node set.

        Args:
            set_name:
                Name of the node set.

        Returns:
            Array of node indices in the set.

        Raises:
            KeyError:
                If ``set_name`` does not exist in ``self.node_sets``.
        """
        return self.node_sets[set_name]

    def HasPointData(self,data_name):
        """
        Check whether point data with the given name exists in ``mesh_io``.

        Args:
            data_name:
                Name of the point data field.

        Returns:
            True if the point data exists, otherwise False.
        """
        return data_name in self.mesh_io.point_data

    def __getitem__(self, key):
        """
        Access a point-data array stored in the underlying ``meshio`` mesh.

        This provides dictionary-like access to ``self.mesh_io.point_data``.

        Args:
            key:
                Name of the point-data field.

        Returns:
            The point-data array associated with ``key``.

        Raises:
            KeyError:
                If ``key`` does not exist in ``mesh_io.point_data``.
        """
        return self.mesh_io.point_data[key]

    def __setitem__(self, key, value):
        """
        Set a point-data array in the underlying ``meshio`` mesh.

        The value is converted to a NumPy array before assignment.

        Args:
            key:
                Name of the point-data field to set.
            value:
                Data to store. Must be broadcastable to the number of mesh nodes.

        Returns:
            None
        """
        self.mesh_io.point_data[key] = np.array(value)

    @print_with_timestamp_and_execution_time
    def Finalize(self,export_dir:str=".",export_format:str="vtk") -> None:
        """
        Export the mesh (and any attached point data) to disk using ``meshio``.

        The output file name is derived from the original ``file_name`` by
        replacing the extension with ``export_format``.

        Args:
            export_dir:
                Output directory.
            export_format:
                Output file format supported by ``meshio`` (e.g. ``vtk``).

        Returns:
            None
        """
        file_name=self.file_name.split('.')[0]+"."+export_format
        self.mesh_io.write(os.path.join(export_dir, file_name),file_format=export_format)

    def __ReadKratosNodes(self, f):
        """
        Read Kratos ``.mdpa`` node block from an open binary file handle.

        This method assumes the file cursor is positioned right after the
        ``Begin Nodes`` line. It reads until ``End Nodes`` to count nodes,
        then uses ``numpy.fromfile`` to parse the numeric block efficiently.

        Args:
            f:
                Open file handle in binary mode.

        Returns:
            None
        """
        pos = f.tell()
        num_nodes = 0
        while True:
            line = f.readline().decode()
            if "End Nodes" in line:
                break
            num_nodes += 1
        f.seek(pos)

        nodes_data = np.fromfile(f, count=num_nodes * 4, sep=" ").reshape((num_nodes, 4))
        self.nodes_coordinates = jnp.array(nodes_data[:, 1:] * self.scale_factor)
        self.node_ids = jnp.arange(len(self.nodes_coordinates))

    def __ReadKratosElements(self, f, environ=None):
        """
        Read a Kratos ``.mdpa`` element block and append to ``elements_nodes``.

        The Kratos entity name is mapped to a meshio element type using the
        ``_mdpa_to_meshio_type`` dictionary.

        Args:
            f:
                Open file handle in binary mode.
            environ:
                The full ``Begin Elements ...`` line used to infer the element type.

        Returns:
            None

        Notes:
            - Kratos node ids are converted to 0-based indexing.
            - If multiple element blocks map to the same meshio type, their
              connectivities are stacked.
        """
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

    def __ReadKratosSubModelPart(self, f, environ=None):
        """
        Read Kratos ``SubModelPart`` node set definitions into ``node_sets``.

        This method supports sub-model parts that define node sets using a
        ``Begin SubModelPartNodes`` block.

        Args:
            f:
                Open file handle in binary mode.
            environ:
                The full ``Begin SubModelPart ...`` line, used to extract the
                sub-model part name.

        Returns:
            None

        Notes:
            - Kratos node ids are converted to 0-based indexing.
            - Only node sets are extracted here; element sets are not processed.
        """
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
