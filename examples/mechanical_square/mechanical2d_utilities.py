import jax.numpy as jnp
from fol.mesh_input_output.mesh import Mesh
import gmsh, os, math, random, meshio
from fol.tools.usefull_functions import *

import gmsh
import meshio
import os
import jax.numpy as jnp

def create_rectangle_tri_mesh(Lx, Ly, case_dir,
                          mesh_size_min=None, mesh_size_max=None):
    """
    Create a triangular mesh of a rectangle (2D, no holes).

    Adds node sets "left" (x≈0) and "right" (x≈Lx).
    Saves both .msh and .vtk in case_dir.
    """
    gmsh.initialize()
    gmsh.model.add("rectangle")

    # --- 1. Rectangle geometry
    rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    gmsh.model.occ.synchronize()

    # --- 2. Mesh settings
    if mesh_size_min is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_min)
    else:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(Lx, Ly) * 0.1)

    if mesh_size_max is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_max)
    else:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", min(Lx, Ly) * 0.2)

    # --- 3. Generate triangular mesh
    gmsh.model.mesh.generate(2)

    # --- 4. Save to files
    os.makedirs(case_dir, exist_ok=True)
    msh_file = os.path.join(case_dir, "rectangle.msh")
    vtk_file = os.path.join(case_dir, "rectangle.vtk")
    gmsh.write(msh_file)
    gmsh.write(vtk_file)
    gmsh.finalize()

    # --- 5. Wrap in FOL Mesh object
    mesh = meshio.read(msh_file)
    fe_mesh = Mesh("rectangle_io", "rectangle.")
    fe_mesh.node_ids = jnp.arange(len(mesh.points))
    fe_mesh.nodes_coordinates = jnp.array(mesh.points[:, :2])  # 2D coords

    if "triangle" not in mesh.cells_dict:
        raise RuntimeError("No triangular cells found. Check meshing settings.")
    fe_mesh.elements_nodes = {"triangle": jnp.array(mesh.cells_dict["triangle"])}

    # --- 6. Define boundary node sets (left & right edges)
    tol = 1e-6
    coords = fe_mesh.nodes_coordinates
    left_ids  = fe_mesh.node_ids[jnp.isclose(coords[:,0], 0.0, atol=tol)]
    right_ids = fe_mesh.node_ids[jnp.isclose(coords[:,0], Lx, atol=tol)]
    fe_mesh.node_sets = {"left": left_ids, "right": right_ids}

    fe_mesh.mesh_io = meshio.Mesh(
        points=jnp.array(fe_mesh.nodes_coordinates),
        cells={"triangle": jnp.array(fe_mesh.elements_nodes["triangle"])}
    )

    fe_mesh.is_initialized = True
    return fe_mesh

