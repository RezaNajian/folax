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


#-----------------------------------------------------------------------------------------------#
#3_HOLES



import os, gmsh, meshio, jax.numpy as jnp
from fol.mesh_input_output.mesh import Mesh

def create_quad_with_holes_tri_mesh(
        case_dir: str,
        p0=(0.0, 0.0), p1=(1.0, 0.5), p2=(1.0, 0.8), p3=(0.0, 0.8),
        holes: list[tuple[float, float, float]] | None = None,   # (cx, cy, R)
        mesh_size_min: float | None = None,
        mesh_size_max: float | None = None,
        # --- adaptive refinement (applies to *all* holes) ---
        adapt_radius: float | None = None,
        adapt_factor: float = 0.3,
        with_gmsh: bool = True,
) -> Mesh:
    """
    Convex quadrilateral + zero/one/many circular holes → triangular mesh.

    *holes* – list of (x-centre, y-centre, radius) tuples, or None for none.  
    With *adapt_radius* ≠ None, elements shrink from SizeMin =
        adapt_factor·mesh_size_min at each hole rim to SizeMax = mesh_size_max
        at distance (R + adapt_radius).
    """
    # ---------------------------------------------------- 1┃init & outer quad
    if with_gmsh:
        gmsh.initialize()
    gmsh.model.add("quad_holes")

    pts_xy = [p0, p1, p2, p3]
    pts = [gmsh.model.occ.addPoint(x, y, 0, 0) for x, y in pts_xy]
    lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    loop  = gmsh.model.occ.addCurveLoop(lines)
    surf  = gmsh.model.occ.addPlaneSurface([loop])

    # ---------------------------------------------------- 2┃subtract holes
    hole_surfs = []
    if holes:
        for cx, cy, R in holes:
            hole_surfs.append((2, gmsh.model.occ.addDisk(cx, cy, 0, R, R)))

        # one boolean cut with *all* disks at once → faster & cleaner topology
        trimmed, _ = gmsh.model.occ.cut([(2, surf)], hole_surfs, removeTool=True)
        surf = trimmed[0][1]

    gmsh.model.occ.synchronize()

    # ---------------------------------------------------- 3┃global size
    span_x = max(x for x, _ in pts_xy) - min(x for x, _ in pts_xy)
    span_y = max(y for _, y in pts_xy) - min(y for _, y in pts_xy)
    ref = min(span_x, span_y)
    h_min = mesh_size_min or 0.1 * ref
    h_max = mesh_size_max or 0.2 * ref
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_max)

    # ---------------------------------------------------- 4┃adaptive sizing
    if adapt_radius is not None and holes:
        # Build one distance field from all hole centres
        ptags = [gmsh.model.occ.addPoint(cx, cy, 0, 0) for cx, cy, _ in holes]
        gmsh.model.occ.synchronize()
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "NodesList", ptags)

        thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr, "InField", dist)
        gmsh.model.mesh.field.setNumber(thr, "SizeMin", adapt_factor * h_min)
        gmsh.model.mesh.field.setNumber(thr, "SizeMax", h_max)
        gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.0)              # at rim
        gmsh.model.mesh.field.setNumber(thr, "DistMax", adapt_radius)     # fade out

        gmsh.model.mesh.field.setAsBackgroundMesh(thr)

    # ---------------------------------------------------- 5┃mesh & export
    gmsh.model.mesh.generate(2)

    os.makedirs(case_dir, exist_ok=True)
    msh = os.path.join(case_dir, "quad_holes.msh")
    vtk = os.path.join(case_dir, "quad_holes.vtk")
    gmsh.write(msh); gmsh.write(vtk)
    if with_gmsh:
        gmsh.finalize()

    # ---------------------------------------------------- 6┃wrap into FOL Mesh
    m = meshio.read(msh)
    if "triangle" not in m.cells_dict:
        raise RuntimeError("No triangular cells generated—check geometry.")

    fol = Mesh("quad_holes_io", "quad_holes.")
    fol.node_ids          = jnp.arange(len(m.points))
    fol.nodes_coordinates = jnp.array(m.points[:, :2])
    fol.elements_nodes    = {"triangle": jnp.array(m.cells_dict["triangle"])}

    tol = 1e-6
    x = fol.nodes_coordinates[:, 0]
    fol.node_sets = {
        "left":  fol.node_ids[jnp.isclose(x, x.min(), atol=tol)],
        "right": fol.node_ids[jnp.isclose(x, x.max(), atol=tol)],
    }
    fol.mesh_io = meshio.Mesh(
        points=jnp.array(fol.nodes_coordinates),
        cells={"triangle": jnp.array(fol.elements_nodes["triangle"])})
    fol.is_initialized = True
    return fol
