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

# ------------------------------------------------------------
#  mesh_helpers.py   (replace the old function with this one)
# ------------------------------------------------------------
import os, gmsh, meshio, jax.numpy as jnp
from fol.mesh_input_output.mesh import Mesh

def create_quad_with_holes_tri_mesh(
        case_dir: str,
        p0=(0.0, 0.0), p1=(1.0, 0.5), p2=(1.0, 0.8), p3=(0.0, 0.8),
        holes: list[tuple[float, float, float]] | None = None,   # (cx, cy, R)
        mesh_size_min: float | None = None,
        mesh_size_max: float | None = None,
        adapt_radius: float | None = None,
        adapt_factor: float = 0.3,
        add_corner_sets: bool = False,
        with_gmsh: bool = True,
) -> Mesh:
    """Quadrilateral + 0/1/… circular holes → triangular FOL mesh.

    Saves **quad_holes.msh**, **quad_holes.vtk** *and now* **quad_holes.brep**  ← NEW
    """
    # ───────────────────────────── OCC geometry ───────────────────────────
    if with_gmsh:
        gmsh.initialize()
    gmsh.model.add("quad_holes")

    quad_pts = [p0, p1, p2, p3]
    pt_tags  = [gmsh.model.occ.addPoint(x, y, 0, 0) for x, y in quad_pts]
    ln_tags  = [gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i + 1) % 4])
                for i in range(4)]
    loop_tag = gmsh.model.occ.addCurveLoop(ln_tags)
    surf_tag = gmsh.model.occ.addPlaneSurface([loop_tag])

    # subtract disks (if any) ---------------------------------------------
    hole_surfs = []
    if holes:
        for cx, cy, R in holes:
            hole_surfs.append((2, gmsh.model.occ.addDisk(cx, cy, 0, R, R)))

        trimmed, _ = gmsh.model.occ.cut([(2, surf_tag)], hole_surfs,
                                        removeTool=True)
        surf_tag = trimmed[0][1]

    gmsh.model.occ.synchronize()

    # ───────────────────────────── mesh parameters ────────────────────────
    span_x = max(x for x, _ in quad_pts) - min(x for x, _ in quad_pts)
    span_y = max(y for _, y in quad_pts) - min(y for _, y in quad_pts)
    ref    = min(span_x, span_y)

    h_min = mesh_size_min or 0.1 * ref
    h_max = mesh_size_max or 0.2 * ref
    h_min = max(h_min, 0.2 * h_max)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_max)

    # refine near holes (optional) ----------------------------------------
    if adapt_radius is not None and holes:
        centre_pts = [gmsh.model.occ.addPoint(cx, cy, 0, 0) for cx, cy, _ in holes]
        gmsh.model.occ.synchronize()

        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "NodesList", centre_pts)

        thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr, "InField", dist)
        gmsh.model.mesh.field.setNumber(thr, "SizeMin", adapt_factor * h_min)
        gmsh.model.mesh.field.setNumber(thr, "SizeMax", h_max)
        thr_dist_min = min(R for *_, R in holes)
        gmsh.model.mesh.field.setNumber(thr, "DistMin", thr_dist_min)
        gmsh.model.mesh.field.setNumber(thr, "DistMax", thr_dist_min + adapt_radius)
        gmsh.model.mesh.field.setAsBackgroundMesh(thr)

    # ───────────────────────────── mesh + export ──────────────────────────
    gmsh.model.mesh.generate(2)

    os.makedirs(case_dir, exist_ok=True)
    msh  = os.path.join(case_dir, "quad_holes.msh")
    vtk  = os.path.join(case_dir, "quad_holes.vtk")
    brep = os.path.join(case_dir, "quad_holes.brep")   # NEW

    gmsh.write(msh)
    gmsh.write(vtk)
    gmsh.write(brep)                                   # NEW
    if with_gmsh:
        gmsh.finalize()

    # ───────────────────────────── wrap into FOL mesh ─────────────────────
    m = meshio.read(msh)
    if "triangle" not in m.cells_dict:
        raise RuntimeError("No triangular cells generated—check geometry")

    fol = Mesh("quad_holes_io", "quad_holes.")
    fol.node_ids          = jnp.arange(len(m.points))
    fol.nodes_coordinates = jnp.array(m.points[:, :2])
    fol.elements_nodes    = {"triangle": jnp.array(m.cells_dict["triangle"])}

    tol = 1e-6
    x, y = fol.nodes_coordinates.T
    fol.node_sets = {
        "left":  fol.node_ids[jnp.isclose(x, x.min(), atol=tol)],
        "right": fol.node_ids[jnp.isclose(x, x.max(), atol=tol)],
    }

    if add_corner_sets:
        bl = fol.node_ids[(jnp.isclose(x, x.min(), atol=tol) &
                           jnp.isclose(y, y.min(), atol=tol))][0]
        tl = fol.node_ids[(jnp.isclose(x, x.min(), atol=tol) &
                           jnp.isclose(y, y.max(), atol=tol))][0]
        fol.node_sets |= {"corner_bl": jnp.array([bl]),
                          "corner_tl": jnp.array([tl])}

    fol.mesh_io = meshio.Mesh(points=jnp.array(fol.nodes_coordinates),
                              cells={"triangle": jnp.array(
                                  fol.elements_nodes["triangle"])})
    fol.is_initialized = True
    return fol


from typing import Tuple

import jax.numpy as jnp  # noqa: F401  (kept for symmetry with other helpers)
from fol.mesh_input_output.mesh import Mesh


def create_quad_with_corner_cuts_tri_mesh(
    case_dir: str,
    p0: Tuple[float, float] = (0.0, 0.0),
    p1: Tuple[float, float] = (1.0, 0.0),
    p2: Tuple[float, float] = (1.0, 1.0),
    p3: Tuple[float, float] = (0.0, 1.0),
    corner_radius: float = 0.1,
    **kwargs,
) -> Mesh:
    """Return a triangular FOL mesh of a quadrilateral with *rounded corners*.

    A quarter‑circle of radius ``corner_radius`` is cut from each corner, using
    :pyfunc:`~mesh_helpers.create_quad_with_holes_tri_mesh` under the hood.

    Parameters
    ----------
    case_dir : str
        Output folder for the generated ``.msh`` + ``.vtk`` files.
    p0, p1, p2, p3 : Tuple[float, float]
        Corner coordinates (counter‑clockwise order). Defaults to the unit square.
    corner_radius : float, default 0.1
        Radius of the circular cut at each corner.
    **kwargs
        Any extra keyword arguments accepted by
        :pyfunc:`~mesh_helpers.create_quad_with_holes_tri_mesh` (``mesh_size_min``,
        ``mesh_size_max``, ``adapt_radius``, ``adapt_factor``, ``add_corner_sets``,
        ``with_gmsh`` …).

    Returns
    -------
    Mesh
        A fully‑initialised :class:`~fol.mesh_input_output.mesh.Mesh` object ready
        for FOL/IFOL workflows.
    """

    if corner_radius <= 0:
        raise ValueError("corner_radius must be positive")

    # assemble 4 identical disks – one per corner -------------------------
    holes = [
        (p0[0], p0[1], corner_radius),
        (p1[0], p1[1], corner_radius),
        (p2[0], p2[1], corner_radius),
        (p3[0], p3[1], corner_radius),
    ]

    # provide a sensible default grading radius if none supplied ----------
    if "adapt_radius" not in kwargs or kwargs["adapt_radius"] is None:
        kwargs["adapt_radius"] = 3 * corner_radius

    return create_quad_with_holes_tri_mesh(
        case_dir=case_dir,
        p0=p0,
        p1=p1,
        p2=p2,
        p3=p3,
        holes=holes,
        **kwargs,
    )





#shapes
"""mesh_corner_cuts.py
========================
One-stop utilities for 2-D meshing with Gmsh → MeshIO → FOL.

* **create_quad_with_custom_cuts_tri_mesh** – main work-horse that takes any
  combination of circular, elliptical, or half-circular *cuts* and returns a
  triangular FOL mesh.
* **create_quad_with_corner_cuts_tri_mesh** – thin wrapper that just rounds the
  four rectangle corners by the same radius and forwards to the function above.

Both helpers auto-enable adaptive grading: if you don’t supply ``adapt_radius``
we estimate it as *three times* the largest radius in the job.
"""


import os
from typing import Any, Dict, List, Tuple

import gmsh
import jax.numpy as jnp
import meshio
from fol.mesh_input_output.mesh import Mesh

# -----------------------------------------------------------------------------
# Low-level helper: add a semicircle surface and return its OCC tag
# -----------------------------------------------------------------------------

def _add_semicircle_surface(cx: float, cy: float, R: float, orientation: str) -> int:
    """Create a semicircular *plane surface* and return its tag.

    ``orientation`` selects where the flat edge lies:
    ``"top" | "bottom" | "left" | "right"``.
    """
    orient = orientation.lower()
    if orient not in {"top", "bottom", "left", "right"}:
        raise ValueError(
            "orientation must be 'top', 'bottom', 'left' or 'right'; "
            f"got {orientation!r}"
        )

    c = gmsh.model.occ.addPoint(cx, cy, 0, 0)
    if orient in {"left", "right"}:  # vertical flat edge
        p0 = gmsh.model.occ.addPoint(cx, cy - R, 0, 0)
        p1 = gmsh.model.occ.addPoint(cx, cy + R, 0, 0)
        arc = (
            gmsh.model.occ.addCircleArc(p1, c, p0)
            if orient == "left"
            else gmsh.model.occ.addCircleArc(p0, c, p1)
        )
    else:  # horizontal flat edge
        p0 = gmsh.model.occ.addPoint(cx - R, cy, 0, 0)
        p1 = gmsh.model.occ.addPoint(cx + R, cy, 0, 0)
        arc = (
            gmsh.model.occ.addCircleArc(p0, c, p1)
            if orient == "top"
            else gmsh.model.occ.addCircleArc(p1, c, p0)
        )

    line = gmsh.model.occ.addLine(p1, p0)
    loop = gmsh.model.occ.addCurveLoop([arc, line])
    return gmsh.model.occ.addPlaneSurface([loop])


# -----------------------------------------------------------------------------
# 1. Generic quad-with-arbitrary-cuts helper
# -----------------------------------------------------------------------------

def create_quad_with_custom_cuts_tri_mesh(
    *,
    case_dir: str,
    p0: Tuple[float, float] = (0.0, 0.0),
    p1: Tuple[float, float] = (1.0, 0.0),
    p2: Tuple[float, float] = (1.0, 1.0),
    p3: Tuple[float, float] = (0.0, 1.0),
    shapes: List[Dict[str, Any]] | None = None,
    mesh_size_min: float | None = None,
    mesh_size_max: float | None = None,
    adapt_radius: float | None = None,
    adapt_factor: float = 0.3,
    add_corner_sets: bool = False,
    with_gmsh: bool = True,
) -> Mesh:
    """Quadrilateral domain with arbitrary circular / elliptical / semicircular cuts.

    ``shapes`` items:

    * circle      – ``{"type": "circle",      "cx": .., "cy": .., "R":  ..}``
    * ellipse     – ``{"type": "ellipse",     "cx": .., "cy": .., "rx": .., "ry": ..}``
    * half circle – ``{"type": "half_circle", "cx": .., "cy": .., "R":  .., "orientation": "top|bottom|left|right"}``
    """

    if with_gmsh:
        gmsh.initialize()
    gmsh.model.add("quad_custom_cuts")

    # ----- base quad geometry -------------------------------------------
    quad_pts = [p0, p1, p2, p3]
    pts = [gmsh.model.occ.addPoint(x, y, 0, 0) for x, y in quad_pts]
    lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    quad_loop = gmsh.model.occ.addCurveLoop(lines)
    quad_surf = gmsh.model.occ.addPlaneSurface([quad_loop])

    # ----- build subtraction surfaces -----------------------------------
    cut_surfs: List[Tuple[int, int]] = []
    centre_pts: List[int] = []

    shapes = shapes or []
    for sh in shapes:
        stype = sh.get("type", "").lower()
        if stype == "circle":
            cx, cy, R = sh["cx"], sh["cy"], sh["R"]
            tag = gmsh.model.occ.addDisk(cx, cy, 0, R, R)
            cut_surfs.append((2, tag))
            centre_pts.append(gmsh.model.occ.addPoint(cx, cy, 0, 0))
        elif stype == "ellipse":
            cx, cy, rx, ry = sh["cx"], sh["cy"], sh["rx"], sh["ry"]
            tag = gmsh.model.occ.addDisk(cx, cy, 0, rx, ry)
            cut_surfs.append((2, tag))
            centre_pts.append(gmsh.model.occ.addPoint(cx, cy, 0, 0))
        elif stype == "half_circle":
            cx, cy, R = sh["cx"], sh["cy"], sh["R"]
            orient = sh.get("orientation", "top")
            tag = _add_semicircle_surface(cx, cy, R, orient)
            cut_surfs.append((2, tag))
            centre_pts.append(gmsh.model.occ.addPoint(cx, cy, 0, 0))
        else:
            raise ValueError(f"Unknown shape type: {stype!r}")

    # ----- Boolean cut ---------------------------------------------------
    if cut_surfs:
        quad_surf = gmsh.model.occ.cut([(2, quad_surf)], cut_surfs, removeTool=True)[0][0][1]
    gmsh.model.occ.synchronize()

    assert len(gmsh.model.getEntities(dim=2)) == 1, "Cut produced disconnected pieces"

    # ----- mesh sizes ----------------------------------------------------
    span_x = max(x for x, _ in quad_pts) - min(x for x, _ in quad_pts)
    span_y = max(y for _, y in quad_pts) - min(y for _, y in quad_pts)
    ref = min(span_x, span_y)

    h_min = mesh_size_min or 0.1 * ref
    h_max = mesh_size_max or 0.2 * ref
    h_min = max(h_min, 0.2 * h_max)  # safeguard

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_max)

    # ----- auto-default adapt_radius ------------------------------------
    if adapt_radius is None and centre_pts:
        max_r = 0.0
        for sh in shapes:
            st = sh["type"].lower()
            if st == "circle":
                max_r = max(max_r, sh["R"])
            elif st == "ellipse":
                max_r = max(max_r, sh["rx"], sh["ry"])
            elif st == "half_circle":
                max_r = max(max_r, sh["R"])
        adapt_radius = 3 * max_r if max_r else None

    # ----- distance field refinement ------------------------------------
    if adapt_radius is not None and centre_pts:
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "NodesList", centre_pts)

        thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr, "InField", dist)
        gmsh.model.mesh.field.setNumber(thr, "SizeMin", adapt_factor * h_min)
        gmsh.model.mesh.field.setNumber(thr, "SizeMax", h_max)
        gmsh.model.mesh.field.setNumber(thr, "DistMin", adapt_radius)
        gmsh.model.mesh.field.setNumber(thr, "DistMax", 3 * adapt_radius)
        gmsh.model.mesh.field.setAsBackgroundMesh(thr)

    # ----- mesh + export -------------------------------------------------
    gmsh.model.mesh.generate(2)

    os.makedirs(case_dir, exist_ok=True)
    msh = os.path.join(case_dir, "quad_custom_cuts.msh")
    vtk = os.path.join(case_dir, "quad_custom_cuts.vtk")
    gmsh.write(msh)
    gmsh.write(vtk)
    if with_gmsh:
        gmsh.finalize()

    # ----- wrap into FOL.Mesh -------------------------------------------
    m = meshio.read(msh)
    if "triangle" not in m.cells_dict:
        raise RuntimeError("No triangular cells generated—check geometry")

    fol = Mesh("quad_custom_cuts_io", "quad_custom_cuts.")
    fol.node_ids = jnp.arange(len(m.points))
    fol.nodes_coordinates = jnp.array(m.points[:, :2])
    fol.elements_nodes = {"triangle": jnp.array(m.cells_dict["triangle"])}

    tol = 1e-6
    x, y = fol.nodes_coordinates.T
    fol.node_sets = {
        "left":  fol.node_ids[jnp.isclose(x, x.min(), atol=tol)],
        "right": fol.node_ids[jnp.isclose(x, x.max(), atol=tol)],
    }

    if add_corner_sets:
        bl = fol.node_ids[(jnp.isclose(x, x.min(), atol=tol) & jnp.isclose(y, y.min(), atol=tol))][0]
        tl = fol.node_ids[(jnp.isclose(x, x.min(), atol=tol) & jnp.isclose(y, y.max(), atol=tol))][0]
        fol.node_sets |= {"corner_bl": jnp.array([bl]), "corner_tl": jnp.array([tl])}

    fol.mesh_io = meshio.Mesh(points=jnp.array(fol.nodes_coordinates),
                              cells={"triangle": jnp.array(fol.elements_nodes["triangle"])} )
    fol.is_initialized = True
    return fol


# -----------------------------------------------------------------------------
# 2. Convenience wrapper: same-radius rounded corners only
# -----------------------------------------------------------------------------

def create_quad_with_corner_cuts_tri_mesh(
    case_dir: str,
    p0: Tuple[float, float] = (0.0, 0.0),
    p1: Tuple[float, float] = (1.0, 0.0),
    p2: Tuple[float, float] = (1.0, 1.0),
    p3: Tuple[float, float] = (0.0, 1.0),
    corner_radius: float = 0.1,
    **kwargs,
) -> Mesh:
    """Shortcut to fillet *all four* corners with the same radius."""

    if corner_radius <= 0:
        raise ValueError("corner_radius must be positive")

    shapes = [
        {"type": "circle", "cx": p0[0], "cy": p0[1], "R": corner_radius},
        {"type": "circle", "cx": p1[0], "cy": p1[1], "R": corner_radius},
        {"type": "circle", "cx": p2[0], "cy": p2[1], "R": corner_radius},
        {"type": "circle", "cx": p3[0], "cy": p3[1], "R": corner_radius},
    ]

    kwargs.setdefault("adapt_radius", 3 * corner_radius)

    return create_quad_with_custom_cuts_tri_mesh(
        case_dir=case_dir,
        p0=p0,
        p1=p1,
        p2=p2,
        p3=p3,
        shapes=shapes,
        **kwargs,
    )

# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------
