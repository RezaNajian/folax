import jax.numpy as jnp
import numpy as np
from fol.mesh_input_output.mesh import Mesh

def create_gyroid(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]
    if tpms_settings.get("constant") is not None:
        const = tpms_settings["constant"]
    else:
        const = 0.

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    sin_pix = jnp.sin(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    sin_piy = jnp.sin(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)
    sin_piz = jnp.sin(fz * jnp.pi * z + phi_z)

    K = cos_piy * sin_pix + cos_piz * sin_piy + cos_pix * sin_piz - const
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K

def create_schwarz_P(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]
    if tpms_settings.get("constant") is not None:
        const = tpms_settings["constant"]
    else:
        const = 0.

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)


    K = cos_piy + cos_piz + cos_pix - const
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K

def create_diamond(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]
    if tpms_settings.get("constant") is not None:
        const = tpms_settings["constant"]
    else:
        const = 0.

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)
    sin_pix = jnp.sin(fx * jnp.pi * x + phi_x)
    sin_piy = jnp.sin(fy * jnp.pi * y + phi_y)
    sin_piz = jnp.sin(fz * jnp.pi * z + phi_z)



    K = sin_pix * sin_piy * sin_piz +\
          sin_pix * cos_piy * cos_piz +\
              cos_pix * sin_piy * cos_piz +\
                  cos_pix * cos_piy * sin_piz - const
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K

def create_lidinoid(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]
    if tpms_settings.get("constant") is not None:
        const = tpms_settings["constant"]
    else:
        const = -0.3

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)
    sin_pix = jnp.sin(fx * jnp.pi * x + phi_x)
    sin_piy = jnp.sin(fy * jnp.pi * y + phi_y)
    sin_piz = jnp.sin(fz * jnp.pi * z + phi_z)
    cos_2pix = jnp.cos(2*(fx * jnp.pi * x + phi_x))
    cos_2piy = jnp.cos(2*(fy * jnp.pi * y + phi_y))
    cos_2piz = jnp.cos(2*(fz * jnp.pi * z + phi_z))
    sin_2pix = jnp.sin(2*(fx * jnp.pi * x + phi_x))
    sin_2piy = jnp.sin(2*(fy * jnp.pi * y + phi_y))
    sin_2piz = jnp.sin(2*(fz * jnp.pi * z + phi_z))



    K = sin_2pix * cos_piy * sin_piz +\
          sin_pix * sin_2piy * cos_piz +\
              cos_pix * sin_piy * sin_2piz -\
                  cos_2pix * cos_2piy - cos_2piy * cos_2piz - cos_2piz * cos_2pix - const
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K

def create_split_p(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)
    sin_pix = jnp.sin(fx * jnp.pi * x + phi_x)
    sin_piy = jnp.sin(fy * jnp.pi * y + phi_y)
    sin_piz = jnp.sin(fz * jnp.pi * z + phi_z)
    cos_2pix = jnp.cos(2*(fx * jnp.pi * x + phi_x))
    cos_2piy = jnp.cos(2*(fy * jnp.pi * y + phi_y))
    cos_2piz = jnp.cos(2*(fz * jnp.pi * z + phi_z))
    sin_2pix = jnp.sin(2*(fx * jnp.pi * x + phi_x))
    sin_2piy = jnp.sin(2*(fy * jnp.pi * y + phi_y))
    sin_2piz = jnp.sin(2*(fz * jnp.pi * z + phi_z))



    K = 1.1*(sin_2pix * cos_piy * sin_piz +
            sin_pix * sin_2piy * cos_piz +
            cos_pix * sin_piy * sin_2piz) - 0.2*(
        cos_2pix * cos_2piy + 
            cos_2piy * cos_2piz + 
            cos_2piz * cos_2pix) - 0.4*(
        cos_2pix + cos_2piy + cos_2piz)
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K


def create_random_periodic_sphere_field(fe_mesh, tpms_settings):
    """
    Generate a periodic field with randomly distributed non-overlapping spheres.

    Parameters:
    - fe_mesh: finite element mesh with node positions.
    - tpms_settings: dict with:
        - "sphere_diameter": float
        - "min": value inside spheres
        - "max": value outside spheres
        - "tolerance": optional fuzzy boundary width
        - "num_spheres": optional int, default=30
    """

    D = tpms_settings["sphere_diameter"]
    r = D / 2.0
    tol = tpms_settings.get("tolerance", 0.02 * D)
    min_val = tpms_settings["min"]
    max_val = tpms_settings["max"]
    num_spheres = tpms_settings.get("num_spheres", 30)

    # Mesh node coordinates
    if tpms_settings.get("section_axis_value") is not None:
        # FE init starts here
        N = int(fe_mesh.GetNumberOfNodes()**0.5)
        # Generate mesh coordinates
        x = jnp.linspace(0, 1, N)
        y = jnp.linspace(0, 1, N)
        z = jnp.linspace(0, 1, N)
        X, Y, Z = jnp.meshgrid(x, y, z)
        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()
        coords = jnp.stack([x, y, z], axis=1)
    else:
        x = fe_mesh.GetNodesX()
        y = fe_mesh.GetNodesY()
        z = fe_mesh.GetNodesZ()
        coords = jnp.stack([x, y, z], axis=1)

    # Generate non-overlapping centers
    centers = []
    max_attempts = 1000
    attempts = 0
    while len(centers) < num_spheres and attempts < max_attempts:
        candidate = jnp.array(np.random.rand(3))  # random in [0, 1]^3
        is_far_enough = True
        for c in centers:
            delta = jnp.abs(candidate - c)
            delta = jnp.minimum(delta, 1.0 - delta)  # periodic
            dist = jnp.linalg.norm(delta)
            if dist < D:  # overlapping
                is_far_enough = False
                break
        if is_far_enough:
            centers.append(candidate)
        attempts += 1

    centers = jnp.array(centers)

    # Compute minimum distance from each node to any sphere center (periodic)
    dist_min = jnp.full((coords.shape[0],), jnp.inf)
    for center in centers:
        delta = jnp.abs(coords - center)
        delta = jnp.minimum(delta, 1.0 - delta)  # periodic wrapping
        dist = jnp.linalg.norm(delta, axis=1)
        dist_min = jnp.minimum(dist_min, dist)

    # Thresholding
    inside = dist_min <= (r + tol)
    binary_K = jnp.where(inside, min_val, max_val)

    if tpms_settings.get("section_axis_value") is not None:
        z_val = tpms_settings.get("section_axis_value")
        Nz = len(jnp.unique(z))
        mask = jnp.isclose(z, z_val, atol=1.0 / Nz / 2)
        K_slice = binary_K[mask]
        return K_slice

    return binary_K

def create_random_fiber_field(fe_mesh, tpms_settings):
    """
    Create a field with elongated super-ellipsoids (fibers) inside a periodic box.
    
    Parameters:
    - fe_mesh: provides node positions.
    - tpms_settings: dict with:
        - "fiber_length": float (along main axis)
        - "fiber_radius": float (minor radius)
        - "min", "max": float
        - "num_fibers": int
        - "tolerance": optional, fuzzy boundary width
    """
    L = tpms_settings["fiber_length"]
    r = tpms_settings["fiber_radius"]
    min_val = tpms_settings["min"]
    max_val = tpms_settings["max"]
    tol = tpms_settings.get("tolerance", 0.02 * r)
    num_fibers = tpms_settings.get("num_fibers", 30)


    # Mesh node coordinates
    if tpms_settings.get("section_axis_value") is not None:
        # FE init starts here
        N = int(fe_mesh.GetNumberOfNodes()**0.5)
        # Generate mesh coordinates
        x = jnp.linspace(0, 1, N)
        y = jnp.linspace(0, 1, N)
        z = jnp.linspace(0, 1, N)
        X, Y, Z = jnp.meshgrid(x, y, z)
        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()
        coords = jnp.stack([x, y, z], axis=1)
    else:
        x = fe_mesh.GetNodesX()
        y = fe_mesh.GetNodesY()
        z = fe_mesh.GetNodesZ()
        coords = jnp.stack([x, y, z], axis=1)

    # Each fiber has center + random unit direction
    centers = []
    directions = []
    max_attempts = 1000
    attempts = 0

    while len(centers) < num_fibers and attempts < max_attempts:
        center = np.random.rand(3)
        theta = np.random.uniform(0, jnp.pi)
        phi = np.random.uniform(0, 2 * jnp.pi)
        direction = jnp.array([
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta)
        ])
        # Skip overlap checking for simplicity (can be added)
        centers.append(jnp.array(center))
        directions.append(direction)
        attempts += 1

    centers = jnp.stack(centers)
    directions = jnp.stack(directions)

    # Distance function for super-ellipsoid (fiber)
    dist_min = jnp.full((coords.shape[0],), jnp.inf)
    for i in range(num_fibers):
        c = centers[i]
        d = directions[i]
        vec = coords - c
        vec = jnp.where(vec > 0.5, vec - 1.0, vec)
        vec = jnp.where(vec < -0.5, vec + 1.0, vec)

        # Project onto fiber axis
        t = jnp.dot(vec, d)
        t = jnp.clip(t, -L/2, L/2)
        proj = jnp.outer(t, d)
        radial = vec - proj
        radial_dist = jnp.linalg.norm(radial, axis=1)

        # inside if radial <= r and |t| <= L/2
        inside = (radial_dist <= (r + tol)) & (jnp.abs(t) <= (L/2 + tol))
        dist_min = jnp.where(inside, 0.0, dist_min)

    binary_K = jnp.where(dist_min == 0.0, min_val, max_val)

    if tpms_settings.get("section_axis_value") is not None:
        z_val = tpms_settings.get("section_axis_value")
        Nz = len(jnp.unique(z))
        mask = jnp.isclose(z, z_val, atol=1.0 / Nz / 2)
        K_slice = binary_K[mask]
        return K_slice
    
    return binary_K


