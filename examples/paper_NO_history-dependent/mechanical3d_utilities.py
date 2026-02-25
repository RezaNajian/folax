import jax.numpy as jnp
import numpy as np
from fol.mesh_input_output.mesh import Mesh
import gmsh, os, math, random, meshio
from fol.tools.usefull_functions import *
from skimage.transform import resize
from scipy.ndimage import zoom
import pickle, h5py

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




def create_cube_with_spheres_mesh(num_spheres, Lx, Ly, Lz, case_dir,
                                  min_radius=0.05, max_radius=0.15,
                                  mesh_size_min=None, mesh_size_max=None,
                                  seed=42):
    """
    Create a tetrahedral mesh of a cube with spherical voids.

    Special placements:
        num_spheres=1 → sphere at cube center
        num_spheres=8 → spheres at 8 corners
        num_spheres=9 → 8 corners + 1 center
        otherwise     → uniform grid, random subset of num_spheres points

    Adds node sets "left" (x≈0) and "right" (x≈Lx).
    Saves both .msh and .vtk in case_dir.
    """

    random.seed(seed)
    gmsh.initialize()
    gmsh.model.add("cube_with_spheres")

    # --- 1. Cube geometry
    box = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)

    # --- 2. Sphere placement logic
    r = (min_radius + max_radius) / 2.0
    centers = []

    if num_spheres == 1:
        centers = [(Lx/2, Ly/2, Lz/2)]

    elif num_spheres == 8:
        centers = [(x, y, z)
                   for x in [0, Lx]
                   for y in [0, Ly]
                   for z in [0, Lz]]

    elif num_spheres == 9:
        centers = [(x, y, z)
                   for x in [0, Lx]
                   for y in [0, Ly]
                   for z in [0, Lz]]
        centers.append((Lx/2, Ly/2, Lz/2))

    else:
        n = math.ceil(num_spheres ** (1/3))
        spacing_x = Lx / (n - 1 if n > 1 else 1)
        spacing_y = Ly / (n - 1 if n > 1 else 1)
        spacing_z = Lz / (n - 1 if n > 1 else 1)

        all_points = [(i*spacing_x, j*spacing_y, k*spacing_z)
                      for i in range(n)
                      for j in range(n)
                      for k in range(n)]
        centers = random.sample(all_points, min(num_spheres, len(all_points)))

    # --- 3. Add spheres + cut
    sphere_tags = [gmsh.model.occ.addSphere(x, y, z, r) for (x, y, z) in centers]
    gmsh.model.occ.cut([(3, box)], [(3, s) for s in sphere_tags],
                       removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    # --- 4. Mesh (tetrahedra)
    if mesh_size_min is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_min)
    else:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", r * 0.5)

    if mesh_size_max is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_max)
    else:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", r * 1.0)

    gmsh.model.mesh.generate(3)

    # --- 5. Save to files
    os.makedirs(case_dir, exist_ok=True)
    msh_file = os.path.join(case_dir, "cube_with_spheres.msh")
    vtk_file = os.path.join(case_dir, "cube_with_spheres.vtk")
    gmsh.write(msh_file)
    gmsh.write(vtk_file)
    gmsh.finalize()

    # --- 6. Wrap in FOL Mesh object
    mesh = meshio.read(msh_file)
    fe_mesh = Mesh("cube_with_spheres_io", "cube_with_spheres.")
    fe_mesh.node_ids = jnp.arange(len(mesh.points))
    fe_mesh.nodes_coordinates = jnp.array(mesh.points)

    if "tetra" not in mesh.cells_dict:
        raise RuntimeError("No tetrahedral cells found. Check meshing settings.")
    fe_mesh.elements_nodes = {"tetra": jnp.array(mesh.cells_dict["tetra"])}
# --- 7. Define boundary node sets (left & right faces)
    tol = 1e-6
    coords = fe_mesh.nodes_coordinates
    left_ids  = fe_mesh.node_ids[jnp.isclose(coords[:,0], 0.0, atol=tol)]
    right_ids = fe_mesh.node_ids[jnp.isclose(coords[:,0], Lx, atol=tol)]
    fe_mesh.node_sets = {"left": left_ids, "right": right_ids}

    # rebuild clean meshio object to avoid bad cell_sets
    fe_mesh.mesh_io = meshio.Mesh(
        points=jnp.array(fe_mesh.nodes_coordinates),
        cells={"tetra": jnp.array(fe_mesh.elements_nodes["tetra"])}
    )

    fe_mesh.is_initialized = True
    return fe_mesh


def create_hex_mesh_with_spheres(Lx, Ly, Lz, nx, ny, nz, sphere_centers, sphere_radii, case_dir):
    """
    Create a structured hexahedral mesh of a box with spherical voids (voxelization style).
    
    Parameters
    ----------
    Lx, Ly, Lz : float
        Box dimensions
    nx, ny, nz : int
        Number of divisions in each direction
    sphere_centers : list of (x,y,z)
        Centers of spheres
    sphere_radii : list of float
        Radii of spheres
    case_dir : str
        Output directory
    """

    os.makedirs(case_dir, exist_ok=True)

    # Grid points
    x = np.linspace(0, Lx, nx+1)
    y = np.linspace(0, Ly, ny+1)
    z = np.linspace(0, Lz, nz+1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Hexahedral connectivity
    hexes = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # indices of cube corners
                n0 = i*(ny+1)*(nz+1) + j*(nz+1) + k
                n1 = n0 + 1
                n2 = n0 + (nz+1)
                n3 = n2 + 1
                n4 = n0 + (ny+1)*(nz+1)
                n5 = n4 + 1
                n6 = n4 + (nz+1)
                n7 = n6 + 1
                hexes.append([n0,n1,n3,n2,n4,n5,n7,n6])
    hexes = np.array(hexes)

    # Remove hexes inside spheres
    hex_centers = points[hexes].mean(axis=1)
    mask = np.ones(len(hexes), dtype=bool)
    for c, r in zip(sphere_centers, sphere_radii):
        dist = np.linalg.norm(hex_centers - np.array(c), axis=1)
        mask &= dist > r  # keep only outside
    hexes = hexes[mask]

    # Write mesh
    mesh = meshio.Mesh(points, [("hexahedron", hexes)])
    msh_file = os.path.join(case_dir, "cube_with_spheres_hex.msh")
    vtk_file = os.path.join(case_dir, "cube_with_spheres_hex.vtk")
    meshio.write(msh_file, mesh)
    meshio.write(vtk_file, mesh)

    return mesh

def plot_norm_iter(data,plot_name='res_norm_iter',type=None):

    
    if type=='1':
        plt.figure(figsize=(10,4))
    else:
        plt.figure(figsize=(2,4))
        
    plt.plot(np.arange(len(data)), data, marker='o', color='black')
    plt.yscale("log")
    plt.grid(which="both", linestyle="--", linewidth=0.7, alpha=0.7)  # clearer grid
    
    plt.xlabel("Iteration",fontdict={"size": 16})
    plt.xlim()
    if type=='1':    
        plt.ylabel("Residual norm",fontdict={"size": 16})
    plt.ylim((1e-9,3e-1))
    
    # # set x-axis ticks every 5
    # plt.xticks(np.arange(0, len(data) + 1, 5))
    
    plt.tight_layout()
    plt.savefig(f"{plot_name}.png")
    print(f"plot saved to {plot_name}")
    plt.close()


def downsample(arr_128:np.ndarray):
    """
    Arguments:
    ---------
    an array of shape (128, 128)

    Returns:
    -------
    an array of shape (64,64)
    """
    # return resize(arr_128, (64, 64), anti_aliasing=True)
    # return zoom(arr_128, zoom=0.5)  # scale down by factor of 2
    return arr_128.reshape(64, 2, 64, 2).mean(axis=(1, 3))

def von_mises_plot(model_pred,ground_truth,error,steps,N,case_id,case_dir,filename,start_id):
    fig, axs = plt.subplots(3, steps, figsize=((3*steps), steps))

    # Define consistent color limits
    vmin_model, vmax_model = model_pred.min(), model_pred.max()
    vmin_fft, vmax_fft = ground_truth.min(), ground_truth.max()
    vmin_err, vmax_err = error.min(), error.max()

    
    for col in range(steps):
        field_model = model_pred[col, :].reshape((N, N))
        field_fft = ground_truth[col, :].reshape((N, N))
        field_err = error[col, :].reshape((N, N))

        # Row 1
        im0 = axs[0, col].imshow(field_model, cmap='viridis',
                                vmin=vmin_model, vmax=vmax_model)
        axs[0, col].set_title(f"Strain {start_id+col+1}%", fontsize=10)
        axs[0, col].set_xticks([]); axs[0, col].set_yticks([])

        # Row 2
        im1 = axs[1, col].imshow(field_fft, cmap='viridis',
                                vmin=vmin_fft, vmax=vmax_fft)
        axs[1, col].set_xticks([]); axs[1, col].set_yticks([])

        # Row 3
        im2 = axs[2, col].imshow(field_err, cmap='gray',
                                vmin=vmin_err, vmax=vmax_err)
        axs[2, col].set_xticks([]); axs[2, col].set_yticks([])

    # --- Add one colorbar per row manually (aligned vertically to the right) ---
    # [left, bottom, width, height] in figure coordinates
    cbar_ax1 = fig.add_axes([0.92, 0.70, 0.01, 0.2])  # Row 1
    cbar_ax2 = fig.add_axes([0.92, 0.40, 0.01, 0.2])  # Row 2
    cbar_ax3 = fig.add_axes([0.92, 0.10, 0.01, 0.2])  # Row 3

    fig.colorbar(im0, cax=cbar_ax1, label='FNO values')
    fig.colorbar(im1, cax=cbar_ax2, label='FFT values')
    fig.colorbar(im2, cax=cbar_ax3, label='Error')

    # --- Row labels ---
    axs[0, 0].set_ylabel("Von Mises FNO", fontsize=12)
    axs[1, 0].set_ylabel("Von Mises FFT", fontsize=12)
    axs[2, 0].set_ylabel("Abs. Error", fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave room for colorbars
    plt.savefig(case_dir + f"/{filename}_case_{case_id}.png", dpi=200)
    plt.close(fig)

def plot_value_1d(pred_field_list:list,gt_field_list:list,y_eval_value_grid:int,time:list[str],case_dir:str,filename:str,start_id):
    
    dim = int(pred_field_list[0].size**0.5)
    fig, axes = plt.subplots(1,3,figsize=(18,6))

    for i in range(3):
        y_value_pred = pred_field_list[i].reshape(dim, dim)[y_eval_value_grid,:]
        y_value_gt = gt_field_list[i].reshape(dim, dim)[y_eval_value_grid,:]

        axes[i].plot(np.arange(len(y_value_pred)), y_value_pred, label='predicted',linestyle='--')
        axes[i].plot(np.arange(len(y_value_gt)), y_value_gt, label='ground truth')

        axes[i].set_title(
            f"strain: {int(time[i])+start_id}%"
        )
        axes[i].legend()
        axes[i].grid('on')
        # axes[i].set_aspect('equal', adjustable='box')   # ← makes plot area square
    fig.suptitle(f"Von-Mises values at grid: {y_eval_value_grid}")
    plt.tight_layout()
    plt.savefig(case_dir + f"/{filename}.png", dpi=200)
    plt.close(fig)


    # ---------------------------------------------------------------
# 1. Von Mises MAE and Relative L1
# ---------------------------------------------------------------
def compute_von_mises_metrics(vm_pred, vm_gt):
    """
    vm_pred, vm_gt : 2D arrays (H,W) in PHYSICAL units.

    Returns
    -------
    mae : float
    rel_l1 : float (%)
    """
    mae = np.mean(np.abs(vm_pred - vm_gt))

    denom = np.mean(np.abs(vm_gt))
    if denom < 1e-9:
        denom = 1e-9

    rel_l1 = (mae / denom) * 100.0
    return mae, rel_l1


# ---------------------------------------------------------------
# 2. Slip Shear Combined Metrics (channels 1–4)
# ---------------------------------------------------------------
def compute_shear_metrics(shear_pred, shear_gt):
    """
    shear_pred, shear_gt : arrays shape (4, H, W)

    Computes MAE over all four slip shear fields combined.

    Returns
    -------
    mae : float
    rel_l1 : float (%)
    """
    mae = np.mean(np.abs(shear_pred - shear_gt))

    denom = np.mean(np.abs(shear_gt))
    if denom < 1e-9:
        denom = 1e-9

    rel_l1 = (mae / denom) * 100.0
    return mae, rel_l1


# ---------------------------------------------------------------
# 3. Quaternion Geodesic Error (channels 5–8)
# ---------------------------------------------------------------
def compute_quaternion_metrics(q_pred, q_gt):
    """
    q_pred, q_gt : arrays shape (4, H, W)
                   q_gt MUST be unit-length per pixel.

    Returns
    -------
    mean_theta_deg : float
    median_theta_deg : float
    p95_theta_deg : float
    """

    # Normalize prediction per pixel
    norm = np.linalg.norm(q_pred, axis=0, keepdims=True) + 1e-12
    q_pred_unit = q_pred / norm

    # Dot product <q_pred, q_gt> at every pixel
    dot = np.sum(q_pred_unit * q_gt, axis=0)

    # q ~ -q ambiguity → take absolute value
    dot = np.abs(dot)
    dot = np.clip(dot, 0.0, 1.0)

    # Geodesic angle on SO(3)
    theta = 2.0 * np.arccos(dot)           # radians
    theta_deg = np.degrees(theta)

    return (
        float(np.mean(theta_deg)),
        float(np.median(theta_deg)),
        float(np.percentile(theta_deg, 95)),
    )


# ---------------------------------------------------------------
# 4. Optional utility: denormalize predictions & GT
# ---------------------------------------------------------------
def denormalize(arr, denorm_min, denorm_max):
    """
    arr        : np.array, shape (C,H,W)
    denorm_min : array shape (C,)
    denorm_max : array shape (C,)
    """
    mn = denorm_min[:, None, None]
    mx = denorm_max[:, None, None]
    return arr * (mx - mn) + mn



## Usage exmaple
"""
# Suppose pred_np and gt_np are (C,H,W) arrays already in PHYSICAL units

vm_pred = pred_np[0]
vm_gt   = gt_np[0]
mae_vm, rel_vm = compute_von_mises_metrics(vm_pred, vm_gt)

shear_pred = pred_np[1:5]
shear_gt   = gt_np[1:5]
mae_shear, rel_shear = compute_shear_metrics(shear_pred, shear_gt)

q_pred = pred_np[5:9]
q_gt   = gt_np[5:9]
th_mean, th_med, th_p95 = compute_quaternion_metrics(q_pred, q_gt)

"""

from typing import Dict, Tuple

def compute_k_edges(Nx: int, Ny: int, Lx: float, Ly: float) -> Tuple[np.ndarray, float]:
    """
    Compute radial |k| bin edges and Nyquist frequency.

    Parameters
    ----------
    Nx, Ny : int
        Grid resolution.
    Lx, Ly : float
        Physical domain lengths.

    Returns
    -------
    edges : np.ndarray
        Bin edges for radial frequency histogram.
    k_nyq : float
        Nyquist frequency of the domain.
    """
    dx = Lx / Nx
    dy = Ly / Ny

    k_nyq = min(1.0 / (2.0 * dx), 1.0 / (2.0 * dy))  # isotropic Nyquist
    dk = min(1.0 / Lx, 1.0 / Ly)                    # radial bin spacing

    edges = np.arange(0.0, k_nyq + dk, dk)
    return edges, k_nyq


def radial_power_spectrum(
    field: np.ndarray,
    Lx: float,
    Ly: float,
    edges: np.ndarray,
    use_window: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute radial power spectrum ρ(k) and its magnitude √ρ(k)
    for a 2D real-valued field on a uniform grid.

    Notes
    -----
    This implementation **exactly** mirrors the project's logic:
      • float64 casting for FFT stability
      • mean subtraction
      • optional Hann window + compensation via window-energy gain
      • FFT normalization: |F|² / (Nx*Ny)²
      • histogram-bin averaging over |k|
      • NaN/Inf → 0 via np.nan_to_num
      • returns counts for later masking/filtering
    """

    assert field.ndim == 2
    Ny, Nx = field.shape
    dx = Lx / Nx
    dy = Ly / Ny

    # --- 1. Preprocess ------------------------------------------------------
    data = field.astype(np.float64, copy=False)
    data = data - np.mean(data)

    if use_window:
        wy = np.hanning(Ny)
        wx = np.hanning(Nx)
        w2d = np.outer(wy, wx)
        gain2 = np.mean(w2d**2)
        data = data * w2d
    else:
        gain2 = 1.0

    # --- 2. FFT & 2D power spectrum ----------------------------------------
    F = np.fft.fft2(data)
    power2d = (np.abs(F) ** 2) / (Nx * Ny)**2

    if gain2 > 0:
        power2d = power2d / gain2

    # --- 3. Build |k| array -------------------------------------------------
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    kx, ky = np.meshgrid(fx, fy, indexing="xy")
    kr = np.sqrt(kx**2 + ky**2)

    # Bin centers
    edges = np.asarray(edges, dtype=float)
    k_centers = 0.5 * (edges[:-1] + edges[1:])

    # Flatten for histogramming
    kr_flat = kr.ravel()
    pw_flat = power2d.ravel()

    # Weighted histogram (sum of power in each radial bin)
    bin_power, _ = np.histogram(kr_flat, bins=edges, weights=pw_flat)
    counts, _    = np.histogram(kr_flat, bins=edges)

    # Avoid NaN/Inf from empty bins
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_k = bin_power / counts

    rho_k = np.nan_to_num(rho_k, nan=0.0, posinf=0.0, neginf=0.0)

    # Magnitude spectrum
    sqrt_rho_k = np.sqrt(rho_k)

    return {
        "k": k_centers,
        "rho": rho_k,
        "sqrt_rho": sqrt_rho_k,
        "counts": counts.astype(np.int64, copy=False),
    }



## Usage exmaple
"""
k_edges, k_nyq = compute_k_edges(Nx=128, Ny=128, Lx=1.0, Ly=1.0)

psd_gt = radial_power_spectrum(gt_field, 1.0, 1.0, k_edges)
psd_pred = radial_power_spectrum(pred_field, 1.0, 1.0, k_edges)

plt.plot(psd_gt["k"], psd_gt["sqrt_rho"], label="GT")
plt.plot(psd_pred["k"], psd_pred["sqrt_rho"], label="Pred")
plt.yscale("log")
plt.legend()

"""

def radial_power_spectrum_plot(
    pred_field,
    gt_field,
    steps,
    N,
    case_id,
    case_dir,
    filename,
    start_id
):
    fig, axs = plt.subplots(1, steps, figsize=((3*steps), 4), sharey=True)

    # Frequency bins
    k_edges, k_nyq = compute_k_edges(Nx=N, Ny=N, Lx=1.0, Ly=1.0)

    for col in range(steps):
        gt   = gt_field[col, :]
        pred = pred_field[col, :]
        # err  = gt - pred  # signed error field

        psd_gt   = radial_power_spectrum(gt,   1.0, 1.0, k_edges)
        psd_pred = radial_power_spectrum(pred, 1.0, 1.0, k_edges)
        # psd_err  = radial_power_spectrum(err,  1.0, 1.0, k_edges)

        ax = axs[col]

        ax.plot(psd_gt["k"],   psd_gt["sqrt_rho"],   label="GT")
        ax.plot(psd_pred["k"], psd_pred["sqrt_rho"], label="Pred")
        # ax.plot(psd_err["k"],  psd_err["sqrt_rho"],  label="Err")

        ax.set_title(f"Strain {start_id+col+1}%", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)

        # Optional: Nyquist line
        ax.axvline(k_nyq, color="k", ls="--", alpha=0.3)

    # Axis labels
    axs[0].set_ylabel("Radial Spectrum √ρ(k)", fontsize=12)
    for ax in axs:
        ax.set_xlabel("|k|", fontsize=10)

    # Single legend (outside)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(
        f"{case_dir}/{filename}_case_{case_id}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)



def rollout(model, x0, steps, N, C, w, case_dir):
    """
    Autoregressive rollout with window.

    Parameters
    ----------
    model : trained nnx
        Predicts increments of shape (N*N*C,)
    x0 : ndarray
        Initial state, shape (N*N*w*C,)
    steps : int
        Number of rollout steps
    N : int
        Grid size
    C : int
        Channels per time step || Ouput Channels
    w : int
        Window length

    Returns
    -------
    outputs : ndarray
        Shape (steps, N*N*w*C)
    """

    Cin = w * C
    x = x0.copy()
    outputs = []
    for _ in range(steps):
        # 1) Reshape to grid
        x_grid = x.reshape(N, N, Cin)
        
        # 2) Extract last time slice u^t
        u_t = x_grid[:, :, -C:]                                     # (N, N, C)

        # 3) Predict increment Δu
        dx = model.Predict(x.reshape(-1, 1).T).reshape(-1).reshape(N,N,C)            # (N, N, C)

        # 4) Compute next state
        u_next = u_t + dx                                           # (N, N, C)

        # 5) Drop oldest time slice
        x_grid_new = x_grid[:, :, C:]                               # remove first C channels

        # 6) Append new time slice
        x_grid_new = np.concatenate([x_grid_new, u_next],axis=-1)   # (N, N, w*C)

        # 7) Flatten for next iteration
        x = x_grid_new.reshape(-1)

        outputs.append(u_next.copy())

    return np.array(outputs)



def plot_to_check(train_set,N,window_lenght, channel_length,indices,case_dir,plot_name,scale=(1.,1.,1.)):
    stride = channel_length * window_lenght
    for id in indices:   # for test set
        random_sample_id = id

        fig, ax = plt.subplots(1, channel_length, figsize=((channel_length*5), 8))

        # First sample
        for i in range(channel_length):
            if (channel_length % 5 == 0 or channel_length % 6 == 0) and i==0:
                im = ax[i].imshow(scale[0] * train_set[random_sample_id, ::stride].reshape(N, N), cmap='viridis')
                ax[i].set_title(f"Current Step Von-Mises Stress (Sample {random_sample_id})")
                fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            elif (channel_length % 6 == 0) and i==5:
                im = ax[i].imshow(scale[2] * train_set[random_sample_id, i::stride].reshape(N, N), cmap='viridis')
                ax[i].set_title(f"Current Step time difference (Sample {random_sample_id})")
                fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            else:
                im = ax[i].imshow(scale[1] * train_set[random_sample_id, i::stride].reshape(N, N), cmap='viridis')
                ax[i].set_title(f"Current Step $alpha_{i}$ (Sample {random_sample_id})")
                fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)

        # Layout & save
        for a in ax.flat:
            a.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, f'{plot_name}_{N}_{id}.png'), dpi=300)
        plt.close()


def batch_rollout(model, x0, steps, N, C, w, case_dir):
    """
    Autoregressive rollout with window.

    Parameters
    ----------
    model : trained nnx
        Predicts increments of shape (B,N*N*C,)
    x0 : ndarray
        Initial state, shape (B,N*N*w*C)
    steps : int
        Number of rollout steps
    N : int
        Grid size
    C : int
        Channels per time step || Ouput Channels
    w : int
        Window length

    Returns
    -------
    outputs : ndarray
        Shape (steps, B, N*N*w*C)
    """

    Cin = w * C
    x = x0.copy()
    outputs = []
    B = x0.shape[0]

    for _ in range(steps):
        # 1) Reshape to grid
        x_grid = x.reshape(B, N, N, Cin)
        
        # 2) Extract last time slice u^t
        u_t = x_grid[:, :, :, -C:]                                     # (B, N, N, C)

        # 3) Predict increment Δu
        dx = model.Predict(x.reshape(B,-1)).reshape(-1).reshape(B,N,N,C)            # (B, N, N, C)

        # 4) Compute next state
        u_next = u_t + dx                                           # (B, N, N, C)

        # 5) Drop oldest time slice
        x_grid_new = x_grid[:, :, :, C:]                               # remove first C channels

        # 6) Append new time slice
        x_grid_new = np.concatenate([x_grid_new, u_next],axis=-1)   # (B, N, N, w*C)

        # 7) Flatten for next iteration
        x = x_grid_new.reshape(-1)

        outputs.append(u_next.copy())

    return np.array(outputs)

def error_time_plot_(error_data_dict,which_channel,w,filename,case_dir):
    # -----------------------------
    # Parameters
    # -----------------------------
    channel = which_channel          # von Mises
    step = 0             # choose which step to visualize
    show_log = False     # optional

    # -----------------------------
    # Extract data
    # -----------------------------
    data = error_data_dict[f"step_{step}"]     # (B, N, N, C)
    data_c = data[..., channel]                # (B, N, N)

    B, N, _ = data_c.shape

    # Flatten spatial dimensions
    data_flat = data_c.reshape(B, -1)           # (B, N*N)

    if show_log:
        data_flat = np.log10(data_flat + 1e-12)

    # -----------------------------
    # Box plot
    # -----------------------------
    box_data = []

    steps = sorted(error_data_dict.keys(), key=lambda x: int(x.split('_')[1]))

    for step_key in steps:
        data = error_data_dict[step_key][..., channel]   # (B, N, N)
        box_data.append(data.reshape(-1))                # (B*N*N,)
    final_step = int(w + len(steps))
    plt.figure(figsize=(12, 5))
    plt.boxplot(box_data, positions=np.arange(w,final_step), showfliers=False)
    plt.xlabel("Time step")
    plt.ylabel("Error (von Mises)")
    plt.title("Error evolution over time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, f'{filename}.png'), dpi=300)
    plt.close()


def rollout_direct(model, x0, steps, N, C, w):
    """
    Autoregressive rollout with window.

    Parameters
    ----------
    model : trained nnx
        Predicts increments of shape (N*N*C,)
    x0 : ndarray
        Initial state, shape (N*N*w*C,)
    steps : int
        Number of rollout steps
    N : int
        Grid size
    C : int
        Channels per time step || Ouput Channels
    w : int
        Window length

    Returns
    -------
    outputs : ndarray
        Shape (steps, B, N, N, C)
    """

    Cout = C
    x = x0.copy()
    outputs = []
    if x0.ndim==2:
        B = x0.shape[0]
    else:
        x0 = x0[None,:]
        B = 1

    # 1) Reshape to grid
    x_grid = x.reshape(B, N, N, C)
    # 2) Extract last time slice u^t
    u_t = x_grid.copy()                                     # (B, N, N, C)
    t_init = np.ones((B,N,N,1))                              # (B, N, N)
    
    for t in range(steps):

        t_channel = ((t+1)/steps) * t_init
        x_grid = np.concatenate((u_t,t_channel),axis=-1)

        # 3) Predict increment Δu
        dx = model.Predict(x_grid).reshape(-1).reshape(B,N,N,Cout)            # (B, N, N, C)

        # 4) Compute next state
        u_next = u_t + dx                                           # (B, N, N, C)

        outputs.append(u_next.copy())

    return np.array(outputs)


def data_loader_hdf5(which:str='von_Mises_stress',sim_id:int=0,increment:int=75,path:str="Cu_textured_dataset.hdf5",excluded_idx:list[int]=None):
    
    """
    The following is borrowed from load_data from `damask_data_utilities.py` with some modification.

    Load HDF5 simulation data into a nested dictionary.

    Parameters
    ----------
    which : str
        String of top-level groups (data types) to load from the HDF5 file.
        Valid strings are as follows: `gamma_slip_(-1-11)`, `gamma_slip_(-11-1)`, `gamma_slip_(1-1-1)`, `gamma_slip_(111)`, `von_Mises_stress`, `orientations`
    increments : int
        Index of increments to load.
    ids : int
        Index of simulation IDs to load.
    path : str, optional
        Path to the HDF5 file.

    Returns
    -------
    Tuple: [numpy.ndarray, dict] | [None,None]
        Numpy 1DArray and an auxilary dictionary with structure:
        (numpy.array, {`"sim_id"`: id, `"increment"`: increment id, 
        `"data"`: which string, `"max"`: max(numpy.array), `"min"`: min(numpy.array)})
    """
    
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None, None

    with h5py.File(path, "r") as f:
        
        if sim_id not in excluded_idx:
            inc_key = f"increment_{increment}"
            sim_key = f"simulation_{sim_id}"
            h5_key = f"{which}/{inc_key}/{sim_key}"
            if h5_key in f:
                data = f[h5_key][...]
            else:
                raise ValueError(f"Missing dataset: {h5_key}")
            if data.ndim == 2:
                data = data[:,:,None]
        
        return data, {"sim_id": sim_id, "increment":increment, "data":which, "max":np.max(data), "min":np.min(data)}
    

def create_autoregressive_set(window_length:int, sim_ids:Tuple[int,int], increments:list[int], which_data:list[str], N:int=128, excluded_idx:list[int]=None,
                              path:str="Cu_textured_dataset.hdf5", dtype:np.dtype=np.float64, normalize:bool= True | False):
    
    excluded_idx = excluded_idx or []
    valid_sim_ids = [s for s in range(*sim_ids) if s not in excluded_idx]
    sim_len = len(valid_sim_ids)
    time_steps = len(increments) - window_length
    x_data_all = np.zeros((sim_len*time_steps,N*N,len(which_data)*window_length),dtype=dtype)
    y_data_all = np.zeros((sim_len*time_steps,N*N,len(which_data)),dtype=dtype)
    x_aux_all = []
    y_aux_all = []
    for incr in range(time_steps):
        for sim_index, sim_id in enumerate(valid_sim_ids):
            for w in range(window_length):
                for channel_id, which in enumerate(which_data):
                    
                    sample_id = incr * sim_len + sim_index
                    channel = w * len(which_data) + channel_id

                    x_arr, _ = data_loader_hdf5(which,sim_id,increments[incr+w],path,excluded_idx=[])
                    assert x_arr.shape[:2] == (N, N)
                    x_data_all[sample_id,:,channel] = x_arr.flatten()

                    # x_aux_all.append(x_aux_dict)
                    if w == 0:
                        y_arr, _ = data_loader_hdf5(which,sim_id,increments[incr+window_length],path,excluded_idx=[])
                        assert y_arr.shape[:2] == (N, N)
                        y_data_all[sample_id,:,channel_id] = y_arr.flatten()
                        # y_aux_all.append(y_aux_dict)

    
    if normalize:
        C = len(which_data)
        x_view = x_data_all.reshape(x_data_all.shape[0], N, N, window_length, C)
        x_scale = np.max(np.abs(x_view), axis=(0, 1, 2, 3))
        y_view = y_data_all.reshape(y_data_all.shape[0], N, N, C)
        y_scale = np.max(np.abs(y_view), axis=(0, 1, 2))
        scales = np.maximum(x_scale, y_scale)
        scales[scales == 0] = 1.0

        x_data_all = (x_view / scales[None, None, None, None, :]).reshape(x_data_all.shape)
        y_data_all = (y_view / scales[None, None, None, :]).reshape(y_data_all.shape)
    else:
        scales = np.ones((1,C))
    return x_data_all.reshape(x_data_all.shape[0], -1), y_data_all.reshape(y_data_all.shape[0], -1), scales

def create_time_channel_set(sim_ids:Tuple[int,int], increments:list[int], which_data:list[str], N:int=128, excluded_idx:list[int]=None,
                              path:str="Cu_textured_dataset.hdf5", dtype:np.dtype=np.float64,normalize:bool=True | False):
    
    excluded_idx = excluded_idx or []
    valid_sim_ids = [s for s in range(*sim_ids) if s not in excluded_idx]
    time_steps = len(increments)
    # Create all (t1, t2) index pairs
    data_index = []
    for sim_idx in valid_sim_ids:
        for t1 in range(time_steps - 1):
            for t2 in range(t1 + 1, time_steps):
                data_index.append((sim_idx, t1, t2))


    def get_item(index):
        sim_idx, t1, t2 = data_index[index]

        def load_frame(t_idx):
            arr = np.zeros((N,N,len(which_data)),dtype=dtype)
            # Load and normalize stress
            for channel, which in enumerate(which_data):
                x_arr, _ = data_loader_hdf5(which,sim_idx,increments[t_idx],path,excluded_idx=[])
                if x_arr is None:
                    raise ValueError(f"x_arr passed as {type(x_arr)}, it should be of type {type(np.array([]))}")

                arr[:,:,channel] = x_arr.squeeze()
            return arr

        x = load_frame(t1)
        y = load_frame(t2)

        dt = (increments[t2] - increments[t1]) / (increments[-1] - increments[0])  # normalized time difference
        inputs_t = np.ones((N, N, 1), dtype=dtype) * dt
        x_with_time = np.concatenate((x, inputs_t), axis=-1)  # shape: (N, N, C+1)

        return x_with_time, y
    
    x_arr = np.zeros((len(data_index), N, N, len(which_data) + 1), dtype=dtype)
    y_arr = np.zeros((len(data_index), N, N, len(which_data)), dtype=dtype)
    # dt_arr = np.zeros((len(data_index), N, N, 1), dtype=dtype)
    for idx in range(len(data_index)):
        x_arr[idx,:,:,:], y_arr[idx,:,:,:]= get_item(idx)

    if normalize:
        C = len(which_data)

        x_view = x_arr.reshape(x_arr.shape[0], N, N, C + 1)
        y_view = y_arr.reshape(y_arr.shape[0], N, N, C)

        x_phys = x_view[..., :C]
        x_time = x_view[..., C:]

        x_phys_scale = np.max(np.abs(x_phys), axis=(0, 1, 2))
        y_phys_scale = np.max(np.abs(y_view), axis=(0, 1, 2))

        phys_scales = np.maximum(x_phys_scale, y_phys_scale)
        phys_scales[phys_scales == 0] = 1.0

        time_scale = np.max(np.abs(x_time))
        if time_scale == 0:
            time_scale = 1.0

    else:
        phys_scales = np.ones(5,)
        time_scale = 1.

    x_phys = x_phys / phys_scales[None, None, None, :]
    y_view = y_view / phys_scales[None, None, None, :]
    x_time = x_time / time_scale

    x_view = np.concatenate([x_phys, x_time], axis=-1)
    x_arr = x_view.reshape(x_arr.shape)
    y_arr = y_view.reshape(y_arr.shape)


    scales = {"physical": phys_scales,  # shape (C,)
                "time": time_scale}        # scalar

    return x_arr.reshape(len(data_index), -1), y_arr.reshape(len(data_index), -1), scales

def gt_loader(which_data:list[str],sim_id:int,increments:list[int],path:str,excluded_idx:list[int],N:int):
    
    assert sim_id not in excluded_idx, f'simulation id is missing! please pick another!'
    time_steps = len(increments)
    channels = len(which_data)
    gt = np.zeros((time_steps,N*N,channels))
        
    for idx, incr in enumerate(increments):
        for ch, which in enumerate(which_data):
            arr,_ = data_loader_hdf5(which,sim_id,incr,path,excluded_idx=[])
            gt[idx,:,ch] = arr.flatten()
    return gt.reshape(time_steps,-1)