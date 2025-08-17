import numpy as np
from skimage import measure
import trimesh

def gyroid_function(x, y, z, fx=2, fy=2, fz=2, phi_x=0, phi_y=0, phi_z=0, c=0):
    return (
        np.cos(fx * np.pi * x + phi_x) * np.sin(fy * np.pi * y + phi_y) +
        np.cos(fy * np.pi * y + phi_y) * np.sin(fz * np.pi * z + phi_z) +
        np.cos(fz * np.pi * z + phi_z) * np.sin(fx * np.pi * x + phi_x)
        - c
    )


# Grid size and domain
nx, ny, nz = 21, 21, 21
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
z = np.linspace(0, 1, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Compute gyroid scalar field
gyroid = gyroid_function(X, Y, Z, c=0)


verts, faces, normals, values = measure.marching_cubes(gyroid, level=0.0)

# Create a mesh object
mesh = trimesh.Trimesh(vertices=verts, faces=faces)

# Export to STL
mesh.export("gyroid_shell.stl")


# import numpy as np
# from skimage import measure
# import trimesh

# def gyroid_function(x, y, z, fx=2, fy=2, fz=2, phi_x=0, phi_y=0, phi_z=0, c=0):
#     return (
#         np.cos(fx * np.pi * x + phi_x) * np.sin(fy * np.pi * y + phi_y) +
#         np.cos(fy * np.pi * y + phi_y) * np.sin(fz * np.pi * z + phi_z) +
#         np.cos(fz * np.pi * z + phi_z) * np.sin(fx * np.pi * x + phi_x)
#         - c
#     )

# # Grid size and domain
# nx, ny, nz = 21, 21, 21
# x = np.linspace(0, 1, nx)
# y = np.linspace(0, 1, ny)
# z = np.linspace(0, 1, nz)
# X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# # Compute gyroid scalar field
# gyroid = gyroid_function(X, Y, Z, c=0)

# # Set thickness (solid region: -thickness < gyroid < +thickness)
# thickness = 0.4
# solid_mask = np.abs(gyroid) < thickness

# # Marching cubes on the binary mask (solid region)
# verts, faces, normals, values = measure.marching_cubes(solid_mask.astype(float), level=0.5)

# # Create mesh and export
# mesh = trimesh.Trimesh(vertices=verts, faces=faces)
# mesh.export("gyroid_solid.stl")


# import numpy as np
# import meshio

# # === Step 1: Define Gyroid Function === #
# def gyroid_function(x, y, z, fx=2, fy=2, fz=2, phi_x=0, phi_y=0, phi_z=0, c=0):
#     return (
#         np.cos(fx * np.pi * x + phi_x) * np.sin(fy * np.pi * y + phi_y) +
#         np.cos(fy * np.pi * y + phi_y) * np.sin(fz * np.pi * z + phi_z) +
#         np.cos(fz * np.pi * z + phi_z) * np.sin(fx * np.pi * x + phi_x)
#         - c
#     )

# # === Step 2: Create Grid === #
# nx, ny, nz = 30, 30, 30  # voxel resolution
# Lx, Ly, Lz = 1.0, 1.0, 1.0  # domain size

# x = np.linspace(0, Lx, nx)
# y = np.linspace(0, Ly, ny)
# z = np.linspace(0, Lz, nz)
# X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# # === Step 3: Evaluate and Threshold === #
# gyroid = gyroid_function(X, Y, Z, c=0)
# threshold = 0.2
# material = np.where(np.abs(gyroid) < threshold, 1.0, 0.1)  # material density or property

# # === Step 4: Build Hexahedral Mesh === #
# dx = Lx / (nx - 1)
# dy = Ly / (ny - 1)
# dz = Lz / (nz - 1)

# points = []
# point_index = {}
# counter = 0

# for i in range(nx):
#     for j in range(ny):
#         for k in range(nz):
#             points.append([x[i], y[j], z[k]])
#             point_index[(i, j, k)] = counter
#             counter += 1

# cells = []
# cell_data = []

# for i in range(nx - 1):
#     for j in range(ny - 1):
#         for k in range(nz - 1):
#             n0 = point_index[(i, j, k)]
#             n1 = point_index[(i+1, j, k)]
#             n2 = point_index[(i+1, j+1, k)]
#             n3 = point_index[(i, j+1, k)]
#             n4 = point_index[(i, j, k+1)]
#             n5 = point_index[(i+1, j, k+1)]
#             n6 = point_index[(i+1, j+1, k+1)]
#             n7 = point_index[(i, j+1, k+1)]

#             cells.append([n0, n1, n2, n3, n4, n5, n6, n7])
#             cell_data.append(material[i, j, k])  # one material value per cell

# # === Step 5: Export to .vtu === #
# mesh = meshio.Mesh(
#     points=np.array(points),
#     cells=[("hexahedron", np.array(cells))],
#     cell_data={"material": [np.array(cell_data)]}
# )

# mesh.write("gyroid_voxel.vtu")

# import numpy as np
# import meshio

# # === Gyroid Scalar Function ===
# def gyroid_function(x, y, z, fx=2, fy=2, fz=2, phi_x=0, phi_y=0, phi_z=0, c=0):
#     return (
#         np.cos(fx * np.pi * x + phi_x) * np.sin(fy * np.pi * y + phi_y) +
#         np.cos(fy * np.pi * y + phi_y) * np.sin(fz * np.pi * z + phi_z) +
#         np.cos(fz * np.pi * z + phi_z) * np.sin(fx * np.pi * x + phi_x)
#         - c
#     )

# # === Grid Settings ===
# nx, ny, nz = 50, 50, 50         # number of points
# Lx, Ly, Lz = 1.0, 1.0, 1.0      # physical domain size
# x = np.linspace(0, Lx, nx)
# y = np.linspace(0, Ly, ny)
# z = np.linspace(0, Lz, nz)
# X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# # === Evaluate Gyroid Function ===
# gyroid = gyroid_function(X, Y, Z, c=0)
# threshold = 0.2
# material = np.where(np.abs(gyroid) < threshold, 1.0, 0.1)  # Material field

# # === Create Node List ===
# points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
# point_index = np.arange(len(points)).reshape((nx, ny, nz))

# # === Create Hexahedral Elements and Cell Data ===
# cells = []
# cell_values = []

# for i in range(nx - 1):
#     for j in range(ny - 1):
#         for k in range(nz - 1):
#             n0 = point_index[i  , j  , k  ]
#             n1 = point_index[i+1, j  , k  ]
#             n2 = point_index[i+1, j+1, k  ]
#             n3 = point_index[i  , j+1, k  ]
#             n4 = point_index[i  , j  , k+1]
#             n5 = point_index[i+1, j  , k+1]
#             n6 = point_index[i+1, j+1, k+1]
#             n7 = point_index[i  , j+1, k+1]
#             cells.append([n0, n1, n2, n3, n4, n5, n6, n7])
#             cell_values.append(material[i, j, k])

# cells = np.array(cells)
# cell_values = np.array(cell_values)

# # === Export to .vtu ===
# mesh = meshio.Mesh(
#     points=points,
#     cells=[("hexahedron", cells)],
#     cell_data={"material": [cell_values]}
# )

# mesh.write("gyroid_solid.vtu")
# print("Exported gyroid_solid.vtu for visualization in ParaView.")


# import numpy as np
# from skimage import measure
# import trimesh

# def gyroid_function(x, y, z, fx=2, fy=2, fz=2, phi_x=0, phi_y=0, phi_z=0, c=0.0):
#     return (
#         np.cos(fx * np.pi * x + phi_x) * np.sin(fy * np.pi * y + phi_y) +
#         np.cos(fy * np.pi * y + phi_y) * np.sin(fz * np.pi * z + phi_z) +
#         np.cos(fz * np.pi * z + phi_z) * np.sin(fx * np.pi * x + phi_x)
#         - c
#     )

# # === Grid resolution and domain size ===
# nx, ny, nz = 41, 41, 41
# x = np.linspace(0, 1, nx)
# y = np.linspace(0, 1, ny)
# z = np.linspace(0, 1, nz)
# X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# # === Define thickness (band width around the zero surface) ===
# thickness = 0.05  # Increase for thicker solid
# fx, fy, fz = 2, 2, 2  # Frequency

# # === Evaluate Gyroid Field ===
# G = gyroid_function(X, Y, Z, fx=fx, fy=fy, fz=fz, c=0.0)

# # === Create binary solid region around the isosurface ===
# solid_mask = np.abs(G) < thickness

# # === Convert solid voxels to a mesh using marching cubes ===
# verts, faces, normals, values = measure.marching_cubes(solid_mask.astype(float), level=0.5)

# # === Scale mesh back to actual domain size ===
# scale = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
# verts = verts * scale

# # === Create Trimesh object and export ===
# mesh = trimesh.Trimesh(vertices=verts, faces=faces)
# mesh.export("solid_gyroid_pure.stl")

# print("✅ Solid gyroid exported as 'solid_gyroid.stl'")


import numpy as np
from skimage import measure
import trimesh

def gyroid_function(x, y, z, fx=2, fy=2, fz=2, c=0.0):
    return (
        np.cos(fx * np.pi * x) * np.sin(fy * np.pi * y) +
        np.cos(fy * np.pi * y) * np.sin(fz * np.pi * z) +
        np.cos(fz * np.pi * z) * np.sin(fx * np.pi * x)
        - c
    )

# Grid settings
nx, ny, nz = 100, 100, 100
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
z = np.linspace(0, 1, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Compute scalar field
G = gyroid_function(X, Y, Z, c=0.0)

# Extract surface using marching cubes
verts, faces, normals, _ = measure.marching_cubes(G, level=0.0)

# Scale to match physical domain
scale = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
verts_scaled = verts * scale

# Offset distance for wall thickness
offset = 0.02

# Offset using the normals
verts_inward = verts_scaled - offset * normals
verts_outward = verts_scaled + offset * normals

# Create two mesh shells
mesh_inward = trimesh.Trimesh(vertices=verts_inward, faces=faces, process=False)
mesh_outward = trimesh.Trimesh(vertices=verts_outward, faces=faces, process=False)

# Flip the inward faces so normals point inward
mesh_inward.faces = mesh_inward.faces[:, ::-1]

# Merge both into one watertight mesh (optional: can do boolean here)
gyroid_shell = trimesh.util.concatenate([mesh_outward, mesh_inward])

# Export as STL
gyroid_shell.export("gyroid_solid_shell.stl")
print("✅ Exported 'gyroid_solid_shell.stl'")

