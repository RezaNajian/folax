'''
Baseline FEM for 3-D periodic micro-mechanics homogenized elasticity tensor evaluation.
'''

import argparse
import os
import sys
from pathlib import Path


def configure_jax_platform_from_argv(argv):
    platform = "gpu"
    for idx, arg in enumerate(argv):
        if arg == "--jax-platform" and idx + 1 < len(argv):
            platform = argv[idx + 1]
            break
        if arg.startswith("--jax-platform="):
            platform = arg.split("=", 1)[1]
            break

    if platform == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = "all"
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            os.environ.pop("CUDA_VISIBLE_DEVICES")
    elif platform == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif platform != "auto":
        raise ValueError("--jax-platform must be one of: gpu, cpu, auto")


configure_jax_platform_from_argv(sys.argv[1:])

import jax
import jax.numpy as jnp
import numpy as np

if "JAX_PLATFORMS" in os.environ:
    jax.config.update("jax_platforms", os.environ["JAX_PLATFORMS"])
if "JAX_PLATFORM_NAME" in os.environ:
    jax.config.update("jax_platform_name", os.environ["JAX_PLATFORM_NAME"])
if "JAX_CUDA_VISIBLE_DEVICES" in os.environ:
    jax.config.update("jax_cuda_visible_devices", os.environ["JAX_CUDA_VISIBLE_DEVICES"])

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fol.controls.fourier_control import FourierControl
from fol.loss_functions.fe_kinematics import b_matrix_3d
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.logging_functions import Logger
from mechanical_periodic_bc import MechanicalLoss3DHexa
from usefull_functions import (
    create_3D_box_mesh_structured,
    plot_displacement_components_pyvista,
    plot_displacement_contours_pyvista,
)

jax.config.update("jax_default_matmul_precision", "highest")

VOIGT_LABELS = ("xx", "yy", "zz", "xy", "yz", "xz")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the 3D homogenized elasticity tensor with the periodic FE solver."
    )
    parser.add_argument("--N", type=int, default=22, help="Number of nodes per direction.")
    parser.add_argument("--L", type=float, default=1.0, help="Box side length.")
    parser.add_argument("--sample-id", type=int, default=0, help="Material sample id to evaluate.")
    parser.add_argument(
        "--coeffs-file",
        type=Path,
        default=THIS_DIR / "3d_coeffs_matrix_0.npy",
        help="Fourier coefficient file used to reconstruct the nodal material field.",
    )
    parser.add_argument(
        "--uniform-k",
        type=float,
        default=None,
        help="Use a constant material multiplier instead of the Fourier coefficient file.",
    )
    parser.add_argument(
        "--strain-amplitude",
        type=float,
        default=1.0,
        help="Engineering strain amplitude used for each unit load case.",
    )
    parser.add_argument(
        "--solver",
        choices=("JAX-direct", "JAX-bicgstab"),
        default="JAX-bicgstab",
        help="Linear solver backend. JAX-bicgstab avoids large cuSolver allocations.",
    )
    parser.add_argument(
        "--jax-platform",
        choices=("gpu", "cpu", "auto"),
        default="gpu",
        help="JAX device platform. gpu is the default; use cpu for debugging.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not apply the same deterministic shuffle used by the FNO script.",
    )
    parser.add_argument(
        "--no-vtk",
        action="store_true",
        help="Skip writing FE displacement fields to VTK.",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        default=True,
        help="Disable displacement contour images (enabled by default).",
    )
    parser.add_argument(
        "--visualization-load-case",
        choices=("all",) + VOIGT_LABELS,
        default="xx",
        help="Load case to visualize, or 'all' for all six cases (default: xx).",
    )
    parser.add_argument(
        "--warp-factor",
        type=float,
        default=1.0,
        help="Displacement scale used to warp the rendered mesh.",
    )
    parser.add_argument(
        "--camera-zoom",
        type=float,
        default=0.85,
        help="PyVista camera zoom used for displacement images.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=THIS_DIR / "periodic_bcs_FE_homogenized_3D",
        help="Directory for logs and tensor outputs.",
    )
    return parser.parse_args()


def voigt_strain_to_macro_deformation_grad(strain_voigt):
    strain_voigt = jnp.asarray(strain_voigt)
    strain_tensor = jnp.array(
        [
            [strain_voigt[0], 0.5 * strain_voigt[3], 0.5 * strain_voigt[5]],
            [0.5 * strain_voigt[3], strain_voigt[1], 0.5 * strain_voigt[4]],
            [0.5 * strain_voigt[5], 0.5 * strain_voigt[4], strain_voigt[2]],
        ]
    )
    return jnp.eye(3) + strain_tensor


def macro_displacement(fe_mesh, macro_deformation_grad):
    return fe_mesh.nodes_coordinates[:, :3] @ (macro_deformation_grad - jnp.eye(3)).T


def build_control_input(k_field, macro_deformation_grad):
    k_field = jnp.asarray(k_field).reshape(-1)
    control_input = jnp.zeros((k_field.shape[0], 10))
    control_input = control_input.at[:, 0].set(k_field)
    control_input = control_input.at[:, 1:4].set(macro_deformation_grad[0])
    control_input = control_input.at[:, 4:7].set(macro_deformation_grad[1])
    control_input = control_input.at[:, 7:10].set(macro_deformation_grad[2])
    return control_input


def load_material_field(args, fe_mesh):
    if args.uniform_k is not None:
        return args.uniform_k * jnp.ones(fe_mesh.GetNumberOfNodes())

    if not args.coeffs_file.exists():
        print(f"{args.coeffs_file} was not found; using a uniform material multiplier of 1.0.")
        return jnp.ones(fe_mesh.GetNumberOfNodes())

    control_settings = {
        "x_freqs": np.array([2, 4, 6]),
        "y_freqs": np.array([2, 4, 6]),
        "z_freqs": np.array([2, 4, 6]),
        "beta": 10,
        "min": 0.1,
        "max": 1.0,
    }
    fourier_control = FourierControl("K", control_settings, fe_mesh)
    fourier_control.Initialize()
    coeffs_matrix = np.load(args.coeffs_file)
    k_matrix = np.asarray(fourier_control.ComputeBatchControlledVariables(coeffs_matrix)).copy()

    if not args.no_shuffle:
        np.random.seed(42)
        np.random.shuffle(k_matrix)

    if args.sample_id < 0 or args.sample_id >= k_matrix.shape[0]:
        raise ValueError(f"sample-id {args.sample_id} is outside [0, {k_matrix.shape[0] - 1}].")

    return jnp.asarray(k_matrix[args.sample_id])


def create_mechanical_loss(fe_mesh, material_dict):
    bc_dict = {dof: {"periodic_corners": 0.0} for dof in ("Ux", "Uy", "Uz")}
    mechanical_loss_3d = MechanicalLoss3DHexa(
        "mechanical_loss_3d",
        loss_settings={
            "dirichlet_bc_dict": bc_dict,
            "num_gp": 2,
            "material_dict": material_dict,
            "periodic_bc_dict": fe_mesh.periodic_node_pairs,
            "macro_deformation_grad": jnp.eye(3),
            "periodic_bcs": True,
        },
        fe_mesh=fe_mesh,
    )
    mechanical_loss_3d.Initialize()
    return mechanical_loss_3d


def create_volume_average_stress_function(fe_mesh, mechanical_loss_3d):
    gp_points, gp_weights = mechanical_loss_3d.fe_element.GetIntegrationData()
    elements_nodes = jnp.asarray(fe_mesh.GetElementsNodes("hexahedron"))
    nodes_coordinates = jnp.asarray(fe_mesh.GetNodesCoordinates())
    D = mechanical_loss_3d.D
    fe_element = mechanical_loss_3d.fe_element

    @jax.jit
    def compute_volume_average_stress(k_field, total_displacement):
        k_field = jnp.asarray(k_field)
        total_displacement = jnp.asarray(total_displacement)

        def compute_element_contribution(element_nodes):
            xyze = nodes_coordinates[element_nodes, :]
            element_k = k_field[element_nodes]
            element_u = total_displacement[element_nodes, :].reshape(-1, 1)

            def compute_gp_contribution(gp_point, gp_weight):
                n_vec = fe_element.ShapeFunctionsValues(gp_point)
                dn_dx = fe_element.ShapeFunctionsGlobalGradients(xyze, gp_point)
                b_mat = b_matrix_3d(dn_dx)
                det_j = jnp.linalg.det(fe_element.Jacobian(xyze, gp_point))
                weight = gp_weight * det_j
                k_gp = jnp.dot(n_vec, element_k)
                strain_gp = (b_mat @ element_u).reshape(6)
                stress_gp = k_gp * (D @ strain_gp)
                return weight * stress_gp, weight

            gp_stress_contributions, gp_volume_contributions = jax.vmap(
                compute_gp_contribution
            )(gp_points, gp_weights)
            return jnp.sum(gp_stress_contributions, axis=0), jnp.sum(gp_volume_contributions)

        element_stress_contributions, element_volume_contributions = jax.vmap(
            compute_element_contribution
        )(elements_nodes)
        total_stress = jnp.sum(element_stress_contributions, axis=0)
        total_volume = jnp.sum(element_volume_contributions)
        return total_stress / total_volume, total_volume

    return compute_volume_average_stress


def compute_volume_average_stress_python(fe_mesh, mechanical_loss_3d, k_field, total_displacement):
    gp_points, gp_weights = mechanical_loss_3d.fe_element.GetIntegrationData()
    elements_nodes = np.asarray(fe_mesh.GetElementsNodes("hexahedron"))
    nodes_coordinates = jnp.asarray(fe_mesh.GetNodesCoordinates())
    total_stress = jnp.zeros(6)
    total_volume = 0.0

    for element_nodes in elements_nodes:
        xyze = nodes_coordinates[element_nodes, :]
        element_k = jnp.asarray(k_field)[element_nodes]
        element_u = jnp.asarray(total_displacement)[element_nodes, :].reshape(-1, 1)

        for gp_point, gp_weight in zip(gp_points, gp_weights):
            n_vec = mechanical_loss_3d.fe_element.ShapeFunctionsValues(gp_point)
            dn_dx = mechanical_loss_3d.fe_element.ShapeFunctionsGlobalGradients(xyze, gp_point)
            b_mat = b_matrix_3d(dn_dx)
            det_j = jnp.linalg.det(mechanical_loss_3d.fe_element.Jacobian(xyze, gp_point))
            weight = gp_weight * det_j
            k_gp = jnp.dot(n_vec, element_k)
            strain_gp = (b_mat @ element_u).reshape(6)
            stress_gp = k_gp * (mechanical_loss_3d.D @ strain_gp)
            total_stress = total_stress + weight * stress_gp
            total_volume = total_volume + weight

    return total_stress / total_volume, total_volume


def solve_homogenized_tensor(
    fe_mesh,
    linear_fe_solver,
    volume_average_stress_function,
    k_field,
    strain_amplitude,
):
    homogenized_tensor = np.zeros((6, 6))
    averaged_stresses = []
    displacement_fields = {}
    zero_dofs = np.zeros(3 * fe_mesh.GetNumberOfNodes())

    for column_id, label in enumerate(VOIGT_LABELS):
        macro_strain = np.zeros(6)
        macro_strain[column_id] = strain_amplitude
        macro_deformation_grad = voigt_strain_to_macro_deformation_grad(macro_strain)
        control_input = build_control_input(k_field, macro_deformation_grad)

        print(f"{label}: solving periodic FE fluctuation problem")
        fluctuation = np.asarray(linear_fe_solver.SolveReduced(control_input, zero_dofs))
        fluctuation = fluctuation.reshape((fe_mesh.GetNumberOfNodes(), 3))
        total_displacement = fluctuation + np.asarray(macro_displacement(fe_mesh, macro_deformation_grad))
        print(f"{label}: volume-averaging stress")
        average_stress, volume = volume_average_stress_function(k_field, total_displacement)

        average_stress = np.asarray(average_stress)
        averaged_stresses.append(average_stress)
        homogenized_tensor[:, column_id] = average_stress / strain_amplitude
        displacement_fields[f"U_FE_fluctuation_{label}"] = fluctuation
        displacement_fields[f"U_FE_total_{label}"] = total_displacement

        print(f"{label}: volume = {float(volume):.8e}, average stress = {average_stress}")

    return homogenized_tensor, np.asarray(averaged_stresses), displacement_fields


def compute_homogeneous_reference_tensor(mechanical_loss_3d, k_field):
    homogeneous_k = float(np.asarray(jnp.mean(k_field)))
    reference_tensor = homogeneous_k * np.asarray(mechanical_loss_3d.D)
    return reference_tensor, homogeneous_k


def visualize_displacement_fields(args, fe_mesh, displacement_fields):
    """Render total and fluctuation FE displacement fields for selected load cases."""
    if args.visualization_load_case == "all":
        labels = VOIGT_LABELS
    else:
        labels = (args.visualization_load_case,)

    for label in labels:
        output_directory = args.case_dir / "visualizations" / label
        output_directory.mkdir(parents=True, exist_ok=True)
        fields_by_type = {
            "total": {
                "Finite element total": displacement_fields[f"U_FE_total_{label}"]
            },
            "fluctuation": {
                "Finite element fluctuation": displacement_fields[
                    f"U_FE_fluctuation_{label}"
                ]
            },
        }

        for field_type, fields in fields_by_type.items():
            filename_prefix = f"displacement_{field_type}"
            plot_displacement_contours_pyvista(
                fe_mesh=fe_mesh,
                displacement_fields=fields,
                output_path=output_directory / f"{filename_prefix}_contours.png",
                warp_factor=args.warp_factor,
                camera_zoom=args.camera_zoom,
            )
            plot_displacement_components_pyvista(
                fe_mesh=fe_mesh,
                displacement_fields=fields,
                output_directory=output_directory,
                filename_prefix=filename_prefix,
                warp_factor=args.warp_factor,
                camera_zoom=args.camera_zoom,
            )


def main():
    args = parse_args()
    args.case_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(args.case_dir / "periodic_bcs_FE_homogenized_3D.log")

    print("Evaluating homogenized 3D elasticity tensor with periodic FE solves")
    print(f"case_dir: {args.case_dir}")
    print(f"linear solver: {args.solver}")
    print(f"JAX devices: {jax.devices()}")

    fe_mesh = create_3D_box_mesh_structured(
        Nx=args.N,
        Ny=args.N,
        Nz=args.N,
        Lx=args.L,
        Ly=args.L,
        Lz=args.L,
    )
    fe_mesh.Initialize()

    material_dict = {"young_modulus": 1.0, "poisson_ratio": 0.3}
    mechanical_loss_3d = create_mechanical_loss(fe_mesh, material_dict)
    # k_field = load_material_field(args, fe_mesh)
    # print(k_field.shape)
    # Here we use a uniform material multiplier of 1.0 for the baseline FE homogenization.
    k_field = jnp.ones(fe_mesh.GetNumberOfNodes())

    fe_settings = {
        "linear_solver_settings": {
            "solver": args.solver,
            "tol": 1e-6,
            "atol": 1e-6,
            "maxiter": 5000,
            "pre-conditioner": "ilu",
        },
        "nonlinear_solver_settings": {
            "rel_tol": 1e-5,
            "abs_tol": 1e-5,
            "maxiter": 10,
            "load_incr": 5,
        },
    }
    linear_fe_solver = FiniteElementLinearResidualBasedSolver(
        "linear_fe_solver", mechanical_loss_3d, fe_settings
    )
    linear_fe_solver.Initialize()
    volume_average_stress_function = create_volume_average_stress_function(fe_mesh, mechanical_loss_3d)

    c_hom, averaged_stresses, displacement_fields = solve_homogenized_tensor(
        fe_mesh,
        linear_fe_solver,
        volume_average_stress_function,
        k_field,
        args.strain_amplitude,
    )
    c_hom_symmetric = 0.5 * (c_hom + c_hom.T)
    c_ref_homogeneous, homogeneous_k_ref = compute_homogeneous_reference_tensor(
        mechanical_loss_3d, k_field
    )
    c_hom_error = c_hom - c_ref_homogeneous
    c_hom_symmetric_error = c_hom_symmetric - c_ref_homogeneous

    np.save(args.case_dir / "homogenized_elasticity_tensor.npy", c_hom)
    np.save(args.case_dir / "homogenized_elasticity_tensor_symmetric.npy", c_hom_symmetric)
    np.save(args.case_dir / "reference_homogeneous_elasticity_tensor.npy", c_ref_homogeneous)
    np.save(args.case_dir / "homogenized_minus_reference_tensor.npy", c_hom_error)
    np.save(args.case_dir / "homogenized_symmetric_minus_reference_tensor.npy", c_hom_symmetric_error)
    np.savetxt(args.case_dir / "homogenized_elasticity_tensor.txt", c_hom, fmt="%.12e")
    np.savetxt(args.case_dir / "homogenized_elasticity_tensor_symmetric.txt", c_hom_symmetric, fmt="%.12e")
    np.savetxt(
        args.case_dir / "reference_homogeneous_elasticity_tensor.txt",
        c_ref_homogeneous,
        fmt="%.12e",
    )
    np.savetxt(args.case_dir / "homogenized_minus_reference_tensor.txt", c_hom_error, fmt="%.12e")
    np.savetxt(
        args.case_dir / "homogenized_symmetric_minus_reference_tensor.txt",
        c_hom_symmetric_error,
        fmt="%.12e",
    )
    np.savez(
        args.case_dir / "homogenized_elasticity_results.npz",
        C_hom=c_hom,
        C_hom_symmetric=c_hom_symmetric,
        C_reference_homogeneous=c_ref_homogeneous,
        C_hom_minus_reference=c_hom_error,
        C_hom_symmetric_minus_reference=c_hom_symmetric_error,
        homogeneous_k_reference=homogeneous_k_ref,
        averaged_stresses=averaged_stresses,
        voigt_labels=np.asarray(VOIGT_LABELS),
        k_field=np.asarray(k_field),
    )

    print("\nHomogenized elasticity tensor C_hom:")
    print(c_hom)
    print("\nSymmetrized homogenized elasticity tensor:")
    print(c_hom_symmetric)
    print(f"\nReference homogeneous elasticity tensor, k = {homogeneous_k_ref:.12e}:")
    print(c_ref_homogeneous)
    print("\nHomogenized minus reference tensor:")
    print(c_hom_error)
    print("\nSymmetrized homogenized minus reference tensor:")
    print(c_hom_symmetric_error)

    if args.visualize:
        visualize_displacement_fields(args, fe_mesh, displacement_fields)

    if not args.no_vtk:
        fe_mesh["K_matrix"] = np.asarray(k_field)
        for field_name, field_value in displacement_fields.items():
            fe_mesh[field_name] = field_value
        fe_mesh.Finalize(export_dir=args.case_dir)


if __name__ == "__main__":
    main()
