"""
Train 3-D physics-informed FNOs and compare homogenized tensors with FEM.

By default, training uses 3,000 Fourier microstructures: 1,000 each from the
``[1, 2, 3]``, ``[2, 4, 6]``, and ``[4, 6, 8]`` frequency families (on all
three coordinate axes).
"""

import sys

# Import the helpers first: they apply the requested JAX platform before JAX is
# imported by the rest of the workflow.
from micro_mechanics_fno_homogenized_elasticity_tensor_3d_functions import (
    compare_models,
    create_problem,
    ensure_fem_dataset,
    load_dataset,
    parse_args,
    train_models,
    validate_args,
)
import jax

from fol.tools.logging_functions import Logger


if __name__ == "__main__":
    # directory and save handling
    args = parse_args()
    validate_args(args)

    args.case_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(args.case_dir / "fno_homogenized_3d.log")
    print("Per-loading-condition FNO surrogate for 3-D periodic homogenization")
    print(f"stage={args.stage}, devices={jax.devices()}, case_dir={args.case_dir}")

    # problem setup and model creation
    mesh, loss, solver, stress_function = create_problem(args)

    # held-out finite-element reference data
    data = None
    if args.stage in ("all", "data", "evaluate"):
        data = ensure_fem_dataset(args, mesh, loss, solver)

    # physics-informed FNO training and comparison
    if args.stage != "data":
        if data is None:
            data = load_dataset(args)
        if args.stage in ("all", "train"):
            train_models(args, data, loss)
        if args.stage in ("all", "compare", "evaluate"):
            compare_models(args, data, mesh, loss, solver, stress_function)
