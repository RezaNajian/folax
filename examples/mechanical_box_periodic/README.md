# 3-D periodic homogenization with a physics-informed FNO

This guide explains how to run
[`micro_mechanics_fno_homogenized_elasticity_tensor_3d.py`](micro_mechanics_fno_homogenized_elasticity_tensor_3d.py)
for the first time.

The example trains six Fourier Neural Operators (FNOs), one for each 3-D
Voigt load case (`xx`, `yy`, `zz`, `xy`, `yz`, and `xz`). Each model maps a
spatially varying material multiplier and a prescribed macroscopic deformation
gradient to the periodic displacement fluctuation field. Training is
physics-informed: it uses the periodic finite-element residual instead of
labelled displacement solutions. FEM solutions are generated only for the
held-out test samples used in the final comparison.

## 1. Install Folax

Run commands from the repository root:

```bash
cd /path/to/folax_multi_input
```

Python 3.11 or newer is required. For a CPU installation:

```bash
python -m pip install -e '.[cpu]'
```

For an NVIDIA CUDA installation:

```bash
python -m pip install -e '.[cuda]'
```

Check that JAX sees the expected device:

```bash
python -c "import jax; print(jax.devices())"
```

The script defaults to GPU execution. Always pass `--jax-platform cpu` on a
machine without a working CUDA installation.

## 2. Run a small CPU smoke test

Start with this deliberately small configuration:

```bash
python examples/mechanical_box_periodic/micro_mechanics_fno_homogenized_elasticity_tensor_3d.py \
    --stage all \
    --jax-platform cpu \
    --N 4 \
    --num-train 6 \
    --num-test 1 \
    --epochs 1 \
    --batch-size 2 \
    --width 8 \
    --modes 3 \
    --layers 2 \
    --solver JAX-direct \
    --benchmark-repeats 1 \
    --coeffs-file examples/mechanical_box_periodic/runs/smoke_cpu/fourier_coefficients.npy \
    --case-dir examples/mechanical_box_periodic/runs/smoke_cpu \
    --no-visualize \
    --no-vtk
```

This command exercises the complete workflow:

1. Generate reproducible Fourier coefficients from `--seed`.
2. Construct the training and held-out material fields.
3. Solve all six FEM load cases for the held-out sample.
4. Train all six FNO models for one epoch.
5. Compare their displacement fields, averaged stresses, homogenized tensors,
   and inference times with FEM.

The smoke-test network is intentionally tiny and is not expected to be
accurate. Its purpose is to verify the installation and workflow. The first
run can spend noticeable time compiling JAX functions.

## 3. Inspect the results

The smoke test writes to
`examples/mechanical_box_periodic/runs/smoke_cpu`. The most useful files are:

```text
smoke_cpu/
├── fno_homogenized_3d.log          complete console log
├── physics_informed_dataset.npz    material fields and held-out FEM targets
├── model_metadata.json             architecture and training metadata
├── models/
│   ├── xx.pkl ... xz.pkl           six reusable FNO checkpoints
│   └── xx/ ... xz/                 per-load-case training artifacts
├── performance_comparison.csv      metrics and timings by load case
├── comparison_summary.json         machine-readable summary
├── fno_fem_comparison.npz          raw comparison arrays
└── fno_fem_comparison.png          summary chart
```

When visualization and VTK export are enabled, the script also creates:

```text
CASE_DIR/visualizations/sample_0000/xx/
CASE_DIR/vtk_comparison/
```

## 4. Run a real experiment

The built-in defaults are a much larger GPU-oriented experiment:

- `N=22` nodes per coordinate direction;
- 3,000 training material fields;
- 20 held-out test fields;
- 500 epochs for each of six FNOs;
- FNO width 32, four layers, and eight Fourier modes;
- three Fourier material families with frequencies `(1,2,3)`, `(2,4,6)`,
  and `(4,6,8)` along every coordinate axis.

These defaults can require substantial GPU memory and runtime. A default run
can be started with:

```bash
python examples/mechanical_box_periodic/micro_mechanics_fno_homogenized_elasticity_tensor_3d.py \
    --stage all \
    --jax-platform gpu \
    --coeffs-file examples/mechanical_box_periodic/runs/fourier_N22/fourier_coefficients.npy \
    --case-dir examples/mechanical_box_periodic/runs/fourier_N22
```

Use a separate `--case-dir` and `--coeffs-file` for each experiment. This keeps
smoke-test data, production data, and checkpoints from being mixed.

## Workflow stages

The `--stage` option lets a long run be split into reusable steps:

| Stage | Behavior |
| --- | --- |
| `all` | Generate or reuse data, train six models, and compare FNO with FEM. |
| `data` | Generate or reuse the material/FEM dataset only. |
| `train` | Load an existing dataset and train the six models. |
| `compare` | Load an existing dataset and checkpoints, then compare them. |
| `evaluate` | Generate missing test data if needed and compare existing checkpoints without training. |

For separate stages, repeat the same mesh, dataset, material, and network
options in every command. In particular, keep `--N`, `--num-train`,
`--num-test`, `--seed`, `--strain-amplitude`, `--material-distribution`,
`--width`, `--modes`, `--layers`, `--case-dir`, and `--model-dir` consistent.

For example, take a successful `--stage all` command and run it as three
commands by changing only the stage, in this order:

```text
--stage data  ->  --stage train  ->  --stage compare
```

Existing compatible datasets are reused, and the comparison stages reuse the
saved checkpoints. Pass `--regenerate-data` with `data`, `all`, or `evaluate`
when the FEM dataset must be rebuilt.

## Material distributions

### Fourier microstructures (default)

No input dataset is required. If `--coeffs-file` does not exist, the script
generates a deterministic normal-random coefficient matrix using `--seed` and
saves it for reuse. The coefficients are transformed into bounded material
fields with values between `0.1` and `1.0`.

An existing coefficient file must be a finite two-dimensional NumPy array with
28 columns and enough rows for the requested split. If a small earlier run
created too few rows, choose a new path with `--coeffs-file`; the script will
generate a correctly sized file there.

### Periodic polycrystals

Select periodic 3-D Voronoi fields with:

```bash
--material-distribution polycrystal --num-grains 16 --grain-values 0.1 1.0
```

The grain seeds and continuous grain values are generated reproducibly from
`--seed`. `--coeffs-file` is not used for polycrystal runs.

## Visualization and VTK output

FE/FNO displacement visualization is enabled by default during `all`,
`compare`, and `evaluate`. It renders total and fluctuation displacement
magnitudes and signed `Ux`, `Uy`, and `Uz` components for held-out sample 0 and
load case `xx`.

Useful controls are:

```bash
--visualization-sample-id 3
--visualization-load-case all
--warp-factor 2.0
--camera-zoom 1.1
--no-visualize
--no-vtk
```

`--no-visualize` disables PyVista PNG rendering. `--no-vtk` independently
disables VTK field export. On a headless machine without a suitable OpenGL
implementation, use `--no-visualize` or configure a supported off-screen
PyVista renderer.

## Important options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--N` | `22` | Nodes along each coordinate direction. Cost grows rapidly in 3-D. |
| `--num-train` | `3000` | Physics-informed training material fields. |
| `--num-test` | `20` | Held-out fields with six FEM reference solves each. |
| `--epochs` | `500` | Epochs for each of the six models. |
| `--batch-size` | `100` | Training batch size. Reduce this first for training OOM errors. |
| `--inference-batch-size` | `1` | Prediction batch size during comparison. |
| `--width` | `32` | Hidden FNO channel width. |
| `--modes` | `8` | Requested Fourier modes per spatial direction. |
| `--layers` | `4` | Number of FNO layers. |
| `--strain-amplitude` | `0.1` | Engineering strain applied in each unit load case. |
| `--solver` | `JAX-bicgstab` | FEM linear solver; iterative mode reduces large direct allocations. |
| `--jax-platform` | `gpu` | JAX backend: `gpu`, `cpu`, or `auto`. |
| `--benchmark-repeats` | `3` | Repetitions used for median online timing. |

Run the following for the complete current option list:

```bash
python examples/mechanical_box_periodic/micro_mechanics_fno_homogenized_elasticity_tensor_3d.py \
    --jax-platform cpu --help
```

## Troubleshooting

### JAX reports that no CUDA platform is available

Run with `--jax-platform cpu`, or reinstall the project with the CUDA extra and
confirm that the NVIDIA driver is visible to JAX.

### The process runs out of memory

Reduce `--batch-size` first. For 3-D inference also keep
`--inference-batch-size 1`. If necessary, reduce `--N`, `--width`, `--modes`,
or `--num-train`. Prefer the default `JAX-bicgstab` solver for a large mesh.

### A dataset or checkpoint is incompatible

Use exactly the same configuration flags for `data`, `train`, and `compare`.
The dataset records its mesh and sampling configuration, while
`model_metadata.json` records the FNO architecture and training distribution.
For a new configuration, the safest approach is a new `--case-dir`.

### A model checkpoint is missing

`compare` and `evaluate` require all six files in `CASE_DIR/models`. Run
`--stage train` or `--stage all` first.

### Visualization fails on a server

PyVista needs a working graphical or off-screen OpenGL backend. Add
`--no-visualize` for a non-rendering run. The numerical comparison chart
`fno_fem_comparison.png` is still generated with Matplotlib.
