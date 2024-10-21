<p align=center><img height="32.125%" width="32.125%" src="https://github.com/RezaNajian/FOL/assets/62375973/0e1ca4e0-0658-4f5d-aad9-1ae7c9f67574"></p>

[![License][license-image]][license] [![CI](https://github.com/RezaNajian/FOL/actions/workflows/ci.yml/badge.svg)](https://github.com/RezaNajian/FOL/actions/workflows/ci.yml)

[license-image]: https://img.shields.io/badge/license-BSD-green.svg?style=flat
[license]: https://github.com/RezaNajian/FOL/LICENSE

# FOL: Efficient Solution and Optimization of PDEs
Finite Operator Learning (FOL) combines neural operators, physics-informed machine learning, and classical numerical methods to solve and optimize parametrically defined partial differential equations (PDEs). In essence, FOL directly utilizes the discretized residuals of governing equations during the backpropagation step of training, enabling the integration of traditional numerical methods—such as finite element methods (FEM). The advantages of this approach are thoroughly studied and detailed in FOL's publication.

# Main Features
- Python-based framework built on [JAX](https://github.com/jax-ml/jax), leveraging key features like auto-vectorization with [jax.vmap()](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap), just-in-time compilation with [jax.jit()](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit), and automatic differentiation with [jax.grad()](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html#jax.grad) for high-efficiency computations on CPUs, GPUs, and TPUs. This framework integrates seamlessly with [Flax](https://github.com/google/flax?tab=readme-ov-file) for building modular and flexible neural networks, [Optax](https://github.com/google-deepmind/optax) for applying state-of-the-art gradient-based optimization algorithms, and [Orbax](https://github.com/google/orbax) for efficient checkpointing and serialization, ensuring scalability and ease of use in large-scale training processes.
- Easily implement the weak form of desired PDEs in Python; the framework handles the rest for highly efficient neural operator learning, PINNs, and finite element simulations.
- FOL utilizes the [mesh_io](https://github.com/nschloe/meshio) library for seamless mesh and result file handling, and [PETSc4py](https://petsc.org/release/) for efficiently solving large-scale linear systems.

# Installation Guide
To install FOL, follow these steps:
   ```sh
   git clone https://github.com/RezaNajian/FOL.git
   cd FOL
   python3 setup.py sdist bdist_wheel
   pip install -e .
   ```
To run the tests:
   ```sh
   pytest -s tests/
   ```
To run the examples:
   ```sh
   cd examples
   python3 examples_runner.py
   ```
# How to cite FOL?
Please, use the following references when citing FOL in your work.
- [Shahed Rezaei, Reza Najian Asl, Kianoosh Taghikhani, Ahmad Moeineddin, Michael Kaliske, and Markus Apel. "Finite Operator Learning: Bridging Neural Operators and Numerical Methods for Efficient Parametric Solution and Optimization of PDEs." arXiv preprint arXiv:2407.04157 (2024).](https://arxiv.org/pdf/2407.04157)
- [Shahed Rezaei, Reza Najian Asl, Shirko Faroughi, Mahdi Asgharzadeh, Ali Harandi, Rasoul Najafi Koopas, Gottfried Laschet, Stefanie Reese, Markus Apel. "A finite operator learning technique for mapping the elastic properties of microstructures to their mechanical deformations." arXiv preprint arXiv:2404.00074 (2024)](https://arxiv.org/pdf/2404.00074)
