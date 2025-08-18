<p align=center><img height="64.125%" width="64.125%" src="https://github.com/RezaNajian/FOL/assets/62375973/0e1ca4e0-0658-4f5d-aad9-1ae7c9f67574"></p>

[![License][license-image]][license] [![CI](https://github.com/RezaNajian/FOL/actions/workflows/ci.yml/badge.svg)](https://github.com/RezaNajian/FOL/actions/workflows/ci.yml)

[license-image]: https://img.shields.io/badge/license-BSD-green.svg?style=flat
[license]: https://github.com/RezaNajian/FOL/LICENSE

# FOL: Solution and Optimization of parameterized PDEs
**F**inite **O**perator **L**earning constitutes a unified numerical framework that seamlessly integrates established numerical methods with advanced scientific machine learning techniques for solving and optimizing parametrized partial differential equations (PDEs).  In constructing a physics-informed operator learning approach, FOL formulates a purely physics-based loss function derived from the Method of Weighted Residuals, allowing discrete residuals—computed using classical PDE solution techniques—to be directly incorporated into backpropagation during network training. This approach ensures that the learned operators rigorously satisfy the underlying governing equations while maintaining consistency with established numerical discretizations. Importantly, this loss formulation is agnostic to the network architecture and has been successfully applied to architectures such as Conditional Neural Fields, Fourier Neural Operators (FNO), and DeepONets. 
FOL includes the following algorithms:
- physics-informed operator learning:
  - A Physics-Informed Meta-Learning Framework for the Continuous Solution of Parametric PDEs on Arbitrary Geometries [[arXiv](https://arxiv.org/abs/2504.02459)].
  - Finite Operator Learning: Bridging Neural Operators and Numerical Methods for Efficient Parametric Solution and Optimization of PDEs [[arXiv](https://arxiv.org/abs/2407.04157)].
  - Digitalizing metallic materials from image segmentation to multiscale solutions via physics informed operator learning [[npj Computational Materials](https://www.nature.com/articles/s41524-025-01718-y)].
  - A Finite Operator Learning Technique for Mapping the Elastic Properties of Microstructures to Their Mechanical Deformations [[Numerical Methods in Eng.](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7637)].

We built upon several widely adopted Python packages, including [JAX](https://github.com/jax-ml/jax) for high-performance array computations on CPUs and GPUs, [PETSc](https://petsc.org/release/) for the efficient solution of large-scale linear systems, [Metis](https://github.com/KarypisLab/METIS) for mesh partitioning (integration forthcoming), [Flax](https://github.com/google/flax?tab=readme-ov-file) for constructing modular and flexible neural networks, [Optax](https://github.com/google-deepmind/optax) for applying state-of-the-art gradient-based optimization algorithms, and [Orbax](https://github.com/google/orbax) for efficient checkpointing and serialization. This foundation ensures scalability, computational efficiency, and ease of use in large-scale training and simulation workflows.
