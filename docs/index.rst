
Folax
=====
.. div:: sd-text-left sd-font-italic

   **F**\ inite **O**\ perator **L**\ earning (FOL) with **JAX**

.. image:: https://img.shields.io/badge/Python-3.10%2B-blue
   :alt: Python version

.. image:: https://img.shields.io/badge/JAX-accelerated-orange
   :alt: JAX accelerated

.. image:: https://img.shields.io/badge/GPU%2FTPU-supported-green
   :alt: GPU TPU support

.. image:: https://img.shields.io/badge/License-MIT-yellow
   :alt: License: MIT

----

🚀 What is Folax?
-----------------

**Folax** is a high-performance Python library designed and developed for
the **solution**, **optimization**, and **surrogate modeling** of
**parametrized partial differential equations (PDEs)**.

It provides a unified computational framework that seamlessly integrates
**classical numerical methods** (e.g. finite element and finite volume
methods) with **modern operator-learning architectures**, including
physics-informed and data-driven neural operators. This unified,
mathematically principled abstraction enables smooth transitions from
**direct PDE simulation** to **learned surrogate operators** without
duplicating physical models or numerical formulations.

Folax is intended for researchers and practitioners who require
**numerical rigor**, **computational efficiency**, and **conceptual
clarity**, supporting end-to-end workflows that span simulation,
sensitivity analysis, and optimization within a single, coherent
software ecosystem.

----

🧩 Core Idea: Weighted Residual Formulation
-------------------------------------------

At the core of Folax lies the **weighted residual formulation** of
parametrized partial differential equations (PDEs), which serves as a
unifying mathematical foundation for both **numerical discretization**
and **operator learning**.

Given a PDE operator

.. math::

   \mathcal{N}(u, p) = 0,

Folax expresses the governing equations through the residual functional

.. math::

   R(u, p; v) = \int_\Omega v \, \mathcal{N}(u, p)\, \mathrm{d}\Omega = 0,

where :math:`u` denotes the solution field, :math:`p` represents
parameters or control variables, and :math:`v` is a test (weighting)
function. The choice of the test function defines the numerical or
learning paradigm, while preserving a common residual-based structure.

By appropriate selection of :math:`v`, this single formulation recovers
a broad class of established and modern methods:

- **Finite Element Method (FEM)**
  Choosing the test function from the same space as the trial solution
  (Galerkin choice, :math:`v = \phi_i`) yields the classical finite-element
  weak form. This formulation enables the automatic construction of
  global residual vectors and tangent (Jacobian) operators, supporting
  Newton-based nonlinear solvers.

- **Finite Volume Method (FVM)**
  Selecting piecewise-constant test functions over control volumes
  results in local integral balance laws, recovering conservative
  finite-volume discretizations based on flux conservation.

- **Physics-Informed Operator Learning**
  Approximating the solution using neural fields :math:`u_\theta` and
  interpreting the weighted residual as a scalar functional yields
  label-free, physics-based loss functions. In this setting, the neural
  field acts simultaneously as trial and test function, and depending
  on the activation functions employed, may satisfy continuity
  requirements up to :math:`C^\infty`.

All approaches are derived from the same residual formulation, allowing
Folax to support classical solvers and learning-based surrogates without
duplicating physical models, numerical logic, or training pipelines.

----

🧠 Operator Learning Capabilities
---------------------------------

Folax supports multiple operator-learning paradigms:

- **Explicit parametric operators**
  Neural networks predict discretized solution fields directly

- **Implicit neural fields**
  Coordinate-based representations conditioned on parameters

- **DeepONet architectures**
  Branch–trunk operator learning with spatial evaluation

- **Fourier Neural Operators (FNOs)**
  Resolution-invariant learning on structured grids

- **Meta-learning extensions**
  Fast adaptation across parameter regimes using learnable inner loops

All formulations support both **data-driven** and
**physics-informed** training, with consistent handling of boundary
conditions and discretized fields.

----

⚡ Performance by Design
------------------------

Folax is implemented entirely in **Python** and leverages:

- **JAX** for composable automatic differentiation and JIT compilation
- **Flax** for flexible neural network definitions
- **Optax** for scalable optimization

Key features include:

- End-to-end **JIT-compiled** execution
- Native **GPU / TPU acceleration**
- Differentiable residuals, solvers, and response functionals

This design allows Folax to scale from classical finite-element
simulations to large-scale operator-learning workloads.

----

🎯 Who Is Folax For?
--------------------

Folax is designed for:

- **Computational mechanics researchers**
  Building FEM solvers fully in Python with minimal boilerplate

- **Scientific machine learning practitioners**
  Training physics-informed or data-driven operator surrogates

- **Optimization and inverse-problem workflows**
  Gradient-based optimization with respect to:

  - state variables
  - control parameters
  - geometric or shape variables

Adjoint-based gradients are supported for large-scale problems.

----

🌱 Philosophy
-------------

**One formulation. One codebase. Many paradigms.**

By unifying numerical discretization, physics-based modeling, and
modern deep learning, **Folax** provides a foundation for
**next-generation scientific computing**, where simulation and learning
are no longer separate worlds.

----

.. toctree::
   :hidden:
   :maxdepth: 3

   installation
   quick_start
   api_reference/index
   3D_thermo_mechanics_example
