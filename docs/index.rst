
Folax
=====
.. div:: sd-text-left sd-font-italic

   **F**\ inite **O**\ perator **L**\ earning (FOL) with **JAX**


----

The **Folax** library is a unified Python framework for **solving partial
differential equations (PDEs)** and for **learning surrogate operators**
arising from them. It is designed at the intersection of
**computational mechanics** and **modern operator learning**, enabling both classical numerical solutions
and data-driven or physics-informed surrogate modeling within a single,
coherent abstraction.

At its core, FOL is built around the **weighted residual formulation**
of PDEs. The same mathematical structure used to derive finite element
residuals and tangent operators is reused to construct
**label-free, unsupervised surrogate models**. As a result, FOL supports
direct PDE solution and physics-informed operator learning without
duplicating model logic or training pipelines.

Weighted residual viewpoint
---------------------------

Given a PDE operator :math:`\mathcal{N}(u, p) = 0`, Folax adopts a weighted
residual formulation of the form

.. math::

   R(u, p; v) = \int_\Omega v \, \mathcal{N}(u, p)\, d\Omega = 0

where :math:`u` is the solution field, :math:`p` denotes parameters or control
variables, and :math:`v` is a test (weighting) function.

By appropriate choices of the test function :math:`v`, this single formulation
recovers several classical and modern approaches:

- **Finite Element Method (FEM)**
  Choosing the test function from the same space as the trial solution
  (Galerkin choice, :math:`v = \phi_i`) yields the standard finite element weak
  form, from which global residual vectors and tangent (Jacobian) matrices are
  derived for Newton-based solvers.

- **Finite Volume Method (FVM)**
  Choosing the test function as piecewise-constant characteristic functions
  over control volumes leads to local integral balance laws, recovering
  finite-volume discretizations based on flux conservation.

- **Physics-informed operator learning**
  Approximating the solution by a neural field :math:`u_\theta` and using the
  weighted residual as a scalar functional yields a label-free, physics-based
  loss. The predicted unknown fields serve simultaneously as trial and test
  functions, and depending on the activation function, the neural field may
  satisfy continuity requirements such as :math:`C^0` or even :math:`C^\infty`.

Operator learning and surrogate modeling
----------------------------------------

Folax provides multiple operator-learning formulations built on top of the
same numerical foundations:

- **Explicit parametric operator learning**, where networks directly
  predict discretized fields.
- **Implicit parametric operator learning**, where coordinate-based neural
  fields represent the solution and are conditioned on parameters.
- **DeepONet-based operator learning**, combining parametric conditioning
  with coordinate evaluation.
- **Fourier Neural Operator (FNO)-based learning**, operating on structured
  grids with resolution-invariant inference.
- **Meta-learning extensions**, including latent inner-loop adaptation and
  learnable update rules for fast generalization across parameter regimes.

All formulations support both **data-driven** and
**physics-informed** training, with consistent handling of Dirichlet
boundary conditions and discretized fields.

High-performance Python implementation
--------------------------------------

Folax is implemented entirely in **Python** and leverages **JAX**, **Flax**,
and **Optax** for high-performance execution. All core operations,
including matrix–vector products, residual evaluations, and gradient
computations, are JIT-compiled and **GPU/TPU accelerated** when available.

This design allows FOL to scale from classical FEM simulations to
large-scale operator-learning workloads while maintaining a clean,
research-friendly API.

Scope and intent
----------------

FoLax is intended for researchers and practitioners who want to:

- Solve PDEs using **classical finite element methods** through a fully
  **Python-based**, **GPU-accelerated** implementation, where users only need
  to define element-level residuals.
- Build **physics-informed or data-driven operator surrogates** for parametric
  PDEs by combining **computational mechanics** and **machine learning** within
  a principled and extensible framework.
- Perform **sensitivity analysis and gradient-based optimization** using response
  functionals evaluated on state and control fields, with automatic
  differentiation support for state, control, and shape variables, and optional
  adjoint-based gradients for large-scale problems.

By unifying numerical discretization, physics-based modeling, and modern
deep learning, FoLax provides a foundation for both **direct simulation** and
**next-generation surrogate modeling** in scientific computing.

----

.. toctree::
   :hidden:
   :maxdepth: 3

   installation
   api_reference/index