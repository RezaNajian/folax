``fol.deep_neural_networks``
============================

The ``fol.deep_neural_networks`` module provides a comprehensive collection of deep learning
architectures and operator-learning frameworks designed for **learning mappings between
function spaces**, with a strong focus on **physics-informed modeling**, **parametric systems**,
and **scientific machine learning**.

This module implements state-of-the-art neural operators such as **DeepONets** and
**Fourier Neural Operators (FNOs)**, together with **Conditional Neural Fields** and
meta-learning strategies. The provided models enable the approximation of complex
solution operators and continuous fields arising in **partial differential equations (PDEs)**,
**multiphysics simulations**, and **high-dimensional parametric problems**, where classical
surrogate models are often insufficient.

Key capabilities of this module include:

- Learning **nonlinear operators** mapping input functions, boundary conditions, or parameters
  to solution fields.
- Supporting **data-driven**, **physics-informed**, and **hybrid** training paradigms.
- Handling **explicit** and **implicit** parametric operator learning formulations.
- Neural operators based on **DeepONets** and **Fourier Neural Operators** for learning
  mappings between infinite-dimensional spaces.
- **Neural field representations** using coordinate-based MLPs, including **SIREN-style**
  sinusoidal networks for high-frequency signal and geometry representation.
- **Conditional neural fields** that model families of functions conditioned on parameters,
  latent variables, or context embeddings.
- **Autoencoding architectures** for learning compact latent representations of solution
  manifolds and parametric fields.
- **Meta-learning and latent-variable methods** for fast adaptation across tasks, parameters,
  and operating regimes.
- Efficient spectral representations and convolutions for scalable, high-dimensional problems.

The module is designed to be extensible and modular, enabling seamless integration
into scientific computing pipelines while leveraging modern deep learning frameworks.

.. automodule:: fol.deep_neural_networks
.. currentmodule:: fol.deep_neural_networks

Deep network base class
-----------------------

.. automodule:: fol.deep_neural_networks.deep_network
.. currentmodule:: fol.deep_neural_networks.deep_network

.. autoclass:: DeepNetwork
   :members:
   :show-inheritance:

Explicit parametric operator learning
-------------------------------------

This submodule implements **explicit parametric operator learning** on discretized
fields, where a fixed-dimensional parametric input space (for example control
variables or Fourier coefficients) is mapped directly to a fixed-dimensional
discretized field such as temperature or displacement.

The learning is **unsupervised or physics-informed**: no direct target fields are
required. Instead, predicted fields are evaluated using physics-based loss
functions (for example weighted residual- or energy-based formulations), with boundary
conditions applied explicitly at the field level.

.. automodule:: fol.deep_neural_networks.explicit_parametric_operator_learning
   :members:
   :undoc-members:
   :show-inheritance:

.. currentmodule:: fol.deep_neural_networks.explicit_parametric_operator_learning

Implicit parametric operator learning
-------------------------------------

This submodule implements **implicit parametric operator learning** using
coordinate-based neural fields. A fixed-dimensional parametric input (for example
control variables or parameterization features such as Fourier coefficients)
conditions a neural field represented by a coordinate-based MLP (the synthesizer),
with conditioning provided by a modulator network according to the coupling modes
implemented by :class:`fol.deep_neural_networks.nns.HyperNetwork`.

The learning is **unsupervised or physics-informed**: predicted discretized fields
are evaluated using physics-based loss functionals (for example residual- or
energy-based formulations), and Dirichlet boundary conditions are enforced by
explicitly inserting prescribed values across the batch.

Although training is typically performed on a fixed FE mesh, the coordinate-based
synthesizer enables multi-resolution inference (and, in principle, multi-resolution
training) by evaluating the conditioned neural field on alternative coordinate sets.

.. automodule:: fol.deep_neural_networks.implicit_parametric_operator_learning
   :members:
   :undoc-members:
   :show-inheritance:

.. currentmodule:: fol.deep_neural_networks.implicit_parametric_operator_learning

Meta-implicit parametric operator learning
------------------------------------------

This submodule implements **meta-implicit parametric operator learning**, which
extends implicit parametric operator learning by introducing **per-sample latent
adaptation**. Instead of directly conditioning the neural field with parametric
inputs, latent variables are optimized in an inner loop to minimize the
physics-based loss, enabling fast adaptation without updating the main network
weights.

The neural field is represented by a coordinate-based synthesizer MLP and is
conditioned through a modulator network using the coupling modes implemented by
:class:`fol.deep_neural_networks.nns.HyperNetwork`. Training remains
**unsupervised or physics-informed**, with explicit boundary-condition
enforcement and support for multi-resolution inference.

.. automodule:: fol.deep_neural_networks.meta_implicit_parametric_operator_learning
   :members:
   :undoc-members:
   :show-inheritance:

.. currentmodule:: fol.deep_neural_networks.meta_implicit_parametric_operator_learning

Meta-alpha-meta implicit parametric operator learning
-----------------------------------------------------

This submodule implements **meta-alpha-meta implicit parametric operator learning**,
which further extends meta-implicit parametric operator learning by introducing a
**learnable latent-step size** in the inner-loop adaptation. As in the meta-implicit
formulation, latent variables are optimized per sample to minimize the
physics-based loss, but in this variant the magnitude of the latent update itself
is learned jointly with the network parameters.

The neural field is represented by a coordinate-based synthesizer MLP and is
conditioned through a modulator network using the coupling modes implemented by
:class:`fol.deep_neural_networks.nns.HyperNetwork`. The latent codes are adapted
using gradient-based updates, while a dedicated trainable step model controls the
latent update size, enabling improved robustness and adaptability across problem
instances.

Training remains **unsupervised or physics-informed**, with explicit enforcement
of boundary conditions and preservation of the coordinate-based formulation, which
allows multi-resolution inference by evaluating the conditioned neural field on
alternative coordinate sets.

.. automodule:: fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning
   :members:
   :undoc-members:
   :show-inheritance:

.. currentmodule:: fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning

Fourier parametric operator learning
------------------------------------

This submodule implements **Fourier parametric operator learning** using a
**Fourier Neural Operator (FNO)** on discretized fields. A fixed-dimensional
parametric input space (for example control variables or parameterization
features) is mapped to grid-aligned input channels and processed by the FNO to
produce discretized field outputs such as temperature or displacement.

The FNO is **not bound to a specific mesh resolution**. Once trained, the learned
operator can be evaluated on different grid resolutions, as long as the mesh is
structured and uniform (for example square grids in 2D or cubic grids in 3D).
This enables resolution-invariant inference across compatible discretizations.

The learning can be **data-driven** or **physics-informed**, depending on the
chosen loss function, with boundary conditions enforced explicitly through the
loss.

.. automodule:: fol.deep_neural_networks.fourier_parametric_operator_learning
   :members:
   :undoc-members:
   :show-inheritance:

.. currentmodule:: fol.deep_neural_networks.fourier_parametric_operator_learning

DeepONet parametric operator learning
-------------------------------------

This submodule implements **DeepONet-based parametric operator learning** on
discretized fields. A fixed-dimensional parametric input space conditions a
DeepONet that is evaluated on FE mesh node coordinates to produce discretized
field outputs such as temperature or displacement.

The learning can be **data-driven** or **physics-informed**, depending on the
chosen loss function, with boundary conditions enforced explicitly through the
loss and inference utilities.

.. automodule:: fol.deep_neural_networks.deep_o_net_parametric_operator_learning
   :members:
   :undoc-members:
   :show-inheritance:

.. currentmodule:: fol.deep_neural_networks.deep_o_net_parametric_operator_learning

Neural fields and hypernetworks
-------------------------------

This module provides building blocks for neural field models (e.g., SIREN-style
MLPs and Fourier-feature MLPs) and DeepONets, and hypernetworks used to modulate or generate
parameters for coordinate-based models. These components are commonly used in
implicit neural representations, conditional neural fields, and meta-learning
workflows.

.. automodule:: fol.deep_neural_networks.nns
   :members:
   :undoc-members:
   :show-inheritance:

.. currentmodule:: fol.deep_neural_networks.nns

