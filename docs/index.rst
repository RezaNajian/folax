
Folax
=====
.. div:: sd-text-left sd-font-italic

   **F**\ inite **O**\ perator **L**\ earning (FOL) with **JAX**


----

Folax constitutes a unified numerical framework that seamlessly integrates established numerical methods in computational mechanics with advanced scientific machine learning techniques for solving and optimizing parametrized partial differential equations (PDEs).

We built upon several widely adopted Python packages, including `JAX <https://github.com/jax-ml/jax>`__ for high-performance array computations on CPUs and GPUs, PETSc `PETSc <https://petsc.org/release/>`__ for the efficient solution of large-scale linear systems, `Metis <https://github.com/KarypisLab/METIS>`__ for mesh partitioning (integration forthcoming), `Flax <https://github.com/google/flax>`__ for constructing modular and flexible neural networks, Optax for applying state-of-the-art gradient-based optimization algorithms, and `Orbax <https://github.com/google/orbax>`__ for efficient checkpointing and serialization. This foundation ensures scalability, computational efficiency, and ease of use in large-scale training and simulation workflows.

Installation
^^^^^^^^^^^^

Install FoLax using ``pip`` based on your intended use case.

CPU installation
----------------
The CPU version is recommended for small-scale problems and for familiarizing
yourself with the FoLax API.

.. code-block:: bash

   pip install folax[cpu]

GPU (CUDA) installation
-----------------------
The CUDA version enables GPU acceleration and is intended for high-performance,
accelerated workloads and large-scale experiments.

.. code-block:: bash

   pip install folax[cuda]

Developer installation
----------------------
If you plan to develop FoLax, first clone the repository and then, from the
project root directory, run:

.. code-block:: bash

   pip install -e .[cuda,dev]

----

.. toctree::
   :hidden:
   :maxdepth: 3

   api_reference/index