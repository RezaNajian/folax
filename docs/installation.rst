Installation
^^^^^^^^^^^^

Folax can be installed using ``pip``. Choose the installation variant
that best matches your intended use case and available hardware.

CPU Installation
----------------

The CPU-only installation is recommended for **small- to medium-scale
problems**, rapid prototyping, and for becoming familiar with the
Folax API without requiring accelerator hardware.

.. code-block:: bash

   pip install folax[cpu]

GPU (CUDA) Installation
-----------------------

The CUDA-enabled installation provides **GPU acceleration** and is
intended for **high-performance workloads**, large-scale simulations,
and operator-learning experiments.

.. code-block:: bash

   pip install folax[cuda]

Developer Installation
----------------------

For development, experimentation with the source code, or contributing
to Folax, clone the repository and install the package in editable mode.
From the project root directory, run:

.. code-block:: bash

   pip install -e .[cuda,dev]
