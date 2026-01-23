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
