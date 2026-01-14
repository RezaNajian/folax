``fol.loss_functions`` module
=============================

Loss functions provided by FoLax.

.. automodule:: fol.loss_functions
.. currentmodule:: fol.loss_functions

Linear mechanical loss functions
--------------------------------

.. automodule:: fol.loss_functions.mechanical
.. currentmodule:: fol.loss_functions.mechanical

.. autoclass:: MechanicalLoss
   :members:
   :show-inheritance:

.. autoclass:: MechanicalLoss2DQuad
   :members:
   :show-inheritance:

.. autoclass:: MechanicalLoss2DTri
   :members:
   :show-inheritance:

.. autoclass:: MechanicalLoss3DHexa
   :members:
   :show-inheritance:

.. autoclass:: MechanicalLoss3DTetra
   :members:
   :show-inheritance:

Nonlinear thermal loss functions
--------------------------------

.. automodule:: fol.loss_functions.thermal
.. currentmodule:: fol.loss_functions.thermal

.. autoclass:: ThermalLoss
   :members:
   :show-inheritance:

.. autoclass:: ThermalLoss2DQuad
   :members:
   :show-inheritance:

.. autoclass:: ThermalLoss2DTri
   :members:
   :show-inheritance:

.. autoclass:: ThermalLoss3DHexa
   :members:
   :show-inheritance:

.. autoclass:: ThermalLoss3DTetra
   :members:
   :show-inheritance:

Allen–Cahn phase-field loss functions
-------------------------------------

.. automodule:: fol.loss_functions.phase_field
.. currentmodule:: fol.loss_functions.phase_field

.. autoclass:: AllenCahnLoss
   :members:
   :show-inheritance:

.. autoclass:: AllenCahnLoss2DQuad
   :members:
   :show-inheritance:

.. autoclass:: AllenCahnLoss2DTri
   :members:
   :show-inheritance:

.. autoclass:: AllenCahnLoss3DHexa
   :members:
   :show-inheritance:

Regression loss function
-------------------------

.. automodule:: fol.loss_functions.regression_loss
.. currentmodule:: fol.loss_functions.regression_loss

.. autoclass:: RegressionLoss
   :members:
   :exclude-members: GetNumberOfUnknowns, Finalize, GetFullDofVector
   :show-inheritance: