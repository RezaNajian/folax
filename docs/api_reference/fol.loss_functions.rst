``fol.loss_functions``
======================

Loss functions provided by FoLax.

.. automodule:: fol.loss_functions
.. currentmodule:: fol.loss_functions

Finite element loss base class
------------------------------

.. automodule:: fol.loss_functions.fe_loss
.. currentmodule:: fol.loss_functions.fe_loss

.. autoclass:: FiniteElementLoss
   :members:
   :show-inheritance:
   :exclude-members: Finalize

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

Saint Venant–Kirchhoff mechanical loss functions
------------------------------------------------

.. automodule:: fol.loss_functions.mechanical_saint_venant
.. currentmodule:: fol.loss_functions.mechanical_saint_venant

.. autoclass:: SaintVenantMechanicalLoss
   :members:
   :show-inheritance:
   :exclude-members: Initialize

.. autoclass:: SaintVenantMechanicalLoss2DQuad
   :members:
   :show-inheritance:

.. autoclass:: SaintVenantMechanicalLoss2DTri
   :members:
   :show-inheritance:

.. autoclass:: SaintVenantMechanicalLoss3DHexa
   :members:
   :show-inheritance:

.. autoclass:: SaintVenantMechanicalLoss3DTetra
   :members:
   :show-inheritance:

Neo-Hookean mechanical loss functions
-------------------------------------

.. automodule:: fol.loss_functions.mechanical_neohooke
.. currentmodule:: fol.loss_functions.mechanical_neohooke

.. autoclass:: NeoHookeMechanicalLoss
   :members:
   :show-inheritance:
   :exclude-members: Initialize

.. autoclass:: NeoHookeMechanicalLoss2DQuad
   :members:
   :show-inheritance:

.. autoclass:: NeoHookeMechanicalLoss2DTri
   :members:
   :show-inheritance:

.. autoclass:: NeoHookeMechanicalLoss3DHexa
   :members:
   :show-inheritance:

.. autoclass:: NeoHookeMechanicalLoss3DTetra
   :members:
   :show-inheritance:

Elastoplasticity loss functions
-------------------------------

.. automodule:: fol.loss_functions.mechanical_elastoplasticity
.. currentmodule:: fol.loss_functions.mechanical_elastoplasticity

.. autoclass:: ElastoplasticityLoss
   :members:
   :show-inheritance:
   :exclude-members: Initialize

.. autoclass:: ElastoplasticityLoss2DQuad
   :members:
   :show-inheritance:

.. autoclass:: ElastoplasticityLoss3DTetra
   :members:
   :show-inheritance:

Kratos small-displacement mechanical loss functions
---------------------------------------------------

.. automodule:: fol.loss_functions.kratos_small_displacement
.. currentmodule:: fol.loss_functions.kratos_small_displacement

.. autoclass:: KratosSmallDisplacement3DTetra
   :members:
   :show-inheritance:
   :exclude-members: ComputeElement

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

Transient thermal loss functions
--------------------------------

.. automodule:: fol.loss_functions.transient_thermal
.. currentmodule:: fol.loss_functions.transient_thermal

.. autoclass:: TransientThermalLoss
   :members:
   :show-inheritance:

.. autoclass:: TransientThermalLoss2DQuad
   :members:
   :show-inheritance:

.. autoclass:: TransientThermalLoss2DTri
   :members:
   :show-inheritance:

.. autoclass:: TransientThermalLoss3DHexa
   :members:
   :show-inheritance:

.. autoclass:: TransientThermalLoss3DTetra
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

Thermo-mechanical loss functions
-------------------------------------

.. automodule:: fol.loss_functions.thermo_mechanics
.. currentmodule:: fol.loss_functions.thermo_mechanics

.. autoclass:: ThermoMechanicsLoss
   :members:
   :show-inheritance:

.. autoclass:: ThermoMechanicsLoss2DQuad
   :members:
   :show-inheritance:

.. autoclass:: ThermoMechanicsLoss2DTri
   :members:
   :show-inheritance:

.. autoclass:: ThermoMechanicsLoss3DHexa
   :members:
   :show-inheritance:

.. autoclass:: ThermoMechanicsLoss3DTetra
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
