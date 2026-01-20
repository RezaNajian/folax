``fol.responses``
=================

Sensitivity analysis and response evaluation provided by FoLax.

This module contains response objects that can be evaluated on top of a finite
element (FE) state and a corresponding control field. A response typically
represents a scalar objective or constraint functional of the form

.. math::

   J(d, u) = \sum_{e=1}^{N_e} \int_{\Omega_e} \phi(d(x), u(x)) \, \mathrm{d}\Omega ,

where ``d`` denotes the control field, ``u`` denotes the FE solution (DOFs),
and the integrals are evaluated numerically using Gauss quadrature.

In addition to evaluating the scalar response value, FoLax responses support
sensitivity analysis with respect to:

- State variables (DOFs), via automatic differentiation.
- Control variables, via automatic differentiation.
- Shape variables (nodal coordinates), via automatic differentiation.
- Adjoint-based gradients for large-scale problems where direct differentiation
  through the solver is expensive.

.. automodule:: fol.responses
.. currentmodule:: fol.responses


Finite element response class
-----------------------------

.. automodule:: fol.responses.fe_response
.. currentmodule:: fol.responses.fe_response

.. autoclass:: FiniteElementResponse
   :members:
   :show-inheritance:


Notes
-----

- Element-wise response evaluation is performed using the FE element available
  in the associated :class:`fol.loss_functions.fe_loss.FiniteElementLoss`
  instance. The response integrand is evaluated at Gauss points and summed over
  all elements.

- Adjoint sensitivities are computed by building the adjoint right-hand side
  from the response derivative with respect to the state, and by using the
  transpose of the FE Jacobian provided by the loss function.

- Finite-difference utilities are provided for verification of both control and
  shape sensitivities. Forward difference (FWD) and central difference (CD) are
  supported.

- The response formula is provided as a string and compiled into a JAX-jitted
  function during initialization. The formula is evaluated at Gauss points using
  the interpolated control value and interpolated DOF values.
