``fol.solvers``
===============

Solve strategies provided by FoLax.

.. automodule:: fol.solvers
.. currentmodule:: fol.solvers

Finite element linear solver class
----------------------------------

.. automodule:: fol.solvers.fe_solver
.. currentmodule:: fol.solvers.fe_solver

.. autoclass:: FiniteElementSolver
   :members:
   :show-inheritance:
   :exclude-members: Finalize

Adjoint-based Finite Element solver
-----------------------------------

.. automodule:: fol.solvers.adjoint_fe_solver
.. currentmodule:: fol.solvers.adjoint_fe_solver

.. autoclass:: AdjointFiniteElementSolver
   :members:
   :show-inheritance:

Linear residual-based Finite Element solver
-------------------------------------------

.. automodule:: fol.solvers.fe_linear_residual_based_solver
.. currentmodule:: fol.solvers.fe_linear_residual_based_solver

.. autoclass:: FiniteElementLinearResidualBasedSolver
   :members:
   :show-inheritance:

Nonlinear finite element solver with incremental Newton–Raphson iterations and Gauss-point state variable updates
-----------------------------------------------------------------------------------------------------------------

.. automodule:: fol.solvers.fe_nonlinear_residual_based_solver_with_history_update
.. currentmodule:: fol.solvers.fe_nonlinear_residual_based_solver_with_history_update

.. autoclass:: FiniteElementNonLinearResidualBasedSolverWithStateUpdate
   :members:
   :show-inheritance:


Nonlinear finite element solver with incremental Newton–Raphson iterations
--------------------------------------------------------------------------

.. automodule:: fol.solvers.fe_nonlinear_residual_based_solver
.. currentmodule:: fol.solvers.fe_nonlinear_residual_based_solver

.. autoclass:: FiniteElementNonLinearResidualBasedSolver
   :members:
   :show-inheritance: