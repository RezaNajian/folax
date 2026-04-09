"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: April, 2026
 License: FOL/LICENSE

 Unit tests for ComputeResidualNorm added to FiniteElementLoss (fe_loss.py).
 Verifies:
   - scalar output for single sample
   - RMS vs L2 branch (relative=True/False)
   - value decreases as prediction improves toward FE solution
   - vmap compatibility across a batch
"""
import pytest
import unittest
import numpy as np
import jax
import jax.numpy as jnp

from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.controls.fourier_control import FourierControl
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.usefull_functions import create_2D_square_mesh, create_random_fourier_samples


class TestComputeResidualNorm(unittest.TestCase):
    """Unit tests for fe_loss.ComputeResidualNorm."""

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self, request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        self.fe_mesh = create_2D_square_mesh(L=1, N=6)

        bc_dict = {"Ux": {"left": 0.0, "right": 0.1},
                   "Uy": {"left": 0.0, "right": 0.0}}
        material_dict = {"young_modulus": 1, "poisson_ratio": 0.3}

        self.loss = NeoHookeMechanicalLoss2DQuad(
            "mech_loss_2d",
            loss_settings={"dirichlet_bc_dict": bc_dict,
                           "num_gp": 2,
                           "material_dict": material_dict},
            fe_mesh=self.fe_mesh,
        )

        fourier_settings = {
            "x_freqs": np.array([1, 2, 3]),
            "y_freqs": np.array([1, 2, 3]),
            "z_freqs": np.array([0]),
            "beta": 2,
        }
        self.control = FourierControl("fourier_ctrl", fourier_settings, self.fe_mesh)

        fe_settings = {
            "linear_solver_settings": {"solver": "PETSc-bcgsl"},
            "nonlinear_solver_settings": {"rel_tol": 1e-8, "abs_tol": 1e-8,
                                          "maxiter": 10, "load_incr": 5},
        }
        self.solver = FiniteElementNonLinearResidualBasedSolver(
            "fe_solver", self.loss, fe_settings
        )

        self.fe_mesh.Initialize()
        self.loss.Initialize()
        self.control.Initialize()
        self.solver.Initialize()

        # Generate one sample and solve to get reference DOFs
        self.coeffs, self.K = create_random_fourier_samples(self.control, 0)
        self.ctrl_vec = self.K[-1, :]
        self.coeff_vec = self.coeffs[-1, :]
        # Solver expects full DOF vector (2 * n_nodes for 2D mechanics)
        total_dofs = 2 * self.fe_mesh.GetNumberOfNodes()
        self.fe_dofs = np.array(
            self.solver.Solve(self.ctrl_vec, np.zeros(total_dofs))
        )
        self.total_dofs = total_dofs

    # Helper: ComputeResidualNorm is designed to be called inside jax.vmap.
    # Wrap single-sample calls in a batch-1 vmap to satisfy tracing requirements.
    def _residual_norm_single(self, ctrl, full_dof, relative=True):
        """Call ComputeResidualNorm via vmap (batch=1), return scalar."""
        return jax.vmap(
            lambda c, f: self.loss.ComputeResidualNorm(c, f, relative=relative)
        )(ctrl[jnp.newaxis, :], full_dof[jnp.newaxis, :])[0]

    # ------------------------------------------------------------------
    # Test 1: return type and shape
    # ------------------------------------------------------------------
    def test_returns_scalar(self):
        full_dof = self.loss.GetFullDofVector(
            self.ctrl_vec[jnp.newaxis, :],
            jnp.array(self.fe_dofs)[jnp.newaxis, :],
        )[0]
        rms = self._residual_norm_single(self.ctrl_vec, full_dof, relative=True)
        self.assertEqual(rms.shape, ())  # scalar
        self.assertTrue(jnp.isfinite(rms))

    # ------------------------------------------------------------------
    # Test 2: RMS < L2  (by definition for vectors with > 1 entry)
    # ------------------------------------------------------------------
    def test_rms_less_than_l2(self):
        full_dof = self.loss.GetFullDofVector(
            self.ctrl_vec[jnp.newaxis, :],
            jnp.array(self.fe_dofs)[jnp.newaxis, :],
        )[0]
        rms = float(self._residual_norm_single(self.ctrl_vec, full_dof, relative=True))
        l2 = float(self._residual_norm_single(self.ctrl_vec, full_dof, relative=False))
        # RMS = L2 / sqrt(n), so RMS <= L2 for n >= 1
        self.assertLessEqual(rms, l2 + 1e-15)

    # ------------------------------------------------------------------
    # Test 3: residual at FE solution is small, at zero DOFs is large
    # ------------------------------------------------------------------
    def test_residual_decreases_toward_solution(self):
        # Bad prediction (zeros) — GetFullDofVector expects (batch, total_dofs)
        full_bad = self.loss.GetFullDofVector(
            self.ctrl_vec[jnp.newaxis, :],
            jnp.zeros((1, self.total_dofs)),
        )[0]
        res_bad = float(self._residual_norm_single(self.ctrl_vec, full_bad, relative=True))

        # Good prediction (converged FE)
        full_good = self.loss.GetFullDofVector(
            self.ctrl_vec[jnp.newaxis, :],
            jnp.array(self.fe_dofs)[jnp.newaxis, :],
        )[0]
        res_good = float(self._residual_norm_single(self.ctrl_vec, full_good, relative=True))

        self.assertLess(res_good, res_bad,
                        "Residual at converged FE solution should be much smaller than at zero DOFs")

    # ------------------------------------------------------------------
    # Test 4: vmap over a batch
    # ------------------------------------------------------------------
    def test_vmap_batch(self):
        batch_size = 3
        ctrl_batch = jnp.tile(self.ctrl_vec, (batch_size, 1))
        dofs_batch = jnp.tile(jnp.array(self.fe_dofs), (batch_size, 1))

        full_batch = jax.vmap(
            lambda c, d: self.loss.GetFullDofVector(c[jnp.newaxis, :], d[jnp.newaxis, :])[0]
        )(ctrl_batch, dofs_batch)

        residuals = jax.vmap(
            lambda c, f: self.loss.ComputeResidualNorm(c, f, relative=True)
        )(ctrl_batch, full_batch)

        self.assertEqual(residuals.shape, (batch_size,))
        # All identical inputs → identical residuals
        np.testing.assert_allclose(
            np.array(residuals),
            np.full(batch_size, float(residuals[0])),
            rtol=1e-5,
        )


if __name__ == '__main__':
    unittest.main()
