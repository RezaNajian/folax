"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: April, 2026
 License: FOL/LICENSE

 Unit tests for the generalized convergence and metrics machinery in DeepNetwork:
   - CheckConvergence with arbitrary criterion keys
   - TrainStepMetrics / TestStepMetrics return full dict
   - _reduce_metric min/max/mean semantics (via integration with Train)
   - training_residual_tracker integration
   - backward compatibility: default total_loss criterion still works
"""
import pytest
import unittest
import os
import shutil
import csv
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fourier_parametric_operator_learning import (
    PhysicsInformedFourierParametricOperatorLearning,
)
from fol.tools.usefull_functions import create_2D_square_mesh, create_random_fourier_samples


# Minimal FNO-like model: just a convolutional passthrough for testing.
# Maps (B, N, N, C_in) -> (B, N, N, C_out) via two 1x1 convolutions.
class TinyConvNet(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_ch, 16, kernel_size=(1, 1), rngs=rngs,
                              kernel_init=nnx.initializers.zeros,
                              bias_init=nnx.initializers.zeros)
        self.conv2 = nnx.Conv(16, out_ch, kernel_size=(1, 1), rngs=rngs,
                              kernel_init=nnx.initializers.zeros,
                              bias_init=nnx.initializers.zeros)

    def __call__(self, x):
        x = self.conv1(x)
        x = jax.nn.swish(x)
        x = self.conv2(x)
        return x


def _build_pi_fno_2d(test_dir):
    """Helper: construct a minimal 2D PI-FNO problem (6×6 quad mesh)."""
    fe_mesh = create_2D_square_mesh(L=1, N=6)

    bc_dict = {"Ux": {"left": 0.0, "right": 0.1},
               "Uy": {"left": 0.0, "right": 0.0}}
    material_dict = {"young_modulus": 1, "poisson_ratio": 0.3}

    loss_fn = NeoHookeMechanicalLoss2DQuad(
        "mech_loss",
        loss_settings={"dirichlet_bc_dict": bc_dict,
                       "num_gp": 2,
                       "material_dict": material_dict},
        fe_mesh=fe_mesh,
    )

    fourier_settings = {
        "x_freqs": np.array([1, 2]),
        "y_freqs": np.array([1, 2]),
        "z_freqs": np.array([0]),
        "beta": 2,
    }
    control = FourierControl("ctrl", fourier_settings, fe_mesh)

    fe_mesh.Initialize()
    loss_fn.Initialize()
    control.Initialize()

    n_nodes = fe_mesh.GetNumberOfNodes()
    N = int(round(n_nodes ** 0.5))
    in_ch = control.GetNumberOfVariables() // n_nodes  # channels per node
    # Fallback: if control output is flat per node, in_ch ~ num_coeffs / n_nodes
    # Safe way: just compute in_ch from actual control output shape
    sample_ctrl = control.ComputeControlledVariables(np.zeros(control.GetNumberOfVariables()))
    in_ch = max(1, int(sample_ctrl.size // n_nodes))
    # out_ch must be number_dofs_per_node (2 for 2D mechanics) so that
    # NN output reshapes to (B, total_number_of_dofs)
    out_ch = 2  # 2D mechanics: Ux, Uy

    net = TinyConvNet(in_ch, out_ch, rngs=nnx.Rngs(42))
    optimizer = optax.adam(1e-3)

    pi_fno = PhysicsInformedFourierParametricOperatorLearning(
        name="test_pi_fno",
        control=control,
        loss_function=loss_fn,
        flax_neural_network=net,
        optax_optimizer=optimizer,
    )
    pi_fno.Initialize()

    coeffs, K = create_random_fourier_samples(control, 2)  # 3 samples total
    train_set = (coeffs,)

    return pi_fno, train_set, control, loss_fn


class TestCheckConvergenceGeneric(unittest.TestCase):
    """Test that CheckConvergence works with any dict key, not just total_loss."""

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self, request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "test_convergence_generic")
        os.makedirs(self.test_dir, exist_ok=True)
        self.pi_fno, _, _, _ = _build_pi_fno_2d(self.test_dir)

    def tearDown(self):
        if self.debug_mode == "false":
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_converges_on_total_loss(self):
        """Default criterion total_loss: converges when below threshold."""
        history = {"total_loss": [0.5, 0.01, 1e-9]}
        settings = {"convergence_criterion": "total_loss",
                    "absolute_error": 1e-6,
                    "relative_error": 1e-100,
                    "num_epochs": 500}
        self.assertTrue(self.pi_fno.CheckConvergence(history, settings))

    def test_converges_on_residual_rms_batch_mean(self):
        """Custom criterion residual_rms_batch_mean below threshold."""
        history = {
            "total_loss": [0.5, 0.1, 0.05],
            "residual_rms_batch_mean": [0.1, 0.01, 5e-5],
        }
        settings = {"convergence_criterion": "residual_rms_batch_mean",
                    "absolute_error": 1e-4,
                    "relative_error": 1e-100,
                    "num_epochs": 500}
        self.assertTrue(self.pi_fno.CheckConvergence(history, settings))

    def test_does_not_converge_above_threshold(self):
        """Custom criterion above threshold → not converged."""
        history = {
            "total_loss": [0.5, 0.1],
            "residual_rms_batch_mean": [0.1, 0.01],
        }
        settings = {"convergence_criterion": "residual_rms_batch_mean",
                    "absolute_error": 1e-4,
                    "relative_error": 1e-100,
                    "num_epochs": 500}
        self.assertFalse(self.pi_fno.CheckConvergence(history, settings))

    def test_relative_convergence_on_custom_key(self):
        """Relative change criterion fires on any key."""
        history = {
            "residual_rms_batch_mean": [0.05, 0.05 + 1e-12],
        }
        settings = {"convergence_criterion": "residual_rms_batch_mean",
                    "absolute_error": 1e-100,
                    "relative_error": 1e-8,
                    "num_epochs": 500}
        self.assertTrue(self.pi_fno.CheckConvergence(history, settings))

    def test_missing_key_raises(self):
        """Missing criterion key raises KeyError."""
        history = {"total_loss": [0.5]}
        settings = {"convergence_criterion": "nonexistent_key",
                    "absolute_error": 1e-4,
                    "relative_error": 1e-8,
                    "num_epochs": 500}
        with self.assertRaises(KeyError):
            self.pi_fno.CheckConvergence(history, settings)


class TestPIFNOMetricsDict(unittest.TestCase):
    """Test that PhysicsInformedFourierParametricOperatorLearning
    ComputeBatchLossValue returns all expected residual keys."""

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self, request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "test_pi_fno_metrics")
        os.makedirs(self.test_dir, exist_ok=True)
        self.pi_fno, self.train_set, _, _ = _build_pi_fno_2d(self.test_dir)

    def tearDown(self):
        if self.debug_mode == "false":
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_batch_loss_returns_residual_keys(self):
        """ComputeBatchLossValue must return total_loss + residual_rms_* keys."""
        batch = (jnp.array(self.train_set[0][:2]),
                 jnp.zeros((2, 1)))  # dummy Y
        loss_val, metrics = self.pi_fno.ComputeBatchLossValue(
            batch, self.pi_fno.flax_neural_network
        )
        required_keys = {"total_loss", "residual_rms_batch_mean",
                         "residual_rms_batch_min", "residual_rms_batch_max"}
        for key in required_keys:
            self.assertIn(key, metrics, f"Missing metric key: {key}")
            val = float(metrics[key])
            self.assertTrue(np.isfinite(val), f"{key} is not finite: {val}")

    def test_train_step_metrics_returns_dict(self):
        """TrainStepMetrics should return a dict, not a scalar."""
        batch = (jnp.array(self.train_set[0][:2]),
                 jnp.zeros((2, 1)))
        state = self.pi_fno.GetState()
        result = self.pi_fno.TrainStepMetrics(state, batch)
        self.assertIsInstance(result, dict)
        self.assertIn("total_loss", result)
        self.assertIn("residual_rms_batch_mean", result)

    def test_test_step_metrics_returns_dict(self):
        """TestStepMetrics should return a dict, not a scalar."""
        batch = (jnp.array(self.train_set[0][:2]),
                 jnp.zeros((2, 1)))
        state = self.pi_fno.GetState()
        result = self.pi_fno.TestStepMetrics(state, batch)
        self.assertIsInstance(result, dict)
        self.assertIn("total_loss", result)
        self.assertIn("residual_rms_batch_mean", result)


class TestTrainWithResidualConvergence(unittest.TestCase):
    """Integration: Train() with convergence_criterion='residual_rms_batch_mean'."""

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self, request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "test_train_residual_conv")
        os.makedirs(self.test_dir, exist_ok=True)
        self.pi_fno, self.train_set, _, _ = _build_pi_fno_2d(self.test_dir)

    def tearDown(self):
        if self.debug_mode == "false":
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_train_runs_with_residual_criterion(self):
        """Train() should complete without errors when using residual_rms_batch_mean."""
        self.pi_fno.Train(
            train_set=self.train_set,
            batch_size=len(self.train_set[0]),  # full-batch for stability
            convergence_settings={
                "num_epochs": 5,
                "convergence_criterion": "residual_rms_batch_mean",
                "relative_error": 1e-100,
                "absolute_error": 1e-100,  # won't actually converge
            },
            plot_settings={
                "plot_list": ["total_loss", "residual_rms_batch_mean"],
                "save_frequency": 100,
            },
            working_directory=self.test_dir,
        )
        # If we got here without error, the full pipeline works:
        # ComputeBatchLossValue → TrainStepMetrics → _reduce_metric → history → CheckConvergence

    def test_train_with_tracker(self):
        """Train() with training_residual_tracker writes CSV + plot."""
        # Import from examples — it's a standalone module
        import sys
        examples_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..",
            "examples", "Transfer_learn_nin_2D_3D_FNO_TL",
        )
        sys.path.insert(0, examples_dir)
        from training_residual_tracker import TrainingResidualTracker

        tracker = TrainingResidualTracker(out_dir=self.test_dir, tag="unit_test")

        self.pi_fno.Train(
            train_set=self.train_set,
            batch_size=len(self.train_set[0]),
            convergence_settings={
                "num_epochs": 3,
                "convergence_criterion": "residual_rms_batch_mean",
                "relative_error": 1e-100,
                "absolute_error": 1e-100,
            },
            plot_settings={
                "plot_list": ["total_loss", "residual_rms_batch_mean"],
                "save_frequency": 100,
            },
            training_residual_tracker=tracker,
            working_directory=self.test_dir,
        )

        # Verify CSV was written with correct structure
        csv_path = os.path.join(self.test_dir, "unit_test_residual_rms.csv")
        self.assertTrue(os.path.exists(csv_path), "Tracker CSV not created")

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ["epoch", "total_loss", "residual_rms_batch_mean"])
            rows = list(reader)
            self.assertEqual(len(rows), 3, "Expected 3 epoch rows in CSV")
            for row in rows:
                self.assertEqual(len(row), 3)
                epoch, tl, rms = int(row[0]), float(row[1]), float(row[2])
                self.assertTrue(np.isfinite(tl))
                self.assertTrue(np.isfinite(rms))
                self.assertGreater(rms, 0)

    def test_backward_compat_total_loss_criterion(self):
        """Train() with default total_loss criterion still works (no regression)."""
        self.pi_fno.Train(
            train_set=self.train_set,
            batch_size=len(self.train_set[0]),
            convergence_settings={"num_epochs": 3},
            working_directory=self.test_dir,
        )


if __name__ == '__main__':
    unittest.main()
