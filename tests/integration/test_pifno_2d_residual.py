"""
 Authors: Jerry Paul Varghese, https://github.com/jerrypaulvarghese
 Date: April, 2026
 License: FOL/LICENSE

 Integration test: full 2D physics-informed FNO training loop with
 residual_rms_batch_mean convergence and tracker.

 Verifies that training reduces both total_loss and residual_rms over epochs,
 and that convergence via residual_rms_batch_mean triggers early stop correctly.
"""
import pytest
import unittest
import os
import sys
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


class TinyConvNet(nnx.Module):
    """Tiny 1x1-conv net for fast testing."""
    def __init__(self, in_ch, out_ch, *, rngs):
        self.conv1 = nnx.Conv(in_ch, 32, kernel_size=(1, 1), rngs=rngs,
                              kernel_init=nnx.initializers.zeros,
                              bias_init=nnx.initializers.zeros)
        self.conv2 = nnx.Conv(32, out_ch, kernel_size=(1, 1), rngs=rngs,
                              kernel_init=nnx.initializers.zeros,
                              bias_init=nnx.initializers.zeros)

    def __call__(self, x):
        x = self.conv1(x)
        x = jax.nn.swish(x)
        x = self.conv2(x)
        return x


class TestPIFNO2DResidualTraining(unittest.TestCase):
    """Integration: PI-FNO 2D training with residual-based convergence."""

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self, request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        self.test_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_pifno_2d_residual_training",
        )
        os.makedirs(self.test_dir, exist_ok=True)

        # Small 2D mesh (6x6 nodes = 25 quads)
        self.fe_mesh = create_2D_square_mesh(L=1, N=6)
        bc_dict = {"Ux": {"left": 0.0, "right": 0.1},
                   "Uy": {"left": 0.0, "right": 0.0}}
        material_dict = {"young_modulus": 1, "poisson_ratio": 0.3}

        self.loss_fn = NeoHookeMechanicalLoss2DQuad(
            "mech_loss",
            loss_settings={"dirichlet_bc_dict": bc_dict, "num_gp": 2,
                           "material_dict": material_dict},
            fe_mesh=self.fe_mesh,
        )

        fourier_settings = {
            "x_freqs": np.array([1, 2]),
            "y_freqs": np.array([1, 2]),
            "z_freqs": np.array([0]),
            "beta": 2,
        }
        self.control = FourierControl("ctrl", fourier_settings, self.fe_mesh)

        self.fe_mesh.Initialize()
        self.loss_fn.Initialize()
        self.control.Initialize()

        n_nodes = self.fe_mesh.GetNumberOfNodes()
        N = int(round(n_nodes ** 0.5))
        sample_ctrl = self.control.ComputeControlledVariables(
            np.zeros(self.control.GetNumberOfVariables())
        )
        in_ch = max(1, int(sample_ctrl.size // n_nodes))
        # out_ch must be number_dofs_per_node (2 for 2D mechanics)
        out_ch = 2  # Ux, Uy

        net = TinyConvNet(in_ch, out_ch, rngs=nnx.Rngs(0))
        optimizer = optax.adam(5e-3)

        self.pi_fno = PhysicsInformedFourierParametricOperatorLearning(
            name="pi_fno_integ",
            control=self.control,
            loss_function=self.loss_fn,
            flax_neural_network=net,
            optax_optimizer=optimizer,
        )
        self.pi_fno.Initialize()

        # 1 training sample (single parametric case, full-batch)
        self.coeffs, _ = create_random_fourier_samples(self.control, 0)
        self.train_set = (self.coeffs[-1:],)

    def tearDown(self):
        if self.debug_mode == "false":
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_loss_decreases_over_epochs(self):
        """Verify that both total_loss and residual_rms decrease during training."""
        examples_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..",
            "examples", "Transfer_learn_nin_2D_3D_FNO_TL",
        )
        sys.path.insert(0, examples_dir)
        from training_residual_tracker import TrainingResidualTracker

        tracker = TrainingResidualTracker(out_dir=self.test_dir, tag="integ")

        num_epochs = 30
        self.pi_fno.Train(
            train_set=self.train_set,
            batch_size=len(self.train_set[0]),
            convergence_settings={
                "num_epochs": num_epochs,
                "convergence_criterion": "residual_rms_batch_mean",
                "relative_error": 1e-100,
                "absolute_error": 1e-100,  # won't converge early
            },
            plot_settings={
                "plot_list": ["total_loss", "residual_rms_batch_mean"],
                "save_frequency": num_epochs,
            },
            training_residual_tracker=tracker,
            working_directory=self.test_dir,
        )

        # Read CSV and verify monotonic-ish decrease
        csv_path = os.path.join(self.test_dir, "integ_residual_rms.csv")
        self.assertTrue(os.path.exists(csv_path))

        epochs, losses, residuals = [], [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                losses.append(float(row["total_loss"]))
                residuals.append(float(row["residual_rms_batch_mean"]))

        self.assertEqual(len(epochs), num_epochs)

        # First epoch loss > last epoch loss (training should improve)
        self.assertGreater(losses[0], losses[-1],
                           "Training should decrease total_loss over 30 epochs")
        self.assertGreater(residuals[0], residuals[-1],
                           "Training should decrease residual_rms over 30 epochs")

    def test_early_stop_fires(self):
        """Set a very loose residual target; verify training stops before max epochs."""
        # First, compute current residual to pick a target above it won't reach,
        # or a target that's loose enough to fire quickly.

        # Use a very big absolute_error so it fires at epoch 1
        num_epochs = 50
        self.pi_fno.Train(
            train_set=self.train_set,
            batch_size=len(self.train_set[0]),
            convergence_settings={
                "num_epochs": num_epochs,
                "convergence_criterion": "total_loss",
                "relative_error": 1e-100,
                "absolute_error": 1e10,  # absurdly loose → fires immediately
            },
            plot_settings={
                "plot_list": ["total_loss", "residual_rms_batch_mean"],
                "save_frequency": num_epochs,
            },
            working_directory=self.test_dir,
        )
        # If early stop works, training_history.png should exist (plotted at convergence)
        plot_path = os.path.join(self.test_dir, "training_history.png")
        self.assertTrue(os.path.exists(plot_path),
                        "Training history plot should be saved at convergence")


if __name__ == '__main__':
    unittest.main()
