"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2025
 License: FOL/LICENSE
"""

from __future__ import annotations

from typing import Tuple
import jax
import jax.numpy as jnp
from flax import nnx
from optax import GradientTransformation

from .deep_network import DeepNetwork
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control


# ============================================================
# Helpers (robustness vs dataset normalization in DeepNetwork)
# ============================================================

def _ensure_batch(x: jnp.ndarray) -> jnp.ndarray:
    """Make sure x has leading batch dim."""
    if x is None:
        return x
    return x[jnp.newaxis, ...] if x.ndim == 1 else x


def _has_targets(y: jnp.ndarray) -> bool:
    """True if y is a non-empty target array."""
    if y is None:
        return False
    try:
        return (getattr(y, "size", 0) > 0) and (len(y) > 0)
    except Exception:
        return False


# ============================================================
# 2D Fourier Parametric Operator Learning
# ============================================================

class FourierParametricOperatorLearning(DeepNetwork):
    """
    2D structured-grid version (expects node count n_nodes = N^2),
    model input reshaped to (B, N, N, C).
    """

    def __init__(
        self,
        name: str,
        control: Control,
        loss_function: Loss,
        flax_neural_network: nnx.Module,
        optax_optimizer: GradientTransformation,
    ):
        super().__init__(name, loss_function, flax_neural_network, optax_optimizer)
        self.control = control

    @print_with_timestamp_and_execution_time
    def Initialize(self, reinitialize: bool = False) -> None:
        if self.initialized and not reinitialize:
            return

        super().Initialize(reinitialize)

        if not self.control.initialized:
            self.control.Initialize(reinitialize)

        self.initialized = True

    def ComputeBatchPredictions(self, batch_X: jnp.ndarray, nn_model: nnx.Module):
        batch_X = _ensure_batch(batch_X)
        batch_size = int(batch_X.shape[0])

        n_nodes = int(self.loss_function.fe_mesh.GetNumberOfNodes())
        N = int(round((n_nodes) ** 0.5))
        if N * N != n_nodes:
            raise ValueError(f"[FOL] 2D grid expected n_nodes=N^2, got n_nodes={n_nodes}, N={N}")

        # batch_X is typically (B, n_nodes*C) or (B, n_nodes) or (B, n_nodes, C)
        num_chs = int(batch_X.size // (batch_size * N * N))
        if batch_size * N * N * num_chs != int(batch_X.size):
            raise ValueError(
                f"[FOL] bad batch_X size for 2D reshape: "
                f"batch_size={batch_size}, N={N}, inferred C={num_chs}, batch_X.shape={batch_X.shape}"
            )

        x = batch_X.reshape(batch_size, N, N, num_chs)
        y = nn_model(x)  # expects (B, N, N, out_channels)
        return y.reshape(batch_size, -1)

    @print_with_timestamp_and_execution_time
    def Predict(self, batch_control: jnp.ndarray):
        batch_control = _ensure_batch(batch_control)
        control_outputs = self.control.ComputeBatchControlledVariables(batch_control)
        preds = self.ComputeBatchPredictions(control_outputs, self.flax_neural_network)
        return self.loss_function.GetFullDofVector(batch_control, preds)

    def Finalize(self):
        pass


class DataDrivenFourierParametricOperatorLearning(FourierParametricOperatorLearning):
    def ComputeBatchLossValue(self, batch: Tuple[jnp.ndarray, jnp.ndarray], nn_model: nnx.Module):
        X, Y = batch  # DeepNetwork may supply dummy Y if you passed only X
        if not _has_targets(Y):
            raise ValueError(
                "[FOL] DataDrivenFourierParametricOperatorLearning requires targets Y. "
                "You passed only X (or empty Y). Use PhysicsInformed* for physics loss."
            )

        control_outputs = self.control.ComputeBatchControlledVariables(X)
        batch_predictions = self.ComputeBatchPredictions(control_outputs, nn_model)

        batch_loss, (batch_min, batch_max, batch_avg) = self.loss_function.ComputeBatchLoss(
            Y, batch_predictions
        )
        loss_name = self.loss_function.GetName()
        return batch_loss, {
            loss_name + "_min": batch_min,
            loss_name + "_max": batch_max,
            loss_name + "_avg": batch_avg,
            "total_loss": batch_loss,
        }


class PhysicsInformedFourierParametricOperatorLearning(FourierParametricOperatorLearning):
    """
    2D physics-informed variant with residual_rms_* metrics for convergence/stoppage.
    Requires loss_function.ComputeResidualNorm(...) to exist.
    """

    def ComputeBatchLossValue(self, batch: Tuple[jnp.ndarray, jnp.ndarray], nn_model: nnx.Module):
        X, _ = batch  # targets ignored (physics-informed training)

        # 1) Controls and NN predictions
        control_outputs = self.control.ComputeBatchControlledVariables(X)
        batch_predictions = self.ComputeBatchPredictions(control_outputs, nn_model)

        # 2) Standard batch loss (physics residual / energy / etc.)
        batch_loss, (batch_min, batch_max, batch_avg) = self.loss_function.ComputeBatchLoss(
            control_outputs, batch_predictions
        )
        loss_name = self.loss_function.GetName()

        # 3) Residual RMS per sample (for custom residual-based convergence)
        def residual_for_sample(ctrl_sample, dofs_sample):
            ctrl_flat = ctrl_sample.reshape(-1)
            dofs_flat = dofs_sample.reshape(-1)

            full_dof_vec = self.loss_function.GetFullDofVector(
                ctrl_flat[jnp.newaxis, :],
                dofs_flat[jnp.newaxis, :],
            )[0]

            return self.loss_function.ComputeResidualNorm(
                ctrl_flat,
                full_dof_vec,
                relative=True,
            )

        residual_rms_per_sample = jax.vmap(residual_for_sample)(control_outputs, batch_predictions)

        metrics = {
            loss_name + "_min": batch_min,
            loss_name + "_max": batch_max,
            loss_name + "_avg": batch_avg,
            "total_loss": batch_loss,
            "residual_rms_batch_mean": jnp.mean(residual_rms_per_sample),
            "residual_rms_batch_min": jnp.min(residual_rms_per_sample),
            "residual_rms_batch_max": jnp.max(residual_rms_per_sample),
        }

        return batch_loss, metrics


# ============================================================
# 3D Fourier Parametric Operator Learning
# ============================================================

class FourierParametricOperatorLearning3D(DeepNetwork):
    """
    3D structured-grid version (expects node count n_nodes = N^3),
    model input reshaped to (B, N, N, N, C).
    """

    def __init__(
        self,
        name: str,
        control: Control,
        loss_function: Loss,
        flax_neural_network: nnx.Module,
        optax_optimizer: GradientTransformation,
    ):
        super().__init__(name, loss_function, flax_neural_network, optax_optimizer)
        self.control = control

    @print_with_timestamp_and_execution_time
    def Initialize(self, reinitialize: bool = False) -> None:
        if self.initialized and not reinitialize:
            return

        super().Initialize(reinitialize)

        if not self.control.initialized:
            self.control.Initialize(reinitialize)

        self.initialized = True

    def ComputeBatchPredictions(self, batch_X: jnp.ndarray, nn_model: nnx.Module):
        batch_X = _ensure_batch(batch_X)
        batch_size = int(batch_X.shape[0])

        n_nodes = int(self.loss_function.fe_mesh.GetNumberOfNodes())
        N = int(round(n_nodes ** (1.0 / 3.0)))
        if N * N * N != n_nodes:
            raise ValueError(f"[FOL] 3D grid expected n_nodes=N^3, got n_nodes={n_nodes}, N={N}")

        # batch_X is typically (B, n_nodes*C) or (B, n_nodes) or (B, n_nodes, C)
        num_chs = int(batch_X.size // (batch_size * N * N * N))
        if batch_size * N * N * N * num_chs != int(batch_X.size):
            raise ValueError(
                f"[FOL] bad batch_X size for 3D reshape: "
                f"batch_size={batch_size}, N={N}, inferred C={num_chs}, batch_X.shape={batch_X.shape}"
            )

        x = batch_X.reshape(batch_size, N, N, N, num_chs)
        y = nn_model(x)  # expects (B, N, N, N, out_channels)
        return y.reshape(batch_size, -1)

    @print_with_timestamp_and_execution_time
    def Predict(self, batch_control: jnp.ndarray):
        batch_control = _ensure_batch(batch_control)
        control_outputs = self.control.ComputeBatchControlledVariables(batch_control)
        preds = self.ComputeBatchPredictions(control_outputs, self.flax_neural_network)
        return self.loss_function.GetFullDofVector(batch_control, preds)

    def Finalize(self):
        pass


class DataDrivenFourierParametricOperatorLearning3D(FourierParametricOperatorLearning3D):
    def ComputeBatchLossValue(self, batch: Tuple[jnp.ndarray, jnp.ndarray], nn_model: nnx.Module):
        X, Y = batch
        if not _has_targets(Y):
            raise ValueError(
                "[FOL] DataDrivenFourierParametricOperatorLearning3D requires targets Y. "
                "You passed only X (or empty Y). Use PhysicsInformed* for physics loss."
            )

        control_outputs = self.control.ComputeBatchControlledVariables(X)
        batch_predictions = self.ComputeBatchPredictions(control_outputs, nn_model)

        batch_loss, (batch_min, batch_max, batch_avg) = self.loss_function.ComputeBatchLoss(
            Y, batch_predictions
        )
        loss_name = self.loss_function.GetName()
        return batch_loss, {
            loss_name + "_min": batch_min,
            loss_name + "_max": batch_max,
            loss_name + "_avg": batch_avg,
            "total_loss": batch_loss,
        }


class PhysicsInformedFourierParametricOperatorLearning3D(FourierParametricOperatorLearning3D):
    """
    3D physics-informed variant with residual_rms_* metrics for convergence/stoppage.
    Requires loss_function.ComputeResidualNorm(...) to exist.
    """

    def ComputeBatchLossValue(self, batch: Tuple[jnp.ndarray, jnp.ndarray], nn_model: nnx.Module):
        X, _ = batch  # targets ignored

        # 1) Controls and NN predictions
        control_outputs = self.control.ComputeBatchControlledVariables(X)
        batch_predictions = self.ComputeBatchPredictions(control_outputs, nn_model)

        # 2) Standard batch loss
        batch_loss, (batch_min, batch_max, batch_avg) = self.loss_function.ComputeBatchLoss(
            control_outputs, batch_predictions
        )
        loss_name = self.loss_function.GetName()

        # 3) Residual RMS per sample
        def residual_for_sample(ctrl_sample, dofs_sample):
            ctrl_flat = ctrl_sample.reshape(-1)
            dofs_flat = dofs_sample.reshape(-1)

            full_dof_vec = self.loss_function.GetFullDofVector(
                ctrl_flat[jnp.newaxis, :],
                dofs_flat[jnp.newaxis, :],
            )[0]

            return self.loss_function.ComputeResidualNorm(
                ctrl_flat,
                full_dof_vec,
                relative=True,
            )

        residual_rms_per_sample = jax.vmap(residual_for_sample)(control_outputs, batch_predictions)

        metrics = {
            loss_name + "_min": batch_min,
            loss_name + "_max": batch_max,
            loss_name + "_avg": batch_avg,
            "total_loss": batch_loss,
            "residual_rms_batch_mean": jnp.mean(residual_rms_per_sample),
            "residual_rms_batch_min": jnp.min(residual_rms_per_sample),
            "residual_rms_batch_max": jnp.max(residual_rms_per_sample),
        }

        return batch_loss, metrics