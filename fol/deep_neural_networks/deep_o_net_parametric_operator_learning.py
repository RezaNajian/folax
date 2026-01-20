"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2025
 License: FOL/LICENSE
"""

from typing import Iterator,Tuple
import jax
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial
from optax import GradientTransformation
from flax import nnx
from .deep_network import DeepNetwork
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *

class DeepONetParametricOperatorLearning(DeepNetwork):
    """
    Parametric operator learning using a DeepONet evaluated on FE mesh coordinates.

    This class implements a DeepONet-based operator-learning workflow where a
    fixed-dimensional parametric input (for example control variables or
    parameterization features) is mapped to a discretized field by evaluating a
    DeepONet on the node coordinates of the FE mesh provided by the loss
    function. The model therefore learns an operator from a parametric space to
    a field space discretized on a mesh.

    The base :class:`DeepNetwork` functionality is used for optimizer and
    checkpoint integration. This class adds coupling to a :class:`Control`
    object and defines a prediction interface that supplies the mesh coordinates
    to the DeepONet.

    During inference, Dirichlet boundary conditions are enforced by inserting
    prescribed values at the Dirichlet indices across the full batch using
    :meth:`Loss.GetFullDofVector`.

    Args:
        name (str):
            Name identifier for the model instance, used for logging and
            checkpointing.
        control (Control):
            Control object defining the parametric input space and mapping raw
            parameters to controlled variables used by the network and loss.
        loss_function (Loss):
            Loss object defining the FE mesh discretization (node coordinates)
            and the DOF/boundary-condition handling used to assemble full field
            vectors.
        flax_neural_network (nnx.Module):
            DeepONet model expected to be callable as
            ``nn_model(batch_inputs, coords)`` where ``coords`` are the FE mesh
            node coordinates. The network output is reshaped to
            ``(batch_size, -1)`` to represent discretized field values.
        optax_optimizer (GradientTransformation):
            Optax optimizer transformation used to construct the optimizer for
            training.

    Raises:
        RuntimeError:
            If the DeepONet or loss configuration is incompatible with the FE
            mesh discretization or boundary-condition assembly required by
            :meth:`Loss.GetFullDofVector`.
    """
    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation):

        super().__init__(name,loss_function,flax_neural_network,
                         optax_optimizer)
        self.control = control

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize model components and the associated control object.

        This method initializes the base :class:`DeepNetwork` components and then
        initializes the associated :class:`Control` instance. Initialization is
        skipped if the model was already initialized, unless ``reinitialize`` is
        set to ``True``.

        Args:
            reinitialize (bool, optional):
                If ``True``, force reinitialization even if the instance was
                previously initialized. Default is ``False``.

        Returns:
            None

        Raises:
            None
        """
        if self.initialized and not reinitialize:
            return

        super().Initialize(reinitialize)

        if not self.control.initialized:
            self.control.Initialize(reinitialize)

        self.initialized = True

    def ComputeBatchPredictions(self,batch_X:jnp.ndarray,nn_model:nnx.Module):
        """
        Compute DeepONet predictions for a batch of parametric inputs.

        The DeepONet is evaluated on the FE mesh node coordinates retrieved from
        the loss function. The returned output is reshaped to
        ``(batch_size, -1)`` so it can be treated as a batch of discretized
        field vectors.

        Args:
            batch_X (jax.numpy.ndarray):
                Batch of parametric (or controlled) inputs. The first dimension
                is the batch size.
            nn_model (nnx.Module):
                DeepONet model evaluated as ``nn_model(batch_X, coords)``.

        Returns:
            jax.numpy.ndarray:
                Batch of discretized field predictions with shape
                ``(batch_size, num_predicted_dofs)``.

        Raises:
            RuntimeError:
                If the FE mesh coordinates cannot be obtained from the loss
                function, or if the network output cannot be reshaped to a
                2D batch array.
        """
        return nn_model(batch_X,self.loss_function.fe_mesh.GetNodesCoordinates()).reshape(batch_X.shape[0],-1)

    @print_with_timestamp_and_execution_time
    @partial(nnx.jit, static_argnums=(0,), donate_argnums=1)
    def Predict(self,batch_X):
        """
        Perform inference for a batch of inputs and apply Dirichlet constraints.

        This method computes controlled variables from ``batch_X``, evaluates the
        DeepONet on the FE mesh coordinates, and assembles the full DOF vector by
        applying Dirichlet boundary conditions across the batch using
        :meth:`Loss.GetFullDofVector`.

        Args:
            batch_X (jax.numpy.ndarray):
                Batch of raw parametric inputs. The first dimension is the batch
                size.

        Returns:
            jax.numpy.ndarray:
                Batch of full discretized field vectors with Dirichlet boundary
                conditions enforced.

        Raises:
            RuntimeError:
                If boundary-condition assembly via :meth:`Loss.GetFullDofVector`
                fails due to inconsistent DOF definitions or incompatible shapes.
        """
        control_outputs = self.control.ComputeBatchControlledVariables(batch_X)
        preds = self.ComputeBatchPredictions(control_outputs,self.flax_neural_network)
        return self.loss_function.GetFullDofVector(batch_X,preds.reshape(preds.shape[0], -1))

    def Finalize(self):
        pass

class DataDrivenDeepONetParametricOperatorLearning(DeepONetParametricOperatorLearning):
    """
    Data-driven parametric operator learning using a DeepONet.

    This class specializes :class:`DeepONetParametricOperatorLearning` for fully
    supervised, data-driven training. The DeepONet predicts discretized fields
    that are directly compared against provided target fields using a supervised
    loss function.

    In this formulation, the operator is learned purely from input–output data
    pairs. No physical constraints are enforced explicitly beyond those embedded
    in the training data. Boundary-condition handling is delegated to the loss
    function and to the inference routine that assembles full DOF vectors.

    Args:
        name (str):
            Name identifier for the model instance, used for logging and
            checkpointing.
        control (Control):
            Control object defining the parametric input space and mapping raw
            parameters to controlled variables supplied to the DeepONet.
        loss_function (Loss):
            Supervised loss function used to compare predicted discretized fields
            against target fields provided in the training data.
        flax_neural_network (nnx.Module):
            DeepONet model evaluated as ``nn_model(inputs, coords)``, where
            ``coords`` are FE mesh node coordinates.
        optax_optimizer (GradientTransformation):
            Optax optimizer transformation used for training.

    Raises:
        RuntimeError:
            If the predicted field shape is incompatible with the target field
            shape expected by the loss function.
    """
    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Compute data-driven batch loss and return loss metrics.

        The batch is interpreted as ``(batch_X, batch_y)`` where ``batch_y`` is a
        target discretized field. Predictions are computed from controlled inputs
        and compared against the targets using the configured loss function.

        Args:
            batch (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Training batch tuple ``(batch_X, batch_y)``. ``batch_X`` contains
                raw parametric inputs and ``batch_y`` contains target fields.
            nn_model (nnx.Module):
                DeepONet used to produce predictions for the batch.

        Returns:
            Tuple[jax.numpy.ndarray, dict]:
                A tuple ``(batch_loss, metrics_dict)`` where ``batch_loss`` is a
                scalar aggregated over the batch and ``metrics_dict`` contains
                loss statistics including the mandatory key ``"total_loss"``.

        Raises:
            RuntimeError:
                If target tensors in ``batch_y`` are incompatible with the loss
                function expectations or the prediction shape.
        """
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        batch_predictions = self.ComputeBatchPredictions(control_outputs,nn_model)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(batch[1],batch_predictions)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})

class PhysicsInformedDeepONetParametricOperatorLearning(DeepONetParametricOperatorLearning):
    """
    Physics-informed parametric operator learning using a DeepONet.

    This class specializes :class:`DeepONetParametricOperatorLearning` for
    unsupervised or physics-informed training. No supervised target fields are
    required. Instead, discretized field predictions produced by the DeepONet
    are evaluated using physics-based loss functionals, such as residual- or
    energy-based formulations.

    The DeepONet is evaluated on FE mesh node coordinates, and the resulting
    predictions are passed directly to the loss function together with the
    controlled variables. Dirichlet boundary conditions are enforced through
    the loss and during inference by assembling full DOF vectors.

    Args:
        name (str):
            Name identifier for the model instance, used for logging and
            checkpointing.
        control (Control):
            Control object defining the parametric input space and mapping raw
            parameters to controlled variables used by the physics-based loss.
        loss_function (Loss):
            Physics-based loss function used to evaluate discretized field
            predictions without supervised target data.
        flax_neural_network (nnx.Module):
            DeepONet model evaluated as ``nn_model(inputs, coords)``, where
            ``coords`` are FE mesh node coordinates.
        optax_optimizer (GradientTransformation):
            Optax optimizer transformation used for training.

    Raises:
        RuntimeError:
            If the physics-based loss evaluation fails due to incompatible
            control variables, mesh definitions, or predicted field shapes.
    """
    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Compute physics-informed batch loss and return loss metrics.

        The physics-informed loss is computed by first mapping raw inputs to
        controlled variables, predicting discretized fields using the DeepONet,
        and then evaluating the physics-based loss against the controlled
        variables (rather than supervised targets).

        Args:
            batch (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Training batch tuple ``(batch_X, batch_y)`` kept for interface
                consistency. ``batch_y`` is typically ``None`` and is not used.
            nn_model (nnx.Module):
                DeepONet used to produce predictions for the batch.

        Returns:
            Tuple[jax.numpy.ndarray, dict]:
                A tuple ``(batch_loss, metrics_dict)`` where ``batch_loss`` is a
                scalar aggregated over the batch and ``metrics_dict`` contains
                loss statistics including the mandatory key ``"total_loss"``.

        Raises:
            RuntimeError:
                If the physics loss evaluation fails due to inconsistent control
                outputs, mesh definitions, or prediction shapes.
        """
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        batch_predictions = self.ComputeBatchPredictions(control_outputs,nn_model)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(control_outputs,batch_predictions)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})
