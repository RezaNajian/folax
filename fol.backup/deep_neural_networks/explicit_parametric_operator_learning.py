"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
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

class ExplicitParametricOperatorLearning(DeepNetwork):
    """
    Explicit parametric operator learning on discretized fields.

    This class implements explicit parametric operator learning where both the
    input parameter space and the output field space are discretized and have
    fixed shape. The input typically represents a low-dimensional parametric
    space, such as control variables of a Fourier parametrization, while the
    output corresponds to a high-dimensional discretized field, for example a
    temperature or displacement field.

    The neural network explicitly represents the mapping from parametric inputs
    to field unknowns. Learning is unsupervised or physics-informed: no direct
    target fields are required. Instead, training is driven by physical loss
    functions evaluated on the predicted fields, such as residual-based or
    energy-based formulations.

    The :class:`Control` object maps parametric inputs to controlled variables,
    and the :class:`Loss` object evaluates the predicted fields after Dirichlet
    boundary conditions are applied. Consistency between network dimensions,
    control variables, and loss unknowns is enforced during initialization.

    Args:
        name (str):
            Name identifier for the model instance, used for logging and
            checkpointing.
        control (Control):
            Control object defining the parametric input space and how raw
            parameters are mapped to controlled variables.
        loss_function (Loss):
            Physics-based loss function used to evaluate predicted fields.
            The loss operates on full discretized fields and does not require
            supervised target data.
        flax_neural_network (nnx.Module):
            Flax/NNX neural network mapping parametric inputs to unknown field
            degrees of freedom. The network must expose ``in_features`` and
            ``out_features`` attributes.
        optax_optimizer (GradientTransformation):
            Optax optimizer transformation used to construct the optimizer for
            training.

    Raises:
        RuntimeError:
            If the neural network input or output dimensions are inconsistent
            with the control variable size or the number of unknowns defined by
            the loss function.
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
        Initialize model, loss, control, optimizer, and consistency checks.

        This method first runs the base-class initialization, which prepares
        the loss function, constructs the Orbax checkpointer, and builds the
        :class:`nnx.Optimizer` around the provided network and Optax
        transformation. It then ensures that the associated control object is
        initialized and finally checks that the neural network input and output
        dimensions are consistent with the control and loss function sizes.

        Concretely, the following consistency checks are performed:

        * ``flax_neural_network.in_features`` is compared to
          ``control.GetNumberOfVariables()`` to ensure that the parametric
          input dimension matches the control space.
        * ``flax_neural_network.out_features`` is compared to
          ``loss_function.GetNumberOfUnknowns()`` to ensure that the network
          output dimension matches the number of unknown field DOFs.

        Args:
            reinitialize (bool, optional):
                If ``True``, force reinitialization of all components, even if
                they have already been initialized. Default is ``False``.

        Returns:
            None

        Raises:
            RuntimeError:
                If the network does not expose ``in_features`` or
                ``out_features``, or if these dimensions are inconsistent with
                the control or loss function sizes. The underlying implementation
                uses ``fol_error`` to signal these conditions.
        """

        if self.initialized and not reinitialize:
            return

        super().Initialize(reinitialize)

        if not self.control.initialized:
            self.control.Initialize(reinitialize)

        self.initialized = True

        # now check if the input output layers size match with
        # loss and control sizes, this is explicit parametric learning
        if not hasattr(self.flax_neural_network, 'in_features'):
            fol_error(f"the provided flax neural netwrok does not have in_features "\
                      "which specifies the size of the input layer ")

        if not hasattr(self.flax_neural_network, 'out_features'):
            fol_error(f"the provided flax neural netwrok does not have out_features "\
                      "which specifies the size of the output layer")

        if self.flax_neural_network.in_features != self.control.GetNumberOfVariables():
            fol_error(f"the size of the input layer is {self.flax_neural_network.in_features} "\
                      f"does not match the size of control variables {self.control.GetNumberOfVariables()}")

        if self.flax_neural_network.out_features != self.loss_function.GetNumberOfUnknowns():
            fol_error(f"the size of the output layer is {self.flax_neural_network.out_features} " \
                      f" does not match the size of unknowns of the loss function {self.loss_function.GetNumberOfUnknowns()}")

    def ComputeBatchPredictions(self,batch_X:jnp.ndarray,nn_model:nnx.Module):
        """
        Compute network predictions for a batch of parametric inputs.

        This helper applies the provided neural network to a batch of
        parametric inputs ``batch_X`` and returns the corresponding batch of
        predicted unknown DOFs. It does not insert Dirichlet values or build
        full field vectors; that is handled at the loss level.

        Args:
            batch_X (jax.numpy.ndarray):
                Batch of parametric inputs (for example control variables
                entering the operator). The leading dimension is the batch
                size, and the second dimension must match
                ``nn_model.in_features``.
            nn_model (nnx.Module):
                Neural network that maps parametric inputs to unknown DOFs.

        Returns:
            jax.numpy.ndarray:
                Batch of predicted unknown DOF vectors, one per row in
                ``batch_X``.

        Raises:
            None:
                This method is a thin wrapper around the network call and does
                not introduce additional failure modes beyond those of
                ``nn_model`` itself.
        """
        return nn_model(batch_X)

    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Compute the batch loss for explicit parametric operator learning.

        This method evaluates the loss in an unsupervised or physics-informed
        setting. The batch is provided as a tuple for consistency with the
        :class:`DeepNetwork` base class interface, but only the first entry is
        used. The second entry of the batch tuple is expected to be ``None`` and
        does not represent supervised targets.

        The computation proceeds by mapping the batch of parametric inputs to
        controlled variables using the associated :class:`Control` object, then
        predicting the unknown degrees of freedom with the neural network. The
        predicted unknowns are inserted into full discretized field vectors, and
        the loss function is evaluated directly on these fields using physical
        constraints or residual-based objectives.

        The returned metrics dictionary always includes the key ``"total_loss"``,
        which is required by the training loop for logging, convergence checks,
        plotting, and checkpointing.

        Args:
            batch (Tuple[jax.numpy.ndarray, None]):
                Batch tuple ``(batch_X, None)`` where ``batch_X`` contains the
                parametric input samples. The second entry is unused and present
                only to maintain a consistent interface with the base class.
            nn_model (nnx.Module):
                Neural network used to infer the unknown degrees of freedom from
                the parametric inputs.

        Returns:
            Tuple[jax.numpy.ndarray, dict]:
                A tuple ``(batch_loss, metrics_dict)`` where ``batch_loss`` is the
                scalar loss aggregated over the batch, and ``metrics_dict``
                contains loss statistics including the mandatory key
                ``"total_loss"``.

        Raises:
            None:
                This method assumes consistency between the control, loss
                function, and network dimensions, which is enforced during
                initialization.
        """
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        batch_unknowns_predictions = self.ComputeBatchPredictions(batch[0],nn_model)
        batch_full_pred = jnp.zeros((batch[0].shape[0],self.loss_function.GetTotalNumberOfDOFs()))
        batch_full_pred = batch_full_pred.at[:,self.loss_function.non_dirichlet_indices].set(batch_unknowns_predictions)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(control_outputs,batch_full_pred)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})

    @print_with_timestamp_and_execution_time
    @partial(nnx.jit, static_argnums=(0,), donate_argnums=1)
    def Predict(self,batch_X):
        """
        Perform inference for a batch of parametric inputs and apply Dirichlet
        boundary conditions.

        This method runs inference by evaluating the trained neural network on a
        batch of parametric inputs to predict the unknown degrees of freedom.
        The predicted unknowns are then embedded into full discretized field
        vectors, and Dirichlet boundary conditions are applied by setting the
        prescribed boundary values at the corresponding DOF indices for every
        sample in the batch.

        The result is a batch of full field predictions that satisfy the imposed
        boundary conditions and are suitable for post-processing or evaluation.

        Args:
            batch_X (jax.numpy.ndarray):
                Batch of parametric inputs used for inference. The leading dimension
                corresponds to the batch size, and the feature dimension must match
                the number of control variables.

        Returns:
            jax.numpy.ndarray:
                Batch of full discretized field vectors obtained by inference,
                with Dirichlet boundary conditions applied consistently across the
                batch.

        Raises:
            None
        """
        batch_unknowns_predictions = self.ComputeBatchPredictions(batch_X,self.flax_neural_network)
        batch_full_pred = jnp.zeros((batch_X.shape[0],self.loss_function.GetTotalNumberOfDOFs()))
        batch_full_pred = batch_full_pred.at[:,self.loss_function.non_dirichlet_indices].set(batch_unknowns_predictions)
        return self.loss_function.GetFullDofVector(batch_X,batch_full_pred.reshape(batch_full_pred.shape[0], -1))

    def Finalize(self):
        pass