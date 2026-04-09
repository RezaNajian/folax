"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: October, 2024
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

class ImplicitParametricOperatorLearning(DeepNetwork):
    """
    Implicit parametric operator learning via hypernetwork-conditioned neural fields.

    This class implements an implicit (coordinate-based) parametric operator-learning
    workflow in which a *neural field* is represented by a coordinate-based MLP
    (the synthesizer network) and is conditioned by a modulator network. The
    synthesizer performs the neural-field task by mapping spatial coordinates to
    discretized field values, while the modulator encodes parametric information
    from a fixed-dimensional input space and conditions the synthesizer according
    to the coupling modes supported by :class:`fol.deep_neural_networks.nns.HyperNetwork`.

    Conceptually, the model represents a conditional neural field. The synthesizer
    evaluates a coordinate-based representation of the solution field, and the
    modulator injects parametric context (for example control variables or feature
    points of a parameterization such as Fourier coefficients/frequencies) that
    selects or shifts the field within a family of solutions.

    Training is typically unsupervised or physics-informed. Supervised target
    fields are not required; instead, discretized field predictions are evaluated
    using a physics-based loss functional (for example residual- or energy-based
    losses). During inference, Dirichlet boundary conditions are enforced by
    inserting prescribed values at Dirichlet indices across the full batch using
    :meth:`fol.loss_functions.loss.Loss.GetFullDofVector`.

    Although training is commonly performed on a fixed FE mesh (with a fixed
    discretization used inside the loss evaluation), the coordinate-based nature
    of the synthesizer network makes multi-resolution inference (and, in principle,
    multi-resolution training) possible. In such cases, the synthesizer can be
    evaluated on different coordinate sets than those used during training, while
    maintaining the same conditioning signal from the modulator.

    The base :class:`fol.deep_neural_networks.deep_network.DeepNetwork` provides
    optimizer and checkpoint integration. This class adds coupling to a
    :class:`fol.controls.control.Control` object and a prediction interface that
    produces discretized fields and then applies boundary conditions.

    Args:
        name (str):
            Name identifier for the model instance (used for logging and
            checkpointing).
        control (Control):
            Control object defining the parametric input space and mapping raw
            parameters to controlled variables used by the loss and boundary
            condition enforcement. Typical inputs are parameterization features
            such as Fourier coefficients/frequencies or other low-dimensional
            control points.
        loss_function (Loss):
            Physics-based loss function evaluated on predicted discretized fields.
            The loss defines the FE mesh, DOF structure, and boundary-condition
            handling required for training and inference.
        flax_neural_network (nnx.Module):
            Flax/NNX model used for implicit evaluation. In the common setup this
            is a hypernetwork-based model in which a modulator conditions a
            coordinate-based synthesizer (neural field) using coupling modes
            implemented by :class:`fol.deep_neural_networks.nns.HyperNetwork`.
            The callable interface must accept coordinates for neural-field
            evaluation, and the output dimension must be consistent with
            ``loss_function.dofs``.
        optax_optimizer (GradientTransformation):
            Optax optimizer transformation used to construct the optimizer for
            training.

    Raises:
        RuntimeError:
            If the neural network does not expose required attributes (for example
            ``in_features`` or ``out_features``) or if its output dimension is
            inconsistent with the DOF definition required by the loss function.
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
        Initialize model components and validate dimensional consistency.

        This method initializes the base :class:`DeepNetwork` components and then
        initializes the associated :class:`Control` object. It also validates that
        the provided neural network is compatible with the loss DOF definition.

        In the current implementation, compatibility is checked by requiring that
        the neural network exposes ``out_features`` and that it matches the
        number of DOF components specified by the loss function.

        Args:
            reinitialize (bool, optional):
                If ``True``, force reinitialization even if the instance was
                previously initialized. Default is ``False``.

        Returns:
            None

        Raises:
            RuntimeError:
                If the neural network does not expose ``in_features`` or
                ``out_features`` attributes, or if ``out_features`` does not
                match the number of DOF components required by the loss.
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

        if self.flax_neural_network.out_features != len(self.loss_function.dofs):
            fol_error(f"the size of the output layer is {self.flax_neural_network.out_features} " \
                      f" does not match the number of the loss function {self.loss_function.dofs}")

        # if self.flax_neural_network.in_features != self.control.GetNumberOfVariables():
        #     fol_error(f"the size of the input layer is {self.flax_neural_network.in_features} "\
        #               f"does not match the input size implicit/neural field which is {self.control.GetNumberOfVariables() + 3}")

    def ComputeBatchPredictions(self,batch_X:jnp.ndarray,nn_model:nnx.Module):
        """
        Compute implicit neural-field predictions for a batch of parametric inputs.

        This method evaluates a coordinate-based neural field (the synthesizer network)
        conditioned by the parametric inputs. The parametric inputs ``batch_X`` provide
        the conditioning signal (typically through a modulator network), while the
        synthesizer is evaluated on the mesh node coordinates supplied by the loss
        function. The result is a batch of discretized field values defined on the
        training mesh.

        Although the mesh coordinates used here correspond to the discretization
        associated with the loss function, the coordinate-based formulation allows the
        same conditioned neural field to be evaluated on alternative coordinate sets for
        multi-resolution inference.

        Args:
            batch_X (jax.numpy.ndarray):
                Batch of parametric inputs used to condition the neural field. The
                leading dimension corresponds to the batch size.
            nn_model (nnx.Module):
                Neural network implementing the conditioned neural field and evaluated
                as ``nn_model(batch_X, coords)``.

        Returns:
            jax.numpy.ndarray:
                Batch of discretized field predictions produced by the conditioned
                neural field on the provided mesh coordinates.

        Raises:
            None
        """
        return nn_model(batch_X,self.loss_function.fe_mesh.GetNodesCoordinates())

    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Compute the batch loss and return loss metrics.

        This method is designed for unsupervised or physics-informed learning.
        The batch is provided as a tuple for interface consistency with the base
        class. In typical physics-based training, the second entry of the tuple
        is unused and may be ``None`` because there are no supervised targets.

        The loss is computed by first mapping parametric inputs to controlled
        variables using :class:`Control`, then predicting discretized fields with
        the neural network, and finally evaluating the physics-based loss on the
        predicted fields.

        Args:
            batch (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Batch tuple ``(batch_X, batch_y)`` used for interface consistency.
                For physics-informed learning, ``batch_y`` is typically ``None``
                and is not used by this method.
            nn_model (nnx.Module):
                Neural network used to produce predictions for the batch.

        Returns:
            Tuple[jax.numpy.ndarray, dict]:
                A tuple ``(batch_loss, metrics_dict)`` where ``batch_loss`` is a
                scalar aggregated over the batch and ``metrics_dict`` contains
                loss statistics including the mandatory key ``"total_loss"``.

        Raises:
            None
        """
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        batch_predictions = self.ComputeBatchPredictions(batch[0],nn_model)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(control_outputs,batch_predictions)
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

        This method runs inference by computing controlled variables from the
        batch inputs, evaluating the neural network on the mesh node coordinates,
        and applying Dirichlet boundary conditions by inserting prescribed values
        at the Dirichlet indices for every sample in the batch using
        :meth:`Loss.GetFullDofVector`.

        Args:
            batch_X (jax.numpy.ndarray):
                Batch of parametric inputs used for inference. The leading
                dimension corresponds to the batch size.

        Returns:
            jax.numpy.ndarray:
                Batch of full discretized field vectors obtained by inference,
                with Dirichlet boundary conditions applied consistently across
                the batch.

        Raises:
            None
        """
        control_outputs = self.control.ComputeBatchControlledVariables(batch_X)
        preds = self.ComputeBatchPredictions(batch_X,self.flax_neural_network)
        return self.loss_function.GetFullDofVector(control_outputs,preds.reshape(preds.shape[0], -1))

    @print_with_timestamp_and_execution_time
    @partial(nnx.jit, donate_argnums=(1,), static_argnums=(0,2))
    def PredictDynamics(self,initial_Batch:jnp.ndarray,num_steps:int):
        """
        Perform autoregressive inference over multiple prediction steps.

        Starting from an initial batch of parametric inputs or states, this
        method repeatedly applies :meth:`Predict` to generate a trajectory. The
        returned array stacks the initial batch as the first entry followed by
        ``num_steps`` predicted batches.

        Args:
            initial_Batch (jax.numpy.ndarray):
                Initial batch used to start the rollout. The exact interpretation
                depends on the model usage, but the leading dimension must
                correspond to the batch size expected by :meth:`Predict`.
            num_steps (int):
                Number of autoregressive prediction steps to perform.

        Returns:
            jax.numpy.ndarray:
                Stacked trajectory array containing the initial batch followed by
                the predicted batches. The first axis indexes time steps.

        Raises:
            ValueError:
                If ``num_steps`` is negative.
        """
        def step_fn(current_state, _):
            """Compute the next state given the current state."""
            next_state = self.Predict(current_state)
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_Batch, None, length=num_steps)

        # Stack the initial state with the predicted trajectory
        return jnp.vstack([jnp.expand_dims(initial_Batch, axis=0), trajectory])

    def Finalize(self):
        pass