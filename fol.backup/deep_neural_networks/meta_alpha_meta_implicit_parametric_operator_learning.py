"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: December, 2024
 License: FOL/LICENSE
"""

from typing import Tuple,Iterator
import jax
import jax.numpy as jnp
import optax
from functools import partial
from optax import GradientTransformation
from flax import nnx
from tqdm import trange
from .implicit_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *
from .nns import HyperNetwork

class LatentStepModel(nnx.Module):
    """
    Trainable latent-step wrapper used for meta-learning the inner-loop step size.

    This small NNX module stores a single scalar parameter ``latent_step`` as an
    :class:`flax.nnx.Param`. It is used by meta-implicit learning methods to
    learn the step size applied during latent-code updates in the inner loop.

    Args:
        init_latent_step_value (float or jax.Array):
            Initial value for the latent-step parameter.

    Returns:
        None

    Raises:
        TypeError:
            If ``init_latent_step_value`` cannot be stored as an NNX parameter.
    """
    def __init__(self, init_latent_step_value):
        self.latent_step = nnx.Param(init_latent_step_value)
    def __call__(self):
        """
        Return the current latent-step parameter.

        Args:
            None

        Returns:
            jax.Array:
                The trainable latent-step value.

        Raises:
            None
        """
        return self.latent_step

class MetaAlphaMetaImplicitParametricOperatorLearning(ImplicitParametricOperatorLearning):
    """
    Meta-implicit parametric operator learning with a learnable latent-step size.

    This class derives from :class:`~fol.deep_neural_networks.implicit_parametric_operator_learning.ImplicitParametricOperatorLearning`
    and implements an implicit, coordinate-based operator learning workflow where
    a neural-field synthesizer is evaluated on FE mesh coordinates. The model is
    typically a :class:`~fol.deep_neural_networks.nns.HyperNetwork`, meaning the
    synthesizer performs the coordinate-based neural-field mapping while a
    modulator conditions the synthesizer according to the coupling modes defined
    by :class:`~fol.deep_neural_networks.nns.HyperNetwork`.

    The main difference from the base implicit class is that predictions are not
    produced directly from parametric inputs. Instead, a latent code is optimized
    per sample in an inner loop by minimizing the physics-based loss. The main
    difference from :class:`~fol.deep_neural_networks.meta_implicit_parametric_operator_learning.MetaImplicitParametricOperatorLearning`
    is that this class also learns the inner-loop update magnitude. A dedicated
    trainable scalar step model is optimized jointly with the network parameters,
    so the latent step size adapts during training.

    Training is usually unsupervised or physics-informed. The loss is evaluated
    on discretized field predictions, and Dirichlet boundary conditions can be
    enforced at inference time by inserting prescribed values across the batch
    using :meth:`~fol.loss_functions.loss.Loss.GetFullDofVector`.

    Args:
        name (str):
            Name identifier used for logging and checkpointing.
        control (Control):
            Control object that maps raw parametric inputs to controlled variables
            used by the loss functional.
        loss_function (Loss):
            Physics-based loss functional evaluated on predicted discretized fields.
            The loss provides access to the FE mesh coordinates and DOF handling.
        flax_neural_network (HyperNetwork):
            HyperNetwork used for implicit neural-field prediction. The synthesizer
            is expected to be coordinate-based, and the modulator provides
            conditioning according to HyperNetwork coupling settings.
        main_loop_optax_optimizer (GradientTransformation):
            Optimizer transformation for updating the neural network parameters.
        latent_step_optax_optimizer (GradientTransformation):
            Optimizer transformation for updating the latent-step parameter.
        latent_step_size (float, optional):
            Initial value for the learnable latent step size. Default is ``1e-2``.
        num_latent_iterations (int, optional):
            Number of latent inner-loop update iterations used during prediction.
            Default is ``3``.

    Returns:
        None

    Raises:
        RuntimeError:
            If the model is used before initialization in workflows that require
            initialized loss/control components.
        ValueError:
            If ``num_latent_iterations`` is negative.
    """

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:HyperNetwork,
                 main_loop_optax_optimizer:GradientTransformation,
                 latent_step_optax_optimizer:GradientTransformation,
                 latent_step_size:float=1e-2,
                 num_latent_iterations:int=3
                 ):
        super().__init__(name,control,loss_function,flax_neural_network,
                         main_loop_optax_optimizer)

        self.latent_step_optimizer = latent_step_optax_optimizer
        self.latent_step_nnx_model = LatentStepModel(latent_step_size)
        self.num_latent_iterations = num_latent_iterations
        self.latent_nnx_optimizer = nnx.Optimizer(self.latent_step_nnx_model,self.latent_step_optimizer,wrt=nnx.Param)

    def ComputeBatchPredictions(self,batch_X:jnp.ndarray,meta_model:Tuple[nnx.Module,nnx.Module]):
        """
        Compute batch predictions using latent-code adaptation with a learnable step size.

        This method performs per-sample latent optimization in an inner loop. For each
        parametric input, controlled variables are computed using ``self.control``.
        Latent codes are initialized and iteratively updated by descending the gradient
        of the physics-based loss with respect to the latent variables. Unlike the
        simpler meta-implicit variant where the latent update step size is fixed, this
        class uses a trainable step model ``latent_step`` that is optimized during the
        outer training loop.

        The neural field is evaluated on FE mesh node coordinates obtained from the
        loss function mesh. The final latent codes are used to produce discretized
        field predictions on the same coordinate set.

        Args:
            batch_X (jax.numpy.ndarray):
                Batch of parametric inputs. The leading dimension corresponds to the
                batch size.
            meta_model (Tuple[nnx.Module, nnx.Module]):
                Tuple ``(nn_model, latent_step_model)`` where ``nn_model`` is the
                HyperNetwork used for predictions and ``latent_step_model`` returns
                the current learnable step size via ``latent_step_model()``.

        Returns:
            jax.numpy.ndarray:
                Batch of discretized field predictions evaluated on the FE mesh node
                coordinates.

        Raises:
            RuntimeError:
                If ``nn_model`` does not expose ``in_features`` required to size the
                latent-code array, or if the model cannot be called as
                ``nn_model(latent_codes, coords)``.
            ValueError:
                If ``self.num_latent_iterations`` is negative.
        """
        nn_model,latent_step = meta_model

        latent_codes = jnp.zeros((batch_X.shape[0],nn_model.in_features))
        control_outputs = self.control.ComputeBatchControlledVariables(batch_X)

        def latent_loss(latent_code,control_output):
            nn_output = nn_model(latent_code[None, :],self.loss_function.fe_mesh.GetNodesCoordinates())
            return self.loss_function.ComputeBatchLoss(control_output,nn_output)[0]

        vec_grad_func = jax.vmap(jax.grad(latent_loss, argnums=0))
        for _ in range(self.num_latent_iterations):
            grads = vec_grad_func(latent_codes,control_outputs)
            latent_codes -= latent_step() * grads

        return nn_model(latent_codes,self.loss_function.fe_mesh.GetNodesCoordinates())

    def GetState(self):
        """
        Return the full meta-learning state required for training and checkpointing.

        The returned tuple includes the main network, its optimizer state, the
        latent-step model, and the latent-step optimizer state.

        Args:
            None

        Returns:
            Tuple[nnx.Module, nnx.Optimizer, nnx.Module, nnx.Optimizer]:
                Tuple ``(flax_neural_network, nnx_optimizer, latent_step_model, latent_step_optimizer)``.

        Raises:
            None
        """
        return (self.flax_neural_network, self.nnx_optimizer, self.latent_step_nnx_model, self.latent_nnx_optimizer)

    @partial(nnx.jit, static_argnums=(0,))
    def TrainStep(self, meta_state, data):
        """
        Perform one meta-training step updating both network parameters and latent-step.

        This step computes gradients of the batch physics loss with respect to both
        the main network parameters and the learnable latent-step parameter. The main
        optimizer updates the network, and the latent optimizer updates the step model.

        Args:
            meta_state (Tuple[nnx.Module, nnx.Optimizer, nnx.Module, nnx.Optimizer]):
                Tuple containing ``(nn_model, main_optimizer, latent_step_model, latent_optimizer)``.
            data (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Batch tuple passed through the base training interface. The second
                element is typically unused for physics-informed learning and may be
                ``None``.

        Returns:
            jax.numpy.ndarray:
                Scalar batch loss value.

        Raises:
            RuntimeError:
                If gradient computation fails due to incompatible model outputs or loss
                evaluation.
        """
        nn_model, main_optimizer, latent_step_model, latent_optimizer = meta_state

        (batch_loss,batch_dict), meta_grads = nnx.value_and_grad(self.ComputeBatchLossValue,argnums=1,has_aux=True) \
                                                                    (data,(nn_model,latent_step_model))

        main_optimizer.update(nn_model,meta_grads[0])
        latent_optimizer.update(latent_step_model,meta_grads[1])
        return batch_loss

    @partial(nnx.jit, static_argnums=(0,))
    def TestStep(self, meta_state, data):
        """
        Compute the batch loss for evaluation using the current meta-state.

        Args:
            meta_state (Tuple[nnx.Module, nnx.Optimizer, nnx.Module, nnx.Optimizer]):
                Tuple containing ``(nn_model, main_optimizer, latent_step_model, latent_optimizer)``.
            data (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Batch tuple used for interface consistency. The second element may be
                ``None`` for physics-informed workflows.

        Returns:
            jax.numpy.ndarray:
                Scalar batch loss value.

        Raises:
            RuntimeError:
                If loss evaluation fails.
        """
        nn_model, main_optimizer, latent_step_model, latent_optimizer = meta_state
        return self.ComputeBatchLossValue(data,(nn_model,latent_step_model))[0]

    @print_with_timestamp_and_execution_time
    @partial(nnx.jit, static_argnums=(0,), donate_argnums=1)
    def Predict(self,batch_X:jnp.ndarray):
        """
        Perform inference for a batch and apply Dirichlet boundary conditions.

        This method computes controlled variables from ``batch_X``, predicts the
        discretized field using latent-code adaptation with a learnable step size,
        and returns the full DOF vector with Dirichlet boundary values inserted
        consistently across the batch via :meth:`Loss.GetFullDofVector`.

        Args:
            batch_X (jax.numpy.ndarray):
                Batch of parametric inputs for inference.

        Returns:
            jax.numpy.ndarray:
                Batch of full discretized field vectors with Dirichlet boundary
                conditions applied.

        Raises:
            RuntimeError:
                If prediction fails due to incompatible model signatures or loss DOF
                handling.
        """
        control_outputs = self.control.ComputeBatchControlledVariables(batch_X)
        preds = self.ComputeBatchPredictions(batch_X,(self.flax_neural_network,self.latent_step_nnx_model))
        return self.loss_function.GetFullDofVector(control_outputs,preds.reshape(preds.shape[0], -1))

    def SaveCheckPoint(self,check_point_type,checkpoint_state_dir):
        """
        Save both the main network state and the latent-step state to disk.

        This method writes two checkpoint subdirectories under ``checkpoint_state_dir``.
        One subdirectory stores the main HyperNetwork state and the other stores the
        latent-step model state. This allows full restoration of meta-training and
        inference behavior, including the learned latent step size.

        Args:
            check_point_type (str):
                Label used for logging the checkpoint type, for example ``"best"`` or
                ``"latest"``.
            checkpoint_state_dir (str):
                Directory path where the checkpoint subdirectories are created.

        Returns:
            None

        Raises:
            OSError:
                If the checkpoint directories cannot be created or written.
            RuntimeError:
                If the underlying checkpointer fails to save the provided states.
        """

        nn_checkpoint_state_dir = checkpoint_state_dir + "/nn"
        absolute_path = os.path.abspath(nn_checkpoint_state_dir)
        self.checkpointer.save(absolute_path, nnx.state(self.flax_neural_network),force=True)

        latent_checkpoint_state_dir = checkpoint_state_dir + "/latent"
        absolute_path = os.path.abspath(latent_checkpoint_state_dir)
        self.checkpointer.save(absolute_path, nnx.state(self.latent_step_nnx_model),force=True)

        fol_info(f"{check_point_type} meta flax nnx state is saved to {checkpoint_state_dir}")

    def RestoreState(self,restore_state_directory:str):
        """
        Restore both the main network state and the latent-step state from disk.

        This method restores the HyperNetwork parameters from the ``nn`` subdirectory
        and restores the latent-step model parameter from the ``latent`` subdirectory.
        After restoration, the in-memory modules are updated in place.

        Args:
            restore_state_directory (str):
                Directory containing the saved meta checkpoint. The directory is
                expected to include ``nn`` and ``latent`` subdirectories.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If the expected checkpoint directories do not exist.
            RuntimeError:
                If the checkpointer fails to restore or update the module states.
        """

        # restore nn
        nn_restore_state_directory = restore_state_directory + "/nn"
        absolute_path = os.path.abspath(nn_restore_state_directory)
        nn_state = nnx.state(self.flax_neural_network)
        restored_state = self.checkpointer.restore(absolute_path, nn_state)
        nnx.update(self.flax_neural_network, restored_state)

        # restore latent
        latent_restore_state_directory = restore_state_directory + "/latent"
        absolute_path = os.path.abspath(latent_restore_state_directory)
        latent_state = nnx.state(self.latent_step_nnx_model)
        restored_state = self.checkpointer.restore(absolute_path, latent_state)
        nnx.update(self.latent_step_nnx_model, restored_state)

        fol_info(f"meta flax nnx state is restored from {restore_state_directory}")
