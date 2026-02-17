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

class FourierParametricOperatorLearning(DeepNetwork):
    """
    Parametric operator learning using a Fourier Neural Operator (FNO).

    This class implements parametric operator learning where the neural model is
    a Fourier Neural Operator operating on discretized fields defined on a
    structured grid. Parametric inputs are first mapped to controlled variables
    using a :class:`Control` object and then reshaped into grid-aligned tensor
    representations suitable for FNO evaluation.

    In this formulation, both training and inference assume a fixed grid
    resolution associated with the finite-element mesh used by the loss
    function. The FNO maps grid-aligned input channels to grid-aligned output
    fields in the spectral domain, after which the outputs are flattened and
    embedded into full DOF vectors with boundary conditions applied by the loss
    function.

    This class provides the common infrastructure for Fourier Neural Operator–
    based parametric operator learning. Concrete training paradigms are defined
    in derived classes, such as data-driven and physics-informed variants.

    Args:
        name (str):
            Name identifier for the model instance, used for logging and
            checkpointing.
        control (Control):
            Control object defining the parametric input space and mapping raw
            parameters to grid-aligned controlled variables.
        loss_function (Loss):
            Loss function defining the discretized field representation, DOF
            structure, mesh topology, and boundary-condition handling.
        flax_neural_network (nnx.Module):
            Fourier Neural Operator mapping grid-aligned inputs to grid-aligned
            output fields.
        optax_optimizer (GradientTransformation):
            Optax optimizer transformation used for training.

    Raises:
        RuntimeError:
            If the control output or loss mesh is incompatible with the assumed
            structured grid required by the Fourier Neural Operator.
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
        Initialize the Fourier parametric operator learning model.

        This method initializes the base :class:`DeepNetwork` components and ensures
        that the associated :class:`Control` object is also initialized. No additional
        dimensional consistency checks are performed here beyond those enforced by
        the base class and the loss function.

        Args:
            reinitialize (bool, optional):
                If ``True``, force reinitialization even if the instance was already
                initialized. Default is ``False``.

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

        total_num_nodes = self.loss_function.fe_mesh.GetNumberOfNodes()
        if self.loss_function.dim == 1:
            self.spatial_shape = (int(total_num_nodes))
        elif self.loss_function.dim == 2:
            dim_mesh_size = jnp.sqrt(total_num_nodes)
            self.spatial_shape = (int(dim_mesh_size),int(dim_mesh_size))
        elif self.loss_function.dim == 3:
            dim_mesh_size = jnp.cbrt(total_num_nodes)
            self.spatial_shape = (int(dim_mesh_size),int(dim_mesh_size),int(dim_mesh_size))

        self.initialized = True

    def ComputeBatchPredictions(self,batch_X:jnp.ndarray,nn_model:nnx.Module):
        """
        Compute grid-based field predictions for a batch of parametric inputs.

        This method reshapes the controlled parametric inputs into a structured
        grid format compatible with Fourier-based neural operators. The neural
        network is then evaluated on the grid-aligned inputs, and the resulting
        grid-aligned outputs are flattened into vectors of unknown DOFs.

        Args:
            batch_X (jax.numpy.ndarray):
                Batch of controlled parametric inputs. The array is expected to
                encode grid-aligned channels for each sample.
            nn_model (nnx.Module):
                Fourier-based neural operator evaluated on reshaped grid inputs.

        Returns:
            jax.numpy.ndarray:
                Batch of flattened field predictions corresponding to the unknown
                degrees of freedom.

        Raises:
            ValueError:
                If the input cannot be reshaped consistently with the mesh size
                inferred from the loss function.
        """
        batch_size = batch_X.shape[0]
        return nn_model(batch_X.reshape(batch_size,*self.spatial_shape,-1)).reshape(batch_size,-1)

    @print_with_timestamp_and_execution_time
    def Predict(self,batch_control:jnp.ndarray):
        """
        Perform inference and apply boundary conditions for a batch of inputs.

        This method computes controlled variables from the raw parametric inputs,
        evaluates the Fourier-based neural operator to predict discretized fields,
        and applies Dirichlet boundary conditions by inserting prescribed values
        across the batch using the loss function.

        Args:
            batch_control (jax.numpy.ndarray):
                Batch of parametric inputs used for inference.

        Returns:
            jax.numpy.ndarray:
                Batch of full discretized field vectors with boundary conditions
                applied.

        Raises:
            None
        """
        control_outputs = self.control.ComputeBatchControlledVariables(batch_control)
        preds = self.ComputeBatchPredictions(control_outputs,self.flax_neural_network)
        return self.loss_function.GetFullDofVector(batch_control,preds)

    def Finalize(self):
        pass

class DataDrivenFourierParametricOperatorLearning(FourierParametricOperatorLearning):
    """
    Data-driven parametric operator learning using a Fourier Neural Operator.

    This class specializes
    :class:`FourierParametricOperatorLearning` for fully supervised,
    data-driven training. The Fourier Neural Operator is trained by directly
    comparing predicted discretized fields against provided target fields using
    a supervised loss function.

    No physical constraints are enforced explicitly in the loss; instead, the
    operator is learned purely from input–output field data. This formulation is
    appropriate when high-fidelity simulation or experimental data are
    available.

    Args:
        name (str):
            Name identifier for the model instance, used for logging and
            checkpointing.
        control (Control):
            Control object defining the parametric input space and mapping raw
            parameters to controlled variables arranged as grid-aligned inputs.
        loss_function (Loss):
            Supervised loss function used to compare predicted discretized fields
            against target fields.
        flax_neural_network (nnx.Module):
            Fourier Neural Operator mapping grid-aligned inputs to grid-aligned
            output fields.
        optax_optimizer (GradientTransformation):
            Optax optimizer transformation used for training.

    Raises:
        None
    """
    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Compute supervised batch loss for data-driven Fourier Neural Operator training.

        The batch consists of parametric inputs and corresponding target discretized
        fields. Predictions are computed using the Fourier Neural Operator and are
        compared directly against the target fields using the provided loss
        function.

        Args:
            batch (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Tuple ``(batch_X, batch_Y)`` where ``batch_X`` contains parametric
                inputs and ``batch_Y`` contains target discretized fields.
            nn_model (nnx.Module):
                Fourier Neural Operator used to generate predictions.

        Returns:
            Tuple[jax.numpy.ndarray, dict]:
                Batch loss value and a dictionary of loss statistics including
                the key ``"total_loss"``.

        Raises:
            None
        """
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        batch_predictions = self.ComputeBatchPredictions(control_outputs,nn_model)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(batch[1],batch_predictions)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})

class PhysicsInformedFourierParametricOperatorLearning(FourierParametricOperatorLearning):
    """
    Physics-informed parametric operator learning using a Fourier Neural Operator.

    This class specializes :class:`FourierParametricOperatorLearning` for unsupervised
    or physics-informed training. The Fourier Neural Operator predicts discretized
    fields that are evaluated using physics-based loss functionals, such as residual-
    or energy-based formulations, rather than supervised target data.

    This formulation is suitable when governing equations are known and labeled
    training data are limited or unavailable. Boundary conditions are enforced
    explicitly through the loss function.

    Args:
        name (str):
            Name identifier for the model instance, used for logging and
            checkpointing.
        control (Control):
            Control object defining the parametric input space and mapping raw
            parameters to controlled variables.
        loss_function (Loss):
            Physics-based loss function evaluated on predicted discretized fields.
        flax_neural_network (nnx.Module):
            Fourier Neural Operator mapping grid-aligned inputs to grid-aligned
            output fields.
        optax_optimizer (GradientTransformation):
            Optax optimizer transformation used for training.

    Raises:
        None
    """
    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Compute physics-informed batch loss for Fourier Neural Operator learning.

        The batch is provided as a tuple for interface consistency, but only the
        parametric inputs are used. Predicted discretized fields are evaluated using
        a physics-based loss functional without supervised target fields.

        Args:
            batch (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Tuple ``(batch_X, None)`` where ``batch_X`` contains parametric
                inputs. The second entry is unused.
            nn_model (nnx.Module):
                Fourier Neural Operator used to generate predictions.

        Returns:
            Tuple[jax.numpy.ndarray, dict]:
                Batch loss value and a dictionary of loss statistics including
                the key ``"total_loss"``.

        Raises:
            None
        """
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        batch_predictions = self.ComputeBatchPredictions(control_outputs,nn_model)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(control_outputs,batch_predictions)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})