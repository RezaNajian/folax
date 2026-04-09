"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: November, 2024
 License: FOL/LICENSE
"""
import jax.numpy as jnp
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh
from jax import jit,grad
from functools import partial
from  .loss import Loss

class RegressionLoss(Loss):
    """
    Regression loss for supervised learning on mesh nodal quantities.

    This class implements a mean-squared-error (MSE) regression loss between
    ground-truth values and predicted values. It is designed to integrate with
    the FoLax loss interface and to work with a finite element mesh context,
    where nodal unknowns (DOFs) define the structure of the regression targets.

    The loss is computed over batches of samples by flattening the input arrays
    to ``(batch_size, -1)`` and evaluating the element-wise squared error. The
    returned scalar is the mean of the squared error over the full batch and
    all output components. A simple error summary (min, max, mean) is also
    returned for monitoring.

    Args:
        name (str):
            Name identifier for the loss instance.
        loss_settings (dict):
            Configuration dictionary. Must include ``"nodal_unknows"`` defining
            the nodal degrees of freedom used by this loss (key spelling
            preserved from the current implementation).
        fe_mesh (Mesh):
            Finite element mesh associated with the nodal unknown structure.

    Attributes:
        loss_settings (dict):
            Loss configuration dictionary.
        fe_mesh (Mesh):
            Finite element mesh used for sizing/indexing.
        dofs:
            Nodal degrees of freedom as provided by ``loss_settings["nodal_unknows"]``.
        non_dirichlet_indices (jax.numpy.ndarray):
            Indices of DOFs treated as trainable/unknown in this loss. For this
            regression loss, these are initialized as a full range over all
            nodal DOFs.
    """

    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh) -> None:
        """
        Create a regression loss instance.

        Args:
            name (str):
                Name identifier for the loss instance.
            loss_settings (dict):
                Configuration dictionary containing ``"nodal_unknows"``.
            fe_mesh (Mesh):
                Finite element mesh associated with the regression targets.
        """
        super().__init__(name)
        self.loss_settings = loss_settings
        self.fe_mesh = fe_mesh
        self.dofs = self.loss_settings["nodal_unknows"]

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize indices and internal state for loss evaluation.

        This method sets up indexing for nodal DOFs based on the number of mesh
        nodes and the configured DOFs. If the instance is already initialized
        and ``reinitialize`` is ``False``, the method returns without modifying
        the current configuration.

        Args:
            reinitialize (bool, optional):
                If ``True``, forces reinitialization even if the object is
                already initialized. Default is ``False``.
        """

        if self.initialized and not reinitialize:
            return

        self.non_dirichlet_indices = jnp.arange(len(self.dofs)*self.fe_mesh.GetNumberOfNodes())

        self.initialized = True

    def GetFullDofVector(self,known_dofs: jnp.array,unknown_dofs: jnp.array) -> jnp.array:
        return unknown_dofs

    def GetNumberOfUnknowns(self) -> int:
        pass

    # @print_with_timestamp_and_execution_time
    def ComputeBatchLoss(self,gt_values:jnp.array,pred_values:jnp.array) -> tuple[float, tuple[float, float, float]]:
        """
        Compute batch regression loss and error summary.

        The batch loss is computed as the mean squared error between ground
        truth and predictions. Inputs are converted to at least 2D arrays and
        flattened to shape ``(batch_size, -1)`` prior to evaluation.

        Args:
            gt_values (jax.numpy.ndarray):
                Ground-truth values. Any additional trailing dimensions are
                flattened per sample.
            pred_values (jax.numpy.ndarray):
                Predicted values. Must be broadcast-compatible with
                ``gt_values`` after flattening.

        Returns:
            Tuple[float, Tuple[float, float, float]]:
                - Mean squared error over the batch and all components.
                - Error summary ``(min_error, max_error, mean_error)`` computed
                  from the element-wise squared errors.
        """
        gt_values = jnp.atleast_2d(gt_values)
        gt_values = gt_values.reshape(gt_values.shape[0], -1)
        pred_values = jnp.atleast_2d(pred_values)
        pred_values = pred_values.reshape(pred_values.shape[0], -1)
        err = (gt_values-pred_values)**2
        return jnp.mean(err),(jnp.min(err),jnp.max(err),jnp.mean(err))

    def Finalize(self) -> None:
        """
        Finalizes the loss computation for the training process.

        This method performs any necessary cleanup or final adjustments to the loss
        at the end of the training process. It is intended to be called only once
        after all training iterations are completed.

        """

        pass



