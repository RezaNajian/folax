"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: December, 2024
 License: FOL/LICENSE
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from functools import partial
from optax import GradientTransformation
from flax import nnx
from .implicit_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *
from .nns import HyperNetwork

class MetaImplicitParametricOperatorLearning(ImplicitParametricOperatorLearning):
    """
    Meta-learning extension of implicit parametric operator learning with latent-code adaptation.

    This class derives from :class:`~fol.deep_neural_networks.implicit_parametric_operator_learning.ImplicitParametricOperatorLearning`
    and adds a meta-learning style inner-loop optimization over latent codes. In the
    base implicit formulation, the parametric input directly conditions a
    coordinate-based neural field (typically via a modulator–synthesizer structure).
    In this meta formulation, the parametric input is first mapped to controlled
    variables, and then a latent code is optimized per sample to minimize the
    physics-based loss. The optimized latent code is subsequently used to evaluate
    the neural field on the mesh coordinates.

    The practical difference is that the conditioning signal is not applied only by
    feeding the parametric input into the network. Instead, the model performs
    test-time (or per-batch) adaptation by iteratively updating latent variables
    using gradients of the physics loss. This enables fast adaptation to new
    parameter regimes or tasks while keeping the main network weights fixed during
    the inner loop.

    The provided ``flax_neural_network`` is expected to be a
    :class:`~fol.deep_neural_networks.nns.HyperNetwork`, where the synthesizer
    represents a coordinate-based MLP neural field evaluated on FE mesh node
    coordinates and the modulator provides conditioning according to supported
    coupling modes. Training and loss evaluation are typically performed on a
    fixed FE mesh, but the coordinate-based synthesizer formulation makes
    multi-resolution inference (and, in principle, multi-resolution training)
    possible by evaluating the conditioned neural field on alternative coordinate
    sets.

    Args:
        name (str):
            Name identifier for the model instance (used for logging and
            checkpointing).
        control (Control):
            Control object defining the parametric input space and mapping raw
            parameters to controlled variables used by the loss. Typical inputs are
            low-dimensional parameterization features such as Fourier coefficients
            or other control points.
        loss_function (Loss):
            Physics-based loss function evaluated on predicted discretized fields.
            The loss defines the FE mesh, DOF structure, and boundary-condition
            handling required for training and inference.
        flax_neural_network (HyperNetwork):
            Hypernetwork-based model used for implicit evaluation. The network is
            evaluated as ``nn_model(latent_codes, coords)``, where ``latent_codes``
            are adapted by the inner loop and ``coords`` are FE mesh node
            coordinates. The synthesizer component performs the coordinate-based
            neural-field mapping and the modulator provides conditioning according
            to the hypernetwork coupling settings.
        main_loop_optax_optimizer (GradientTransformation):
            Optax optimizer transformation used for the outer (main) optimization
            loop over network parameters.
        latent_step_size (float, optional):
            Step size used for gradient-based latent-code updates in the inner loop.
            Default is ``1e-2``.
        num_latent_iterations (int, optional):
            Number of inner-loop latent optimization iterations performed per batch.
            Default is ``3``.

    Raises:
        RuntimeError:
            If the provided neural network does not expose required attributes
            (for example ``in_features``) needed to size the latent codes, or if
            the network interface is incompatible with ``nn_model(latent, coords)``.
    """

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:HyperNetwork,
                 main_loop_optax_optimizer:GradientTransformation,
                 latent_step_size:float=1e-2,
                 num_latent_iterations:int=3):
        super().__init__(name,control,loss_function,flax_neural_network,
                         main_loop_optax_optimizer)

        self.latent_step = latent_step_size
        self.num_latent_iterations = num_latent_iterations

    def ComputeBatchPredictions(self,batch_X:jnp.ndarray,nn_model:nnx.Module):
        """
        Compute batch predictions using latent-code inner-loop optimization.

        This method overrides the base implicit prediction behavior by introducing a
        per-sample latent adaptation step. For each parametric input in ``batch_X``,
        the method initializes a latent code and iteratively updates it using the
        gradient of the physics-based loss with respect to the latent variables. The
        neural network is evaluated on the FE mesh node coordinates at each iteration
        to compute the loss, and the latent code is updated using a normalized gradient
        step. After ``num_latent_iterations`` updates, the final latent codes are used
        to produce the discretized field predictions on the mesh.

        Compared to :meth:`~fol.deep_neural_networks.implicit_parametric_operator_learning.ImplicitParametricOperatorLearning.ComputeBatchPredictions`,
        where parametric inputs directly condition the neural field through the network
        interface, this meta formulation conditions the neural field indirectly through
        optimized latent variables that minimize the physics loss for the given control
        outputs.

        Args:
            batch_X (jax.numpy.ndarray):
                Batch of parametric inputs. The leading dimension corresponds to the
                batch size. These inputs are transformed into controlled variables by
                ``self.control`` for physics-based loss evaluation.
            nn_model (nnx.Module):
                Neural model evaluated as ``nn_model(latent_codes, coords)``, where
                ``latent_codes`` are adapted in the inner loop and ``coords`` are FE
                mesh node coordinates obtained from the loss function mesh. The model
                must expose ``in_features`` to define the latent code dimension.

        Returns:
            jax.numpy.ndarray:
                Batch of discretized field predictions evaluated on the FE mesh node
                coordinates using the optimized latent codes.

        Raises:
            RuntimeError:
                If ``nn_model`` does not expose ``in_features`` required to size the
                latent-code array, or if the model cannot be called as
                ``nn_model(latent_codes, coords)``.
            ValueError:
                If ``self.num_latent_iterations`` is negative or if the latent-gradient
                normalization encounters invalid norms.
        """
        latent_codes = jnp.zeros((batch_X.shape[0],nn_model.in_features))
        control_outputs = self.control.ComputeBatchControlledVariables(batch_X)

        def latent_loss(latent_code,control_output):
            nn_output = nn_model(latent_code[None, :],self.loss_function.fe_mesh.GetNodesCoordinates())
            return self.loss_function.ComputeBatchLoss(control_output,nn_output)[0]

        vec_grad_func = jax.vmap(jax.grad(latent_loss, argnums=0))
        for _ in range(self.num_latent_iterations):
            grads = vec_grad_func(latent_codes,control_outputs)
            grads_norms =  jnp.linalg.norm(grads, axis=1, keepdims=True)
            norm_grads = grads/grads_norms
            latent_codes -= self.latent_step * norm_grads

        return nn_model(latent_codes,self.loss_function.fe_mesh.GetNodesCoordinates())

