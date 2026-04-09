"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""
import os
from abc import ABC,abstractmethod
from typing import Tuple,Iterator
from tqdm import trange
import matplotlib.pyplot as plt
import jax.numpy as jnp
from functools import partial
from flax import nnx
import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import orbax.checkpoint as orbax
from optax import GradientTransformation
import orbax.checkpoint as ocp
from fol.loss_functions.loss import Loss
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *

class DeepNetwork(ABC):
    """
    Base abstract class for deep learning models.

    This class provides a common training/testing workflow for Flax/NNX neural
    networks optimized with Optax. It centralizes model initialization, JIT-compiled
    train/test steps, convergence checks, plotting of training curves, and Orbax
    checkpointing.

    Subclasses must implement :meth:`ComputeBatchLossValue` to define how a batch
    loss is computed (including any auxiliary loss terms and metrics) and
    :meth:`Finalize` to run any end-of-training procedures.

    The training loop performs:

    - Optional restoration of a previously saved model state.
    - Optional data/model sharding across devices.
    - Batched training with randomized batch indices.
    - Periodic evaluation on a test set.
    - Convergence checks based on configured criteria.
    - Optional checkpointing (least-loss, interval, and final state).
    - Optional plotting of loss histories.

    Args:
        name (str):
            Name identifier for the model instance. Used for logging and to
            organize checkpoint folders.
        loss_function (Loss):
            Loss function object that defines the optimization objective and
            provides any required initialization.
        flax_neural_network (nnx.Module):
            Flax/NNX module defining the neural network architecture.
        optax_optimizer (GradientTransformation):
            Optax optimizer transformation used to update the model parameters.

    Attributes:
        name (str):
            Model name used for identification and checkpointing.
        loss_function (Loss):
            Loss function used to compute objective values.
        flax_neural_network (nnx.Module):
            Flax/NNX neural network module.
        optax_optimizer (GradientTransformation):
            Optax optimizer transformation.
        initialized (bool):
            Indicates whether :meth:`Initialize` has been called.
        checkpointer (ocp.StandardCheckpointer):
            Orbax checkpointer used for saving/restoring model states.
        nnx_optimizer (nnx.Optimizer):
            NNX optimizer wrapper holding parameters and optimizer state.

    Notes:
        The subclass implementation of :meth:`ComputeBatchLossValue` must return
        a metrics dictionary containing the key ``"total_loss"``. This value is
        used for logging, convergence checks, plotting, and checkpointing.
    """
    default_convergence_settings = {"num_epochs":100,"convergence_criterion":"total_loss",
                                    "relative_error":1e-8,"absolute_error":1e-8}
    default_plot_settings = {"plot_list":["total_loss"],"plot_frequency":1,"save_frequency":100,"save_directory":"."}
    default_restore_nnx_state_settings = {"restore":False,"state_directory":"flax_state"}
    default_train_checkpoint_settings = {"least_loss_checkpointing":False,"least_loss":np.inf,"frequency":100,"state_directory":"flax_train_state"}
    default_test_checkpoint_settings = {"least_loss_checkpointing":False,"least_loss":np.inf,"frequency":100,"state_directory":"flax_test_state"}
    default_save_nnx_state_settings = {"save_final_state":True,"final_state_directory":"flax_final_state",
                                       "interval_state_checkpointing":False,"interval_state_checkpointing_frequency":0,"interval_state_checkpointing_directory":"."}
    default_data_model_sharding_settings = {"sharding":False,"num_data_devices":1,"num_nnx_model_devices":1}

    def __init__(self,
                 name:str,
                 loss_function:Loss,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation):
        self.name = name
        self.loss_function = loss_function
        self.flax_neural_network = flax_neural_network
        self.optax_optimizer = optax_optimizer
        self.initialized = False

    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize the loss function, checkpointer, and NNX optimizer wrapper.

        This method prepares the model for training and testing. It initializes
        ``self.loss_function`` (if needed or if ``reinitialize=True``), creates an
        Orbax ``StandardCheckpointer``, and constructs an ``nnx.Optimizer`` that couples
        the Flax/NNX model with the Optax optimizer transformation.

        Args:
            reinitialize (bool, optional):
                If ``True``, force reinitialization of the loss function even if it was
                previously initialized. Default is ``False``.

        Returns:
            None
        """

        # initialize inputs
        if not self.loss_function.initialized:
            self.loss_function.Initialize(reinitialize)

        # create orbax checkpointer
        self.checkpointer = ocp.StandardCheckpointer()

        # initialize the nnx optimizer
        self.nnx_optimizer = nnx.Optimizer(self.flax_neural_network, self.optax_optimizer, wrt=nnx.Param)

    def GetName(self) -> str:
        """
        Return the model name.

        Returns:
            str:
                The name identifier of this model instance.
        """
        return self.name

    @abstractmethod
    def ComputeBatchLossValue(self,batch_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Compute the loss for a single batch and return loss metrics.

        Subclasses must implement this method. It is used by both training and testing
        steps. Implementations are expected to perform a forward pass of ``nn_model``
        on the batch input, evaluate the configured loss function and any auxiliary
        loss terms, and return a scalar batch loss together with a metrics dictionary.

        The returned metrics dictionary must include the key ``"total_loss"`` because
        the training loop uses it for logging, convergence checks, plotting, and
        checkpoint decisions.

        Args:
            batch_set (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Batch tuple ``(inputs, targets)``. Exact shapes depend on the problem
                setup.
            nn_model (nnx.Module):
                The Flax/NNX model to evaluate.

        Returns:
            Tuple[jax.numpy.ndarray, dict]:
                A tuple ``(loss_value, metrics_dict)`` where ``loss_value`` is the scalar
                loss used for differentiation and optimization, and ``metrics_dict``
                contains batch-level metrics including the required key
                ``"total_loss"``.

        Raises:
            KeyError:
                If the returned metrics dictionary does not contain ``"total_loss"``.
        """
        pass

    @partial(nnx.jit, static_argnums=(0,))
    def TrainStep(self, state, data):
        """
        Execute one JIT-compiled training step.

        This step calls :meth:`ComputeBatchLossValue` with gradients enabled,
        applies the parameter update using the internal ``nnx.Optimizer``, and
        returns the scalar ``"total_loss"`` for logging.

        Args:
            state (tuple[nnx.Module, nnx.Optimizer]):
                Training state as ``(model, optimizer)``.
            data (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Batch tuple ``(inputs, targets)``.

        Returns:
            jax.numpy.ndarray:
                Scalar total loss for the batch (``metrics["total_loss"]``).
        """
        nn, opt = state
        (_,batch_dict), batch_grads = nnx.value_and_grad(self.ComputeBatchLossValue,argnums=1,has_aux=True) \
                                                                    (data,nn)
        opt.update(nn,batch_grads)
        return batch_dict["total_loss"]

    @partial(nnx.jit, static_argnums=(0,))
    def TrainStepMetrics(self, state, data):
        """
        Execute one JIT-compiled training step returning full metrics dict.

        Like :meth:`TrainStep` but returns the complete ``batch_dict`` from
        :meth:`ComputeBatchLossValue` instead of only ``"total_loss"``.

        Args:
            state (tuple[nnx.Module, nnx.Optimizer]):
                Training state as ``(model, optimizer)``.
            data (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Batch tuple ``(inputs, targets)``.

        Returns:
            dict:
                Full batch metrics dictionary from :meth:`ComputeBatchLossValue`.
        """
        nn, opt = state
        (_,batch_dict), batch_grads = nnx.value_and_grad(self.ComputeBatchLossValue,argnums=1,has_aux=True) \
                                                                    (data,nn)
        opt.update(nn,batch_grads)
        return batch_dict

    @partial(nnx.jit, static_argnums=(0,))
    def TestStep(self, state, data):
        """
        Execute one JIT-compiled evaluation (test) step.

        This step calls :meth:`ComputeBatchLossValue` without updating parameters
        and returns the scalar ``"total_loss"`` for logging.

        Args:
            state (tuple[nnx.Module, nnx.Optimizer]):
                State tuple ``(model, optimizer)``. The optimizer is unused but
                kept for interface consistency.
            data (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Batch tuple ``(inputs, targets)``.

        Returns:
            jax.numpy.ndarray:
                Scalar total loss for the batch (``metrics["total_loss"]``).
        """
        nn,_ = state
        (_,batch_dict) = self.ComputeBatchLossValue(data,nn)
        return batch_dict["total_loss"]

    @partial(nnx.jit, static_argnums=(0,))
    def TestStepMetrics(self, state, data):
        """
        Execute one JIT-compiled evaluation step returning full metrics dict.

        Like :meth:`TestStep` but returns the complete ``batch_dict`` from
        :meth:`ComputeBatchLossValue` instead of only ``"total_loss"``.

        Args:
            state (tuple[nnx.Module, nnx.Optimizer]):
                State tuple ``(model, optimizer)``.
            data (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Batch tuple ``(inputs, targets)``.

        Returns:
            dict:
                Full batch metrics dictionary from :meth:`ComputeBatchLossValue`.
        """
        nn,_ = state
        (_,batch_dict) = self.ComputeBatchLossValue(data,nn)
        return batch_dict

    def GetState(self):
        """
        Return the state tuple consumed by :meth:`TrainStep` and :meth:`TestStep`.

        Returns:
            tuple[nnx.Module, nnx.Optimizer]:
                A tuple ``(self.flax_neural_network, self.nnx_optimizer)``.
        """
        return (self.flax_neural_network, self.nnx_optimizer)

    @print_with_timestamp_and_execution_time
    def Train(self,
              train_set:Tuple[jnp.ndarray, jnp.ndarray],
              test_set:Tuple[jnp.ndarray, jnp.ndarray] = (jnp.array([]), jnp.array([])),
              test_frequency:int=100,
              batch_size:int=100,
              convergence_settings:dict={},
              plot_settings:dict={},
              restore_nnx_state_settings:dict={},
              train_checkpoint_settings:dict={},
              test_checkpoint_settings:dict={},
              save_nnx_state_settings:dict={},
              data_model_sharding_settings:dict={},
              working_directory='.',
              training_residual_tracker=None,
              jit_donate_argnums=(0, 1)):
        """
        Train the model using mini-batch optimization and optional evaluation.

        This method orchestrates the full training loop:
        - Applies default settings and merges user settings dictionaries,
        - Optionally restores a saved model state,
        - Optionally shards data/model across devices,
        - Runs multiple epochs of randomized mini-batch updates,
        - Periodically evaluates on ``test_set``,
        - Checks convergence criteria,
        - Optionally plots and saves loss histories,
        - Optionally saves checkpoints (least-loss, interval, and/or final state).

        Args:
            train_set (Tuple[jax.numpy.ndarray, jax.numpy.ndarray]):
                Training dataset tuple ``(inputs, targets)``.
            test_set (Tuple[jax.numpy.ndarray, jax.numpy.ndarray], optional):
                Test dataset tuple ``(inputs, targets)``. If empty arrays are
                provided, testing is skipped. Default is empty arrays.
            test_frequency (int, optional):
                Evaluate test loss every ``test_frequency`` epochs. Default is
                ``100``.
            batch_size (int, optional):
                Requested batch size. May be adjusted to evenly divide the
                training set size for batching/parallelization. Default is
                ``100``.
            convergence_settings (dict, optional):
                Convergence configuration (epochs, criterion name, tolerances).
                Missing keys are filled from defaults.
            plot_settings (dict, optional):
                Plot configuration (metrics list, plot frequency, save paths).
                Missing keys are filled from defaults.
            restore_nnx_state_settings (dict, optional):
                Restore configuration. If ``restore=True``, restores model state
                from ``state_directory``.
            train_checkpoint_settings (dict, optional):
                Checkpoint configuration for tracking least training loss.
            test_checkpoint_settings (dict, optional):
                Checkpoint configuration for tracking least test loss.
            save_nnx_state_settings (dict, optional):
                Final and interval checkpoint configuration.
            data_model_sharding_settings (dict, optional):
                Sharding configuration including device partitioning.
            working_directory (str, optional):
                Base directory used to store plots and checkpoints. Default is
                ``"."``.
            training_residual_tracker (object, optional):
                An optional tracker object with ``log_epoch(epoch, total_loss,
                residual_rms)`` and ``finalize()`` methods. When provided, the
                training loop logs ``residual_rms_batch_mean`` each epoch and
                calls ``finalize()`` after training. Default is ``None``.
            jit_donate_argnums (tuple, optional):
                Argument indices to donate in the JIT-compiled scan step.
                Default is ``(0, 1)``.

        Returns:
            None

        Raises:
            ValueError:
                If provided settings dictionaries are missing required keys after
                default merging (implementation-dependent).
        """
        # Robust dataset normalization
        def _normalize_dataset(ds):
            if ds is None:
                return (jnp.array([]), jnp.array([]))
            if isinstance(ds, (tuple, list)):
                if len(ds) == 2:
                    return (ds[0], ds[1])
                if len(ds) == 1:
                    X = ds[0]
                    Y = jnp.zeros((len(X), 1), dtype=X.dtype) if hasattr(X, "dtype") else jnp.zeros((len(X), 1))
                    return (X, Y)
                fol_error(f"dataset tuple/list must be length 1 or 2, got len={len(ds)}")
            X = ds
            Y = jnp.zeros((len(X), 1), dtype=X.dtype) if hasattr(X, "dtype") else jnp.zeros((len(X), 1))
            return (X, Y)

        train_set = _normalize_dataset(train_set)
        test_set = _normalize_dataset(test_set)

        def _has_test_data(ds):
            try:
                return (ds is not None) and (ds[0] is not None) and (getattr(ds[0], "size", 0) > 0) and (len(ds[0]) > 0)
            except Exception:
                return False

        has_test = _has_test_data(test_set)

        convergence_settings = UpdateDefaultDict(self.default_convergence_settings,convergence_settings)
        fol_info(f"convergence settings:{convergence_settings}")

        default_plot_settings = copy.deepcopy(self.default_plot_settings)
        default_plot_settings["save_directory"] = working_directory
        plot_settings = UpdateDefaultDict(default_plot_settings,plot_settings)
        plot_settings["test_frequency"] = test_frequency
        fol_info(f"plot settings:{plot_settings}")

        # ensure total_loss and convergence criterion are always tracked
        plot_list = list(plot_settings["plot_list"])
        if "total_loss" not in plot_list:
            plot_list.insert(0, "total_loss")
        crit = str(convergence_settings.get("convergence_criterion", "total_loss"))
        if crit not in plot_list:
            plot_list.append(crit)
        plot_settings["plot_list"] = plot_list

        fol_info(f"jit donate_argnums: {jit_donate_argnums}")

        default_restore_nnx_state_settings = copy.deepcopy(self.default_restore_nnx_state_settings)
        default_restore_nnx_state_settings["state_directory"] = working_directory + "/" + default_restore_nnx_state_settings["state_directory"]
        restore_nnx_state_settings = UpdateDefaultDict(default_restore_nnx_state_settings,restore_nnx_state_settings)
        fol_info(f"restore settings:{restore_nnx_state_settings}")

        default_train_checkpoint_settings = copy.deepcopy(self.default_train_checkpoint_settings)
        default_train_checkpoint_settings["state_directory"] = working_directory + "/" + default_train_checkpoint_settings["state_directory"]
        train_checkpoint_settings = UpdateDefaultDict(default_train_checkpoint_settings,train_checkpoint_settings)
        fol_info(f"train checkpoint settings:{train_checkpoint_settings}")

        default_test_checkpoint_settings = copy.deepcopy(self.default_test_checkpoint_settings)
        default_test_checkpoint_settings["state_directory"] = working_directory + "/" + default_test_checkpoint_settings["state_directory"]
        test_checkpoint_settings = UpdateDefaultDict(default_test_checkpoint_settings,test_checkpoint_settings)
        fol_info(f"test checkpoint settings:{test_checkpoint_settings}")

        default_save_nnx_state_settings = copy.deepcopy(self.default_save_nnx_state_settings)
        default_save_nnx_state_settings["final_state_directory"] = working_directory + "/" + default_save_nnx_state_settings["final_state_directory"]
        default_save_nnx_state_settings["interval_state_checkpointing_directory"] = working_directory + "/" + default_save_nnx_state_settings["interval_state_checkpointing_directory"]
        save_nnx_state_settings = UpdateDefaultDict(default_save_nnx_state_settings,save_nnx_state_settings)
        fol_info(f"save nnx state settings:{save_nnx_state_settings}")

        sharding_settings = UpdateDefaultDict(self.default_data_model_sharding_settings,data_model_sharding_settings)
        fol_info(f"sharding settings:{sharding_settings}")

        # restore state if needed
        if restore_nnx_state_settings['restore']:
            self.RestoreState(restore_nnx_state_settings["state_directory"])

        # adjust batch for parallization reasons
        adjusted_batch_size = next(i for i in range(batch_size, 0, -1) if len(train_set[0]) % i == 0)
        if adjusted_batch_size!=batch_size:
            fol_info(f"for the parallelization of batching, the batch size is changed from {batch_size} to {adjusted_batch_size}")
            batch_size = adjusted_batch_size

        # sharding & data-model parallelization
        if sharding_settings["sharding"]:
            num_data_devices = sharding_settings["num_data_devices"]
            num_model_devices = sharding_settings["num_nnx_model_devices"]
            if num_data_devices * num_model_devices != jax.local_device_count():
                fol_error(
                    f"number of available devices (i.e., {jax.local_device_count()}) does not match with "
                    f"num_data_devices*num_model_devices (i.e., {(num_data_devices, num_model_devices)}) !"
                )

            if len(train_set[0]) % num_data_devices != 0:
                fol_error(
                    f"size/shape of train_set (i.e., {train_set[0].shape}) is not a multiplier of data devices "
                    f"(i.e.,{num_data_devices}) for sharding !"
                )

            if has_test:
                if len(test_set[0]) % num_data_devices != 0:
                    fol_error(
                        f"size/shape of test_set (i.e., {test_set[0].shape}) is not a multiplier of data devices "
                        f"(i.e.,{num_data_devices}) for sharding !"
                    )

            sharding_mesh = jax.sharding.Mesh(
                devices=np.array(jax.devices()).reshape(num_data_devices, num_model_devices),
                axis_names=("data", "model"),
            )

            data_sharding = jax.NamedSharding(sharding_mesh, jax.sharding.PartitionSpec("data"))

            # data sharding (train/test are pytrees)
            train_set = jax.tree_util.tree_map(lambda x: jax.device_put(x, data_sharding), train_set)
            if has_test:
                test_set = jax.tree_util.tree_map(lambda x: jax.device_put(x, data_sharding), test_set)

            # nnx model sharding
            with sharding_mesh:
                state = nnx.state(self.flax_neural_network)
                pspecs = nnx.get_partition_spec(state)
                sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
                nnx.update(self.flax_neural_network, sharded_state)

            try:
                fol_info("train set is sharded as ")
                jax.debug.visualize_array_sharding(train_set[0])
                if has_test:
                    fol_info("test set is sharded as ")
                    jax.debug.visualize_array_sharding(test_set[0])
            except Exception:
                pass

        def train_loop():
            train_history_dict = {name: [] for name in plot_list}
            test_history_dict = {name: [] for name in plot_list}

            pbar = trange(convergence_settings["num_epochs"])
            converged = False
            rng, _ = jax.random.split(jax.random.PRNGKey(0))

            state = self.GetState()

            # Helpers: reduce per-batch metric -> epoch metric
            def _reduce_metric(name: str, x):
                if x is None:
                    return None
                x = jnp.asarray(x)
                if x.ndim == 0:
                    return x
                if x.ndim > 1:
                    within_axes = tuple(range(1, x.ndim))
                    per_batch = jnp.mean(x, axis=within_axes)
                else:
                    per_batch = x
                lname = str(name).lower()
                if ("_min" in lname) or ("batch_min" in lname):
                    return jnp.min(per_batch)
                if ("_max" in lname) or ("batch_max" in lname):
                    return jnp.max(per_batch)
                return jnp.mean(per_batch)

            # Train: scan over minibatches, returns stacked metrics dict
            train_multiple_steps_with_idxs = nnx.jit(
                lambda st, dat, idxs: nnx.scan(
                    lambda st2, idxs2: (
                        st2,
                        self.TrainStepMetrics(st2, jax.tree_util.tree_map(lambda a: a[idxs2], dat)),
                    )
                )(st, idxs),
                donate_argnums=jit_donate_argnums,
            )

            # Eval: scan over minibatches, returns stacked metrics dict
            eval_multiple_steps_with_idxs = nnx.jit(
                lambda st, dat, idxs: nnx.scan(
                    lambda st2, idxs2: (
                        st2,
                        self.TestStepMetrics(st2, jax.tree_util.tree_map(lambda a: a[idxs2], dat)),
                    )
                )(st, idxs)
            )

            # Ensure test batching is also divisible
            test_batch_size = batch_size
            if has_test and (len(test_set[0]) % test_batch_size != 0):
                test_batch_size = next(i for i in range(test_batch_size, 0, -1) if len(test_set[0]) % i == 0)
                fol_info(
                    f"for the parallelization of batching (test), the batch size is changed from "
                    f"{batch_size} to {test_batch_size}"
                )

            for epoch in pbar:
                # update least values in case of restore
                if restore_nnx_state_settings["restore"] and epoch == 0:
                    if train_checkpoint_settings["least_loss_checkpointing"]:
                        idxs = jnp.arange(len(train_set[0])).reshape(-1, batch_size)
                        state, m = eval_multiple_steps_with_idxs(state, train_set, idxs)
                        train_checkpoint_settings["least_loss"] = float(_reduce_metric("total_loss", m["total_loss"]))
                    if has_test and test_checkpoint_settings["least_loss_checkpointing"]:
                        idxs = jnp.arange(len(test_set[0])).reshape(-1, test_batch_size)
                        state, m = eval_multiple_steps_with_idxs(state, test_set, idxs)
                        test_checkpoint_settings["least_loss"] = float(_reduce_metric("total_loss", m["total_loss"]))

                # parallel batching and train step
                rng, sub = jax.random.split(rng)
                order = jax.random.permutation(sub, len(train_set[0])).reshape(-1, batch_size)
                state, train_batch_metrics = train_multiple_steps_with_idxs(state, train_set, order)

                # Minibatch-aggregated epoch metrics (train)
                epoch_train_total = None
                epoch_residual_mean = None

                if epoch == 0:
                    if crit not in train_batch_metrics:
                        fol_error(
                            f"convergence_criterion='{crit}' not produced by ComputeBatchLossValue(). "
                            f"Available metrics: {list(train_batch_metrics.keys())}"
                        )

                for name in plot_list:
                    if name not in train_batch_metrics:
                        if name == crit:
                            fol_error(
                                f"convergence_criterion='{crit}' missing in train metrics at epoch={epoch}. "
                                f"Available metrics: {list(train_batch_metrics.keys())}"
                            )
                        train_history_dict[name].append(float("nan"))
                        continue

                    v = _reduce_metric(name, train_batch_metrics[name])
                    v_f = float(v)

                    if name == crit and np.isnan(v_f):
                        fol_error(
                            f"convergence_criterion='{crit}' evaluated to NaN at epoch={epoch}. "
                            f"Check ComputeBatchLossValue() numerics."
                        )

                    train_history_dict[name].append(v_f)

                    if name == "total_loss":
                        epoch_train_total = v_f
                    if name == "residual_rms_batch_mean":
                        epoch_residual_mean = v_f

                if epoch_train_total is None:
                    fol_error("Metric 'total_loss' missing from train metrics; ComputeBatchLossValue must return it.")

                if training_residual_tracker is not None and epoch_residual_mean is not None:
                    training_residual_tracker.log_epoch(int(epoch), float(epoch_train_total), float(epoch_residual_mean))

                # test step
                did_test = False
                if has_test and ((epoch) % test_frequency == 0 or epoch == convergence_settings["num_epochs"] - 1):
                    did_test = True
                    idxs = jnp.arange(len(test_set[0])).reshape(-1, test_batch_size)
                    state, test_batch_metrics = eval_multiple_steps_with_idxs(state, test_set, idxs)

                    for name in plot_list:
                        if name not in test_batch_metrics:
                            test_history_dict[name].append(float("nan"))
                            continue
                        v = _reduce_metric(name, test_batch_metrics[name])
                        test_history_dict[name].append(float(v))

                # print step
                if has_test:
                    test_last = (
                        test_history_dict["total_loss"][-1] if len(test_history_dict["total_loss"]) > 0 else float("nan")
                    )
                    print_dict = {"train_loss": train_history_dict["total_loss"][-1], "test_loss": test_last}
                else:
                    print_dict = {"train_loss": train_history_dict["total_loss"][-1]}

                pbar.set_postfix(print_dict)

                # check converged
                converged = self.CheckConvergence(train_history_dict, convergence_settings)

                # plot the histories
                if (epoch > 0 and epoch % plot_settings["save_frequency"] == 0) or converged:
                    self.PlotHistoryDict(plot_settings, train_history_dict, test_history_dict)

                # train checkpointing
                if (
                    train_checkpoint_settings["least_loss_checkpointing"]
                    and epoch > 0
                    and (epoch) % train_checkpoint_settings["frequency"] == 0
                    and train_history_dict["total_loss"][-1] < train_checkpoint_settings["least_loss"]
                ):
                    fol_info(
                        f"train total_loss improved from {train_checkpoint_settings['least_loss']} "
                        f"to {train_history_dict['total_loss'][-1]}"
                    )
                    train_checkpoint_settings["least_loss"] = train_history_dict["total_loss"][-1]
                    self.SaveCheckPoint("train", train_checkpoint_settings["state_directory"])

                # test checkpointing
                if (
                    has_test
                    and test_checkpoint_settings["least_loss_checkpointing"]
                    and did_test
                    and epoch > 0
                    and (epoch) % test_checkpoint_settings["frequency"] == 0
                    and test_history_dict["total_loss"][-1] < test_checkpoint_settings["least_loss"]
                ):
                    fol_info(
                        f"test total_loss improved from {test_checkpoint_settings['least_loss']} "
                        f"to {test_history_dict['total_loss'][-1]}"
                    )
                    test_checkpoint_settings["least_loss"] = test_history_dict["total_loss"][-1]
                    self.SaveCheckPoint("test", test_checkpoint_settings["state_directory"])

                # interval checkpointing
                if (
                    save_nnx_state_settings["interval_state_checkpointing"]
                    and epoch > 0
                    and (epoch) % save_nnx_state_settings["interval_state_checkpointing_frequency"] == 0
                ):
                    self.SaveCheckPoint(
                        f"interval {epoch}",
                        save_nnx_state_settings["interval_state_checkpointing_directory"]
                        + "/flax_train_state_epoch_"
                        + str(epoch),
                    )

                if epoch < convergence_settings["num_epochs"] - 1 and converged:
                    break

            # final best-checkpoint saves
            if (
                train_checkpoint_settings["least_loss_checkpointing"]
                and train_history_dict["total_loss"][-1] < train_checkpoint_settings["least_loss"]
            ):
                fol_info(
                    f"train total_loss improved from {train_checkpoint_settings['least_loss']} "
                    f"to {train_history_dict['total_loss'][-1]}"
                )
                self.SaveCheckPoint("train", train_checkpoint_settings["state_directory"])

            if (
                has_test
                and test_checkpoint_settings["least_loss_checkpointing"]
                and test_history_dict["total_loss"][-1] < test_checkpoint_settings["least_loss"]
            ):
                fol_info(
                    f"test total_loss improved from {test_checkpoint_settings['least_loss']} "
                    f"to {test_history_dict['total_loss'][-1]}"
                )
                self.SaveCheckPoint("test", test_checkpoint_settings["state_directory"])

            if save_nnx_state_settings["save_final_state"]:
                self.SaveCheckPoint("final", save_nnx_state_settings["final_state_directory"])

            self.checkpointer.close()

        if sharding_settings["sharding"]:
            with sharding_mesh:
                train_loop()
        else:
            train_loop()

        if training_residual_tracker is not None:
            training_residual_tracker.finalize()

    def CheckConvergence(self,train_history_dict:dict,convergence_settings:dict):
        """
        Determine whether training has converged based on loss history.

        Convergence is checked using:
        - Absolute threshold on the most recent value of the configured criterion,
        - Relative change between the last two criterion values,
        - Or reaching the configured maximum number of epochs.

        Args:
            train_history_dict (dict):
                Dictionary of training histories. Must contain the key specified
                by ``convergence_settings["convergence_criterion"]`` whose value
                is a list of scalar history values.
            convergence_settings (dict):
                Dictionary containing:
                - ``"convergence_criterion"`` (str): key in ``train_history_dict``
                - ``"absolute_error"`` (float): absolute threshold
                - ``"relative_error"`` (float): relative change threshold
                - ``"num_epochs"`` (int): maximum epochs

        Returns:
            bool:
                ``True`` if the convergence conditions are met, otherwise
                ``False``.

        Raises:
            KeyError:
                If the configured convergence criterion is missing from
                ``train_history_dict``.
        """
        convergence_criterion = convergence_settings["convergence_criterion"]
        absolute_error = convergence_settings["absolute_error"]
        relative_error = convergence_settings["relative_error"]
        num_epochs = convergence_settings["num_epochs"]
        current_epoch = len(train_history_dict[convergence_criterion])
        # check for absolute and relative errors and convergence
        if abs(train_history_dict[convergence_criterion][-1])<absolute_error:
            return True
        if current_epoch>1:
            if abs(train_history_dict[convergence_criterion][-1] -
                   train_history_dict[convergence_criterion][-2])<relative_error:
                return True
            elif current_epoch>=num_epochs:
                return True
            else:
                return False
        else:
            return False

    def RestoreState(self,restore_state_directory:str):
        """
        Restore the model state from a checkpoint directory.

        This loads a previously saved NNX state using Orbax and updates the
        in-memory Flax/NNX model parameters accordingly.

        Args:
            restore_state_directory (str):
                Directory containing the saved Orbax checkpoint state.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If the checkpoint directory does not exist (may be raised by the
                checkpointer backend).
            ValueError:
                If the restored state is incompatible with the current model
                structure.
        """

        absolute_path = os.path.abspath(restore_state_directory)
        # get the state
        nn_state = nnx.state(self.flax_neural_network)
        # restore
        restored_state = self.checkpointer.restore(absolute_path, nn_state)
        # now update the model with the loaded state
        nnx.update(self.flax_neural_network, restored_state)
        fol_info(f"flax nnx state is restored from {restore_state_directory}")

    def SaveCheckPoint(self,check_point_type,checkpoint_state_dir):
        """
        Save the current model state to a checkpoint directory.

        This writes the current NNX model state (parameters and related state)
        using Orbax to the provided directory and forces the write.

        Args:
            check_point_type (str):
                Human-readable label for logging (e.g., ``"train"``, ``"test"``,
                ``"final"``, or ``"interval <epoch>"``).
            checkpoint_state_dir (str):
                Directory where the checkpoint will be saved.

        Returns:
            None

        Raises:
            OSError:
                If the directory cannot be created or written to.
        """

        absolute_path = os.path.abspath(checkpoint_state_dir)
        self.checkpointer.save(absolute_path, nnx.state(self.flax_neural_network),force=True)
        fol_info(f"{check_point_type} flax nnx state is saved to {checkpoint_state_dir}")

    def PlotHistoryDict(self,plot_settings:dict,train_history_dict:dict,test_history_dict:dict):
        """
        Plot and save training/test history curves.

        This method creates a semilog-y plot for selected metrics from
        ``train_history_dict`` and ``test_history_dict`` and saves the figure as
        ``training_history.png`` in ``plot_settings["save_directory"]``.

        Args:
            plot_settings (dict):
                Dictionary containing plot configuration. Expected entries include
                ``plot_frequency``, ``plot_list``, ``save_directory``, and
                ``test_frequency``.
            train_history_dict (dict):
                Training history mapping metric names to lists of recorded values.
            test_history_dict (dict):
                Test history mapping metric names to lists of recorded values.

        Returns:
            None

        Raises:
            KeyError:
                If required entries are missing from ``plot_settings``.
        """
        plot_rate = plot_settings["plot_frequency"]
        plot_list = plot_settings["plot_list"]

        plt.figure(figsize=(10, 5))
        train_max_length = 0

        for key,value in train_history_dict.items():
            if len(value)>0 and (len(plot_list)==0 or key in plot_list):
                train_max_length = max(train_max_length, len(value))
                plt.semilogy(value[::plot_rate], label=f"train_{key}")

        for key,value in test_history_dict.items():
            if len(value)>0 and (len(plot_list)==0 or key in plot_list):
                test_length = len(value)
                x_value = [i * plot_settings["test_frequency"] for i in range(max(test_length-1, 0))]
                x_value.append(train_max_length-1 if train_max_length > 0 else 0)
                plt.semilogy(x_value[::plot_rate], value[::plot_rate], label=f"test_{key}")

        plt.title("Training History")
        plt.xlabel(str(plot_rate) + " Epoch")
        plt.ylabel("Log Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_settings["save_directory"],"training_history.png"), bbox_inches='tight')
        plt.close()

    @abstractmethod
    def Finalize(self) -> None:
        """
        Finalize the model after training.

        Subclasses implement this hook to perform any final steps that must occur
        once at the end of training (for example, releasing resources, computing
        final diagnostics, exporting artifacts, or post-processing).

        Returns:
            None
        """
        pass





