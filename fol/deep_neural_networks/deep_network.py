"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""
import os
import copy
import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple
from tqdm import trange

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx

import orbax.checkpoint as ocp
from optax import GradientTransformation

from fol.loss_functions.loss import Loss
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *


class DeepNetwork(ABC):
    """
    Base abstract class for deep learning models.

    Validation notes (comments only):
      - To confirm no full-batch metric sweeps remain in the epoch loop:
          grep -n "ComputeBatchLossValue(train_set" -n fol/deep_neural_networks/deep_network.py
          grep -n "ComputeBatchLossValue(test_set"  -n fol/deep_neural_networks/deep_network.py
        These should NOT appear inside the epoch loop.

      - Residual-based stopping requires ComputeBatchLossValue() to return a batch_dict key
        matching convergence_settings["convergence_criterion"], e.g. "residual_rms_batch_mean".
    """

    default_convergence_settings = {
        "num_epochs": 100,
        "convergence_criterion": "total_loss",
        "relative_error": 1e-8,
        "absolute_error": 1e-8,
    }
    default_plot_settings = {
        "plot_list": ["total_loss"],
        "plot_frequency": 1,
        "save_frequency": 100,
        "save_directory": ".",
    }
    default_restore_nnx_state_settings = {"restore": False, "state_directory": "flax_state"}
    default_train_checkpoint_settings = {
        "least_loss_checkpointing": False,
        "least_loss": np.inf,
        "frequency": 100,
        "state_directory": "flax_train_state",
    }
    default_test_checkpoint_settings = {
        "least_loss_checkpointing": False,
        "least_loss": np.inf,
        "frequency": 100,
        "state_directory": "flax_test_state",
    }
    default_save_nnx_state_settings = {
        "save_final_state": True,
        "final_state_directory": "flax_final_state",
        "interval_state_checkpointing": False,
        "interval_state_checkpointing_frequency": 0,
        "interval_state_checkpointing_directory": ".",
    }
    default_data_model_sharding_settings = {"sharding": False, "num_data_devices": 1, "num_nnx_model_devices": 1}

    def __init__(
        self,
        name: str,
        loss_function: Loss,
        flax_neural_network: nnx.Module,
        optax_optimizer: GradientTransformation,
    ):
        self.name = name
        self.loss_function = loss_function
        self.flax_neural_network = flax_neural_network
        self.optax_optimizer = optax_optimizer
        self.initialized = False

    def Initialize(self, reinitialize: bool = False) -> None:
        # initialize inputs
        if not self.loss_function.initialized:
            self.loss_function.Initialize(reinitialize)

        # create orbax checkpointer
        self.checkpointer = ocp.StandardCheckpointer()

        # initialize the nnx optimizer
        self.nnx_optimizer = nnx.Optimizer(self.flax_neural_network, self.optax_optimizer, wrt=nnx.Param)

    def GetName(self) -> str:
        return self.name

    @abstractmethod
    def ComputeBatchLossValue(self, batch_set: Tuple[jnp.ndarray, jnp.ndarray], nn_model: nnx.Module):
        pass

    @partial(nnx.jit, static_argnums=(0,))
    def TrainStep(self, state, data):
        nn, opt = state
        (_, batch_dict), batch_grads = nnx.value_and_grad(self.ComputeBatchLossValue, argnums=1, has_aux=True)(
            data, nn
        )
        opt.update(nn, batch_grads)
        return batch_dict["total_loss"]

    # ------------------------------------------------------------
    # ✅ Train step that returns full metrics dict (not just loss)
    # ------------------------------------------------------------
    @partial(nnx.jit, static_argnums=(0,))
    def TrainStepMetrics(self, state, data):
        nn, opt = state
        (_, batch_dict), batch_grads = nnx.value_and_grad(self.ComputeBatchLossValue, argnums=1, has_aux=True)(
            data, nn
        )
        opt.update(nn, batch_grads)
        return batch_dict

    @partial(nnx.jit, static_argnums=(0,))
    def TestStep(self, state, data):
        nn, _ = state
        (_, batch_dict) = self.ComputeBatchLossValue(data, nn)
        return batch_dict["total_loss"]

    # ------------------------------------------------------------
    # ✅ Test step that returns full metrics dict (not just loss)
    # ------------------------------------------------------------
    @partial(nnx.jit, static_argnums=(0,))
    def TestStepMetrics(self, state, data):
        nn, _ = state
        (_, batch_dict) = self.ComputeBatchLossValue(data, nn)
        return batch_dict

    def GetState(self):
        return (self.flax_neural_network, self.nnx_optimizer)

    @print_with_timestamp_and_execution_time
    def Train(
        self,
        train_set: Tuple[jnp.ndarray, jnp.ndarray],
        test_set: Tuple[jnp.ndarray, jnp.ndarray] = (jnp.array([]), jnp.array([])),
        test_frequency: int = 100,
        batch_size: int = 100,
        convergence_settings: dict = {},
        plot_settings: dict = {},
        restore_nnx_state_settings: dict = {},
        train_checkpoint_settings: dict = {},
        test_checkpoint_settings: dict = {},
        save_nnx_state_settings: dict = {},
        data_model_sharding_settings: dict = {},
        working_directory=".",
        training_residual_tracker=None,  # <<< NEW
        jit_donate_argnums=(0, 1),  # <<< NEW
    ):

        # ------------------------------------------------------------
        # ✅ Robust dataset normalization:
        # Accepts: X  OR (X,) OR (X,Y). Always returns (X,Y).
        # This prevents "tuple has no attribute shape" bugs downstream.
        # ------------------------------------------------------------
        def _normalize_dataset(ds):
            if ds is None:
                return (jnp.array([]), jnp.array([]))

            # ds is (X,Y) or (X,)
            if isinstance(ds, (tuple, list)):
                if len(ds) == 2:
                    X, Y = ds
                    return (X, Y)
                if len(ds) == 1:
                    X = ds[0]
                    Y = jnp.zeros((len(X), 1), dtype=X.dtype) if hasattr(X, "dtype") else jnp.zeros((len(X), 1))
                    return (X, Y)
                fol_error(f"dataset tuple/list must be length 1 or 2, got len={len(ds)}")

            # ds is X only
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

        convergence_settings = UpdateDefaultDict(self.default_convergence_settings, convergence_settings)
        fol_info(f"convergence settings:{convergence_settings}")

        default_plot_settings = copy.deepcopy(self.default_plot_settings)
        default_plot_settings["save_directory"] = working_directory
        plot_settings = UpdateDefaultDict(default_plot_settings, plot_settings)
        plot_settings["test_frequency"] = test_frequency
        fol_info(f"plot settings:{plot_settings}")

        # make sure total_loss always tracked
        plot_list = list(plot_settings["plot_list"])
        if "total_loss" not in plot_list:
            plot_list.insert(0, "total_loss")

        # ✅ ALSO ensure convergence criterion is tracked every epoch
        crit = str(convergence_settings.get("convergence_criterion", "total_loss"))
        if crit not in plot_list:
            plot_list.append(crit)
        plot_settings["plot_list"] = plot_list

        fol_info(f"jit donate_argnums: {jit_donate_argnums}")

        default_restore_nnx_state_settings = copy.deepcopy(self.default_restore_nnx_state_settings)
        default_restore_nnx_state_settings["state_directory"] = (
            working_directory + "/" + default_restore_nnx_state_settings["state_directory"]
        )
        restore_nnx_state_settings = UpdateDefaultDict(default_restore_nnx_state_settings, restore_nnx_state_settings)
        fol_info(f"restore settings:{restore_nnx_state_settings}")

        default_train_checkpoint_settings = copy.deepcopy(self.default_train_checkpoint_settings)
        default_train_checkpoint_settings["state_directory"] = (
            working_directory + "/" + default_train_checkpoint_settings["state_directory"]
        )
        train_checkpoint_settings = UpdateDefaultDict(default_train_checkpoint_settings, train_checkpoint_settings)
        fol_info(f"train checkpoint settings:{train_checkpoint_settings}")

        default_test_checkpoint_settings = copy.deepcopy(self.default_test_checkpoint_settings)
        default_test_checkpoint_settings["state_directory"] = (
            working_directory + "/" + default_test_checkpoint_settings["state_directory"]
        )
        test_checkpoint_settings = UpdateDefaultDict(default_test_checkpoint_settings, test_checkpoint_settings)
        fol_info(f"test checkpoint settings:{test_checkpoint_settings}")

        default_save_nnx_state_settings = copy.deepcopy(self.default_save_nnx_state_settings)
        default_save_nnx_state_settings["final_state_directory"] = (
            working_directory + "/" + default_save_nnx_state_settings["final_state_directory"]
        )
        default_save_nnx_state_settings["interval_state_checkpointing_directory"] = (
            working_directory + "/" + default_save_nnx_state_settings["interval_state_checkpointing_directory"]
        )
        save_nnx_state_settings = UpdateDefaultDict(default_save_nnx_state_settings, save_nnx_state_settings)
        fol_info(f"save nnx state settings:{save_nnx_state_settings}")

        sharding_settings = UpdateDefaultDict(self.default_data_model_sharding_settings, data_model_sharding_settings)
        fol_info(f"sharding settings:{sharding_settings}")

        # restore state if needed
        if restore_nnx_state_settings["restore"]:
            self.RestoreState(restore_nnx_state_settings["state_directory"])

        # adjust batch for parallelization reasons
        adjusted_batch_size = next(i for i in range(batch_size, 0, -1) if len(train_set[0]) % i == 0)
        if adjusted_batch_size != batch_size:
            fol_info(
                f"for the parallelization of batching, the batch size is changed from {batch_size} to {adjusted_batch_size}"
            )
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

            # optional debug visualize (DON'T assume model internals)
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

            # ---------------------------------------------
            # Helpers: reduce per-batch metric -> epoch metric
            # ---------------------------------------------
            def _reduce_metric(name: str, x):
                """
                x is stacked over batches by nnx.scan.
                Typical: (num_batches,) for scalar metrics.
                Possible: (num_batches, k, ...) for vector/matrix metrics.

                We first reduce within-batch dims -> per-batch scalar, then reduce across batches:
                  - *_min / batch_min -> min over batches
                  - *_max / batch_max -> max over batches
                  - otherwise         -> mean over batches
                """
                if x is None:
                    return None
                x = jnp.asarray(x)
                if x.ndim == 0:
                    return x  # already scalar

                # reduce within-batch dims to get (num_batches,) scalars (if needed)
                if x.ndim > 1:
                    within_axes = tuple(range(1, x.ndim))
                    per_batch = jnp.mean(x, axis=within_axes)
                else:
                    per_batch = x  # (num_batches,)

                lname = str(name).lower()
                if ("_min" in lname) or ("batch_min" in lname):
                    return jnp.min(per_batch)
                if ("_max" in lname) or ("batch_max" in lname):
                    return jnp.max(per_batch)
                return jnp.mean(per_batch)

            # -----------------------------
            # Train: scan over minibatches
            # returns stacked metrics dict
            # -----------------------------
            train_multiple_steps_with_idxs = nnx.jit(
                lambda st, dat, idxs: nnx.scan(
                    lambda st2, idxs2: (
                        st2,
                        self.TrainStepMetrics(st2, jax.tree_util.tree_map(lambda a: a[idxs2], dat)),
                    )
                )(st, idxs),
                donate_argnums=jit_donate_argnums,
            )

            # -----------------------------
            # Eval: scan over minibatches
            # returns stacked metrics dict
            # -----------------------------
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
                    # ✅ init least_loss via minibatch eval (no full-batch call)
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

                # IMPORTANT: because donate_argnums donates state buffers, always rebind returned state.
                state, train_batch_metrics = train_multiple_steps_with_idxs(state, train_set, order)

                # ✅ minibatch-aggregated epoch metrics (train)
                epoch_train_total = None
                epoch_residual_mean = None

                # sanity: ensure convergence criterion exists in metrics, otherwise convergence will be bogus
                if epoch == 0:
                    if crit not in train_batch_metrics:
                        fol_error(
                            f"convergence_criterion='{crit}' not produced by ComputeBatchLossValue(). "
                            f"Available metrics: {list(train_batch_metrics.keys())}"
                        )

                for name in plot_list:
                    if name not in train_batch_metrics:
                        # convergence criterion must never be missing
                        if name == crit:
                            fol_error(
                                f"convergence_criterion='{crit}' missing in train metrics at epoch={epoch}. "
                                f"Available metrics: {list(train_batch_metrics.keys())}"
                            )
                        # keep history length consistent; also avoids KeyError in plotting
                        train_history_dict[name].append(float("nan"))
                        continue

                    v = _reduce_metric(name, train_batch_metrics[name])
                    v_f = float(v)

                    # convergence criterion should not be NaN (fail loudly)
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
                    # show latest available test_loss (from last test eval)
                    test_last = (
                        test_history_dict["total_loss"][-1] if len(test_history_dict["total_loss"]) > 0 else float("nan")
                    )
                    print_dict = {"train_loss": train_history_dict["total_loss"][-1], "test_loss": test_last}
                else:
                    print_dict = {"train_loss": train_history_dict["total_loss"][-1]}
                pbar.set_postfix(print_dict)

                # check converged
                converged = self.CheckConvergence(train_history_dict, convergence_settings)

                # plot histories
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

    def CheckConvergence(self, train_history_dict: dict, convergence_settings: dict):
        convergence_criterion = convergence_settings["convergence_criterion"]
        absolute_error = convergence_settings["absolute_error"]
        relative_error = convergence_settings["relative_error"]
        num_epochs = convergence_settings["num_epochs"]
        current_epoch = len(train_history_dict[convergence_criterion])

        if abs(train_history_dict[convergence_criterion][-1]) < absolute_error:
            return True

        if current_epoch > 1:
            if abs(train_history_dict[convergence_criterion][-1] - train_history_dict[convergence_criterion][-2]) < relative_error:
                return True
            elif current_epoch >= num_epochs:
                return True
            else:
                return False
        else:
            return False

    def RestoreState(self, restore_state_directory: str):
        absolute_path = os.path.abspath(restore_state_directory)
        nn_state = nnx.state(self.flax_neural_network)
        restored_state = self.checkpointer.restore(absolute_path, nn_state)
        nnx.update(self.flax_neural_network, restored_state)
        fol_info(f"flax nnx state is restored from {restore_state_directory}")

    def SaveCheckPoint(self, check_point_type, checkpoint_state_dir):
        absolute_path = os.path.abspath(checkpoint_state_dir)
        self.checkpointer.save(absolute_path, nnx.state(self.flax_neural_network), force=True)
        fol_info(f"{check_point_type} flax nnx state is saved to {checkpoint_state_dir}")

    def PlotHistoryDict(self, plot_settings: dict, train_history_dict: dict, test_history_dict: dict):
        plot_rate = plot_settings["plot_frequency"]
        plot_list = plot_settings["plot_list"]

        plt.figure(figsize=(10, 5))
        train_max_length = 0

        for key, value in train_history_dict.items():
            if len(value) > 0 and (len(plot_list) == 0 or key in plot_list):
                train_max_length = max(train_max_length, len(value))
                plt.semilogy(value[::plot_rate], label=f"train_{key}")

        for key, value in test_history_dict.items():
            if len(value) > 0 and (len(plot_list) == 0 or key in plot_list):
                test_length = len(value)
                x_value = [i * plot_settings["test_frequency"] for i in range(max(test_length - 1, 0))]
                x_value.append(train_max_length - 1 if train_max_length > 0 else 0)

                # ✅ keep x and y sampling consistent with plot_rate
                plt.semilogy(x_value[::plot_rate], value[::plot_rate], label=f"test_{key}")

        plt.title("Training History")
        plt.xlabel(str(plot_rate) + " Epoch")
        plt.ylabel("Log Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_settings["save_directory"], "training_history.png"), bbox_inches="tight")
        plt.close()

    @abstractmethod
    def Finalize(self) -> None:
        pass