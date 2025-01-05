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
import orbax.checkpoint as orbax
from optax import GradientTransformation
import orbax.checkpoint as ocp
from fol.loss_functions.loss import Loss
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *

class DeepNetwork(ABC):
    """
    Base abstract class for deep learning models.

    This class serves as a foundation for deep neural networks. It provides a 
    structure to initialize essential components such as the network, optimizer, 
    loss function, and checkpoint settings. The class is abstract and intended 
    to be extended by specific model implementations.

    Attributes:
    ----------
    name : str
        The name of the model, used for identification and checkpointing.
    loss_function : Loss
        The loss function that the model will optimize during training. 
        It defines the objective that the network is learning to minimize.
    flax_neural_network : nnx.Module
        The Flax neural network module that defines the model's architecture.
    optax_optimizer : GradientTransformation
        The Optax optimizer used to update model parameters during training.
    checkpoint_settings : dict, optional
        Dictionary that stores settings for saving and restoring checkpoints. 
        Defaults to an empty dictionary.

    """

    def __init__(self,
                 name:str,
                 loss_function:Loss,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation,
                 checkpoint_settings:dict={},
                 working_directory='.'):
        self.name = name
        self.loss_function = loss_function
        self.flax_neural_network = flax_neural_network
        self.optax_optimizer = optax_optimizer
        self.checkpoint_settings = checkpoint_settings
        self.working_directory = working_directory
        self.initialized = False
        self.default_checkpoint_settings = {"restore_state":False,
                                            "state_directory":'./flax_state'}

    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize the deep learning model, its components, and checkpoint settings.

        This method handles the initialization of essential components for the deep network. 
        It ensures that the loss function is initialized, sets up checkpointing 
        for saving and restoring model states, and manages reinitialization if needed. 
        The function is responsible for restoring the model's state from a previous checkpoint, 
        if specified in the checkpoint settings.

        Attributes:
        ----------
        reinitialize : bool, optional
            If True, forces reinitialization of the model and its components even if 
            they have been initialized previously. Default is False.

        Raises:
        -------
        AssertionError:
            If the restored neural network state does not match the current state 
            (based on a comparison using `np.testing.assert_array_equal`).
        """

        # initialize inputs
        if not self.loss_function.initialized:
            self.loss_function.Initialize(reinitialize)

        # create orbax checkpointer
        self.checkpointer = ocp.StandardCheckpointer()

        self.checkpoint_settings = UpdateDefaultDict(self.default_checkpoint_settings,
                                                     self.checkpoint_settings)
        
        # restore flax nn.Module from the file
        if self.checkpoint_settings["restore_state"]:
            self.RestoreCheckPoint(self.checkpoint_settings)

        # initialize the nnx optimizer
        self.nnx_optimizer = nnx.Optimizer(self.flax_neural_network, self.optax_optimizer)

    def CreateBatches(self,data: Tuple[jnp.ndarray, jnp.ndarray], batch_size: int) -> Iterator[jnp.ndarray]:
        """
        Creates batches from the input dataset.

        This method splits the input data into batches of a specified size, 
        yielding input data and optionally target labels if provided.

        Parameters
        ----------
        data : Tuple[jnp.ndarray, jnp.ndarray]
            A tuple of input data and target labels.
        batch_size : int
            The number of samples per batch.

        Yields
        ------
        Iterator[jnp.ndarray]
            Batches of input data and optionally target labels.
        """

        # Unpack data into data_x and data_y
        if len(data) > 1:
            data_x, data_y = data  
            if data_x.shape[0] != data_y.shape[0]:
                fol_error("data_x and data_y must have the same number of samples.")
        else:
            data_x = data[0]

        # Iterate over the dataset and yield batches of data_x and data_y
        for i in range(0, data_x.shape[0], batch_size):
            batch_x = data_x[i:i+batch_size, :]
            if len(data) > 1:
                batch_y = data_y[i:i+batch_size, :]
                yield batch_x, batch_y
            else:
                yield batch_x,

    def GetName(self) -> str:
        """
        Returns the name of the model.

        Returns
        -------
        str
            The name of the deep learning model.
        """
        return self.name
    
    @abstractmethod
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Computes the loss value for a single data point.

        This method computes the network's output for a single input data point, 
        applies the control parameters, and evaluates the loss function.

        Parameters
        ----------
        x_set : Tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing the input data and corresponding target labels.
        nn_model : nnx.Module
            The Flax neural network model.

        Returns
        -------
        jnp.ndarray
            The loss value for the single data point.
        """
        pass
    
    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLossValue(self,batch_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Computes the loss values for a batch of data.

        This method computes the network's output for a batch of input data, applies the control parameters,
        and evaluates the loss function for the entire batch. It aggregates the results and returns
        summary statistics (min, max, avg) for the batch losses.

        Parameters
        ----------
        batch_set : Tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing a batch of input data and corresponding target labels.
        nn_model : nnx.Module
            The Flax neural network model.

        Returns
        -------
        Tuple[jnp.ndarray, dict]
            The mean loss for the batch and a dictionary of loss statistics (min, max, avg, total).
        """

        batch_losses,(batch_mins,batch_maxs,batch_avgs) = jax.vmap(self.ComputeSingleLossValue,(0,None))(batch_set,nn_model)
        loss_name = self.loss_function.GetName()
        total_mean_loss = jnp.mean(batch_losses)
        return total_mean_loss, ({loss_name+"_min":jnp.min(batch_mins),
                                         loss_name+"_max":jnp.max(batch_maxs),
                                         loss_name+"_avg":jnp.mean(batch_avgs),
                                         "total_loss":total_mean_loss})

    @partial(jax.jit, static_argnums=(0,))
    def TrainStep(self, nnx_graphdef:nnx.GraphDef, nxx_state:nnx.GraphState, 
                        train_batch:Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Performs a single training step.

        This method computes the loss for a batch of data, calculates gradients, and updates 
        the model's parameters using the provided optimizer.

        Parameters
        ----------
        nn_model : nnx.Module
            The Flax neural network model.
        optimizer : nnx.Optimizer
            The flax optimizer to apply the gradients to the model.
        batch_set : Tuple[jnp.ndarray, jnp.ndarray]
            A batch of input data and corresponding target labels.

        Returns
        -------
        dict
            A dictionary containing information about the training step, such as loss values.
        """

        nnx_model, nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)

        (batch_loss, batch_dict), batch_grads = nnx.value_and_grad(self.ComputeBatchLossValue,argnums=1,has_aux=True) \
                                                                    (train_batch,nnx_model)
        nnx_optimizer.update(batch_grads)
        _, new_state = nnx.split((nnx_model, nnx_optimizer))
        return batch_dict,new_state
    
    @partial(jax.jit, static_argnums=(0,))
    def TestStep(self, nnx_graphdef:nnx.GraphDef, nxx_state:nnx.GraphState,
                       test_batch:Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Performs a single test step.

        This method evaluates the model's performance on a test batch by computing the loss and
        returning additional metrics without updating the model parameters.

        Parameters
        ----------
        nnx_graphdef : nnx.GraphDef
            The neural network graph definition, specifying the model architecture and configurations.
        nxx_state : nnx.GraphState
            The state of the neural network, including parameters and optimizer state.
        test_batch : Tuple[jnp.ndarray, jnp.ndarray]
            A batch of input data and corresponding target labels for testing.

        Returns
        -------
        tuple
            A tuple containing:
            - test_batch_dict (dict): A dictionary with information about the test step, such as loss values
            and other computed metrics.
            - state (nnx.GraphState): The updated state of the model after processing the test batch.
        """

        nnx_model, nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)
        (test_loss, test_batch_dict) = self.ComputeBatchLossValue(test_batch,nnx_model)
        _, state = nnx.split((nnx_model, nnx_optimizer))
        return test_batch_dict,state

    @print_with_timestamp_and_execution_time
    def Train(self, train_set:Tuple[jnp.ndarray, jnp.ndarray], test_set:Tuple[jnp.ndarray, jnp.ndarray] = (jnp.array([]), jnp.array([])), 
              batch_size:int=100, test_settings:dict={},convergence_settings:dict={}, plot_settings:dict={}, save_settings:dict={}):

        """
        Trains the neural network model over multiple epochs with batch processing, testing, and optional model checkpointing.

        This method optimizes the model's parameters using gradient descent on the provided training dataset, evaluates 
        performance on a test set (if provided), tracks training and testing metrics, and saves intermediate states 
        and results as specified in the input settings. It also supports convergence checking and history plotting 
        during training.

        Parameters
        ----------
        train_set : Tuple[jnp.ndarray, jnp.ndarray]
            Training dataset consisting of input data and corresponding target labels.
        test_set : Tuple[jnp.ndarray, jnp.ndarray], optional
            Test dataset for validation. Defaults to empty arrays if no test set is provided.
        batch_size : int, optional
            Number of samples per batch for training. Default is 100.
        test_settings : dict, optional
            Configuration for test evaluation during training. Default is an empty dictionary. 
            Supported keys include:
            - `test_frequency` (int): Frequency (in epochs) for test evaluation. Default is 100.
        convergence_settings : dict, optional
            Settings to control the convergence criteria. Default is an empty dictionary.
            Default values include:
            - `num_epochs` (int): Maximum number of training epochs. Default is 100.
            - `convergence_criterion` (str): Metric to check for convergence, e.g., "total_loss". Default is "total_loss".
            - `relative_error` (float): Relative error threshold for convergence. Default is 1e-8.
            - `absolute_error` (float): Absolute error threshold for convergence. Default is 1e-8.
        plot_settings : dict, optional
            Configuration for plotting training and testing histories. Default is an empty dictionary.
            Default values include:
            - `plot_list` (list): Metrics to plot, e.g., ["total_loss"]. Default is ["total_loss"].
            - `plot_rate` (int): Frequency (in epochs) for plotting updates. Default is 1.
            - `plot_save_rate` (int): Frequency (in epochs) for saving plots. Default is 100.
        save_settings : dict, optional
            Configuration for saving the trained model and checkpoints. Default is an empty dictionary.
            Default values include:
            - `save_nn_model` (bool): Whether to save the model after training. Default is True.
            - `best_model_checkpointing` (bool): Whether to save the model when performance improves. Default is False.
            - `best_model_checkpointing_frequency` (int): Frequency (in epochs) for checkpointing the best model. Default is 100.

        Returns
        -------
        None
            The training process updates the model's state in place, saves the training history, and optionally 
            saves the model and its checkpoints to disk.
        """

        self.default_convergence_settings = {"num_epochs":100,"convergence_criterion":"total_loss",
                                             "relative_error":1e-8,"absolute_error":1e-8}
        convergence_settings = UpdateDefaultDict(self.default_convergence_settings,convergence_settings)
        
        self.default_plot_settings = {"plot_list":["total_loss"],"plot_rate":1,"plot_save_rate":100}
        plot_settings = UpdateDefaultDict(self.default_plot_settings,plot_settings)

        self.default_save_settings = {"save_nn_model":True,
                                      "best_model_checkpointing_frequency":100,
                                      "best_model_checkpointing":False}
        save_settings = UpdateDefaultDict(self.default_save_settings,save_settings)
        save_settings["best_loss"] = np.inf

        self.default_test_settings = {"test_frequency":100}
        test_settings = UpdateDefaultDict(self.default_test_settings,test_settings)
        plot_settings["test_frequency"] = test_settings["test_frequency"]

        def update_batch_history_dict(batches_hist_dict,batch_dict,batch_index):
            # fill the batch dict
            if batch_index == 0:
                for key, value in batch_dict.items():
                    batches_hist_dict[key] = [value]
            else:
                for key, value in batch_dict.items():
                    batches_hist_dict[key].append(value)

            return batches_hist_dict
        
        def update_history_dict(hist_dict,batch_hist_dict):
            for key, value in batch_hist_dict.items():
                if "max" in key:
                    batch_hist_dict[key] = [max(value)]
                elif "min" in key:
                    batch_hist_dict[key] = [min(value)]
                elif "avg" in key:
                    batch_hist_dict[key] = [sum(value)/len(value)]
                elif "total" in key:
                    batch_hist_dict[key] = [sum(value)]

            if len(hist_dict.keys())==0:
                hist_dict = batch_hist_dict
            else:
                for key, value in batch_hist_dict.items():
                    hist_dict[key].extend(value)

            return hist_dict

        train_history_dict = {}
        test_history_dict = {}
        pbar = trange(convergence_settings["num_epochs"])
        converged = False

        # here split according to https://github.com/google/flax/discussions/4224
        nnx_graphdef, nxx_state = nnx.split((self.flax_neural_network, self.nnx_optimizer))

        for epoch in pbar:
            train_set_hist_dict = {}
            test_set_hist_dict = {}
            # now loop over batches
            batch_index = 0 
            for batch_set in self.CreateBatches(train_set, batch_size):
                batch_dict,nxx_state = self.TrainStep(nnx_graphdef, nxx_state,batch_set)
                train_set_hist_dict = update_batch_history_dict(train_set_hist_dict,batch_dict,batch_index)                
                batch_index += 1
            
            if len(test_set[0])>0 and ((epoch)%test_settings["test_frequency"]==0 or epoch==convergence_settings["num_epochs"]-1):
                test_batch_dict,nxx_state = self.TestStep(nnx_graphdef,nxx_state,test_set)
                test_set_hist_dict = update_batch_history_dict(test_set_hist_dict,test_batch_dict,0)

            train_history_dict = update_history_dict(train_history_dict,train_set_hist_dict)
            print_dict = {"train_loss":train_history_dict["total_loss"][-1]}
            if len(test_set[0])>0:
                if ((epoch)%test_settings["test_frequency"]==0 or epoch==convergence_settings["num_epochs"]-1):
                    test_history_dict = update_history_dict(test_history_dict,test_set_hist_dict)
                print_dict = {"train_loss":train_history_dict["total_loss"][-1],
                              "test_loss":test_history_dict["total_loss"][-1]}

            pbar.set_postfix(print_dict)

            # check converged
            converged = self.CheckConvergence(train_history_dict,convergence_settings)

            # plot the histories
            if (epoch>0 and epoch %plot_settings["plot_save_rate"] == 0) or converged:
                self.PlotHistoryDict(plot_settings,train_history_dict,test_history_dict)

            # save checkpoint
            if save_settings["best_model_checkpointing"] and epoch>0 and \
                (epoch)%save_settings["best_model_checkpointing_frequency"] == 0 and \
                train_history_dict["total_loss"][-1] < save_settings["best_loss"]:
                fol_info(f"total_loss improved from {save_settings['best_loss']} to {train_history_dict['total_loss'][-1]}")
                save_settings["best_loss"] = train_history_dict["total_loss"][-1]
                # merge before saving
                self.flax_neural_network, self.nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)
                self.SaveCheckPoint()
                # split again
                nnx_graphdef, nxx_state = nnx.split((self.flax_neural_network, self.nnx_optimizer))

            if epoch<convergence_settings["num_epochs"]-1 and converged:
                break    

        # now we need to merge the model again
        self.flax_neural_network, self.nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)        

        if save_settings["best_model_checkpointing"] and \
            train_history_dict["total_loss"][-1] < save_settings['best_loss']:
            fol_info(f"total_loss improved from {save_settings['best_loss']} to {train_history_dict['total_loss'][-1]}")
            self.SaveCheckPoint()
        elif not save_settings["best_model_checkpointing"] and save_settings["save_nn_model"]:
            self.SaveCheckPoint()

    def CheckConvergence(self,train_history_dict:dict,convergence_settings:dict):
        """
        Checks whether the training process has converged.

        This method evaluates the training history based on the defined convergence 
        criterion, absolute error, or relative error. If the conditions are met, 
        it returns True, indicating convergence.

        Parameters
        ----------
        train_history_dict : dict
            The history of the training loss values.
        convergence_settings : dict
            The settings that define when convergence occurs, including absolute error 
            and relative error thresholds.

        Returns
        -------
        bool
            True if the model has converged, False otherwise.
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

    def RestoreCheckPoint(self,checkpoint_settings:dict):
        """
        Restores the state of the neural network from a saved checkpoint.

        This method retrieves the saved state of the neural network from a specified directory and updates the model 
        to reflect the restored state.

        Parameters
        ----------
        checkpoint_settings : dict
            A dictionary containing the settings for checkpoint restoration.
            Expected keys:
            - `state_directory` (str): The directory path where the checkpoint is saved.

        Returns
        -------
        None
            The neural network's state is restored and updated in place. A message is logged to confirm the restoration process.

        Notes
        -----
        - Ensure the `state_directory` key is included in the `checkpoint_settings` dictionary, and the specified directory exists.
        - This method uses `nnx.state` to retrieve the current state of the model and updates it with the restored state.
        - Logs the restoration process using `fol_info`.
        """
        if "state_directory" in checkpoint_settings.keys():
            state_directory = checkpoint_settings["state_directory"]
            absolute_path = os.path.abspath(state_directory)
            # get the state
            nn_state = nnx.state(self.flax_neural_network)
            # restore
            restored_state = self.checkpointer.restore(absolute_path, nn_state)
            # now update the model with the loaded state
            nnx.update(self.flax_neural_network, restored_state)
            fol_info(f"flax nnx state is restored from {state_directory}")

    def SaveCheckPoint(self):
        """
        Saves the current state of the neural network to a specified directory.

        This method stores the state of the neural network model in a designated directory, ensuring the model's 
        state can be restored later.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The current state of the neural network is saved to the specified directory. A confirmation message is 
            logged to indicate the successful save operation.

        Notes
        -----
        - The directory for saving the checkpoint is specified in the `checkpoint_settings` attribute under the 
        `state_directory` key.
        - The directory path is converted to an absolute path before saving.
        - Uses the `self.checkpointer.save` method to store the state and forces the save operation.
        - Logs the save operation using `fol_info`.
        """
        state_directory = self.checkpoint_settings["state_directory"]
        absolute_path = os.path.abspath(state_directory)
        self.checkpointer.save(absolute_path, nnx.state(self.flax_neural_network),force=True)
        self.checkpointer.close()  # Close resources properly

        fol_info(f"flax nnx state is saved to {state_directory}")

    def PlotHistoryDict(self,plot_settings:dict,train_history_dict:dict,test_history_dict:dict):
        """
        Plots the training and testing history.

        This method generates and saves a plot of the training and test history based on 
        the specified settings. It supports logging various metrics, such as loss, 
        across training epochs and allows customization of which metrics to plot.

        Parameters
        ----------
        plot_settings : dict
            Dictionary containing settings for the plot, such as:
            - 'plot_rate': int, how often to plot the history (in terms of epochs).
            - 'plot_list': list of str, the metrics to be plotted (e.g., 'total_loss').
        train_history_dict : dict
            A dictionary where keys are metric names (e.g., 'total_loss') and values 
            are lists of the corresponding metric values during training.
        test_history_dict : dict
            A dictionary where keys are metric names (e.g., 'total_loss') and values 
            are lists of the corresponding metric values during testing.

        Returns
        -------
        None
            The function does not return any values but saves the plot to a file 
            in the working directory as 'training_history.png'.
        """
        plot_rate = plot_settings["plot_rate"]
        plot_list = plot_settings["plot_list"]

        plt.figure(figsize=(10, 5))
        train_max_length = 0
        for key,value in train_history_dict.items():
            if len(value)>0 and (len(plot_list)==0 or key in plot_list):
                train_max_length = len(value)
                plt.semilogy(value[::plot_rate], label=f"train_{key}") 

        for key,value in test_history_dict.items():
            if len(value)>0 and (len(plot_list)==0 or key in plot_list):
                test_length = len(value)
                x_value = [ i * plot_settings["test_frequency"] for i in range(test_length-1)]
                x_value.append(train_max_length-1)
                plt.semilogy(x_value,value[::plot_rate], label=f"test_{key}") 

        plt.title("Training History")
        plt.xlabel(str(plot_rate) + " Epoch")
        plt.ylabel("Log Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.working_directory,"training_history.png"), bbox_inches='tight')
        plt.close()

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the network.

        This method finalizes the network. This is only called once in the whole training process.

        """
        pass





