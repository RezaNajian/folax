"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""
import os
from abc import ABC,abstractmethod
from typing import Iterator,Tuple
from tqdm import trange
import matplotlib.pyplot as plt
import jax.numpy as jnp
from functools import partial
from flax import nnx
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
                                            "save_state":True,
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

            state_directory = self.checkpoint_settings["state_directory"]
            absolute_path = os.path.abspath(state_directory)

            # get the state
            nn_state = nnx.state(self.flax_neural_network)

            # restore
            restored_state = self.checkpointer.restore(absolute_path, nn_state)

            # now update the model with the loaded state
            nnx.update(self.flax_neural_network, restored_state)

        # initialize the nnx optimizer
        self.nnx_optimizer = nnx.Optimizer(self.flax_neural_network, self.optax_optimizer)

    def GetName(self) -> str:
        """
        Returns the name of the model.

        Returns
        -------
        str
            The name of the deep learning model.
        """
        return self.name
    
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

    @partial(nnx.jit, static_argnums=(0,))
    def TrainStep(self, nn_model:nnx.Module, optimizer:nnx.Optimizer, batch_set:Tuple[jnp.ndarray, jnp.ndarray]):
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

        (batch_loss, batch_dict), batch_grads = nnx.value_and_grad(self.ComputeBatchLossValue,argnums=1,has_aux=True) \
                                                                    (batch_set,nn_model)
        optimizer.update(batch_grads)
        return batch_dict

    @print_with_timestamp_and_execution_time
    def Train(self, train_set:Tuple[jnp.ndarray, jnp.ndarray], test_set:Tuple[jnp.ndarray, jnp.ndarray] = (jnp.array([]), jnp.array([])), 
              batch_size:int=100, convergence_settings:dict={}, plot_settings:dict={}, save_settings:dict={}):

        """
        Trains the neural network model over multiple epochs.

        This method trains the model on the provided training dataset, evaluates performance on the test set, 
        updates model parameters using gradient descent, and tracks training history. It also checks for convergence 
        and saves the model state during the process.

        Parameters
        ----------
        train_set : Tuple[jnp.ndarray, jnp.ndarray]
            Training dataset consisting of input data and corresponding target labels.
        test_set : Tuple[jnp.ndarray, jnp.ndarray], optional
            Test dataset for validation, defaults to empty arrays.
        batch_size : int, optional
            Number of samples per batch, default is 100.
        convergence_settings : dict, optional
            Settings to control the convergence criteria, defaults to an empty dict.
        plot_settings : dict, optional
            Settings to control the plotting of training history, defaults to an empty dict.
        save_settings : dict, optional
            Settings to control saving of the trained model, defaults to an empty dict.
        """

        self.default_convergence_settings = {"num_epochs":100,"convergence_criterion":"total_loss",
                                             "relative_error":1e-8,"absolute_error":1e-8}
        convergence_settings = UpdateDefaultDict(self.default_convergence_settings,convergence_settings)
        
        self.default_plot_settings = {"plot_list":["total_loss"],"plot_rate":1,"plot_save_rate":100}
        plot_settings = UpdateDefaultDict(self.default_plot_settings,plot_settings)

        self.default_save_settings = {"save_nn_model":True}
        save_settings = UpdateDefaultDict(self.default_save_settings,save_settings)

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
        for epoch in pbar:
            train_set_hist_dict = {}
            test_set_hist_dict = {}
            # now loop over batches
            batch_index = 0 
            for batch_set in self.CreateBatches(train_set, batch_size):
                batch_dict = self.TrainStep(self.flax_neural_network,self.nnx_optimizer,batch_set)
                train_set_hist_dict = update_batch_history_dict(train_set_hist_dict,batch_dict,batch_index)

                if len(test_set[0])>0:
                    _,test_dict = self.ComputeBatchLossValue(test_set,self.flax_neural_network)
                    test_set_hist_dict = update_batch_history_dict(test_set_hist_dict,test_dict,batch_index)
                else:
                    test_dict = {}
                
                batch_index += 1

            train_history_dict = update_history_dict(train_history_dict,train_set_hist_dict)
            print_dict = {"train_loss":train_history_dict["total_loss"][-1]}
            if len(test_set[0])>0:
                test_history_dict = update_history_dict(test_history_dict,test_set_hist_dict)
                print_dict = {"train_loss":train_history_dict["total_loss"][-1],
                              "test_loss":test_history_dict["total_loss"][-1]}

            pbar.set_postfix(print_dict)

            # check converged
            converged = self.CheckConvergence(train_history_dict,convergence_settings)

            # plot the histories
            if (epoch>0 and epoch %plot_settings["plot_save_rate"] == 0) or converged:
                self.PlotHistoryDict(plot_settings,train_history_dict,test_history_dict)

            if epoch<convergence_settings["num_epochs"]-1 and converged:
                break    

        # Save the flax model
        if save_settings["save_nn_model"]:
            state_directory = self.checkpoint_settings["state_directory"]
            absolute_path = os.path.abspath(state_directory)
            checkpointer = orbax.PyTreeCheckpointer()
            checkpointer.save(absolute_path, nnx.state(self.flax_neural_network),
                              force=True)

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
        for key,value in train_history_dict.items():
            if len(value)>0 and (len(plot_list)==0 or key in plot_list):
                plt.semilogy(value[::plot_rate], label=f"train_{key}") 

        for key,value in test_history_dict.items():
            if len(value)>0 and (len(plot_list)==0 or key in plot_list):
                plt.semilogy(value[::plot_rate], label=f"test_{key}") 
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





