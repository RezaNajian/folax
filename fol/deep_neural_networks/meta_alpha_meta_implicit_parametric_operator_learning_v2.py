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

class MetaAlphaMetaImplicitParametricOperatorLearningV2(ImplicitParametricOperatorLearning):
    """
    A class for implementing meta-learning techniques in the context of implicit parametric operator learning.

    This class extends the `ImplicitParametricOperatorLearning` class and incorporates 
    meta-learning functionality for optimizing latent variables. It supports custom loss functions, 
    neural network models, and optimizers. Additionally, this class optimizes both the latent code 
    and the latent step size during the process of latent finding and optimization.

    Attributes:
        name (str): Name of the learning instance.
        control (Control): Control object to manage configurations and settings.
        loss_function (Loss): Loss function used for optimization.
        flax_neural_network (HyperNetwork): Neural network model for operator learning.
        main_loop_optax_optimizer (GradientTransformation): Optimizer for the main training loop.
        latent_step_optax_optimizer (GradientTransformation): Optimizer for updating latent variables.
        latent_step (float): Step size for latent updates.
        num_latent_iterations (int): Number of iterations for latent variable optimization.
        checkpoint_settings (dict): Settings for checkpointing, such as saving and restoring states.
        working_directory (str): Directory for saving files and logs.
        latent_step_optimizer_state: Internal state of the latent step optimizer.
        default_checkpoint_settings (dict): Default checkpoint settings, including directories and restore options.
    """

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:HyperNetwork,
                 main_loop_optax_optimizer:GradientTransformation,
                 latent_step_optax_optimizer:GradientTransformation,
                 latent_step_size:float=1e-2,
                 num_latent_iterations:int=3,
                 checkpoint_settings:dict={},
                 working_directory='.'
                 ):
        """
        Initializes the MetaAlphaMetaImplicitParametricOperatorLearning instance.

        Args:
            name (str): Name of the learning instance.
            control (Control): Control object to manage configurations and settings.
            loss_function (Loss): Loss function used for optimization.
            flax_neural_network (HyperNetwork): Neural network model for operator learning.
            main_loop_optax_optimizer (GradientTransformation): Optimizer for the main training loop.
            latent_step_optax_optimizer (GradientTransformation): Optimizer for updating latent variables and step size.
            latent_step_size (float, optional): Initial step size for latent updates. Default is 1e-2.
            num_latent_iterations (int, optional): Number of iterations for latent variable optimization. Default is 3.
            checkpoint_settings (dict, optional): Settings for checkpointing, such as saving and restoring states. 
                                                  Default is an empty dictionary.
            working_directory (str, optional): Directory for saving files and logs. Default is '.'.

        Notes:
            This class not only finds the optimal latent code but also optimizes the latent step size 
            during the process of latent finding and optimization. This dual optimization ensures better 
            convergence and adaptability for varying problem conditions.
        """
        super().__init__(name,control,loss_function,flax_neural_network,
                         main_loop_optax_optimizer,checkpoint_settings,
                         working_directory)
        
        self.latent_step_optimizer = latent_step_optax_optimizer
        self.latent_step = latent_step_size
        self.num_latent_iterations = num_latent_iterations
        self.latent_step_optimizer_state = self.latent_step_optimizer.init(self.latent_step)
        self.default_checkpoint_settings = {"restore_state":False,
                                            "state_directory":'./flax_state',
                                            "meta_state_directory":'./meta_state'}

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,latent_and_control:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        latent_code = latent_and_control[0]
        control_output = self.control.ComputeControlledVariables(latent_and_control[1])
        nn_output = nn_model(latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        return self.loss_function.ComputeSingleLoss(control_output,nn_output)

    def Finalize(self):
        pass

    def CreateRandomBatch(self,data: Tuple[jnp.ndarray, jnp.ndarray], batch_size: int) -> Iterator[jnp.ndarray]:
        key = jax.random.PRNGKey(int.from_bytes(os.urandom(4), "big"))
        random_subset = jax.random.permutation(key, data[0].shape[0])[:batch_size]
        return data[0][random_subset]
    
    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLatent(self,batch_X:jnp.ndarray,flax_neural_network:nnx.Module,latent_step:float):
        @nnx.jit
        def compute_single_latent(sample_x:jnp.ndarray):

            latent_code = jnp.zeros(flax_neural_network.in_features)
            control_output = self.control.ComputeControlledVariables(sample_x)

            @nnx.jit
            def loss(input_latent_code):
                nn_output = flax_neural_network(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
                return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]

            loss_latent_grad_fn = jax.grad(loss)
            for _ in range(self.num_latent_iterations):
                latent_code -= latent_step * loss_latent_grad_fn(latent_code)

            return latent_code

        return jnp.array(jax.vmap(compute_single_latent)(batch_X))    

    @partial(nnx.jit, static_argnums=(0,))
    def LatentStep(self,batch_X:jnp.ndarray,flax_neural_network:nnx.Module,latent_step:float):
        @nnx.jit
        def compute_batch_loss(batch_X,latent_step,flax_neural_network):
            latent_codes = self.ComputeBatchLatent(batch_X,flax_neural_network,latent_step)
            return self.ComputeBatchLossValue((latent_codes,batch_X),flax_neural_network)[0]
        
        return nnx.value_and_grad(compute_batch_loss,argnums=1) (batch_X,latent_step,flax_neural_network)

    def RestoreCheckPoint(self,checkpoint_settings:dict):
        """
        Restores the model state, including the latent step size, from a specified checkpoint.

        Args:
            checkpoint_settings (dict): 
                A dictionary containing checkpoint configuration settings, including:
                - "meta_state_directory" (str, optional): Path to the directory containing the latent step checkpoint.

        Returns:
            None

        Notes:
            - Ensures that both model parameters and the latent step size are restored when checkpointing.
            - Uses `self.checkpointer.restore` for restoring the latent step size from the specified directory.
        """
        super().RestoreCheckPoint(checkpoint_settings)
        if "meta_state_directory" in checkpoint_settings.keys():
            meta_state_directory = checkpoint_settings["meta_state_directory"]
            absolute_path = os.path.abspath(meta_state_directory)
            self.latent_step = self.checkpointer.restore(absolute_path,{"latent_step":self.latent_step})["latent_step"]
            fol_info(f"latent_step {self.latent_step} is restored from {meta_state_directory}")

    def SaveCheckPoint(self):
        """
        Saves the current model state, including the latent step size, to a specified checkpoint.

        Args:
            None

        Returns:
            None

        Notes:
            - Ensures that both model parameters and the latent step size are checkpointed.
            - Forces the save operation to overwrite existing checkpoint data for the latent step size.
        """
        # save meta learning latent_step
        state_directory = self.checkpoint_settings["meta_state_directory"]
        absolute_path = os.path.abspath(state_directory)
        self.checkpointer.save(absolute_path, {"latent_step":self.latent_step},force=True)
        fol_info(f"latent_step {self.latent_step} is saved to {state_directory}")
        super().SaveCheckPoint()

    @print_with_timestamp_and_execution_time
    def Train(self, train_set:Tuple[jnp.ndarray, jnp.ndarray], test_set:Tuple[jnp.ndarray, jnp.ndarray] = (jnp.array([]), jnp.array([])), 
              batch_size:int=100, test_settings:dict={},convergence_settings:dict={}, plot_settings:dict={}, save_settings:dict={}):

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
                    batch_hist_dict[key] = [sum(value)/len(value)]

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
                # update latent step size
                self.flax_neural_network, self.nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)
                loss_val, loss_grad_latent = self.LatentStep(batch_set[0],self.flax_neural_network,self.latent_step)
                # print(f"loss_val:{loss_val}, self.latent_step:{self.latent_step}")
                latent_step_update, self.latent_step_optimizer_state = self.latent_step_optimizer.update(loss_grad_latent, self.latent_step_optimizer_state)
                self.latent_step = optax.apply_updates(self.latent_step, latent_step_update)

                # compute latent codes with the updated step size
                latent_codes = self.ComputeBatchLatent(batch_set[0],self.flax_neural_network,self.latent_step)

                # now update the NN
                nnx_graphdef, nxx_state = nnx.split((self.flax_neural_network, self.nnx_optimizer))
                batch_dict,nxx_state = self.TrainStep(nnx_graphdef, nxx_state,(latent_codes,batch_set[0]))
                train_set_hist_dict = update_batch_history_dict(train_set_hist_dict,batch_dict,batch_index)

                batch_index += 1                
            
            if len(test_set[0])>0 and ((epoch)%test_settings["test_frequency"]==0 or epoch==convergence_settings["num_epochs"]-1):
                self.flax_neural_network, self.nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)
                # compute latent codes with the updated step size
                test_latent_codes = self.ComputeBatchLatent(test_set[0],self.flax_neural_network,self.latent_step)
                nnx_graphdef, nxx_state = nnx.split((self.flax_neural_network, self.nnx_optimizer))
                test_batch_dict,nxx_state = self.TestStep(nnx_graphdef,nxx_state,(test_latent_codes,test_set[0]))
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

        self.checkpointer.close()  # Close resources properly


    @print_with_timestamp_and_execution_time
    def Predict(self,batch_X:jnp.ndarray):
        @nnx.jit
        def predict_single_sample(sample_x:jnp.ndarray,flax_neural_network:nnx.Module,latent_step):

            latent_code = jnp.zeros(flax_neural_network.in_features)
            control_output = self.control.ComputeControlledVariables(sample_x)

            @nnx.jit
            def loss(input_latent_code):
                nn_output = flax_neural_network(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
                return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]

            loss_latent_grad_fn = jax.grad(loss)
            for _ in range(self.num_latent_iterations):
                latent_code -= latent_step * loss_latent_grad_fn(latent_code)

            nn_output = flax_neural_network(latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]

            return self.loss_function.GetFullDofVector(sample_x,nn_output)

        return jnp.array(jax.vmap(predict_single_sample,(0,None,None))(batch_X,self.flax_neural_network,self.latent_step)) 









        def predict_single_sample(sample_x:jnp.ndarray):

            latent_code = jnp.zeros(self.flax_neural_network.in_features)
            control_output = self.control.ComputeControlledVariables(sample_x)

            @jax.jit
            def loss(input_latent_code):
                nn_output = self.flax_neural_network(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
                return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]

            loss_latent_grad_fn = jax.grad(loss)
            for _ in range(self.num_latent_iterations):
                latent_code -= self.latent_step * loss_latent_grad_fn(latent_code)

            nn_output = self.flax_neural_network(latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]

            return self.loss_function.GetFullDofVector(sample_x,nn_output)

        return jnp.array(jax.vmap(predict_single_sample)(batch_X))