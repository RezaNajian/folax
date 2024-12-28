"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: December, 2024
 License: FOL/LICENSE
"""

from typing import Tuple 
import jax
import jax.numpy as jnp
import optax
from functools import partial
from optax import GradientTransformation
from flax import nnx
from .implicit_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *
from .nns import HyperNetwork

class MetaAlphaMetaImplicitParametricOperatorLearning(ImplicitParametricOperatorLearning):
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
    def ComputeSingleLossValue(self,orig_features:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module,latent_opt_step:float):
        """
        Computes the single loss value for a given input feature set, neural network model, 
        and latent optimization step size. This method optimizes the latent code iteratively 
        and evaluates the loss.

        Args:
            orig_features (Tuple[jnp.ndarray, jnp.ndarray]): 
                A tuple containing the original feature set, where:
                - The first element is the input features for the neural network.
                - The second element is auxiliary data, such as labels or other metadata.
            nn_model (nnx.Module): 
                The neural network model used for computation. It should support 
                evaluation with input latent codes and coordinates.
            latent_opt_step (float): 
                Step size for updating the latent code during optimization.

        Returns:
            float: The computed single loss value after optimizing the latent code.

        Notes:
            - Initializes the latent code as a zero vector with a size equal to the 
              input dimensions of the neural network.
            - Uses the control object to compute controlled variables based on the 
              original features.
            - Iteratively updates the latent code using the gradient of the loss function.
            - Computes the final loss based on the optimized latent code and the neural network output.

        Optimization Process:
            1. Define a loss function based on the neural network's output and the 
               controlled variables.
            2. Compute the gradient of the loss with respect to the latent code.
            3. Perform a specified number of latent optimization iterations, updating 
               the latent code using the step size.
            4. Return the final loss value using the optimized latent code.
        """     
        latent_code = jnp.zeros(nn_model.in_features)
        control_output = self.control.ComputeControlledVariables(orig_features[0])

        @jax.jit
        def loss(input_latent_code):
            nn_output = nn_model(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
            return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]

        loss_latent_grad_fn = jax.grad(loss)
        for _ in range(self.num_latent_iterations):
            latent_code -= latent_opt_step * loss_latent_grad_fn(latent_code)
        
        nn_output = nn_model(latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        return self.loss_function.ComputeSingleLoss(control_output,nn_output)

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLossValue(self,batch_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module,latent_opt_step:float):
        """
        Computes the batch loss value for a given set of input features, neural network model, 
        and latent optimization step size. This method evaluates the loss over a batch of samples 
        and returns aggregated metrics.

        Args:
            batch_set (Tuple[jnp.ndarray, jnp.ndarray]): 
                A tuple containing the batch of input data, where:
                - The first element is a batch of input features for the neural network.
                - The second element is auxiliary data, such as labels or other metadata, 
                  for each sample in the batch.
            nn_model (nnx.Module): 
                The neural network model used for computation. It should support evaluation 
                with input latent codes and coordinates.
            latent_opt_step (float): 
                Step size for updating the latent code during optimization.

        Returns:
            Tuple[float, dict]:
                - The total mean loss across the batch.
                - A dictionary of aggregated metrics, including:
                    - "{loss_name}_min": Minimum loss value across the batch.
                    - "{loss_name}_max": Maximum loss value across the batch.
                    - "{loss_name}_avg": Average loss value across the batch.
                    - "total_loss": The total mean loss.

        """

        batch_losses,(batch_mins,batch_maxs,batch_avgs) = jax.vmap(self.ComputeSingleLossValue,(0,None,None))(batch_set,nn_model,latent_opt_step)
        loss_name = self.loss_function.GetName()
        total_mean_loss = jnp.mean(batch_losses)
        return total_mean_loss, ({loss_name+"_min":jnp.min(batch_mins),
                                         loss_name+"_max":jnp.max(batch_maxs),
                                         loss_name+"_avg":jnp.mean(batch_avgs),
                                         "total_loss":total_mean_loss})

    @partial(jax.jit, static_argnums=(0,))
    def TrainMetaStep(self, nnx_graphdef:nnx.GraphDef, nxx_state:nnx.GraphState, 
                      latent_opt_state:optax.OptState,latent_step:float,
                      train_batch:Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Executes a single meta-training step, optimizing the neural network model parameters, 
        latent step size, and latent step optimizer state based on a given training batch.

        Args:
            nnx_graphdef (nnx.GraphDef): 
                The neural network graph definition containing the model architecture.
            nxx_state (nnx.GraphState): 
                The state of the neural network, including parameters and optimizer states.
            latent_opt_state (optax.OptState): 
                The current state of the optimizer for the latent step size.
            latent_step (float): 
                The current latent step size used for latent code optimization.
            train_batch (Tuple[jnp.ndarray, jnp.ndarray]): 
                A tuple containing the training batch, where:
                - The first element is a batch of input features for the neural network.
                - The second element is auxiliary data, such as labels or other metadata.

        Returns:
            Tuple[dict, nnx.GraphState, float, optax.OptState]:
                - A dictionary containing batch-level loss metrics, including aggregated statistics.
                - The updated state of the neural network (parameters and optimizer states).
                - The updated latent step size after applying optimization updates.
                - The updated latent step optimizer state.

        Workflow:
            1. Merge the graph definition and state into a neural network model and optimizer.
            2. Compute the batch loss and gradients using `ComputeBatchLossValue`.
            3. Use the latent step optimizer to compute updates for the latent step size based on gradients.
            4. Apply the updates to the latent step size and latent optimizer state.
            5. Update the neural network optimizer with the computed gradients.
            6. Split the updated model and optimizer back into a new state.
            7. Return the batch metrics, updated neural network state, updated latent step size, 
               and updated latent optimizer state.

        Notes:
            - Uses `jax.jit` for just-in-time compilation to improve performance.
            - Supports auxiliary outputs (e.g., batch-level statistics) along with loss and gradient computations.
            - Handles simultaneous optimization of neural network parameters and latent step size.
        """
        nnx_model, nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)

        (batch_loss, batch_dict), batch_grads = nnx.value_and_grad(self.ComputeBatchLossValue,argnums=(1,2),has_aux=True) \
                                                                    (train_batch,nnx_model,self.latent_step)
        
        latent_step_update, new_latent_opt_state = self.latent_step_optimizer.update(batch_grads[1], latent_opt_state)
        updated_latent_step = optax.apply_updates(latent_step, latent_step_update)

        nnx_optimizer.update(batch_grads[0])
        _, new_state = nnx.split((nnx_model, nnx_optimizer))
        return batch_dict,new_state,updated_latent_step,new_latent_opt_state
    
    def TrainStep(self, nnx_graphdef:nnx.GraphDef, nxx_state:nnx.GraphState, 
                        train_batch:Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Executes a single training step for the neural network model and updates 
        the latent step size and optimizer state based on the given training batch.

        Args:
            nnx_graphdef (nnx.GraphDef): 
                The neural network graph definition containing the model architecture.
            nxx_state (nnx.GraphState): 
                The state of the neural network, including parameters and optimizer states.
            train_batch (Tuple[jnp.ndarray, jnp.ndarray]): 
                A tuple containing the training batch, where:
                - The first element is a batch of input features for the neural network.
                - The second element is auxiliary data, such as labels or other metadata.

        Returns:
            Tuple[dict, nnx.GraphState]:
                - A dictionary containing batch-level loss metrics, including aggregated statistics.
                - The updated state of the neural network (parameters and optimizer states).

        Workflow:
            1. Calls `TrainMetaStep` to execute a single meta-training step, which includes:
                - Optimizing the neural network parameters.
                - Updating the latent step size and optimizer state.
            2. Updates the internal latent step size and latent optimizer state attributes.
            3. Returns the batch loss metrics and updated neural network state.

        Notes:
            - Simplifies the process by abstracting the details of meta-training into `TrainMetaStep`.
            - Useful for performing iterative training over multiple batches in a loop.
        """
        batch_dict,new_state,self.latent_step,self.latent_step_optimizer_state = self.TrainMetaStep(nnx_graphdef,
                                                                                           nxx_state,
                                                                                           self.latent_step_optimizer_state,
                                                                                           self.latent_step,
                                                                                           train_batch)
        return batch_dict,new_state

    @partial(jax.jit, static_argnums=(0,))
    def TestStep(self, nnx_graphdef:nnx.GraphDef, nxx_state:nnx.GraphState,
                       test_batch:Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Executes a single testing step, evaluating the loss and performance metrics 
        for a given test batch without updating model parameters or latent variables.

        Args:
            nnx_graphdef (nnx.GraphDef): 
                The neural network graph definition containing the model architecture.
            nxx_state (nnx.GraphState): 
                The state of the neural network, including parameters and optimizer states.
            test_batch (Tuple[jnp.ndarray, jnp.ndarray]): 
                A tuple containing the test batch, where:
                - The first element is a batch of input features for the neural network.
                - The second element is auxiliary data, such as labels or other metadata.

        Returns:
            Tuple[dict, nnx.GraphState]:
                - A dictionary containing batch-level loss metrics, including aggregated statistics.
                - The state of the neural network (parameters and optimizer states), unchanged.

        Notes:
            - This function is intended for evaluation purposes and does not update model parameters 
              or latent step variables.
            - Uses `jax.jit` for just-in-time compilation to improve performance.
        """

        nnx_model, nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)
        (test_loss, test_batch_dict) = self.ComputeBatchLossValue(test_batch,nnx_model,self.latent_step)
        _, state = nnx.split((nnx_model, nnx_optimizer))
        return test_batch_dict,state

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
        super().SaveCheckPoint()
        # save meta learning latent_step
        state_directory = self.checkpoint_settings["meta_state_directory"]
        absolute_path = os.path.abspath(state_directory)
        self.checkpointer.save(absolute_path, {"latent_step":self.latent_step},force=True)
        fol_info(f"latent_step {self.latent_step} is saved to {state_directory}")

    @print_with_timestamp_and_execution_time
    def Predict(self,batch_X:jnp.ndarray):
        """
        Predicts the output for a batch of input samples by optimizing the latent code for each sample.

        Args:
            batch_X (jnp.ndarray): 
                A batch of input features, where each row corresponds to a single input sample.

        Returns:
            jnp.ndarray:
                An array of predicted outputs for the input batch, where each row corresponds 
                to the prediction for a single input sample.

        Workflow:
            1. Defines a helper function `predict_single_sample` that performs prediction for a single input sample:
                - Initializes the latent code as a zero vector.
                - Computes controlled variables based on the input sample.
                - Defines a loss function based on the neural network output and the controlled variables.
                - Iteratively optimizes the latent code using the gradient of the loss function.
                - Computes the neural network output for the optimized latent code.
                - Converts the neural network output to a full degree-of-freedom vector using the loss function.
            2. Uses `jax.vmap` to vectorize `predict_single_sample` over the input batch.
            3. Returns the predictions for the batch as a `jnp.ndarray`.

        Notes:
            - The method uses just-in-time (JIT) compilation for improved performance in optimizing the latent code.
            - The latent code optimization process is repeated for each input sample in the batch.
            - The prediction output is based on the optimized latent code and the neural network model.

        """
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

    def Finalize(self):
        pass