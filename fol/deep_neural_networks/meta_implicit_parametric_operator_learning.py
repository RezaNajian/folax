"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: November, 2024
 License: FOL/LICENSE
"""

from typing import Tuple 
import jax
import jax.numpy as jnp
import copy
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

class MetaImplicitParametricOperatorLearning(ImplicitParametricOperatorLearning):
    """
    A class for meta-learning implicit parametric operators using deep neural networks.

    This class extends the `ImplicitParametricOperatorLearning` base class and is designed specifically 
    for learning parametric operators with enhanced meta-learning capabilities. It introduces additional 
    optimization techniques, such as latent loop optimization, to improve model performance.

    Attributes:
        name (str): The name assigned to the neural network model for identification purposes.
        control (Control): An instance of the `Control` class that manages the parametric learning process.
        loss_function (Loss): An instance of the `Loss` class representing the objective function to minimize during training.
        flax_neural_network (HyperNetwork): The Flax-based hypernetwork model that defines the architecture and forward pass.
        latent_optimizer (GradientTransformation): The Optax optimizer used for latent loop optimization during training.
        main_loop_optimizer (GradientTransformation): The Optax optimizer used for the primary optimization loop.
        checkpoint_settings (dict): A dictionary containing configurations for managing checkpoints, 
            including saving and loading model states. Defaults to an empty dictionary.
        working_directory (str): The path to the working directory where model outputs and checkpoints are saved. Defaults to the current directory.
    """

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:HyperNetwork,
                 latent_loop_optax_optimizer:GradientTransformation,
                 main_loop_optax_optimizer:GradientTransformation,
                 checkpoint_settings:dict={},
                 working_directory='.'
                 ):
        """
        Initializes the `MetaImplicitParametricOperatorLearning` class.

        Args:
            name (str): The name assigned to the neural network model for identification purposes.
            control (Control): An instance of the `Control` class that manages the parametric learning process.
            loss_function (Loss): An instance of the `Loss` class representing the objective function to minimize.
            flax_neural_network (HyperNetwork): The Flax-based hypernetwork model defining the architecture and forward pass.
            latent_loop_optax_optimizer (GradientTransformation): The Optax optimizer for latent loop optimization.
            main_loop_optax_optimizer (GradientTransformation): The Optax optimizer for the primary optimization loop.
            checkpoint_settings (dict, optional): Configurations for managing checkpoints. Defaults to an empty dictionary.
            working_directory (str, optional): The path to the working directory for saving outputs and checkpoints. Defaults to the current directory.
        """
        super().__init__(name,control,loss_function,flax_neural_network,
                         main_loop_optax_optimizer,checkpoint_settings,
                         working_directory)
        
        self.latent_optimizer = latent_loop_optax_optimizer
        
    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,orig_features:Tuple[jnp.ndarray, jnp.ndarray],latent_code:jnp.ndarray,nn_model:nnx.Module):
        """
        Computes the single loss value for a given input using the latent code and the neural network model.

        This function calculates the loss by comparing the neural network's output for the given latent code
        with the control output derived from the original features. The loss computation considers only the
        non-Dirichlet indices as defined in the loss function.

        Args:
            orig_features (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing the original input features, 
                where the first element is used for control variable computation.
            latent_code (jnp.ndarray): The latent code input to the neural network.
            nn_model (nnx.Module): The neural network model used for prediction.

        Returns:
            jnp.ndarray: The computed loss value as a scalar.
        """        
        nn_output = nn_model(latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        control_output = self.control.ComputeControlledVariables(orig_features[0])
        return self.loss_function.ComputeSingleLoss(control_output,nn_output)

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLossValue(self,batch_orig_input_features:jnp.ndarray, batch_latent_codes:jnp.ndarray,nn_model:nnx.Module):
        """
        Computes the batch loss value for a set of input features and latent codes.

        This function evaluates the loss for a batch of inputs by leveraging the `ComputeSingleLossValue` method in a
        vectorized manner. It computes the loss for each instance in the batch and aggregates statistics such as 
        minimum, maximum, and average loss values for further analysis.

        Args:
            batch_orig_input_features (jnp.ndarray): A batch of original input features used for control variable computation.
            batch_latent_codes (jnp.ndarray): A batch of latent codes corresponding to the input features.
            nn_model (nnx.Module): The neural network model used for predictions.

        Returns:
            Tuple[float, dict]: A tuple containing:
                - `total_mean_loss` (float): The mean loss value across the batch.
                - A dictionary with the following keys:
                    - `<loss_name>_min`: The minimum loss value in the batch.
                    - `<loss_name>_max`: The maximum loss value in the batch.
                    - `<loss_name>_avg`: The average loss value in the batch.
                    - `total_loss`: The overall mean loss for the batch.
        """        
        batch_losses,(batch_mins,batch_maxs,batch_avgs) = jax.vmap(self.ComputeSingleLossValue,(0,0,None))(batch_orig_input_features,batch_latent_codes,nn_model)
        loss_name = self.loss_function.GetName()
        total_mean_loss = jnp.mean(batch_losses)
        return total_mean_loss, ({loss_name+"_min":jnp.min(batch_mins),
                                    loss_name+"_max":jnp.max(batch_maxs),
                                    loss_name+"_avg":jnp.mean(batch_avgs),
                                    "total_loss":total_mean_loss})

    def ComputeSingleLatentCode(self,orig_features:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module,nn_optimizer:GradientTransformation)->jnp.ndarray:
        """
        Computes the latent code for a given set of original features using an optimization loop.

        This function initializes a latent code and iteratively updates it by minimizing the loss function.
        The optimization process uses a provided optimizer and computes gradients for the latent code over a fixed
        number of iterations.

        Args:
            orig_features (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing the original input features, where the first 
                element is used for control variable computation.
            nn_model (nnx.Module): The neural network model used for predictions.
            nn_optimizer (GradientTransformation): The optimizer used for updating the latent code.

        Returns:
            jnp.ndarray: The optimized latent code after the iterative process.
        """        
        sample_optimizer = copy.deepcopy(nn_optimizer)
        sample_code = 1e-6*jnp.ones(self.flax_neural_network.in_features)
        opt_state = sample_optimizer.init(sample_code)
        for i in range(3):
            (encoding_loss, (loss_min,loss_max,loss_avg)), grads = nnx.value_and_grad(self.ComputeSingleLossValue,argnums=1,has_aux=True) \
                                                                        (orig_features,sample_code,nn_model)
            updates, opt_state = sample_optimizer.update(grads, opt_state, sample_code)
            sample_code = optax.apply_updates(sample_code, updates)
        
        return sample_code
    
    @partial(jax.jit, static_argnums=(0,))
    def OuterLoopStep(self, nnx_graphdef:nnx.GraphDef, nxx_state:nnx.GraphState, batch_orig_input_feature:jnp.ndarray, batch_latent_codes:jnp.ndarray):
        """
        Performs a single step of the outer optimization loop.

        This function updates the state of the neural network and optimizer by computing the batch loss and its gradients.
        The updated model state is returned along with a dictionary containing batch loss statistics.

        Args:
            nnx_graphdef (nnx.GraphDef): The graph definition of the neural network model.
            nxx_state (nnx.GraphState): The current state of the neural network and optimizer.
            batch_orig_input_feature (jnp.ndarray): A batch of original input features used for control variable computation.
            batch_latent_codes (jnp.ndarray): A batch of latent codes corresponding to the input features.

        Returns:
            Tuple[dict, nnx.GraphState]: A tuple containing:
                - `batch_dict` (dict): A dictionary with batch loss statistics, such as minimum, maximum, and average loss values.
                - `new_state` (nnx.GraphState): The updated state of the neural network and optimizer.
        """
        nnx_model, nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)

        (batch_loss, batch_dict), batch_grads = nnx.value_and_grad(self.ComputeBatchLossValue,argnums=2,has_aux=True) \
                                                                    (batch_orig_input_feature,batch_latent_codes,nnx_model)
        nnx_optimizer.update(batch_grads)
        _, new_state = nnx.split((nnx_model, nnx_optimizer))
        return batch_dict,new_state

    def TrainStep(self, nnx_graphdef:nnx.GraphDef, nxx_state:nnx.GraphState, train_batch:Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Performs a single training step for the model.

        This function computes the latent codes for a given training batch using `ComputeSingleLatentCode`.
        It then performs an outer loop update step using the computed latent codes and the training data.

        Args:
            nnx_graphdef (nnx.GraphDef): The graph definition of the neural network model.
            nxx_state (nnx.GraphState): The current state of the neural network and optimizer.
            train_batch (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing:
                - The original input features of the training batch.
                - The corresponding targets or labels.

        Returns:
            Tuple[dict, nnx.GraphState]: A tuple containing:
                - `batch_dict` (dict): A dictionary with batch loss statistics from the outer loop step.
                - `new_state` (nnx.GraphState): The updated state of the neural network and optimizer after the training step.
        """
        nnx_model,_ = nnx.merge(nnx_graphdef, nxx_state)

        latent_codes = jax.vmap(self.ComputeSingleLatentCode,(0,None,None))(train_batch,   
                                                                            nnx_model,
                                                                            self.latent_optimizer)

        return self.OuterLoopStep(nnx_graphdef,nxx_state,train_batch,latent_codes)

    def TestStep(self, nnx_graphdef:nnx.GraphDef, nxx_state:nnx.GraphState,test_batch:Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Evaluates the model on a test batch.

        This function computes the latent codes for a given test batch using `ComputeSingleLatentCode`. 
        It then evaluates the batch loss using the computed latent codes and returns the loss statistics.

        Args:
            nnx_graphdef (nnx.GraphDef): The graph definition of the neural network model.
            nxx_state (nnx.GraphState): The current state of the neural network and optimizer.
            test_batch (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing:
                - The original input features of the test batch.
                - The corresponding targets or labels.

        Returns:
            Tuple[dict, nnx.GraphState]: A tuple containing:
                - `test_batch_dict` (dict): A dictionary with batch loss statistics, such as minimum, maximum, and average loss values.
                - `state` (nnx.GraphState): The updated state of the neural network and optimizer after the evaluation step.
        """        
        nnx_model,nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)

        latent_codes = jax.vmap(self.ComputeSingleLatentCode,(0,None,None))(test_batch,   
                                                                            nnx_model,
                                                                            self.latent_optimizer)
        (test_loss, test_batch_dict) = self.ComputeBatchLossValue(test_batch,latent_codes,nnx_model)
        _, state = nnx.split((nnx_model, nnx_optimizer))
        return test_batch_dict,state

    @print_with_timestamp_and_execution_time
    def Predict(self,batch_X:jnp.ndarray):
        """
        Generates predictions for a batch of input data.

        This method computes predictions for a batch of input features by first computing the latent code for each sample.
        The predicted outputs are then generated using the network and mapped to the full degree of freedom (DoF) vector
        based on the loss function's mapping.

        This process involves:
        1. Computing the latent code for each input sample.
        2. Generating the neural network output using the latent code.
        3. Mapping the network output to the full DoF vector, considering both Dirichlet and non-Dirichlet indices.

        Args:
            batch_X (jnp.ndarray): A batch of input data for which predictions are required.

        Returns:
            jnp.ndarray: The predicted outputs for the batch, with each prediction mapped to the full DoF vector.
        """
        def predict_single_sample(sample_x:jnp.ndarray):
            computed_sample_code = self.ComputeSingleLatentCode((sample_x,),
                                                          nn_model=self.flax_neural_network,
                                                          nn_optimizer=self.latent_optimizer)
            nn_output = self.flax_neural_network(computed_sample_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
            return self.loss_function.GetFullDofVector(sample_x,nn_output)

        return jnp.array(jax.vmap(predict_single_sample)(batch_X))

    def Finalize(self):
        pass