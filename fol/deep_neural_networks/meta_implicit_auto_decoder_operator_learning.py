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
import orbax.checkpoint as orbax
import orbax.checkpoint as ocp
from .implicit_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *
from .nns import HyperNetwork

class MetaImplicitAutodecoderOperatorLearning(ImplicitParametricOperatorLearning):
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
                 main_loop_optax_optimizer:GradientTransformation,
                 latent_step_optax_optimizer:GradientTransformation,
                 latent_step_size:float,
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
        
        self.latent_step_optimizer = latent_step_optax_optimizer
        self.latent_step = latent_step_size
        self.latent_step_optimizer_state = self.latent_step_optimizer.init(self.latent_step)
        self.default_checkpoint_settings = {"restore_state":False,
                                            "state_directory":'./flax_state',
                                            "meta_state_directory":'./meta_state'}

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,orig_features:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module,latent_opt_step:float):
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
        latent_code = jnp.zeros(nn_model.in_features)
        control_output = self.control.ComputeControlledVariables(orig_features[0])

        @jax.jit
        def loss(input_latent_code):
            nn_output = nn_model(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
            return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]

        loss_latent_grad_fn = jax.grad(loss)
        for _ in range(3):
            latent_code -= latent_opt_step * loss_latent_grad_fn(latent_code)
        
        nn_output = nn_model(latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        return self.loss_function.ComputeSingleLoss(control_output,nn_output)

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLossValue(self,batch_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module,latent_opt_step:float):
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

        batch_dict,new_state,self.latent_step,self.latent_step_optimizer_state = self.TrainMetaStep(nnx_graphdef,
                                                                                           nxx_state,
                                                                                           self.latent_step_optimizer_state,
                                                                                           self.latent_step,
                                                                                           train_batch)
        return batch_dict,new_state

    # def TestStep(self, nnx_graphdef:nnx.GraphDef, nxx_state:nnx.GraphState,test_batch:Tuple[jnp.ndarray, jnp.ndarray]):
    #     """
    #     Evaluates the model on a test batch.

    #     This function computes the latent codes for a given test batch using `ComputeSingleLatentCode`. 
    #     It then evaluates the batch loss using the computed latent codes and returns the loss statistics.

    #     Args:
    #         nnx_graphdef (nnx.GraphDef): The graph definition of the neural network model.
    #         nxx_state (nnx.GraphState): The current state of the neural network and optimizer.
    #         test_batch (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing:
    #             - The original input features of the test batch.
    #             - The corresponding targets or labels.

    #     Returns:
    #         Tuple[dict, nnx.GraphState]: A tuple containing:
    #             - `test_batch_dict` (dict): A dictionary with batch loss statistics, such as minimum, maximum, and average loss values.
    #             - `state` (nnx.GraphState): The updated state of the neural network and optimizer after the evaluation step.
    #     """        
    #     nnx_model,nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)

    #     latent_codes = jax.vmap(self.ComputeSingleLatentCode,(0,None,None))(test_batch,   
    #                                                                         nnx_model,
    #                                                                         self.latent_optimizer)
    #     (test_loss, test_batch_dict) = self.ComputeBatchLossValue(test_batch,latent_codes,nnx_model)
    #     _, state = nnx.split((nnx_model, nnx_optimizer))
    #     return test_batch_dict,state

    def RestoreCheckPoint(self,checkpoint_settings:dict):
        super().RestoreCheckPoint(checkpoint_settings)
        if "meta_state_directory" in checkpoint_settings.keys():
            meta_state_directory = checkpoint_settings["meta_state_directory"]
            absolute_path = os.path.abspath(meta_state_directory)
            self.latent_step = self.checkpointer.restore(absolute_path,{"latent_step":self.latent_step})["latent_step"]
            fol_info(f"latent_step {self.latent_step} is restored from {meta_state_directory}")

    def SaveCheckPoint(self):
        super().SaveCheckPoint()
        # save meta learning latent_step
        state_directory = self.checkpoint_settings["meta_state_directory"]
        absolute_path = os.path.abspath(state_directory)
        self.checkpointer.save(absolute_path, {"latent_step":self.latent_step},force=True)
        fol_info(f"latent_step {self.latent_step} is saved to {state_directory}")

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

            latent_code = jnp.zeros(self.flax_neural_network.in_features)
            control_output = self.control.ComputeControlledVariables(sample_x)

            @jax.jit
            def loss(input_latent_code):
                nn_output = self.flax_neural_network(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
                return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]

            loss_latent_grad_fn = jax.grad(loss)
            for _ in range(3):
                latent_code -= self.latent_step * loss_latent_grad_fn(latent_code)

            nn_output = self.flax_neural_network(latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]

            return self.loss_function.GetFullDofVector(sample_x,nn_output)

        return jnp.array(jax.vmap(predict_single_sample)(batch_X))

    def Finalize(self):
        pass