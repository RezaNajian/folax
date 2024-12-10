"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: November, 2024
 License: FOL/LICENSE
"""

from typing import Iterator,Tuple 
import jax
import jax.numpy as jnp
from jax import jit,vmap
from tqdm import trange
import copy
import optax
import orbax.checkpoint as orbax
from functools import partial
from optax import GradientTransformation
from flax import nnx
from .implicit_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *
from .nns import HyperNetwork,MLP

class MetaImplicitParametricOperatorLearning(ImplicitParametricOperatorLearning):
    """
    A class for explicit parametric operator learning in deep neural networks.

    This class extends the `DeepNetwork` base class and is designed specifically 
    for learning parametric operators where spatial fields like predicted displacement
    are explicitly modeled. It inherits all the attributes and methods from `DeepNetwork` and introduces 
    additional components to handle control parameters.

    Attributes:
        name (str): The name assigned to the neural network model for identification purposes.
        control (Control): An instance of the Control class used for the parametric learning.
        loss_function (Loss): An instance of the Loss class representing the objective function to be minimized during training.
        flax_neural_network (Module): The Flax neural network model (inherited from flax.nnx.Module) that defines the architecture and forward pass of the network.
        optax_optimizer (GradientTransformation): The Optax optimizer used to compute and apply gradients during the training process.
        checkpoint_settings (dict): A dictionary of configurations used to manage checkpoints, saving model states and parameters during or after training. Defaults to an empty dictionary.
     
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
        super().__init__(name,control,loss_function,flax_neural_network,
                         main_loop_optax_optimizer,checkpoint_settings,
                         working_directory)
        
        self.inner_optax_optimizer = latent_loop_optax_optimizer
        
    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,orig_features:Tuple[jnp.ndarray, jnp.ndarray],coded_features:jnp.ndarray,nn_model:nnx.Module):
        nn_output = nn_model(coded_features,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        control_output = self.control.ComputeControlledVariables(orig_features[0])
        return self.loss_function.ComputeSingleLoss(control_output,nn_output)

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLossValue(self,batch_orig_input_features:jnp.ndarray, batch_learned_input_features:jnp.ndarray,nn_model:nnx.Module):
        batch_losses,(batch_mins,batch_maxs,batch_avgs) = jax.vmap(self.ComputeSingleLossValue,(0,0,None))(batch_orig_input_features,batch_learned_input_features,nn_model)
        loss_name = self.loss_function.GetName()
        total_mean_loss = jnp.mean(batch_losses)
        return total_mean_loss, ({loss_name+"_min":jnp.min(batch_mins),
                                    loss_name+"_max":jnp.max(batch_maxs),
                                    loss_name+"_avg":jnp.mean(batch_avgs),
                                    "total_loss":total_mean_loss})

    def ComputeSingleLatentCode(self,orig_features:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module,nn_optimizer:GradientTransformation)->jnp.ndarray:
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
    def OuterLoopStep(self, nnx_graphdef:nnx.GraphDef, nxx_state:nnx.GraphState, batch_orig_input_feature:jnp.ndarray, batch_learned_input_feature:jnp.ndarray):

        nnx_model, nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)

        (batch_loss, batch_dict), batch_grads = nnx.value_and_grad(self.ComputeBatchLossValue,argnums=2,has_aux=True) \
                                                                    (batch_orig_input_feature,batch_learned_input_feature,nnx_model)
        nnx_optimizer.update(batch_grads)
        _, new_state = nnx.split((nnx_model, nnx_optimizer))
        return batch_dict,new_state

    def TrainStep(self, nnx_graphdef:nnx.GraphDef, nxx_state:nnx.GraphState, train_batch:Tuple[jnp.ndarray, jnp.ndarray]):

        nnx_model,_ = nnx.merge(nnx_graphdef, nxx_state)

        latent_codes = jax.vmap(self.ComputeSingleLatentCode,(0,None,None))(train_batch,   
                                                                            nnx_model,
                                                                            self.inner_optax_optimizer)

        return self.OuterLoopStep(nnx_graphdef,nxx_state,train_batch,latent_codes)

    def TestStep(self, nnx_graphdef:nnx.GraphDef, nxx_state:nnx.GraphState,test_batch:Tuple[jnp.ndarray, jnp.ndarray]):
        
        nnx_model,nnx_optimizer = nnx.merge(nnx_graphdef, nxx_state)

        latent_codes = jax.vmap(self.ComputeSingleLatentCode,(0,None,None))(test_batch,   
                                                                            nnx_model,
                                                                            self.inner_optax_optimizer)
        (test_loss, test_batch_dict) = self.ComputeBatchLossValue(test_batch,latent_codes,nnx_model)
        _, state = nnx.split((nnx_model, nnx_optimizer))
        return test_batch_dict,state

    @print_with_timestamp_and_execution_time
    def Predict(self,batch_X:jnp.ndarray,num_latent_iterations:int):
        """
        Generates predictions for a batch of input data.

        This method computes the network's predictions for a batch of input data. 
        It maps the network outputs to the full degree of freedom (DoF) vector using the loss function.

        Parameters
        ----------
        batch_X : jnp.ndarray
            A batch of input data.

        Returns
        -------
        jnp.ndarray
            The predicted outputs, mapped to the full DoF vector.
        """
        def predict_single_sample(sample_x:jnp.ndarray):
            computed_sample_code = self.ComputeSingleLatentCode((sample_x,),
                                                          nn_model=self.flax_neural_network,
                                                          nn_optimizer=self.inner_optax_optimizer)
            nn_output = self.flax_neural_network(computed_sample_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
            return self.loss_function.GetFullDofVector(sample_x,nn_output)

        return jnp.array(jax.vmap(predict_single_sample)(batch_X))

    def Finalize(self):
        pass