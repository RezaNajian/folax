"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Iterator
import matplotlib.pyplot as plt
from jax import jit
from functools import partial
from jax.example_libraries import optimizers
import jaxopt
from tqdm import trange
from jax.flatten_util import ravel_pytree
import os

class DeepNetwork(ABC):
    """Base abstract deep network class.

    The base abstract deep network class has the following responsibilities.
        1. Initalizes and finalizes the model.

    """
    def __init__(self,control_name:str,load_NN_params:bool,NN_params_file_name:str,working_directory:str) -> None:
        self.__name = control_name
        self.working_directory = working_directory
        self.load_NN_params = load_NN_params
        self.NN_params_file_name = NN_params_file_name
        self.train_history_dict = {}
        self.initialized = False

    def GetName(self) -> str:
        return self.__name
    
    def CreateBatches(self,data: jnp.ndarray, batch_size: int) -> Iterator[jnp.ndarray]:
        """Creates batches for the given inputs.

        """
        for i in range(0, data.shape[0], batch_size):
            yield data[i:i+batch_size, :]

    @abstractmethod
    def Initialize(self) -> None:
        """Initializes the network.

        This method initializes the network. This is only called once in the whole training process.

        """
        pass

    def InitializeParameters(self) -> None:
        """Initializes the networks parameters.

        This method initializes the network. This is only called once in the whole training process.

        """
        if self.load_NN_params:
            _, unravel_params = ravel_pytree(self.NN_params)
            self.NN_params = unravel_params(jnp.load(os.path.join(self.working_directory, self.NN_params_file_name)))

    @partial(jit, static_argnums=(0,))
    def StepAdam(self,opt_itr,opt_state,x_batch,NN_params):
        (total_loss, batch_dict), final_grads = self.ComputeTotalLossValueAndGrad(NN_params,x_batch)
        updated_state = self.opt_update(opt_itr, final_grads, opt_state)
        updated_NN_params = self.get_params(updated_state)
        return updated_NN_params,updated_state,batch_dict
    
    @partial(jit, static_argnums=(0,))
    def StepLBFGS(self,opt_itr,opt_state,x_batch,NN_params):
        updated_NN_params, updated_state = self.solver.update(params=NN_params, state=opt_state, batch_input=x_batch)
        return updated_NN_params,updated_state,updated_state.aux

    def Run(self,X_train,batch_size,num_epochs,convergence_criterion,relative_error,
            absolute_error,plot_list,plot_rate,plot_save_rate):
        
        pbar = trange(num_epochs)
        step_iteration = 0
        converged = False
        for epoch in pbar:
            batches_dict = {}
            # now loop over batches
            batch_index = 0 
            for batch in self.CreateBatches(X_train, batch_size):
                self.NN_params,self.opt_state,step_dict = self.step_function(step_iteration,self.opt_state,batch,self.NN_params)
                # fill the batch dict
                if batch_index == 0:
                    for key, value in step_dict.items():
                        batches_dict[key] = [value]
                else:
                    for key, value in step_dict.items():
                        batches_dict[key].append(value)
                step_iteration += 1
                batch_index += 1

            for key, value in batches_dict.items():
                if "max" in key:
                    batches_dict[key] = [max(value)]
                elif "min" in key:
                    batches_dict[key] = [min(value)]
                elif "avg" in key:
                    batches_dict[key] = [sum(value)/len(value)]
                elif "total" in key:
                    batches_dict[key] = [sum(value)]

            if not self.train_history_dict:
                self.train_history_dict.update(batches_dict)
            else:
                for key, value in batches_dict.items():
                    self.train_history_dict[key].extend(value)

            pbar.set_postfix({convergence_criterion: self.train_history_dict[convergence_criterion][-1]})

            # check for absolute and relative errors and convergence
            if self.train_history_dict[convergence_criterion][-1]<absolute_error:
                converged = True
            elif epoch>0 and abs(self.train_history_dict[convergence_criterion][-1] -
                 self.train_history_dict[convergence_criterion][-2])<relative_error:
                converged = True

            # plot the histories
            if epoch %plot_save_rate == 0 or epoch==num_epochs-1 or converged:
                self.PlotTrainHistory(plot_list,plot_rate)

            if converged:
                break

    def Train(self,X_train, batch_size=100, num_epochs=1000, learning_rate=0.01,optimizer="adam",
              convergence_criterion="total_loss",relative_error=1e-8,absolute_error=1e-8,plot_list=[],
              plot_rate=1,plot_save_rate=1000,save_NN_params=True, NN_params_save_file_name="NN_params.npy") -> None:
        """Trains the network.

        This method trains the network.

        """

        # here specify the optimizer
        if optimizer =="adam":
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
            self.opt_state = self.opt_init(self.NN_params)
            self.step_function = self.StepAdam
        elif optimizer=="LBFGS":
            self.solver = jaxopt.LBFGS(fun=self.ComputeTotalLossValueAndGrad,value_and_grad=True,has_aux=True,stepsize=-1,
                                       linesearch="backtracking",stop_if_linesearch_fails=True,maxiter=num_epochs,verbose=False)
            self.opt_state = self.solver.init_state(init_params=self.NN_params,batch_input=X_train)
            self.step_function = self.StepLBFGS

        # now run the training
        self.Run(X_train,batch_size,num_epochs,convergence_criterion,
                 relative_error,absolute_error,plot_list,plot_rate,plot_save_rate)

        # save optimized NN parameters
        if save_NN_params:
            flat_params, _ = ravel_pytree(self.NN_params)
            jnp.save(os.path.join(self.working_directory,NN_params_save_file_name), flat_params)

    def ReTrain(self,X_train,batch_size=100,num_epochs=1000,convergence_criterion="total_loss",
                relative_error=1e-8,absolute_error=1e-8,reset_train_history=False,plot_list=[],
                plot_rate=1,plot_save_rate=1000,save_NN_params=True,NN_params_save_file_name="NN_params.npy") -> None:
        """ReTrains the network.

        This method retrains the network.

        """
        if reset_train_history:
            self.train_history_dict = {}

        # now run the training
        self.Run(X_train,batch_size,num_epochs,convergence_criterion,relative_error,
                 absolute_error,plot_list,plot_rate,plot_save_rate)

        # save optimized NN parameters
        if save_NN_params:
            flat_params, _ = ravel_pytree(self.NN_params)
            jnp.save(os.path.join(self.working_directory,NN_params_save_file_name), flat_params)

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the network.

        This method finalizes the network. This is only called once in the whole training process.

        """
        pass

    def PlotTrainHistory(self,plot_list,plot_rate):
        plt.figure(figsize=(10, 5))
        for key,value in self.train_history_dict.items():
            if len(value)>0 and (len(plot_list)==0 or key in plot_list):
                plt.semilogy(value[::plot_rate], label=key) 
        plt.title("Training History")
        plt.xlabel(str(plot_rate) + " Epoch")
        plt.ylabel("Log Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.working_directory,"training_history.png"), bbox_inches='tight')
        plt.close()



