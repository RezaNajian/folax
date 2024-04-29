from typing import Optional
from  .loss import Loss
import jax
import jax.numpy as jnp
from jax import jit,grad,vmap,jit,jacfwd,jacrev
from functools import partial
from abc import ABC, abstractmethod

class FiniteElementLoss(Loss):
    """FE-based losse

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model):
        super().__init__(name)
        self.fe_model = fe_model

    def Initialize(self) -> None:
        pass

    def Finalize(self) -> None:
        pass

    @abstractmethod
    def ComputeElementEnergy(self):
        pass
    
    @partial(jit, static_argnums=(0,))
    def compute_R(self,total_control_vars,total_primal_vars):
        return grad(self.compute_total_energy,argnums=1)(total_control_vars,total_primal_vars)
    
    @partial(jit, static_argnums=(0,))
    def compute_DR_DP(self,total_control_vars,total_primal_vars):
        return jax.jacfwd(self.compute_R,argnums=1)(total_control_vars,total_primal_vars)
    
    @partial(jit, static_argnums=(0,))
    def compute_DR_DC(self,total_control_vars,total_primal_vars):
        return jax.jacfwd(self.compute_R,argnums=0)(total_control_vars,total_primal_vars)

