"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""
from abc import ABC, abstractmethod
import jax.numpy as jnp
from functools import partial
from jax import jit,jacfwd
import jax
from fol.tools.decoration_functions import *

class Control(ABC):
    """
    Base abstract class for control parameterizations.

    This class defines the common interface for control objects used in FoLax.
    A control maps a vector of *control variables* to a set of *controlled
    variables* that are typically consumed by a loss function or solver
    (for example, spatially varying material parameters or boundary values).

    Concrete control implementations derive from this class and must implement
    the abstract methods :meth:`Initialize`, :meth:`ComputeControlledVariables`,
    and :meth:`Finalize`.

    The base class manages:
    - A unique control name.
    - Initialization state tracking.
    - Bookkeeping for the number of control variables and controlled variables.
    - Convenience methods for batched evaluation and Jacobian computation.

    Derived classes are expected to set ``num_control_vars`` and
    ``num_controlled_vars`` during initialization.
    """
    def __init__(self, control_name: str) -> None:
        self.__name = control_name
        self.initialized = False
        self.num_control_vars = None
        self.num_controlled_vars = None

    def GetName(self) -> str:
        """
        Return the name of the control.

        Returns:
            str:
                Name identifier of the control instance.
        """
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        """
        Initialize the control.

        This method prepares all internal data structures required to evaluate
        the control. It is intended to be called once before the control is
        used, typically at the beginning of a training or optimization process.

        Derived classes must implement this method and set:
        - ``self.num_control_vars``
        - ``self.num_controlled_vars``

        Returns:
            None
        """
        pass

    def GetNumberOfVariables(self):
        """
        Return the number of control variables.

        Returns:
            int:
                Number of independent control variables.
        """
        return self.num_control_vars

    def GetNumberOfControlledVariables(self):
        """
        Return the number of controlled variables.

        Returns:
            int:
                Number of variables produced by the control mapping.
        """
        return self.num_controlled_vars

    @abstractmethod
    def ComputeControlledVariables(self,variable_vector:jnp.array) -> None:
        """
        Compute controlled variables from a control vector.

        This method maps the input control vector to the corresponding
        controlled variables. The interpretation of both the input vector
        and the output array is defined by the derived control class.

        Args:
            variable_vector (jax.numpy.ndarray):
                Control variable vector of length ``num_control_vars``.

        Returns:
            jax.numpy.ndarray:
                Controlled variables with length ``num_controlled_vars``.
        """
        pass

    def ComputeBatchControlledVariables(self,batch_variable_vector:jnp.array) -> None:
        """
        Compute controlled variables for a batch of control vectors.

        This method applies :meth:`ComputeControlledVariables` independently
        to each entry in the batch using vectorized evaluation.

        Args:
            batch_variable_vector (jax.numpy.ndarray):
                Batch of control vectors with shape
                ``(batch_size, num_control_vars)``.

        Returns:
            jax.numpy.ndarray:
                Batch of controlled variables with shape
                ``(batch_size, num_controlled_vars)``.
        """
        return jax.vmap(self.ComputeControlledVariables,(0))(batch_variable_vector)

    @partial(jit, static_argnums=(0,))
    def ComputeJacobian(self,control_vec):
        """
        Compute the Jacobian of the control mapping.

        This method computes the Jacobian matrix of
        :meth:`ComputeControlledVariables` with respect to the control
        variables using automatic differentiation.

        Args:
            control_vec (jax.numpy.ndarray):
                Control vector of length ``num_control_vars``.

        Returns:
            jax.numpy.ndarray:
                Jacobian matrix with shape
                ``(num_controlled_vars, num_control_vars)``.
        """
        return jnp.squeeze(jacfwd(self.ComputeControlledVariables,argnums=0)(control_vec))

    @abstractmethod
    def Finalize(self) -> None:
        """
        Finalize the control.

        This method is called once at the end of a training or optimization
        process. Derived classes may override it to release resources or
        perform cleanup. The default base class does not implement any
        finalization logic.

        Returns:
            None
        """
        pass



