"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: January, 2025
 License: FOL/LICENSE
"""
from abc import ABC, abstractmethod
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *
from functools import partial
from jax import jit

class Geometry(ABC):

    def __init__(self, geometry_name: str) -> None:
        self.__name = geometry_name
        self.integration_method = "GI_GAUSS_1"
        self.gauss_integration_function = self.GaussIntegration1

    def GetName(self) -> str:
        return self.__name
    
    def SetGaussIntegrationMethod(self,gi_method:str) -> None:
        if gi_method == "GI_GAUSS_1":
            self.integration_method = "GI_GAUSS_1"
            self.gauss_integration_function = self.GaussIntegration1
        elif gi_method == "GI_GAUSS_2":
            self.integration_method = "GI_GAUSS_2"
            self.gauss_integration_function = self.GaussIntegration2
        elif gi_method == "GI_GAUSS_3":
            self.integration_method = "GI_GAUSS_2"
            self.gauss_integration_function = self.GaussIntegration3
        elif gi_method == "GI_GAUSS_4":
            self.integration_method = "GI_GAUSS_4"
            self.gauss_integration_function = self.GaussIntegration4
        elif gi_method == "GI_GAUSS_5":
            self.integration_method = "GI_GAUSS_5"
            self.gauss_integration_function = self.GaussIntegration5
        else:
            fol_error(f"{gi_method} integration method is not implemented.")

    def GetIntegrationData(self) -> None:
        return self.gauss_integration_function()

    @abstractmethod
    def GaussIntegration1(self) -> None:
        # gaussian integration with order 1 each geometry
        # supposed to have different integration method for
        # integrating.
        pass

    @abstractmethod
    def GaussIntegration2(self) -> None:
        # gaussian integration with order 2 each geometry
        # supposed to have different integration method for
        # integrating.
        pass

    @abstractmethod
    def GaussIntegration3(self) -> None:
        # gaussian integration with order 2 each geometry
        # supposed to have different integration method for
        # integrating.
        pass

    @abstractmethod
    def GaussIntegration4(self) -> None:
        # gaussian integration with order 2 each geometry
        # supposed to have different integration method for
        # integrating.
        pass

    @abstractmethod
    def GaussIntegration5(self) -> None:
        # gaussian integration with order 2 each geometry
        # supposed to have different integration method for
        # integrating.
        pass

    @abstractmethod
    def ShapeFunctionsValues(self,local_coordinates:jnp.ndarray) -> jnp.ndarray:
        pass
    
    @abstractmethod
    def ShapeFunctionsLocalGradients(self,local_coordinates:jnp.ndarray) -> jnp.ndarray:
        pass

    @partial(jit, static_argnums=(0,))
    def ShapeFunctionsGlobalGradients(self,points_coordinates:jnp.ndarray,local_coordinates:jnp.ndarray) -> jnp.ndarray:
        jac = self.Jacobian(points_coordinates,local_coordinates)
        inv_jac = jnp.linalg.inv(jac)
        local_gradients = self.ShapeFunctionsLocalGradients(local_coordinates)
        return jnp.dot(local_gradients,inv_jac)

    @partial(jit, static_argnums=(0,))
    def Jacobian(self,points_coordinates:jnp.ndarray,local_coordinates:jnp.ndarray) -> jnp.ndarray:
        dN_dxi = self.ShapeFunctionsLocalGradients(local_coordinates.flatten())
        return jnp.dot(dN_dxi.T, points_coordinates).T