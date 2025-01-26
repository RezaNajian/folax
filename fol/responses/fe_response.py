"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: January, 2025
 License: FOL/LICENSE
"""
from  .response import Response
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.loss_functions.fe_loss import FiniteElementLoss

class FiniteElementResponse(Response):

    def __init__(self, name: str, response_settings: dict, fe_loss: FiniteElementLoss):
        super().__init__(name)
        self.response_settings = response_settings
        self.fe_loss = fe_loss

    def Initialize(self,reinitialize=False) -> None:

        if self.initialized and not reinitialize:
            return
        self.initialized = True

    def ComputeAdjointBasedGradients(self):
        pass

    def Finalize(self) -> None:
        pass

