"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: January, 2025
 License: FOL/LICENSE
"""
from  .fe_response import FiniteElementResponse
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.loss_functions.fe_loss import FiniteElementLoss

class MechanicalResponse(FiniteElementResponse):

    def __init__(self, name: str, response_settings: dict, fe_loss: FiniteElementLoss):
        super().__init__(name,response_settings,fe_loss)


    def ComputeValue(self):
        pass

