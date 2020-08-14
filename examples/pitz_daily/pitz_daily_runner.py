# -*- coding: utf-8 -*-
"""Pitz Daily

This case uses the pitzDaily example from the OpenFOAM tutorials 
and varies two parameters: Reynolds number and height of the inlet. 
It returns the pressure difference between inlet and outlet.

"""

import numpy as np
from active_learning_cfd.cfd_case import CFDCase

import os


class PitzDaily(CFDCase):
    mesher = "blockMesh"
    solver = "simpleFoam"
    template = "pitzDaily"
    parameter_names = ("reynolds", "entryHeight")
    output_list = (("deltaP", "subtract\(p\) = (.+)"),)

    def __call__(self, parameters):
        assert len(parameters) == len(self.parameter_names)
        parameter_dict = dict(zip(self.parameter_names, parameters))
        parameter_dict["reynolds"] = np.power(10, parameter_dict["reynolds"])
        self.solve(parameter_dict)
        return self.results["deltaP"]


if __name__ == "__main__":
    case = PitzDaily()
    reynolds = 50800.0
    entryHeight = 25.4
    print("deltaP = {}".format(case([np.log10(reynolds), entryHeight])))
