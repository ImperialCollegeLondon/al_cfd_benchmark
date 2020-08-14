# -*- coding: utf-8 -*-

import numpy as np
from active_learning_cfd.cfd_case import CFDCase

import os


class Orifice(CFDCase):
    mesher = "blockMesh", "extrudeMesh","potentialFoam"
    solver = "simpleFoam"
    template = "orifice"
    parameter_names = ("sigma","sTod")
    output_list = (("deltaP", "subtract\(p\) = (.+)"),)

    def __call__(self, parameters):
        assert len(parameters) == len(self.parameter_names)
        parameter_dict = dict(zip(self.parameter_names, parameters))
        self.solve(parameter_dict)
        return self.results["deltaP"]


if __name__ == "__main__":
    case = Orifice()
    sigma = 0.54
    sTod = 0.2
    print("deltaP = {}".format(case([sigma, sTod])))
