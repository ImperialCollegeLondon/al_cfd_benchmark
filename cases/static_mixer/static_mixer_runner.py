# -*- coding: utf-8 -*-

import numpy as np
from active_learning_cfd.cfd_case import CFDCase

import os


class StaticMixer(CFDCase):
    mesher = "blockMesh", "extrudeMesh"
    solver = "simpleFoam"
    template = "staticMixer"
    parameter_names = ("lengthOverDiameter", "angle")
    output_list = (("CoV", "Mixing CoV at outlet: (.+)"),)

    def __call__(self, parameters):
        assert len(parameters) == len(self.parameter_names)
        parameter_dict = dict(zip(self.parameter_names, parameters))
        self.solve(parameter_dict)
        return np.log(self.results["CoV"])


if __name__ == "__main__":
    case = StaticMixer()
    lengthOverDiameter = 0.8
    angle = 1.0
    print("CoV = {}".format(case([length, angle])))
