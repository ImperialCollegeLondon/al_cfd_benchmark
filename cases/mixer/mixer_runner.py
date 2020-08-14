# -*- coding: utf-8 -*-

import numpy as np
from active_learning_cfd.cfd_case import CFDCase

import os


class Mixer(CFDCase):
    mesher = "blockMesh"
    solver = "SRFSimpleFoam"
    template = "mixer"
    parameter_names = ("reynolds", "nonDimGap", "n", "nBlades")
    clone_files = ("*.orig", "parameters", "*.py", "initial", "dynamicCode")
    output_list = (("Cm", "Cm    = (.*)"),)

    def __call__(self, parameters):
        assert len(parameters) == len(self.parameter_names)
        parameter_dict = dict(zip(self.parameter_names, parameters))
        parameter_dict["reynolds"] = np.exp(parameter_dict["reynolds"])
        self.solve(parameter_dict)
        return np.log(2.0 * np.pi * self.results["Cm"])


if __name__ == "__main__":
    case = Mixer()
    reynolds = 10.0
    nonDimGap = 0.2
    n = 1.0
    nBlades = 4
    Np = np.exp(case([np.log(reynolds), nonDimGap, n, nBlades]))
    print(f"Np = {Np}")
