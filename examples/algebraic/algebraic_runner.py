# -*- coding: utf-8 -*-
"""Algebraic function

This class defines a complicated function with a smooth step change.

"""

import numpy as np
from active_learning_cfd.cfd_case import CFDCase


class TwoParameterFunction(CFDCase):
    def __call__(self, features):
        x = features[0]
        y = features[1]
        return [
            (3.0 * np.sin(np.sin(3 * y)) - np.sin(2 * (x + y)) + (x ** 2 + y ** 2) / 4)
            * (0.5 * (np.tanh(10 * (x - 0.5)) + 1))
        ]
