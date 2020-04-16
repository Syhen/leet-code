# -*- coding: utf-8 -*-
"""
create on 2020-04-16 16:10
author @66492
"""
import numpy as np


def sigmoid(x):
    """
    `sigmoid(x) = \frac{1}{1 + e^{-x}}`
    :param x: number or vector.
    :return: np.array.
    """
    return 1 / (1 + np.exp(-x))
