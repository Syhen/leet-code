# -*- coding: utf-8 -*-
"""
create on 2020-04-16 17:45
author @66492
"""
import numpy as np


def binary_crossentropy(y_true, y_pred, eta=1e-7):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return - np.mean(y_true * np.log(y_pred + eta) + (1 - y_true) * np.log(1 - y_pred + eta))


if __name__ == '__main__':
    import keras

    print(binary_crossentropy([1, 0, 1] * 100 + [0, 1], [1, 1, 0] * 100 + [1, 0]))
    print(np.log(1.000000001))
    print(keras.losses.binary_crossentropy([1, 0, 1] * 100 + [1, 1], [1, 1, 0] * 100 + [1, 0]))
