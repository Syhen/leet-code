# -*- coding: utf-8 -*-
"""
create on 2020-04-17 16:02
author @66492
"""
import numpy as np

from leet.ml.tree import entropy, information_gain

from excamples.utils import preprocess_titanic_for_tree

X_train, X_test, y_train, y_test = preprocess_titanic_for_tree()

for i in range(X_train.shape[1]):
    print(i, information_gain(X_train[:, i], np.array(y_train).reshape(-1, ), normalized=True, func="entropy"))
