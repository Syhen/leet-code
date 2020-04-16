# -*- coding: utf-8 -*-
"""
create on 2020-04-16 12:23
author @66492
"""
import os
import pandas as pd

from leet.settings import BASE_PATH


def load_titanic(return_test_groundtrue=False):
    COLUMNS = "Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked".split(",")
    TARGET = ["Survived"]
    path = os.path.join(BASE_PATH, "ml", "datasets", "titanic")
    df_train = pd.read_csv(os.path.join(path, "train.csv"))
    X_train = df_train[COLUMNS]
    y_train = df_train[TARGET]
    df_test = pd.read_csv(os.path.join(path, "test_groundtrue.csv"))
    X_test = df_test[COLUMNS]
    y_test = df_test[TARGET]
    if return_test_groundtrue:
        return X_train, X_test, y_train, y_test
    y_test[TARGET] = 0
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    load_titanic(return_test_groundtrue=True)
