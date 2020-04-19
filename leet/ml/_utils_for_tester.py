# -*- coding: utf-8 -*-
"""
create on 2020-04-19 14:49
author @66492
"""
from sklearn.preprocessing import LabelEncoder, StandardScaler

from leet.ml.datasets import load_titanic


def preprocess_titanic_for_tree():
    X_train, X_test, y_train, y_test = load_titanic(return_test_groundtrue=True)
    columns = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked", "Sex"]
    for col in ["Sex", "Embarked"]:
        label_encoder = LabelEncoder()
        # label_encoder.fit(np.array(list(set(X_train[col].fillna("null")))).reshape(-1, 1))
        label_encoder.fit(X_train[col].fillna("null").astype(str))
        train_sex = label_encoder.transform(X_train[col].fillna("null").astype(str))
        test_sex = label_encoder.transform(X_test[col].fillna("null").astype(str))
        X_train[col] = train_sex
        X_test[col] = test_sex

    X_train = X_train[columns].fillna(-1)
    X_test = X_test[columns].fillna(-1)
    return X_train, X_test, y_train, y_test


def preprocess_titanic_for_linear():
    X_train, X_test, y_train, y_test = load_titanic(return_test_groundtrue=True)
    columns = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked", "Sex"]
    for col in ["Sex", "Embarked"]:
        label_encoder = LabelEncoder()
        # label_encoder.fit(np.array(list(set(X_train[col].fillna("null")))).reshape(-1, 1))
        label_encoder.fit(X_train[col].fillna("null").astype(str))
        train_sex = label_encoder.transform(X_train[col].fillna("null").astype(str))
        test_sex = label_encoder.transform(X_test[col].fillna("null").astype(str))
        X_train[col] = train_sex
        X_test[col] = test_sex

    scaler = StandardScaler()
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    scaler.fit(X_train[columns])
    X_train = scaler.transform(X_train[columns])
    X_test = scaler.transform(X_test[columns])
    X_train = X_train[columns].fillna(-1)
    X_test = X_test[columns].fillna(-1)
    return X_train, X_test, y_train, y_test
