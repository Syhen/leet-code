# -*- coding: utf-8 -*-
"""
create on 2020-04-16 16:09
author @66492
"""
import warnings

import numpy as np

from leet.ml import activations, losses


def _check_X_y(X, y):
    return


class LogisticRegression(object):
    def __init__(self, *, C=1.0, fit_interception=True, tol=1e-4, max_iter=100):
        self.C = C
        self.fit_interception = fit_interception
        self.coef_ = None
        self.tol = tol
        self.max_iter = max_iter
        self.intercept_ = None

    def _init_coef(self, X):
        coef = np.random.random((X.shape[1], 1))
        self.intercept_ = np.zeros((1,)) / 100
        if self.fit_interception:
            size = (1,)
            self.intercept_ = np.random.random(size) / 100
        return coef

    def _fit_one_iter(self, X, y, coef):
        # keepdims is important!!!!! or y_hat - y will be broadcast
        logit = np.sum(np.dot(X, coef), axis=1, keepdims=True) + self.intercept_
        y_hat = activations.sigmoid(logit)
        loss = losses.binary_crossentropy(y, y_hat)
        # dW vector implement has no `sum equation`
        dW = np.dot(X.T, y_hat - y) / y.shape[0]
        db = None
        if self.fit_interception:
            db = np.sum(y_hat - y) / y.shape[0]
        return loss, dW, db

    def _fit(self, X, y, learning_rate=1):
        X = np.array(X)
        y = np.array(y)
        coef = self._init_coef(X)
        pre_loss = 100
        loss = pre_loss
        for _ in range(self.max_iter):
            loss, dW, db = self._fit_one_iter(X, y, coef)
            # add verbose
            # print(_, ":", loss)
            coef -= learning_rate * dW
            if self.fit_interception:
                self.intercept_ -= learning_rate * db
            if 0 < pre_loss - loss < self.tol:
                break
            pre_loss = loss
        else:
            warnings.warn("model may not converge", RuntimeWarning)
        self.loss = loss
        self.coef_ = coef

    def fit(self, X, y):
        _check_X_y(X, y)
        self._fit(X, y)
        return self

    def predict(self, X):
        logit = np.sum(np.dot(X, self.coef_), axis=1) + self.intercept_
        y_hat = activations.sigmoid(logit)
        return (y_hat > 0.5).astype(int)


if __name__ == '__main__':
    from leet.ml.datasets.load_titanic import load_titanic
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    X_train, X_test, y_train, y_test = load_titanic(return_test_groundtrue=True)
    cabins = X_train.append(X_train).Embarked.value_counts()
    columns = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked", "Sex"]
    for col in ["Sex", "Embarked"]:
        label_encoder = LabelEncoder()
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

    model = LogisticRegression(max_iter=100, tol=1e-4)

    print("train model")
    model.fit(X_train, y_train)
    print(model.coef_)
    print(model.intercept_)
    print("my model:", accuracy_score(y_train, model.predict(X_train)))
    print("my model:", accuracy_score(y_test, model.predict(X_test)))

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(model.coef_)
    print(model.intercept_)
    print("sklearn:", accuracy_score(y_train, model.predict(X_train)))
    print("sklearn:", accuracy_score(y_test, model.predict(X_test)))
