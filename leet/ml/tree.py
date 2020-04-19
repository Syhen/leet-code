# -*- coding: utf-8 -*-
"""
create on 2020-04-16 11:26
author @66492
"""
from collections import Counter
import logging

import numpy as np
import pandas as pd
from copy import deepcopy

from leet.base.tree import BinaryTreeNode

FORMAT = '%(asctime)-15s: %(message)s'
logging.basicConfig(format=FORMAT, level="DEBUG")
logger = logging.getLogger(__name__)


def entropy(y, x=None):
    num = len(y)
    if x is None:
        counter = Counter(y)
        return -sum(p / num * np.log(p / num) for p in counter.values())
    groups = pd.DataFrame({"x": x, "y": y}).groupby("x")
    return sum([group.shape[0] / num * entropy(group.y.values) for _, group in groups])


def gini(y, x=None):
    num = len(y)
    if x is None:
        counter = Counter(y)
        return 1 - sum([(p / num) ** 2 for p in counter.values()])
    groups = pd.DataFrame({"x": x, "y": y}).groupby("x")
    return sum([group.shape[0] / num * gini(group.y.values) for _, group in groups])


def information_gain(x, y, normalized=True):
    """Information gain

    :param x: array.
    :param y: array.
    :param normalized: bool. default True. if True, return information gain ratio
    :return: float.
    """
    gain = entropy(y) - entropy(y, x)
    if not normalized:
        return gain
    groups = pd.DataFrame({"x": x, "y": y}).groupby("x")
    num = len(y)
    return gain / - sum([group.shape[0] / num * np.log(group.shape[0] / num) for _, group in groups])


class SplitPointFinder(object):
    ALGORITHMS = ("dichotomy",)

    def __init__(self, criterion="gini", algorithm="dichotomy"):
        if algorithm not in self.ALGORITHMS:
            raise ValueError("algorithm should be %s" % ", ".join(self.ALGORITHMS))
        self.criterion = criterion
        self.criterion_function_ = gini if criterion == "gini" else information_gain
        self.splitter_function_ = min if criterion == "gini" else max
        self.algorithm = algorithm

    def _find_point_from_middle(self, feature: np.ndarray, target: np.ndarray):
        """二分法寻找最佳切分点

        二分法找的点，泛化性能不好

        :param feature:
        :param target:
        :return: float.
        """
        feature = feature.reshape(-1, )
        target = target.reshape(-1, )
        distinct_vals = sorted(set(feature))
        logger.debug("distinct values: %s" % distinct_vals)
        scores = []
        for i in range(len(distinct_vals) - 1):
            point = (distinct_vals[i] + distinct_vals[i + 1]) / 2
            score = self.criterion_function_((feature <= point).astype(int), target)
            scores.append((point, score))
        return self.splitter_function_(scores, key=lambda x: x[1])[0]

    def find_split_point(self, feature, target):
        if self.algorithm == "dichotomy":
            return self._find_point_from_middle(feature, target)


# 决策树生成算法
# ID3、C4.5、CART
class DecisionTreeClassifier(object):
    # 对于连续性变量，主要使用分箱、排序后分割
    # 对于类别行变量，主要是：每一个类别都生成新的分支、将类别也进行划分
    # 某个值 大于 小于多少
    def __init__(self, criterion="gini", max_depth=None, category_columns=None, min_sample_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.category_columns = category_columns
        self._splitter = SplitPointFinder(criterion=criterion)
        self.root_ = None
        self.min_sample_split = min_sample_split

    def _choose_feature(self, X, y, feature_indexes):
        logger.debug("feature_indexes: %s" % feature_indexes)
        features = []
        for i in feature_indexes:
            score = self._splitter.criterion_function_(X[:, i], y)
            features.append((i, score))
        logger.debug("features: %s" % features)
        return self._splitter.splitter_function_(features, key=lambda x: x[1])[0]

    def _stop_generate_node(self, X, y: np.ndarray, feature_indexes, feature_idx=None):
        # TODO: 添加min_node_leaf
        if not feature_indexes:
            return True, Counter(y).most_common(1)[0][0]
        if len(np.unique(y)) == 1:
            return True, y[0]
        if feature_idx is not None:
            if len(np.unique(X[:, feature_idx])) == 1:
                return True, Counter(y).most_common(1)[0][0]
        if len(np.unique(X)) == 1:
            return True, Counter(y).most_common(1)[0][0]
        return False, None

    def _change_feature_index(self, X, y, feature_indexes, feature_idx):
        features = deepcopy(feature_indexes)
        if len(np.unique(y)) == 1:
            features.remove(feature_idx)
        elif len(np.unique(X[:, feature_idx])) == 1:
            features.remove(feature_idx)
        return features

    def _fit(self, X, y, feature_indexes):
        # TODO: feature_indexes 多次利用
        if X.shape[0] < self.min_sample_split:
            return BinaryTreeNode(
                "leaf",
                {
                    "n_samples": len(X),
                    "class": Counter(y).most_common(1)[0][0]
                }, 1
            )
        stop_, class_ = self._stop_generate_node(X, y, feature_indexes)
        if stop_:
            return BinaryTreeNode(
                "leaf",
                {
                    "n_samples": len(X),
                    "class": class_
                }, 1
            )
        feature_idx = self._choose_feature(X, y, feature_indexes)
        stop_, class_ = self._stop_generate_node(X, y, feature_indexes, feature_idx)
        if stop_:
            return BinaryTreeNode(
                "leaf",
                {
                    "n_samples": len(X),
                    "class": class_
                }, 1
            )
        best_point = self._splitter.find_split_point(X[:, feature_idx], y)
        logger.debug("feature_index: %s, best_point: %s" % (feature_idx, best_point))
        node = BinaryTreeNode(
            "{feature_idx}<={best_point}".format(feature_idx=feature_idx, best_point=best_point),
            {
                "n_samples": len(X),
                "class": self._stop_generate_node(X, y, feature_indexes)[1],
                "feature": feature_idx,
                "best_point": best_point
            }, 1
        )

        X1, y1 = X[np.where(X[:, feature_idx] <= best_point)], y[np.where(X[:, feature_idx] <= best_point)]
        X2, y2 = X[np.where(X[:, feature_idx] > best_point)], y[np.where(X[:, feature_idx] > best_point)]

        feature_indexes1 = self._change_feature_index(X1, y1, feature_indexes, feature_idx)
        logger.debug("feature left after change: %s" % feature_indexes1)
        node.left = self._fit(X1, y1, feature_indexes1)
        feature_indexes2 = self._change_feature_index(X2, y2, feature_indexes, feature_idx)
        logger.debug("feature right after change: %s" % feature_indexes2)
        node.right = self._fit(X2, y2, feature_indexes2)
        return node

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, )
        self.root_ = self._fit(X, y, list(range(X.shape[1])))
        return self

    def _predict(self, i, node):
        if node is None:
            return
        if node.val["class"] is not None:
            return node.val["class"]
        feature_idx = node.val["feature"]
        best_point = node.val["best_point"]
        if i[feature_idx] <= best_point:
            return self._predict(i, node.left)
        else:
            return self._predict(i, node.right)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict(i, self.root_) for i in X])


if __name__ == '__main__':
    import time

    from sklearn.metrics import accuracy_score

    from leet.ml._utils_for_tester import preprocess_titanic_for_tree

    logger.setLevel("INFO")

    X_train, X_test, y_train, y_test = preprocess_titanic_for_tree()
    t1 = time.time()
    model = DecisionTreeClassifier(criterion="gini", min_sample_split=4)
    model.fit(X_train, y_train)
    print(time.time() - t1)

    print("my model:", accuracy_score(y_train, model.predict(X_train)))
    print("my model:", accuracy_score(y_test, model.predict(X_test)))

    from sklearn.tree import DecisionTreeClassifier

    t2 = time.time()
    model = DecisionTreeClassifier(min_samples_split=4, min_samples_leaf=1)
    model.fit(X_train, y_train)
    print(time.time() - t2)

    print("sklearn:", accuracy_score(y_train, model.predict(X_train)))
    print("sklearn:", accuracy_score(y_test, model.predict(X_test)))

    # def _to_dict(node: BinaryTreeNode):
    #     if node is None:
    #         return {}
    #     d = {
    #         node.key: node.val,
    #         "children": [_to_dict(node.left), _to_dict(node.right)]
    #     }
    #     return d
    #
    # print(_to_dict(model.root_))
