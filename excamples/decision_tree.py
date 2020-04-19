# -*- coding: utf-8 -*-
"""
create on 2020-04-19 14:50
author @66492
"""
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from excamples.utils import preprocess_titanic_for_tree

X_train, X_test, y_train, y_test = preprocess_titanic_for_tree()

# model = HistGradientBoostingClassifier()
model = DecisionTreeClassifier(min_samples_leaf=2)
model.fit(X_train, y_train)
print("sklearn tree:", accuracy_score(y_train, model.predict(X_train)))
print("sklearn tree:", accuracy_score(y_test, model.predict(X_test)))

# from sklearn.tree import export_graphviz
# import pydotplus
#
# dot_data = export_graphviz(model, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png("tree.png")
