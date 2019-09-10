#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

# data
iris = datasets.load_iris()

# model
ab = AdaBoostClassifier(n_estimators = 100)

# hold out split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
													test_size = 0.3,
													random_state = 27149)
ab.fit(X_train, y_train)
accuracy = ab.score(X_test, y_test)
print("Hold Out 分割数据集模型在测试集上的准确率为：%0.03f" % accuracy)
