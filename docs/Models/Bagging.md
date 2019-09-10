#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier


KNN = KNeighborsClassifier()
bagging_KNN = BaggingClassifier(base_estimator = KNN,
								n_estimators = 500,
								max_samples = 0.5,
								max_features = 0.5,
								bootstrap = True,
								bootstrap_features = False,
								oob_score = True,
								warm_start = False,
								n_jobs = None,
								random_state = None,
								verbose = 0)



DT = DecisionTreeClassifier()
bagging_DT = BaggingClassifier(base_estimator = DT,
							   n_estimators = 500,
							   max_samples = 0.5,
							   max_features = 0.5,
							   bootstrap = True,
							   bootstrap_features = False,
							   oob_score = False,
							   warm_start = False,
							   n_jobs = None,
							   random_state = None,
							   verbose = 0)







