import pandas as pd
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate


def fit_rf(X_train, X_test, y_train, max_depth = 2, random_state = 1):
    clf = RandomForestClassifier(max_depth=2, random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, clf

def fit_rf_gs_cv(X_train, X_test, y_train, param_grid = {
    'max_depth' : [2,3,4,5],
    'criterion' :['gini', 'entropy'],
    'random_state' : [69]
}, cv = 5):
    clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=cv)
    clf.fit(X=X_train, y=y_train)
    y_pred = clf.predict(X_test)
    return y_pred, clf

def generate_classification_report(y_test, y_pred_test):
    print(classification_report(y_test, y_pred_test))
