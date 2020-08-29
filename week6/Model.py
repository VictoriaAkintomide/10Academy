#!/usr/bin/env python
# coding: utf-8

#Machine Learning Models

# This file contains all functions and classes to generate the three models

#importing all required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_recall_curve, f1_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from Data import data_loader
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier

def logisticregr(X_train,X_test, y_train, y_test):
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print('Logistic Regression F1-Score: %.3f' % score)


def xgboost_classifier(X_train,X_test, y_train, y_test):
    xgb_model = XGBoostClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print('XGBoost F1-Score: %.3f' % score)
    
    
def svc(X_train,X_test, y_train, y_test):
    svc_model = SVC(kernel = 'linear', C = 1.0)
    svc_model.fit(X_train, y_train)
    y_pred = svc_model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print('SVM F1-Score: %.3f' % score)


def decision_tree(X_train,X_test, y_train, y_test):
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print('Decision Tree F1-Score: %.3f' % score)

def random_forest(X_train,X_test, y_train, y_test):
    fr_model = RandomForestClassifier()
    fr_model.fit(X_train, y_train)
    y_pred = fr_model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print('Random Forest F1-Score: %.3f' % score)


