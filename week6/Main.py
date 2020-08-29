#!/usr/bin/env python
# coding: utf-8

# This file imports all the necessary classes and functions from other files and
# automates the process of pre-processing, model training, and model prediction
import pandas as pd 
from Data import data_loader
from Model import logisticregr
from Model import xgboost_classifier
from Model import svc
from Model import decision_tree
from Model import random_forest

data = pd.read_csv('./data/preprocessed_data.csv')
X_train, X_test, y_train, y_test = data_loader(data)

#Models F1 Score
logisticregr(X_train, X_test, y_train, y_test)
xgboost_classifier(X_train, X_test, y_train, y_test)
svc(X_train, X_test, y_train, y_test)
decision_tree(X_train, X_test, y_train, y_test)
random_forest(X_train, X_test, y_train, y_test)



