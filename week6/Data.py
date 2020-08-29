#!/usr/bin/env python
# coding: utf-8

#Data Preprocessing

#This file contains all functions and classes for the pre-processing of the data

#import required libraries
from sklearn.model_selection import train_test_split

def data_loader(data):
    y = data['y']
    X = data.drop(columns=['y'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=49)
    return X_train,X_test, y_train, y_test







