#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:22:21 2019

@author: marinettechen
"""
import numpy as np
def calcClosedFormSolution(X,y):

    n= X.shape[0]
    # X = 3500 x 784 and y = 3500x1
    # (748x3500)x(3500x784) = (784x784)
    # (784x784)x(784x3500) = (784x3500)
    # (784x3500)x(3500x1) = (784x1)
    # Target dimensions should be 784x1
    X = np.reshape(X,(n,-1))
    #print(np.shape(X),np.shape(y))
    w_l = np.linalg.inv((X.transpose() @ X)) @ X.transpose()
    w_l = w_l @ y
    return(w_l)