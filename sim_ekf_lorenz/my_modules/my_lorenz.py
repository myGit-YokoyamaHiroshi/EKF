# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:54:06 2020

@author: yokoyama
"""

from copy import deepcopy
from numpy.matlib import repmat
from numpy.random import randn, rand
import numpy as np

#%%
##############################################################################


def func_model(X, param):
    dX   = np.zeros(len(X))
    s    = param[0]
    b    = param[1]
    r    = param[2]
    
    x    = X[0]
    y    = X[1]
    z    = X[2]

    # sort order of dy
    dX[0] =    s * (y - x)
    dX[1] = -x*z + r*x - y
    dX[2] =  x*y - b*z
    
    return dX

def runge_kutta(dt, func, X_now, param):
    k1   = func(X_now, param)
    
    X_k2 = X_now + (dt/2)*k1
    k2   = func(X_k2, param)
    
    X_k3 = X_now + (dt/2)*k2
    k3   = func(X_k3, param)
    
    X_k4 = X_now + dt*k3
    k4   = func(X_k4, param)
    
    X_next = X_now + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return X_next

def euler_maruyama(dt, func, y, param, noise_scale):
    p      = noise_scale
    dw     = np.random.randn(y.shape[0])
    
    y_next = y + func(y, param) * dt + np.sqrt(dt) * p * dw
    
    return y_next
#%%
