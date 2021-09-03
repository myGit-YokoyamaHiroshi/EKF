#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:49:09 2021

@author: user
"""

import numpy as np
import scipy 
import math

def inv_use_cholensky(M):
    L     = np.linalg.cholesky(M)
    L_inv = np.linalg.inv(L)
    M_inv = np.dot(L_inv.T, L_inv)
    
    return M_inv



########################################################
def Jacobian(X, s, b, r):
    x     = X[0]
    y     = X[1]
    z     = X[2]
    
    #### Calculate Jacobian Jx = df/dx
    J_x = np.array([
                    [ -s,  s,  0],
                    [r-z, -1, -x],
                    [  y,  x, -b]
                    ])
    
    #### Calculate Jacobian Jpar = df/d par    
    J_par = np.array([
                      [y-x, 0, 0],
                      [  0, 0, x],
                      [  0,-z, 0]
                      ])
    
    ##### combine two Jacobian matrix
    J     = np.hstack((J_x, J_par))
    return J


###################
def predict(X, P, Q, dt):
    x     = X[:3]
    par   = X[3:]
    
    Npar  = len(par)
    Nx    = len(x)
    
    s     = par[0]
    b     = par[1]
    r     = par[2]
    
    J     = Jacobian(X, s, b, r)
    
    fx    = J @ X
    x     = x + fx * dt + 0.1 * np.sqrt(dt) * np.random.randn(x.shape[0])
    par   = par 
    
    tmp   = np.hstack((np.zeros((Npar,Nx)), np.eye(Npar)))
    F     = np.vstack((J, tmp))
    
    XPred = np.hstack((x, par))
    PPred = F @ P @ F.T + Q
    
    return XPred, PPred

def update(z, X, P, R, alp):
    Ndim    = len(z)
    H       = np.hstack((np.eye(3), np.zeros((3,3))))
    
    zPred   = H @ X
    y       = z - zPred # prediction error of observation model
    
    R       = (1 - alp) * R + alp * np.dot(y, y)
    
    S       = H @ P @ H.T + R
    
    S_inv   = np.linalg.inv(S)
        
    K       = P @ H.T @ S_inv
    X_new   = X + K @ y
    P_new   = P - K @ H @ P # P - K @ H @ K
    
    ### log-likelihood
    _, logdet = np.linalg.slogdet(S)
    
    loglike   = -0.5 * (np.log(2*np.pi) + logdet + y @ S_inv@ y)
    
    return X_new, P_new, zPred, S, R, loglike

def ekf_estimation(z_now, X_now, P_now, Q, R, UT, dt):
    
    # Prediction step (estimate state variable)
    XPred, PPred = predict(X_now, P_now, Q, dt)
    
    # Update state (Update parameters)
    X_new, P_new, z_hat, S, R, loglike = update(z_now, XPred, PPred, R, UT)
    
    return X_new, P_new, z_hat, S, R, loglike
