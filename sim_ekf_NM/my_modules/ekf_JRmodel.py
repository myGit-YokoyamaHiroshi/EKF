#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:49:09 2021

@author: user
"""

import numpy as np
from copy import deepcopy
def inv_use_cholensky(M):
    L     = np.linalg.cholesky(M)
    L_inv = np.linalg.inv(L)
    M_inv = np.dot(L_inv.T, L_inv)
    
    return M_inv

def Sigm(v):
    v0   = 6
    vmax = 5
    r    = 0.56
    sigm = vmax / (1 + np.exp(r * ( v0 - v )))
    
    return sigm

def Sigm_diff(v):
    r    = 0.56
    vmax = 5
    sigm_diff = r/vmax * Sigm(v) * (vmax - Sigm(v))
    
    return sigm_diff

def Jacobian(x, par, dt):
    c1    = 135
    c2    = 0.8  * c1
    c3    = 0.25 * c1
    c4    = 0.25 * c1
    
    A       = par[0]
    a       = par[1]
    B       = par[2]
    b       = par[3]
    u       = par[4]
    
    #### Calculate Jacobian Jx = df/dx
    O     = np.zeros((3,3))
    I     = np.eye(3)
    diag  = np.diag([-2*a, -2*a, -2*b])
    dGdx  = np.array([
                      [                        -a*a, A*a*Sigm_diff(x[1]-x[2]),  -A*a*Sigm_diff(x[1]-x[2])],
                      [A*a*c1*c2*Sigm_diff(c1*x[0]),                     -a*a,                          0],
                      [B*b*c3*c4*Sigm_diff(c3*x[0]),                        0,                       -b*b]
                      ])
    
    term1 = np.hstack((O, I))
    term2 = np.hstack((dGdx, diag))
    
    J_x = np.vstack((term1,term2))
    #### Calculate Jacobian Jpar = df/d par
    term3 = np.zeros((3, len(par)))
    term4 = np.array([
                      [     a*Sigm(x[1]-x[2]),      A*Sigm(x[1]-x[2])-2*x[3]-2*a*x[0],                   0,                                   0,   0],
                      [a*(u+c2*Sigm(c1*x[0])), A*(u+c2*Sigm(c1*x[0]))-2*x[4]-2*a*x[1],                   0,                                   0, A*a],
                      [                     0,                                      0,  b*c4*Sigm(c3*x[0]),  B*c4*Sigm(c3*x[0])-2*x[5]-2*b*x[2],   0]
                      ])
    J_par = np.vstack((term3, term4))
    ##### combine two Jacobian matrix
    J     = np.hstack((J_x, J_par))
    return J

###################
def predict(X, P, Q, dt):    
    x       = deepcopy(X[:6])
    par     = deepcopy(X[6:])
    
    Npar    = len(par)
    Nx      = len(x)
    
    J       = Jacobian(x, par, dt)
    tmp     = np.zeros((Npar, Nx + Npar))#np.hstack((np.zeros((Npar,Nx)), np.eye(Npar)))
    F       = np.vstack((J, tmp))
    F       = np.eye(len(X)) + F*dt  # ~=  exp (F * dt)
    X_new   = F @ X # X0 * exp (F * dt)
    
    x_new   = X_new[:6]#x + X_new[:6] * dt + eta * np.sqrt(dt) * np.random.normal(loc=0, scale=1, size=Nx)
    par_new = X_new[6:]
    
    XPred = np.hstack((x_new, par_new))
    PPred = F @ P @ F.T + Q
    
    
    return XPred, PPred

def update(z, X, P, R, alp):
    H     = np.array([[0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0]])
    zPred = H @ X
    y     = z - zPred # prediction error of observation model
    S     = H @ P @ H.T + R
    
    S_inv = np.linalg.inv(S)
        
    K     = P @ H.T @ S_inv
    X_new = X + K @ y
    P_new = P - K @ H @ P # P - K @ S @ K.T
    
    R     = (1-alp) * R + alp * y**2
    
    ### log-likelihood
    _, logdet = np.linalg.slogdet(S)
    
    loglike   = -0.5 * (np.log(2*np.pi) + logdet + y @ S_inv@ y)
    
    return X_new, P_new, zPred, S, R, loglike

def ekf_estimation(z_now, X_now, P_now, Q, R, UT, dt):
    
    # Prediction step (estimate state variable)
    XPred, PPred = predict(X_now, P_now, Q, dt)
    
    # Update state (Update parameters)
    X_new, P_new, z_hat, S, R, loglike = update(z_now, XPred, PPred, R, UT)
    
    
    # # Update state (Update parameters)
    # X_, P_, z_hat, S, R = update(z_now, X_now, P_now, R, UT)
    
    # # Prediction step (estimate state variable)
    # X_new, P_new = predict(X_, P_, Q, dt)
    
    return X_new, P_new, z_hat, S, R, loglike
