#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:37:21 2021

@author: Hiroshi Yokoyama
"""
from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
if os.name == 'posix': # for linux
    os.chdir('/home/user/Documents/Python_Scripts/sim_Kalman_lorenz/')
elif os.name == 'nt': # for windows
    os.chdir('D:/GitHub/nm_coupling_sim/sim_Kalman_lorenz/')


current_path = os.getcwd()
fig_save_dir = current_path + '/figures/demo_lorenz/' 
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

current_path = os.getcwd()
param_path   = current_path + '/save_data/' 
if os.path.exists(param_path)==False:  # Make the directory for figures
    os.makedirs(param_path)

    
import matplotlib.pylab as plt
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 26 # Font size
#%%
import sys
sys.path.append(current_path)

from my_modules.my_lorenz import *
from my_modules.ekf_lorenz import *
from scipy import signal as sig
import scipy.linalg
import math

import numpy as np
import joblib
import random

np.random.seed(0)
#%% load synthetic data
name     = []
ext      = []
for file in os.listdir(param_path):
    split_str = os.path.splitext(file)
    name.append(split_str[0])
    ext.append(split_str[1])
    
    print(split_str)
    
fullpath   = param_path + name[0] + ext[0]
param_dict = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
y          = param_dict['y']
time       = param_dict['t']
fs         = param_dict['fs']
dt         = param_dict['dt']
Nt         = len(y)
param_true = param_dict['param'] # exact value of satate variables 2 (parameters of Neural mass model)
Nstate     = (y.shape[1]) + param_true.shape[1]
#%%
# def main():
print(__file__ + " start!!")

#%%
UT         = 0.5
Q          = UT * np.diag([1,1,1,1,1,1])#
R          = (1 - UT) * np.eye(3) 

# Estimation parameter of EKF 
zEst       = np.zeros(3)
xEst       = np.array([0.0,0.0,0.0,20.0,10.0,0.0])
PEst       = Q

# history
x_pred      = np.zeros((Nt, Nstate))
y_pred      = np.zeros((Nt, 3))
loglike     = np.zeros(Nt)
x_pred[0,:] = xEst
y_obs       =  y + 0.1 * np.random.randn(Nt, 3)
for t in range(0,Nt):
    z = y_obs[t,:]
    xEst, PEst, zEst, S, R, LL = ekf_estimation(z, xEst, PEst, Q, R, UT, dt)
    # store data history
    x_pred[t,:] = xEst
    y_pred[t,:] = zEst
    loglike[t]  = LL
    print(t+1)
#%%
plt.plot(time, y[:,0]);  
plt.plot(time, y_pred[:,0]);
plt.xlabel('time (s)')
plt.ylabel('amplitude (a.u.)')
plt.show()
#%%
plt.plot(y[:,0], y[:,1], label='exact', linestyle = '--', zorder=2);
plt.plot(y_pred[:,0], y_pred[:,1],label='estimated', zorder=1);
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
plt.show()
#%%
plt.plot(time, param_true[:,-1], label='exact');
plt.plot(time, x_pred[:,-1], label='estimated')
plt.ylim(0, 30)
plt.xlabel('time (s)')
plt.ylabel('amplitude (a.u.)')
plt.title('parameter $r$')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
plt.show()
#%%
plt.plot(time, param_true[:,1], label='exact');
plt.plot(time, x_pred[:,-2], label='estimated')
plt.ylim(0, 30)
plt.xlabel('time (s)')
plt.ylabel('amplitude (a.u.)')
plt.title('parameter $b$')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
plt.show()
#%%
plt.plot(time, param_true[:,0], label='exact');
plt.plot(time, x_pred[:,-3], label='estimated')
plt.ylim(0, 30)
plt.xlabel('time (s)')
plt.ylabel('amplitude (a.u.)')
plt.title('parameter $\sigma$')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
plt.show()

# if __name__ == '__main__':
#     main()