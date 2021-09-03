#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 14:53:08 2021

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
    os.chdir('D:/GitHub/nm_ekf/sim_Kalman_lorenz/')


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
from scipy import signal as sig
import numpy as np
import joblib
import random

np.random.seed(0)
#%%
# paramter settings for beta oscillation (Jansen & Rit, 1995)

# alpha oscillation
# param = np.array([10, 8/3, 28])
# param = np.array([5.55936242,  1.04726008, 24.84312008])
param = np.array([5.56,  1.05, 24.80])
#%%
fs          = 1000
dt          = 1/fs
Nt          = int(100*fs)# + 100
noise_scale = 1
t           = np.arange(0,Nt,1)/fs

y           = np.zeros((Nt, 3))
dy          = np.zeros((Nt, 3))
param_save  = np.zeros((Nt, 3))

y_init      = np.array([0.5622,0.7893,0.3509])
y[0,:]      = y_init
dy[0, :]    = func_model(y_init, param)
param_save[0,:] = param

for i in range(1, Nt):  
    y_now      = y[i-1, :]
    # y_next     = euler_maruyama(dt, func_model, y_now, param, noise_scale)
    y_next     = runge_kutta(dt, func_model, y_now, param)
    dy[i, :]   = func_model(y_now, param)
    y[i, :]    = y_next
    param_save[i,:] = param
    
#%%
############# save_data
param_dict          = {} 
param_dict['fs']    = fs
param_dict['dt']    = dt
param_dict['Nt']    = Nt
param_dict['param'] = param_save
param_dict['t']     = t
param_dict['dy']    = dy
param_dict['y']     = y

save_name   = 'synthetic_data'
fullpath_save   = param_path + save_name 
np.save(fullpath_save, param_dict)
#%%
plt.plot(t, y)
plt.xlabel('time (s)')
plt.ylabel('simulated amp. (a.u.)')
plt.show()
#%%
plt.plot(y[:,0], y[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')