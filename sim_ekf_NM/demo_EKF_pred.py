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
    os.chdir('/home/user/Documents/Python_Scripts/sim_Kalman_JansenRit/')
elif os.name == 'nt': # for windows
    os.chdir('D:/GitHub/nm_coupling_sim/sim_Kalman_JansenRit/')


current_path = os.getcwd()
fig_save_dir = current_path + '/figures/demo_JRmodel/' 
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

from my_modules.my_jansen_rit import *
from my_modules.ekf_JRmodel import *
from scipy import signal as sig
import scipy.linalg
import math

import numpy as np
import joblib
import random


def main():
    np.random.seed(0)
    #%% load synthetic data
    name     = []
    ext      = []
    for file in os.listdir(param_path):
        split_str = os.path.splitext(file)
        name.append(split_str[0])
        ext.append(split_str[1])
        
        print(split_str)
        
    fullpath       = param_path + name[0] + ext[0]
    param_dict     = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
    eeg            = param_dict['eeg']
    time           = param_dict['t']
    fs             = param_dict['fs']
    dt             = param_dict['dt']
    Nt             = len(eeg)
    x_true         = param_dict['y']     # exact value of satate variables 1 (numerical solution of Neural mass model)
    param_true     = param_dict['param'] # exact value of satate variables 2 (parameters of Neural mass model)
    
    # eeg            = eeg - eeg.mean()
    Nstate         = (x_true.shape[1]) + param_true.shape[1]
    #%%
    
    print(__file__ + " start!!")
    # Estimation parameter of EKF 
    
    A          = 3.25
    a          = 100
    B          = 22
    b          = 50
    p          = 220
    
    UT         = 1E-6
    Q          = UT * np.eye(Nstate)
    R          = (1-UT) * 0.1 + UT * 1
    # Q          = np.diag(np.hstack((1E-3 * np.ones(6), 1E-3*np.array([A**2, a**2, B**2, b**2, p**2]))))
    # R          = (0.2 * eeg.std())**2
    
    xEst       = np.zeros(Nstate)
    PEst       = Q
    xEst[6:]   = np.array([A, a, B, b, p]) 
    
    #%%
    # history
    x_pred    = np.zeros((Nt, Nstate))
    eeg_pred  = np.zeros(Nt)
    
    eeg_observe = eeg
    x_pred[0,:] = xEst
    loglike     = np.zeros(Nt)
    for t in range(1,Nt):
        z = eeg_observe[t-1]#np.array([eeg_observe[t], 100, 50, 220])#np.array([eeg_observe[t], 5, 5])
        xEst, PEst, zEst, S, R, LL = ekf_estimation(z, xEst, PEst, Q, R, UT, dt)
        # store data history
        x_pred[t,:] = xEst
        eeg_pred[t] = zEst[0]
        loglike[t]  = LL
        
        print(t+1)
        #%%
    param_pred = x_pred[:,6:]
    #%%
    plt.plot(time, eeg_observe, label='exact', zorder=1);
    plt.plot(time, eeg_pred,    label='estimated', zorder=2);
    plt.xlabel('time (s)')
    plt.ylabel('amplitude (a.u.)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
    plt.savefig(fig_save_dir + 'eeg.png', bbox_inches="tight")
    plt.savefig(fig_save_dir + 'eeg.svg', bbox_inches="tight")
    plt.show()
    #%%
    fig_name = ['A', 'a', 'B', 'b', 'p']
    ylims    = np.array([[3, 4.5],
                         [98, 102],
                         [20.5, 22.5],
                         [48, 52],
                         [120, 320]])
    
    for i in range(len(fig_name)):
        plt.plot(time, param_true[:,i], label='exact', zorder=1); 
        plt.plot(time, param_pred[:,i], label='estimated', zorder=2);
        plt.xlabel('time (s)')
        plt.ylabel('amplitude (a.u.)')
        plt.ylim(ylims[i,:])
        plt.title('$' + fig_name[i] + '(t)$')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
        plt.savefig(fig_save_dir + 'param_' + str(i+1) +'.png', bbox_inches="tight")
        plt.savefig(fig_save_dir + 'param_' + str(i+1) +'.svg', bbox_inches="tight")
        plt.show()

#%%
if __name__ == '__main__':
    main()