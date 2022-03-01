#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:50:45 2021

@author: gdc17
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import matplotlib as mpl
import scipy.optimize as op

# Time parameters 
dt = 0.001
N = 20000
K = 20000
T = N*dt
time = np.linspace(-T/2,T/2,N)
i_minus = N/2
i_plus =  N
dt_samp = 100
N_samp = int((i_plus - i_minus)/(dt_samp))

# Parameters
sig = 1
beta = 3.4

# Initialise arrays
Z = np.zeros((2,K))
r0 = np.sqrt(np.random.uniform(0, 3, size=K))
thet = np.random.uniform(0, 2*np.pi, size=K)
Z[0,:] = r0*np.cos(thet); Z[1,:] = r0*np.sin(thet)
Z_vid = np.zeros((2, N_samp, K))
y = np.random.uniform(-1,1, size=(2,K))

# Define the random vector field 
def f(Z,W):
    x = Z[0]; y = Z[1]
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    noise1 = sig*W[0]
    noise2 = sig*W[1]
    f1 = x*(1-r)*(beta-r)/beta - y*(1+3*r2) + noise1
    f2 = y*(1-r)*(beta-r)/beta + x*(1+3*r2) + noise2
    return np.array([f1,f2])

#----------------- Solve system -----------------------------------------------        
for i in range(N-1):
    print(i)
    y_0 = y
    Z_o = Z
    y = y_0 + np.sqrt(dt)*np.random.normal(size=(2,K))
    Z = Z_o + dt*f(Z_o, np.sin(y)) 
    # save data for video 
    if i >= i_minus and i < i_plus and np.mod(i-i_minus, dt_samp)==0:
        j = int((i-i_minus)/dt_samp)
        Z_vid[:,j,:] = Z
        

#---------------- find mis ----------------------------------------------------
def g_plus(r):
    return r*(1-r)*(1-r/beta) + sig

def g_minus(r):
    return r*(1-r)*(1-r/beta)/beta - sig

r_plus = op.fsolve(g_plus,1)
r_minus = op.fsolve(g_minus,1)
theta = np.linspace(0,2*np.pi, num=1000)
E_plus =r_plus*[np.cos(theta), np.sin(theta)]
E_minus =r_minus*[np.cos(theta), np.sin(theta)]

#-------------- Plotting ------------------------------------------------------
Z1 = np.ndarray.flatten(Z_vid[0,:,:])
Z2 = np.ndarray.flatten(Z_vid[1,:,:])

plt.hist2d(Z1, Z2, bins=200, normed=True, norm=mpl.colors.LogNorm(),
           cmap=mpl.cm.Blues );
plt.plot(E_plus[0], E_plus[1], color = 'red', label="$\partial E$")
#plt.plot(E_minus[0], E_minus[1], color = 'red')
plt.xlim([-2.5,2.5]); plt.ylim([-2.5,2.5])
plt.colorbar()
plt.legend()

plt.figure()
r = np.linspace(0, beta, num = 1000)
plt.plot(r, r*(1-r)*(1-r/beta))
plt.plot(r, r*(1-r)*(1-r/beta)-sig)
plt.plot(r, r*(1-r)*(1-r/beta)+sig)
plt.axhline(y=0)
