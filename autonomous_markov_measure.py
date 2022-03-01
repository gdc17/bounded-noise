#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 13:33:19 2021

@author: gdc17
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import matplotlib as mpl

def W():
    y = np.zeros((N,2))
    for i in range(N-1):
        dW = np.sqrt(dt)*np.random.normal(size=2)
        y[i+1] = y[i] + dW
    y =   np.sin(y)
    return y

W = W()

# Time parameters 
dt = 0.001
N = 50000
K = 500000
T = N*dt
time = np.linspace(-T/2,T/2,N)
i_minus = int(N/2) - 50000
i_plus =  int(N/2) + 50000
dt_samp = 100
N_samp = int((i_plus - i_minus)/(dt_samp))

# Parameters
sig = 1
beta = 3.5

# Initialise arrays
Z = np.zeros((2,K))
r0 = np.sqrt(np.random.uniform(0, 1.3, size=K))
thet = np.random.uniform(0, 2*np.pi, size=K)
Z[0,:] = r0*np.cos(thet); Z[1,:] = r0*np.sin(thet)
Z_vid = np.zeros((2, N_samp, K))

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

#----------------- Solve system ------------------------------------------        
for i in range(N-1):
    Wi = np.tile(W[i], (K,1))
    Z_o = Z
    Z = Z_o + dt*f(Z_o, Wi.T)  
    # save data for video 
    if i >= i_minus and i < i_plus and np.mod(i-i_minus, dt_samp)==0:
        j = int((i-i_minus)/dt_samp)
        Z_vid[:,j,:] = Z
  
#-------------------- Plotting ------------------------------------------------        
x = Z_vid[0,:,:];  y = Z_vid[1,:,:]  
video = 0
if video == 1:
    # Set spatial resolution
    x_res = 425
    y_res = 425
    xedges = np.linspace(-2,2, num=x_res+1)
    yedges = np.linspace(-2,2, num=y_res+1)
    # Convert to histogram like data
    H  = np.zeros((len(x), y_res,x_res, 3))
    for i in range(len(x)):
        print(i)
        H[i,:,:,1] = np.histogram2d(y[i,:], x[i,:], bins=(yedges, xedges))[0]
    H[H>0] = 1 ; H = H*255
    H=H.astype(dtype='uint8')
    imageio.mimwrite('periodic_rtipping.mp4', H , fps = 25)


plt.hist2d(Z[0], Z[1], bins=150, norm=mpl.colors.LogNorm(), 
           normed=True)
plt.plot(E_plus[0], E_plus[1], color = 'red', label="$\partial E$")
plt.xlim([-2.5,2.5]); plt.ylim([-2.5,2.5])
plt.legend()
plt.colorbar()


plt.figure()
r = np.linspace(0, beta, num = 1000)
plt.plot(r, r*(1-r)*(beta-r)/beta)
plt.plot(r, r*(1-r)*(beta-r)/beta-sig)
plt.plot(r, r*(1-r)*(beta-r)/beta+sig)
plt.axhline(y=0)


