#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 01:10:08 2021

@author: gdc17
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio

#---------------- Generate swtiching noise process ----------------------------
def W():
    y = np.zeros((N,2))
    for i in range(N-1):
        dW = np.sqrt(dt)*np.random.normal(size=2)
        y[i+1] = y[i] + dW
    y =   np.sin(y)
    return y

W = W()

def TP(r,W):
    # Time parameters 
    dt = 0.0005
    N = 200000
    K = 100000
    T = N*dt
    time = np.linspace(-T/2,T/2,N)
    i_minus = int(N/2) - 50000
    i_plus =  int(N/2) + 50000
    dt_samp = 100
    N_samp = int((i_plus - i_minus)/(dt_samp))

    # Parameters
    r = r; C = 10; sig = 1; beta = 3

    # Initialise arrays
    Z = np.zeros((2,K))
    r0 = np.sqrt(np.random.uniform(0, 1, size=K))
    thet = np.random.uniform(0, 2*np.pi, size=K)
    Z[0,:] = -C+r0*np.cos(thet); Z[1,:] = r0*np.sin(thet)
    Z_vid = np.zeros((2, N_samp, K))

    def q(t):                                 
        return C*(np.tanh(r*t + 1))

    def f(Z,W):
        x = Z[0]; y = Z[1]
        r2 = x**2 + y**2
        r = np.sqrt(r2)
        noise1 = sig*W[0]
        noise2 = sig*W[1]
        f1 = x*(1-r)*(1-r/beta) - y*(1+2*r2) + noise1
        f2 = y*(1-r)*(1-r/beta) + x*(1+2*r2) + noise2
        return np.array([f1,f2])

#----------------- Solve system ------------------------------------------        
    for i in range(N-1):
        Wi = np.tile(W[i], (K,1))
        qi = q(time[i])*np.ones(K)
        qix =  np.stack((qi, np.zeros(K)), 1).T
        Z_o = Z
        Z = Z_o + dt*f(Z_o - qix, Wi.T)  
        # save data for video 
        if i >= i_minus and i < i_plus and np.mod(i-i_minus, dt_samp)==0:
            j = int((i-i_minus)/dt_samp)
            Z_vid[:,j,:] = Z
  
#-------------------- Plotting ------------------------------------------------        
    tip_ind = np.argwhere(np.isnan(Z[0,:])).T[0]
    notip_ind = np.argwhere(np.isnan(Z[0,:])==False).T[0]
    Z_tip = Z_vid[:,:,tip_ind]
    Z_notip = Z_vid[:,:,notip_ind]
    x_nt = Z_notip[0,:,:]; y_nt = Z_notip[1,:,:]
    x_t = Z_tip[0,:,:];    y_t = Z_tip[1,:,:]
    
    prob_tip = np.size(tip_ind)/K
    return prob_tip

L = 201
K = 6
r = np.linspace(1,11,L)
tp = np.zeros((K,L))
for k in range(K):
    print(k)
    W1 = W()
    for l in range(L):
        print(l)
        tp[k,l] = TP(r[l], W1)


plt.plot(r,pt2, label='$\mathrm{TP}(r)$')
plt.plot(r,tp.T, linewidth=0.3)
plt.xlabel('$r$ (rate)'); plt.ylabel('Tipping Probability')
plt.legend()

plt.figure(figsize=(7,7))
N_plot = 10
plt.scatter(100,100, color='green', label='Non-Tipping Trajectory')
plt.scatter(100,100, color='red', label='Tipping Trajectory')
plt.scatter(x_nt[N_plot,:], y_nt[N_plot,:], s=0.001, color='green')
plt.scatter(x_t[N_plot,:], y_t[N_plot,:], s=0.001, color='red')
x1 = np.min(x_nt[N_plot,:]); x2 = np.max(x_nt[N_plot,:])
y1 = np.min(y_nt[N_plot,:]); y2 = np.max(y_nt[N_plot,:])
plt.ylim([y1-1,y2+1]); plt.xlim([x1-1,x2+1])
T_plot = time[i_minus + N_plot*dt_samp]
plt.text(x1, y1-.9, '$t = $%1.1f' %T_plot, fontsize=14 )
plt.legend()

video = 1
if video == 1:
    # Set spatial resolution
    x_res = 800
    y_res = 350
    xedges = np.linspace(-16,16, num=x_res+1)
    yedges = np.linspace(-5,2, num=y_res+1)
    # Convert to histogram like data
    H  = np.zeros((len(x_t), y_res,x_res, 3))
    for i in range(len(x_t)):
        print(i)
        H[i,:,:,1] = np.histogram2d(y_nt[i,:], x_nt[i,:], bins=(yedges, xedges))[0]
        H[i,:,:,0] = np.histogram2d(y_t[i,:], x_t[i,:], bins=(yedges, xedges))[0]
    H[H>0] = 1 ; H = H*255
    H=H.astype(dtype='uint8')
    imageio.mimwrite('random_rtipping.mp4', H , fps = 25)

