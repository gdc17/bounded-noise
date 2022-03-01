#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:23:43 2021

@author: gdc17
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio

def prob_tip(r):

    # Time parameters 
    dt = 0.0005
    N = 120000
    K = 50000
    T = N*dt
    time = np.linspace(-T/2,T/2,N)
    i_minus = int(N/2) - 50000
    i_plus =  int(N/2) + 50000
    dt_samp = 100
    N_samp = int((i_plus - i_minus)/(dt_samp))

    # Parameters
    r = r; C = 10; sig = 1; beta = 3

    # Initialise arrays
    Z = np.zeros((2,K)); Y = np.zeros((2,K))
    r0 = np.sqrt(np.random.uniform(0, 1.3, size=K))
    thet = np.random.uniform(0, 2*np.pi, size=K)
    Z[0,:] = r0*np.cos(thet); Z[1,:] = r0*np.sin(thet)
    Z_vid = np.zeros((2, N_samp, K))

    def q(t):                                 
        return C*(np.tanh(r*(t + 0*np.cos(t))) + 1)

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
        qi = q(time[i])*np.ones(K)
        qix =  np.stack((qi, np.zeros(K)), 1).T
        Z_o = Z; Y_o = Y
        Y = Y_o + np.random.normal(size=(2,K))*np.sqrt(dt)
        Z = Z_o + dt*f(Z_o - qix, np.sin(Y))  
        # save data for video 
        if i >= i_minus and i < i_plus and np.mod(i-i_minus, dt_samp)==0:
            j = int((i-i_minus)/dt_samp)
            Z_vid[:,j,:] = Z
  
    #-------------------- Plotting ------------------------------------------------        

    tip_ind = np.argwhere(np.isnan(Z[0,:])).T[0]
    notip_ind = np.argwhere(np.isnan(Z[0,:])==False).T[0]
    return np.size(tip_ind)/K

r = np.linspace(2,11, 201)
pt2 = np.zeros(201)
for i in range(201):
    print(i)
    pt2[i] = prob_tip(r[i])
    
plt.plot(r,pt2)
plt.ylabel('$\mathrm{TP}(r)$ (tipping probability)')
plt.xlabel('$r$ (rate)' )


Z_tip = Z_vid[:,:,tip_ind]
Z_notip = Z_vid[:,:,notip_ind]
x_nt = Z_notip[0,:,:]; y_nt = Z_notip[1,:,:]
x_t = Z_tip[0,:,:];    y_t = Z_tip[1,:,:]

plt.figure()
N_plot = 0
plt.scatter(x_nt[N_plot,:], y_nt[N_plot,:], s=0.05, label="does no tip")
plt.scatter(x_t[N_plot,:], y_t[N_plot,:], s=0.05, label="tips")
plt.xlim([-1.8,1.8]); plt.ylim([-1.8,1.8])

video = 1
if video == 1:
    # Set spatial resolution
    x_res = 600
    y_res = 425
    xedges = np.linspace(-1.7,5.7, num=x_res+1)
    yedges = np.linspace(-2,2, num=y_res+1)
    # Convert to histogram like data
    H  = np.zeros((len(x_t), y_res,x_res, 3))
    for i in range(len(x_t)):
        print(i)
        H[i,:,:,1] = np.histogram2d(y_nt[i,:], x_nt[i,:], bins=(yedges, xedges))[0]
        H[i,:,:,2] = np.histogram2d(y_t[i,:], x_t[i,:], bins=(yedges, xedges))[0]
    H[H>0] = 1 ; H = H*255
    H=H.astype(dtype='uint8')
    imageio.mimwrite('periodic_rtipping.mp4', H , fps = 25)

##----------------- Forwards in time -------------------------------------------
#for k in range(K):
#    print(k)
#    for i in range(N-1):
#        if dt*f1(x[i,k],y[i,k]) > 0.1:
#            print('time-step too big!!!!')
#        x[i+1,k] = x[i,k] + dt*f1(x[i,k]-q(time[i]) , y[i,k]) + dt*sig*W1[i]
#        y[i+1,k] = y[i,k] + dt*f2(x[i,k]-q(time[i]) , y[i,k]) + dt*sig*W2[i]
#        if x[i+1,k]**2 > 1000 or y[i+1,k]**2 > 1000:
#            x[i+1:N,k] = np.NaN
#            y[i+1:N,k] = np.NaN
#            break
#
##------------------ Backwards in time -----------------------------------------
#back = 1
#if back == 1:
#    xr = np.zeros((N,K)); yr = np.zeros((N,K))
#    r0 = np.random.uniform(2,3, size=K)
#    thet = np.random.normal(0, 2*np.pi, size=K)
#    xr[N-1,:] = r0*np.cos(thet)
#    yr[N-1,:] = r0*np.sin(thet)
#    for k in range(K):
#        print(k)
#        for i in range(N-1):
#            ir = N-1-i
#            if dt*f1(x[i,k],y[i,k]) > 0.1:
#                print('time-step too big!!!!')
#            xr[ir-1,k] = xr[ir,k] - dt*f1(xr[ir,k]-q(time[ir]), yr[ir,k]) - dt*sig*W1[ir]
#            yr[ir-1,k] = yr[ir,k] - dt*f2(xr[ir,k]-q(time[ir]), yr[ir,k]) - dt*sig*W2[ir]
