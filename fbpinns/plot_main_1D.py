#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 23:18:57 2021

@author: bmoseley
"""

# This module defines plotting functions for 1D FBPINN / PINN problems

# This module is used by plot_main.py (and subsequently main.py)

import numpy as np
import torch
import matplotlib.pyplot as plt

import plot_domain



def lim(mi,ma,factor=1,zero_center=False):
    c = 0 if zero_center else (mi+ma)/2
    w = factor*(ma-mi)/2
    return (c-w, c+w)

def _to_numpy(f):
    "Decorator which converts input tensors to numpy"
    def recurse(obj):
        if isinstance(obj, (list, tuple)):
            return [recurse(o) for o in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().copy()
        else:
            return obj
    def wrapper(*args):
        return f(*recurse(args))
    return wrapper


def _plot_setup(x_test, yj_true, yj_full, yj_test_losses, c):
    
    # get limits of plot
    yjlims = [lim(t.min(),t.max(), *c.PLOT_LIMS) for t in yj_true]
    xlim = np.min(x_test, axis=0), np.max(x_test, axis=0)
    
    # number of solution + derivative terms
    n_yj = len(yj_true)
    
    # loss data
    yj_test_losses = np.array(yj_test_losses)
    
    # boundary condition
    boundary = c.P.boundary_condition(torch.from_numpy(x_test), 
                                      *[torch.ones(t.shape)*c.Y_N[1]+c.Y_N[0] for t in yj_full], *c.BOUNDARY_N)
    
    return xlim, yjlims, n_yj, boundary, yj_test_losses


@_to_numpy
def plot_1D_FBPINN(x_test, yj_true,   xs, yjs, yjs_sum,   yj_full, yjs_full, ys_full_raw,   yj_test_losses,   c, D, i):
    
    _, yjlims, n_yj, boundary, yj_test_losses = _plot_setup(x_test, yj_true, yj_full, yj_test_losses, c)
    
    shape = (2+2*n_yj, 2)# nrows, ncols
    f1 = plt.figure(figsize=(12,2.5*shape[0]))
    
    # TRAIN PLOT
    
    # domain plot
    plt.subplot2grid(shape,(0,0))
    plot_domain.plot_1D(c.SUBDOMAIN_XS, D, create_fig=False)
    
    # individual models
    plt.subplot2grid(shape,(1,0))

    plt.plot(x_test[:,0], yj_true[0][:,0], label="Ground truth")
    
    for im,i1 in D.active_fixed_neighbours_ims:        
        plt.scatter(xs[i1][:,0], yjs[i1][0][:,0], s=20, color=plot_domain.colors[im], alpha=0.6)
        
    plt.legend()
    plt.title("[%i] Individual models before sum and BCs"%(i))
    
    # individual models after sum and BC
    for j in range(n_yj):# plot all yjs
        plt.subplot2grid(shape,(2+j,0))

        plt.plot(x_test[:,0], yj_true[j][:,0], label="Ground truth")
        
        for im,i1 in D.active_ims:            
            plt.scatter(xs[i1][:,0], yjs_sum[i1][j][:,0], s=20, color=plot_domain.colors[im], alpha=0.6)
            
        plt.ylim(*yjlims[j])
        plt.legend()
        plt.title("[%i] Individual models after sum and BCs $yj_{%i}$"%(i,j))
    
    # TEST PLOT
    
    # domain plot
    plt.subplot2grid(shape,(0,1))
    plot_domain.plot_1D(c.SUBDOMAIN_XS, D, create_fig=False)
    
    # individual models
    plt.subplot2grid(shape,(1,1))

    plt.plot(x_test[:,0], yj_true[0][:,0], label="Ground truth")
    
    for j,im in enumerate(D.active_fixed_ims):
        
        plt.plot(x_test[:,0], yjs_full[j][0][:,0], color=plot_domain.colors[im])
        
    plt.ylim(*yjlims[0])
    plt.legend()
    plt.title("[%i] Individual models after BCs (no sum)"%(i))
    
    # full model after sum and BC
    for j in range(n_yj):# plot all yjs
        plt.subplot2grid(shape,(2+j,1))
    
        plt.plot(x_test[:,0], yj_true[j][:,0], label="Ground truth")
        plt.plot(x_test[:,0], yj_full[j][:,0], label="Full solution")
        
        plt.ylim(*yjlims[j])
        plt.legend()
        plt.title("[%i] Full solution - $yj_{%i}$"%(i,j))
    
    # individual raw NN plot
    plt.subplot2grid(shape, (2+n_yj,0))
    
    for j,im in enumerate(D.active_fixed_ims):
        plt.plot(x_test[:,0], ys_full_raw[j][:,0], color=plot_domain.colors[im])
    
    plt.title("[%i] Individual models - Raw"%(i,))
    
    # Boundary response
    plt.subplot2grid(shape, (2+n_yj+1,0))
    
    plt.plot(x_test[:,0], boundary[0][:,0])
    
    plt.title("[%i] Boundary condition"%(i,))
    
    # loss plot
    for j in range(n_yj):# plot all yjs
        plt.subplot2grid(shape,(2+n_yj+j,1))
        
        plt.plot(yj_test_losses[:,0], yj_test_losses[:,3+j])
        
        plt.title("[%i] Test loss $yj_{%i}$"%(i,j))
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    return (("train-test",f1),)
    

@_to_numpy
def plot_1D_PINN(x_test, yj_true,   x, yj,   yj_full, y_full_raw,   yj_test_losses,   c, i):
    
    _, yjlims, n_yj, boundary, yj_test_losses = _plot_setup(x_test, yj_true, yj_full, yj_test_losses, c)
    
    shape = (2*n_yj, 2)# nrows, ncols
    f1 = plt.figure(figsize=(12,2.5*shape[0]))
    
    # TRAIN PLOT
    
    # individual models after sum and BC
    for j in range(n_yj):# plot all yjs
        plt.subplot2grid(shape,(j,0))
    
        plt.plot(x_test[:,0], yj_true[j][:,0], label="Ground truth")
        plt.scatter(  x[:,0],      yj[j][:,0], label="Individual solution", alpha=0.6)
        
        plt.ylim(*yjlims[j])
        plt.legend()
        plt.title("[%i] Full solution - $yj_{%i}$"%(i,j))
    
    # TEST PLOT
    
    # full model after sum and BC
    for j in range(n_yj):# plot all yjs
        plt.subplot2grid(shape,(j,1))
    
        plt.plot(x_test[:,0], yj_true[j][:,0], label="Ground truth")
        plt.plot(x_test[:,0], yj_full[j][:,0], label="Full solution")
        
        plt.ylim(*yjlims[j])
        plt.legend()
        plt.title("[%i] Full solution - $yj_{%i}$"%(i,j))
    
    # raw NN plot
    plt.subplot2grid(shape, (n_yj,0))
    
    plt.plot(x_test[:,0], y_full_raw[:,0])
    
    plt.title("[%i] Individual model - Raw"%(i,))
    
    # Boundary response
    plt.subplot2grid(shape, (n_yj+1,0))
    
    plt.plot(x_test[:,0], boundary[0][:,0])
    
    plt.title("[%i] Boundary condition"%(i,))
    
    # loss plot
    for j in range(n_yj):# plot all yjs
        plt.subplot2grid(shape,(n_yj+j,1))
        
        plt.plot(yj_test_losses[:,0], yj_test_losses[:,3+j])
        
        plt.title("[%i] Test loss $yj_{%i}$"%(i,j))
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
        
    return (("train-test",f1),)