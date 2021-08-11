#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:31:37 2021

@author: bmoseley
"""

# This module defines plotting functions for 2D FBPINN / PINN problems

# This module is used by plot_main.py (and subsequently main.py)

import matplotlib.pyplot as plt
import matplotlib.tri as tri

import plot_domain
from plot_main_1D import _to_numpy, _plot_setup

import sys
sys.path.insert(0, '../shared_modules/')
from helper import Timer


def _plot_test_im(y, xlim, ylim, c):
    plt.imshow(y.reshape(c.BATCH_SIZE_TEST).T, # need to transpose because torch.meshgrid uses np indexing="ij"
               origin="lower", extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
               cmap="viridis", vmin=ylim[0], vmax=ylim[1])

def _fix_plot(xlim):
    plt.colorbar()
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")


@_to_numpy
def plot_2D_FBPINN(x_test, yj_true,   xs, yjs, yjs_sum,   yj_full, yjs_full, ys_full_raw,   yj_test_losses,   c, D, i):
    
    xlim, yjlims, n_yj, boundary, yj_test_losses = _plot_setup(x_test, yj_true, yj_full, yj_test_losses, c)
    
    shape = (2+n_yj, max(4,n_yj))# nrows, ncols
    f1 = plt.figure(figsize=(4*shape[1],3*shape[0]))
    
    # domain plot
    plt.subplot2grid(shape,(0,0))
    plot_domain.plot_2D(c.SUBDOMAIN_XS, D, create_fig=False)
    
    # TRAIN / TEST PLOT
    
    with Timer("tri"):
        triangles = {im:tri.Triangulation(xs[i1][:,0], xs[i1][:,1]) for im,i1 in D.active_ims}
    
    for j in range(n_yj):# plot all 
        
        # individual models after sum and BC
        plt.subplot2grid(shape,(1+j,0))
        
        for im,i1 in D.active_ims:
            plt.tricontourf(triangles[im], yjs_sum[i1][j][:,0], cmap="viridis", vmin=yjlims[j][0], vmax=yjlims[j][1])
        
        _fix_plot(xlim)
        plt.title("[%i] Individual models after sum and BCs $yj_{%i}$"%(i,j))
        
        # full model after sum and BC
        plt.subplot2grid(shape,(1+j,1))
        
        _plot_test_im(yj_full[j][:,0], xlim, yjlims[j], c)
        
        _fix_plot(xlim)
        plt.title("[%i] Full solution - $yj_{%i}$"%(i,j))
        
        # ground truth
        plt.subplot2grid(shape,(1+j,2))
        
        _plot_test_im(yj_true[j][:,0], xlim, yjlims[j], c)
        
        _fix_plot(xlim)
        plt.title("[%i] Ground truth - $yj_{%i}$"%(i,j))
        
        # difference
        plt.subplot2grid(shape,(1+j,3))
        
        _plot_test_im(yj_full[j][:,0]-yj_true[j][:,0], xlim, yjlims[j], c)
        
        _fix_plot(xlim)
        plt.title("[%i] Difference - $yj_{%i}$"%(i,j))
        
    # raw NN plot
    plt.subplot2grid(shape, (0,1))
    
    for j,im in enumerate(D.active_fixed_ims):
        plt.hist(ys_full_raw[j][:,0], bins=100, color=plot_domain.colors[im], alpha=0.6)
    
    plt.yticks([])
    plt.title("[%i] Individual models - Raw"%(i,))
    
    # Boundary response
    plt.subplot2grid(shape, (0,2))
    
    _plot_test_im(boundary[0][:,0], xlim, (None, None), c)
    
    _fix_plot(xlim)
    plt.title("[%i] Boundary condition"%(i,))
    
    # loss plot
    for j in range(n_yj):# plot all yjs
        plt.subplot2grid(shape,(1+n_yj,j))
        
        plt.plot(yj_test_losses[:,0], yj_test_losses[:,3+j])
        
        plt.title("[%i] Test loss $yj_{%i}$"%(i,j))
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    return (("train-test",f1),)


@_to_numpy
def plot_2D_PINN(x_test, yj_true,   x, yj,   yj_full, y_full_raw,   yj_test_losses,   c, i):
    
    xlim, yjlims, n_yj, boundary, yj_test_losses = _plot_setup(x_test, yj_true, yj_full, yj_test_losses, c)
    
    shape = (2+n_yj, max(4,n_yj))# nrows, ncols
    f1 = plt.figure(figsize=(4*shape[1],3*shape[0]))
    
    # TRAIN / TEST PLOT
    
    with Timer("tri"):
        triangle = tri.Triangulation(x[:,0], x[:,1])
    
    for j in range(n_yj):# plot all 
    
        # individual models after sum and BC
        plt.subplot2grid(shape,(1+j,0))
    
        plt.tricontourf(triangle, yj[j][:,0], cmap="viridis", vmin=yjlims[j][0], vmax=yjlims[j][1])
        
        _fix_plot(xlim)
        plt.title("[%i] Full solution - $yj_{%i}$"%(i,j))
        
        # full model after sum and BC
        plt.subplot2grid(shape,(1+j,1))
        
        _plot_test_im(yj_full[j][:,0], xlim, yjlims[j], c)
        
        _fix_plot(xlim)
        plt.title("[%i] Full solution - $yj_{%i}$"%(i,j))
        
        # ground truth
        plt.subplot2grid(shape,(1+j,2))
        
        _plot_test_im(yj_true[j][:,0], xlim, yjlims[j], c)
        
        _fix_plot(xlim)
        plt.title("[%i] Ground truth - $yj_{%i}$"%(i,j))
        
        # difference
        plt.subplot2grid(shape,(1+j,3))
        
        _plot_test_im(yj_full[j][:,0]-yj_true[j][:,0], xlim, yjlims[j], c)
        
        _fix_plot(xlim)
        plt.title("[%i] Difference - $yj_{%i}$"%(i,j))
        
    # raw NN plot
    plt.subplot2grid(shape, (0,1))
    
    _plot_test_im(y_full_raw[:,0], xlim, (None, None), c)
    
    _fix_plot(xlim)
    plt.title("[%i] Individual model - Raw"%(i,))
    
    # Boundary response
    plt.subplot2grid(shape, (0,2))
    
    _plot_test_im(boundary[0][:,0], xlim, (None, None), c)
    
    _fix_plot(xlim)
    plt.title("[%i] Boundary condition"%(i,))
    
    # loss plot
    for j in range(n_yj):# plot all yjs
        plt.subplot2grid(shape,(1+n_yj,j))
        
        plt.plot(yj_test_losses[:,0], yj_test_losses[:,3+j])
        
        plt.title("[%i] Test loss $yj_{%i}$"%(i,j))
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    return (("train-test",f1),)



