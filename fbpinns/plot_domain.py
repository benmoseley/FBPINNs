#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:57:45 2021

@author: bmoseley
"""

# This module defines helper plotting functions for plotting FBPINN domains

# This module is used by the plot_main_ND.py modules

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from domains import itergrid

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']*100


# 1D plot domain in x space

def plot_1D(subdomain_xs, D, create_fig=True):
    "Plots a 1D domain"
    
    if create_fig: f = plt.figure(figsize=(12,5))
    else: f = plt.gcf()
        
    for im,ii in itergrid(D.nm):
    
        # plot subdomain
        plt.hlines(0, subdomain_xs[0][ii[0]], subdomain_xs[0][ii[0]+1], 
                   linewidth=1, colors=colors[im], alpha=0.5)
        
        # plot all active segments
        isegs = D.m[im]
        if isegs:# there can be duplicates across models fyi
            iiis = np.stack(np.unravel_index(isegs, D.onm), -1)# grid index of segments
            for iii in iiis:# for each segment
                s = D.segments[iii[0]]# (2,nd,nm)
                plt.hlines(0.1*(iii[0]+1), s[0,0,iii[1]], s[1,0,iii[1]], 
                           linewidth=2, colors=colors[iii[0]])
                
        # plot all active/fixed window functions
        if D.active[ii]:
            x = np.linspace(subdomain_xs[0][ii[0]], subdomain_xs[0][ii[0]+1], 100).reshape((100,1))
            w = D.w[im](torch.from_numpy(x))
            w = w.reshape((100,))
            plt.plot(x, w, color=colors[im])
        
        # plot all active/fixed mu, sigma
        if D.active[ii]:
            plt.scatter(D.n[im][0][0], 0, color=colors[im])
            plt.scatter(D.n[im][0][0]+D.n[im][1][0], 0, color=colors[im], s=200, alpha=0.4)
        
    return f
    
    
# 2D cross plot domain in x space

def plot_2D_cross_section(subdomain_xs, D, iaxes, create_fig=True):
    "Plots a 2D cross section of a nd>=2 domain"
    
    if len(iaxes) != 2: raise Exception("ERROR: iaxes incorrect format %s"%(iaxes))
    if D.nd <2: raise Exception("ERROR: requires D.nd >= 2!")
    a,b = iaxes
    
    if create_fig: f = plt.figure(figsize=(12,12))
    else: f = plt.gcf()

    ax = plt.gca()
    for im,ii in itergrid(D.nm):
        
        # plot subdomain
        rect = patches.Rectangle((subdomain_xs[a][ii[a]], subdomain_xs[b][ii[b]]), #xy
                                  subdomain_xs[a][ii[a]+1]-subdomain_xs[a][ii[a]], #width
                                  subdomain_xs[b][ii[b]+1]-subdomain_xs[b][ii[b]], #height
                                 linewidth=1, edgecolor=colors[im], facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        
        # plot all active segments
        isegs = D.m[im]
        if isegs:# there can be duplicates across models fyi
            iiis = np.stack(np.unravel_index(isegs, D.onm), -1)# grid index of segments
            for iii in iiis:# for each segment
                iii = tuple(iii)
                s = D.segments[iii[0]]# (2,nd,nm)
                rect = patches.Rectangle((s[(0,a)+iii[1:]],s[(0,b)+iii[1:]]), #xy (x,y)
                                          s[(1,a)+iii[1:]]-s[(0,a)+iii[1:]], #width (along x)
                                          s[(1,b)+iii[1:]]-s[(0,b)+iii[1:]], #height (along y)
                                         linewidth=2, edgecolor=colors[iii[0]], facecolor='none')
                ax.add_patch(rect)
                
        # plot all active/fixed window functions
        if D.active[ii]:
            xs = [np.linspace(subdomain_xs[i][ii[i]], subdomain_xs[i][ii[i]+1], 100) for i in range(D.nd)]
            xx = np.stack(np.meshgrid(*xs, indexing="ij"), -1)# (100,)xnd, nd
            x = xx.reshape((100**D.nd,D.nd))
            w = D.w[im](torch.from_numpy(x))
            w = w.reshape((100,)*D.nd)# (100,)xnd
            sl = tuple(slice(None) if i in [a,b] else 50 for i in range(D.nd))# slice out cross section axes, at middle location
            p = plt.pcolormesh(xx[sl+(a,)], xx[sl+(b,)], w[sl], shading="gouraud", cmap="gray", vmin=0.25, vmax=1)# as we can't overlap imshows
            
        # plot all active/fixed mu, sigma
        if D.active[ii]:
            plt.scatter(D.n[im][0][a],D.n[im][0][b], color=colors[im])
            plt.scatter(D.n[im][0][a]+D.n[im][1][a], D.n[im][0][b]+D.n[im][1][b], color=colors[im],s=200, alpha=0.4)
        
    ax.set_aspect("equal")
    getlim = lambda x: (x.min()-0.05*(x.max()-x.min()), x.max()+0.05*(x.max()-x.min()))
    (xmin, xmax), (ymin, ymax) = getlim(subdomain_xs[a]), getlim(subdomain_xs[b])
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.box(on=None)
    plt.xlabel(a); plt.ylabel(b)
    plt.colorbar(p)
    
    return f

# 2D plot domain in x space

def plot_2D(subdomain_xs, D, create_fig=True):
    "Plots a 2D domain"
    
    return plot_2D_cross_section(subdomain_xs, D, [0,1], create_fig)