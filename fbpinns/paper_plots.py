#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 20:43:25 2021

@author: bmoseley
"""

# This module contains fancy helper functions for the paper plots

# This module is used by Paper plots.ipynb

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from domains import itergrid
from plot_main_1D import lim

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray', 'tab:pink', 'tab:olive', 'tab:cyan']*100
letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)", "(l)", "(m)", "(n)", "(o)", "(p)", "(q)", "(r)", "(s)", "(t)", "(u)", "(v)", "(w)", "(x)", "(y)", "(z)"]


# helper plotting functions used across all examples

def savefig(f, tag, ext="pdf", dpi=100):
    f.savefig("plots/%s.%s"%(tag,ext), bbox_inches='tight', pad_inches=0.1, dpi=dpi)

def get_lim(x_test, yj_true, c):
    yjlims = np.array([lim(t.min(),t.max(), *c.PLOT_LIMS) for t in yj_true])
    xlim = np.min(x_test.numpy(), axis=0), np.max(x_test.numpy(), axis=0)
    return xlim, yjlims

def plot_loss(losses_labels_icolors, title, loc, istep=0):
    for loss,label,icolor in losses_labels_icolors:
        plt.plot(loss[:,0+istep], loss[:,3], color=colors[icolor], label=label)
    plt.yscale("log")
    if loc != "off": 
        plt.legend(loc=loc)
    plt.xlabel({0:"Training step", 1:"Number of weight updates", 2:"FLOPS"}[istep])
    plt.ylabel("L1 loss")
    plt.title(title)

def plot_im(y, xlim, ylim, batch_size, title, it=None, cblabel=None, xlabel="$x_{1}$", ylabel="$x_{2}$"):
    y = y.numpy().reshape(batch_size)
    if it is not None: y = y[:,:,it]
    plt.imshow(y.T,# need to transpose because torch.meshgrid uses np indexing="ij"
               origin="lower", extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
               cmap="viridis", vmin=ylim[0], vmax=ylim[1])
    cb = plt.colorbar()
    if cblabel is not None:
        cb.set_label(cblabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def plot_2D_domain_cross_section(subdomain_xs, D, title, iaxes=[0,1], loc=None, ncol=1, line_labels=True):
    a,b = iaxes
    icolor = {1:-4,2:-3,4:-2,8:-1}
    linewidths = {0:1, 1:2, 2:2}
    linestyles = {0:"-", 1:"-", 2:"--"}
    segment_n = [D.segments_models[ioa].shape[0] for ioa in range(D.N_ORDERS)]
    ax = plt.gca()
    
    for im,ii in itergrid(D.nm):
        
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
                                         linewidth=2, edgecolor=None, 
                                         facecolor=colors[icolor[segment_n[iii[0]]]])
                ax.add_patch(rect)
        
        # plot all subdomains
        linewidth, linestyle = linewidths[D.active[ii]], linestyles[D.active[ii]]# style by active
        plt.vlines([subdomain_xs[a][ii[a]], subdomain_xs[a][ii[a]+1]],
                    subdomain_xs[b][ii[b]], subdomain_xs[b][ii[b]+1],
                   linewidth=linewidth, linestyle=linestyle, edgecolor="k")
        plt.hlines([subdomain_xs[b][ii[b]], subdomain_xs[b][ii[b]+1]],
                    subdomain_xs[a][ii[a]], subdomain_xs[a][ii[a]+1],
                   linewidth=linewidth, linestyle=linestyle, edgecolor="k")
    
    # set axes
    ax.set_aspect("equal")
    getlim = lambda x: (x.min()-0.01*(x.max()-x.min()), x.max()+0.01*(x.max()-x.min()))
    (xmin, xmax), (ymin, ymax) = getlim(subdomain_xs[a]), getlim(subdomain_xs[b])
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
    plt.box(on=None)
    plt.xlabel(["$x_{1}$", "$x_{2}$", "$t$"][a])
    plt.ylabel(["$x_{1}$", "$x_{2}$", "$t$"][b])
    plt.title(title)
    
    # add legend
    if loc is not None:
        
        # subdomain labels
        if line_labels:
            for label, a in [("inactive model", 0), ("active model", 1), ("fixed model", 2)]:
                if a in D.active:
                    plt.hlines(-99,1,2, linewidth=linewidths[a], linestyle=linestyles[a], 
                               color="k", label=label)# dummy glyphs
        
        # segment labels
        for ic in list(icolor.keys())[::-1]:
            if ic in segment_n:
                label = "1 model" if ic == 1 else "%i overlapping models"%(ic)
                rect = patches.Rectangle((-99,-99),1,1, facecolor=colors[icolor[ic]], label=label)# dummy glyphs
                ax.add_patch(rect)
    
        handles, labels = ax.get_legend_handles_labels()
        order = list(range(len(handles)))[::-1]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=loc, ncol=ncol)


        
# helper plotting functions used for 1D examples

def plot_1D_output(x_test, yj_true, yj, label, title, j=0, ylabel="$u$", icolor=1):
    plt.plot(x_test[:,0], yj_true[j][:,0], label="Exact solution")
    plt.plot(x_test[:,0], yj[j][:,0], color=colors[icolor], label=label)
    plt.xlabel("$x$")
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    plt.title(title)

def plot_1D_FBPINN_individual(x_test, yjs, title, active=None):
    for im,yj in enumerate(yjs):
        if active is not None:
            if active[im] == 1: lw = 2; ls = "-"
            elif active[im] == 2: lw = 2; ls = "--"
            else: lw = 0; ls = "-"
        else: lw = None; ls = "-"
        plt.plot(x_test[:,0], yj[0][:,0], ls, linewidth=lw)
    if active is not None:
        plt.plot(0, "-", color="k", label="Active models")
        plt.plot(0, "--", color="k", label="Fixed models")
        plt.legend(loc="upper right")
    plt.xlabel("$x$")
    plt.ylabel("$u$")
    plt.title(title)

def plot_1D_domain(subdomain_xs, subdomain_ws, D, title):
    
    # plot subdomain
    subdomain_ws = [np.array(w).copy() for w in subdomain_ws]
    subdomain_ws[0][0] = subdomain_ws[0][-1] = 0
    for im,ii in itergrid(D.nm):
        plt.hlines(-0.55 if im%2 else -0.45, 
                   subdomain_xs[0][ii[0]]-subdomain_ws[0][ii[0]]/2, 
                   subdomain_xs[0][ii[0]+1]+subdomain_ws[0][ii[0]+1]/2, 
                   linewidth=6, colors=colors[im])
        
    # plot all segments
    for ioa in range(D.N_ORDERS):
        s = D.segments[ioa]# (2,nd,nm)
        for _,ii in itergrid(s.shape[2:]):
            plt.hlines(-1, s[0,0,ii[0]], s[1,0,ii[0]], linewidth=6+5*ioa, colors=colors[-4:][ioa])
            
    # plot all window functions
    for im,ii in itergrid(D.nm):
        x = torch.linspace(subdomain_xs[0].min(), subdomain_xs[0].max(), 1000).view(-1,1)
        w = D.w[im](x)
        plt.plot(x[:,0], w[:,0], color=colors[im])
    
    plt.xlabel("$x$")
    plt.yticks([-1,-0.5,0,0.5,1], ["Overlapping\nmodels","Subdomain\ndefinition",0,"Window\nfunction",1])
    plt.title(title)

    
    