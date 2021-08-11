#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:10:09 2021

@author: bmoseley
"""

# This module defines the window functions which are applied to the output of each subdomain neural network
# note the windows are defined using torch such that we can autodiff through them during training

# This module is used by domain.py

import torch


def _create_kernel(xmin,xmax,smin,smax):
    "Creates a 1D kernel function"
    
    tol = 1e-10# for numerical stability when evaluating gradients
    clamp = lambda x: torch.clamp(x, min=tol)
    
    if xmax is None and xmin is None:
        kernel = lambda x: torch.ones_like(x)
    elif xmax is None:
        if smin <= 0: raise Exception("ERROR smin <= 0 (%s)!"%(smin))
        kernel = lambda x: clamp(torch.sigmoid((x-xmin)/smin))
    elif xmin is None:
        if smax <= 0: raise Exception("ERROR smax <= 0 (%s)!"%(smax))
        kernel = lambda x: clamp(torch.sigmoid((xmax-x)/smax))
    else:
        if xmin>xmax: raise Exception("ERROR: xmin (%s) > xmax (%s)!"%(xmin, xmax))
        if smin <= 0: raise Exception("ERROR smin <= 0 (%s)!"%(smin))
        if smax <= 0: raise Exception("ERROR smax <= 0 (%s)!"%(smax))
        kernel = lambda x: clamp(clamp(torch.sigmoid((x-xmin)/smin))*clamp(torch.sigmoid((xmax-x)/smax)))
        
    return kernel


def construct_window_function_ND(xs_min, xs_max, scales_min, scales_max):
    "Constructs a ND window function"
    
    if not (len(xs_min) == len(xs_max) == len(scales_min) == len(scales_max)):
        raise Exception("ERROR input lengths do not match!")
    
    kernels = [_create_kernel(*args) for args in zip(xs_min,xs_max,scales_min,scales_max)]
    nd = len(xs_min)
    
    def window_function(x):
        
        if x.ndim != 2: raise Exception("ERROR!: x.ndim (%s) != 2!"%(x.shape,))
        if x.shape[-1] != nd: raise Exception("ERROR!: x.shape[1] (%s) != nd (%s)"%(x.shape[1], nd))
        
        xs = x.unbind(-1)# separate out dims
        ws = [kernels[i](x) for i,x in enumerate(xs)]
        w = torch.stack(ws, -1)
        w = torch.prod(w, keepdim=True, dim=-1)# get product of windows over each dimension
        
        return w
    
    return window_function
    


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    
    ## 1D test
    
    x = np.expand_dims(np.arange(-10,10, 0.1), -1).astype(np.float32)
    x = torch.from_numpy(x)
    
    window_function = construct_window_function_ND([-1], [6], [1.2], [0.5])
    w1 = window_function(x)
    
    window_function = construct_window_function_ND([None], [-1], [0.5], [0.5])
    w2 = window_function(x)
    
    window_function = construct_window_function_ND([6], [None], [0.1], [0.1])
    w3 = window_function(x)
    
    plt.figure()
    plt.plot(x, w1)
    plt.plot(x, w2)
    plt.plot(x, w3)
    plt.plot(x, w1 + w2 + w3, color="k", alpha=0.4)
    plt.show()
    
    
    ## 2D test
    
    x = np.linspace(-20,20,220)
    y = np.linspace(-15,18,200)
    xx = np.stack(np.meshgrid(x,y,indexing="ij"), -1)
    x = xx.reshape((220*200,-1))
    
    window_function = construct_window_function_ND([0, -10], [15, -5], [4, 1], [0.2, 0.4])
    w1 = window_function(torch.from_numpy(x))
    w1 = w1.reshape((220,200))
    
    plt.figure()
    plt.imshow(w1.T, origin="lower", extent=(x.min(), x.max(), y.min(), y.max()))
    plt.colorbar()
    plt.show()
    