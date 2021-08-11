#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 12:26:08 2021

@author: bmoseley
"""

# This module contains a set of helper functions for applying hard boundary conditions to the FBPINN / PINN ansatz

# This module is used by problems.py

import torch


# helper analytical functions

def tanh_1(x, mu, sd):
    "Compute solution/gradients of y=tanh((x-mu)/sd)"
    
    xn = (x-mu)/sd
    
    tanh = torch.tanh(xn)
    sech2 = (1-tanh**2)
    
    t   = tanh
    jt  = (1/sd)    * sech2
    
    return t, jt

def tanh2_2(x, mu, sd):
    "Compute solution/gradients of y=tanh^2((x-mu)/sd)"
    
    xn = (x-mu)/sd
    
    tanh = torch.tanh(xn)
    sech2 = (1-tanh**2)
    
    t2   = tanh**2
    jt2  = (1/sd)    * ( 2*tanh*sech2 )
    jjt2 = (1/sd**2) * ( 2*(sech2**2) - 4 * (tanh**2) * sech2 )
    
    return t2, jt2, jjt2

def tanhtanh_2(x, mu1, mu2, sd):
    "Compute solution/gradients of y=tanh((x-mu1)/sd)*tanh((x-mu2)/sd)"
    
    xn_1 = (x-mu1)/sd
    xn_2 = (x-mu2)/sd
    
    tanh_1 = torch.tanh(xn_1)
    tanh_2 = torch.tanh(xn_2)
    sech2_1 = (1-tanh_1**2)
    sech2_2 = (1-tanh_2**2)
    
    t =  tanh_1*tanh_2
    jt =  (1/sd)    * ( tanh_1*sech2_2 + sech2_1*tanh_2 )
    jjt = (1/sd**2) * ( 2*sech2_1*sech2_2 - 2*tanh_1*tanh_2*(sech2_1 + sech2_2) )
    
    return t, jt, jjt

def sigmoid_2(x, mu, sd):
    "Compute solution/gradients of y=sigmoid((x-mu)/sd)"
    
    xn = (x-mu)/sd
    
    sig = torch.sigmoid(xn)
    
    s   = sig
    js  = (1/sd)    * sig*(1-sig)
    jjs = (1/sd**2) * sig*(1-sig)*(1-2*sig)
    
    return s, js, jjs

# helper analytical functions (fused)

def tanh_tanh2_2(x, mu, sd):
    "Compute solution/gradients of y=tanh((x-mu)/sd) and y=tanh^2((x-mu)/sd)"
    
    xn = (x-mu)/sd
    
    tanh = torch.tanh(xn)
    sech2 = (1-tanh**2)
    
    t   = tanh
    jt  = (1/sd)    * sech2
    jjt = (1/sd**2) * -2*tanh*sech2
    
    t2   = tanh**2
    jt2  = (1/sd)    * ( 2*tanh*sech2 )
    jjt2 = (1/sd**2) * ( 2*(sech2**2) - 4 * (tanh**2) * sech2 )
    
    return t, jt, jjt, t2, jt2, jjt2


# helper apply functions

def A_1D_1(x, y, j, A, mu, sd):
    "Apply y = tanh((x-mu)/sd)*NN + A ansatz"
    
    t, jt = tanh_1(x, mu, sd)
    
    y_new = t*y        + A
    j_new = jt*y + t*j
    
    return y_new, j_new

def AB_1D_2(x, y, j, jj, A, B, mu, sd):
    "Apply y = tanh^2((x-mu)/sd)*NN + B*sd*tanh((x-mu)/sd) + A ansatz"
    
    t, jt, jjt, t2, jt2, jjt2 = tanh_tanh2_2(x, mu, sd)
    B = B*sd
    
    y_new  = t2*y                     + B*t     + A
    j_new  = jt2*y  +    t2*j         + B*jt
    jj_new = jjt2*y + 2*jt2*j + t2*jj + B*jjt
    
    return y_new, j_new, jj_new
