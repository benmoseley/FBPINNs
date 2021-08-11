#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 23:48:06 2021

@author: bmoseley
"""

# This script reproduces all of the paper results

import sys

import numpy as np

import problems
from active_schedulers import AllActiveSchedulerND, PointActiveSchedulerND, LineActiveSchedulerND, PlaneActiveSchedulerND
from constants import Constants, get_subdomain_xs, get_subdomain_ws
from main import FBPINNTrainer, PINNTrainer
from trainersBase import train_models_multiprocess

sys.path.insert(0, '../shared_modules/')
import multiprocess



# constants constructors

def run_PINN():
    sampler = "r" if random else "m"
    c = Constants(
                  RUN="final_PINN_%s_%sh_%sl_%sb_%s"%(P.name, n_hidden, n_layers, batch_size[0], sampler),
                  P=P,
                  SUBDOMAIN_XS=subdomain_xs,
                  BOUNDARY_N=boundary_n,
                  Y_N=y_n,
                  N_HIDDEN=n_hidden,
                  N_LAYERS=n_layers,
                  BATCH_SIZE=batch_size,
                  RANDOM=random,
                  N_STEPS=n_steps,
                  BATCH_SIZE_TEST=batch_size_test,
                  PLOT_LIMS=plot_lims,
                  )
    return c, PINNTrainer

def run_FBPINN():
    sampler = "r" if random else "m"
    c = Constants(
                  RUN="final_FBPINN_%s_%sh_%sl_%sb_%s_%sw_%s"%(P.name, n_hidden, n_layers, batch_size[0], sampler, width, A.name),
                  P=P,
                  SUBDOMAIN_XS=subdomain_xs,
                  SUBDOMAIN_WS=subdomain_ws,
                  BOUNDARY_N=boundary_n,
                  Y_N=y_n,
                  ACTIVE_SCHEDULER=A,
                  ACTIVE_SCHEDULER_ARGS=args,
                  N_HIDDEN=n_hidden,
                  N_LAYERS=n_layers,
                  BATCH_SIZE=batch_size,
                  RANDOM=random,
                  N_STEPS=n_steps,
                  BATCH_SIZE_TEST=batch_size_test,
                  PLOT_LIMS=plot_lims,
                  )
    return c, FBPINNTrainer


# DEFINE PROBLEMS


# below uses 200 points per w


runs = []

plot_lims = (1.1, False)
random = False


# 1D PROBLEMS

# Cos w=1
    
P = problems.Cos1D_1(w=1, A=0)
subdomain_xs = get_subdomain_xs([np.array([2,3,2,4,3])], [2*np.pi])
boundary_n = (1/P.w,)
y_n = (0,1/P.w)
batch_size = (200,)
batch_size_test = (1000,)

n_steps = 50000
n_hidden, n_layers = 16, 2
runs.append(run_PINN())

n_steps = 50000
n_hidden, n_layers = 16, 2
A, args = AllActiveSchedulerND, ()
for width in [0.1, 0.5, 0.7, 0.9]:
    subdomain_ws = get_subdomain_ws(subdomain_xs, width)
    runs.append(run_FBPINN())

# Cos w=15

P = problems.Cos1D_1(w=15, A=0)
subdomain_xs = get_subdomain_xs([np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])], [2*np.pi])
boundary_n = (1/P.w,)
y_n = (0,1/P.w)
batch_size = (3000,)
batch_size_test = (5000,)

n_steps = 50000
for n_hidden, n_layers in [(16, 2), (32, 3), (64, 4), (128, 5)]:
    runs.append(run_PINN())

n_hidden, n_layers = 16, 2
width = 0.7
subdomain_ws = get_subdomain_ws(subdomain_xs, width)
for A,args,n_steps in [(AllActiveSchedulerND, (), 50000), (PointActiveSchedulerND, (np.array([0,]),), 100000)]:
    runs.append(run_FBPINN())

# Cos multi w1=1 w2=15

P = problems.Cos_multi1D_1(w1=1, w2=15, A=0)
subdomain_xs = get_subdomain_xs([np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])], [2*np.pi])
boundary_n = (1/P.w2,)
y_n = (0,2)
batch_size = (3000,)
batch_size_test = (5000,)

n_steps = 50000
for n_hidden, n_layers in [(16, 2), (32, 3), (64, 4), (128, 5)]:
    runs.append(run_PINN())

n_hidden, n_layers = 16, 2
width = 0.7
subdomain_ws = get_subdomain_ws(subdomain_xs, width)
for A,args,n_steps in [(AllActiveSchedulerND, (), 50000), (PointActiveSchedulerND, (np.array([0,]),), 100000)]:
    runs.append(run_FBPINN())
    
# Sin w=1

P = problems.Sin1D_2(w=1, A=0, B=-1/1)
subdomain_xs = get_subdomain_xs([np.array([2,3,2,4,3])], [2*np.pi])
boundary_n = (1/P.w,)
y_n = (0,1/P.w**2)
batch_size = (200,)
batch_size_test = (1000,)

n_steps = 50000
n_hidden, n_layers = 16, 2
runs.append(run_PINN())

n_hidden, n_layers = 16, 2
width = 0.7
subdomain_ws = get_subdomain_ws(subdomain_xs, width)
for A,args,n_steps in [(AllActiveSchedulerND, (), 50000), (PointActiveSchedulerND, (np.array([0,]),), 100000)]:
    runs.append(run_FBPINN())

# Sin w=15
    
P = problems.Sin1D_2(w=15, A=0, B=-1/15)
subdomain_xs = get_subdomain_xs([np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])], [2*np.pi])
boundary_n = (1/P.w,)
y_n = (0,1/P.w**2)
batch_size = (3000,)
batch_size_test = (5000,)

n_steps = 100000
n_hidden, n_layers = 128, 5
runs.append(run_PINN())

n_hidden, n_layers = 16, 2
width = 0.7
subdomain_ws = get_subdomain_ws(subdomain_xs, width)
for A,args,n_steps in [(AllActiveSchedulerND, (), 100000), (PointActiveSchedulerND, (np.array([0,]),), 500000)]:
    runs.append(run_FBPINN())
    



if __name__ == "__main__":# required for multiprocessing

    import socket
    
    # GLOBAL VARIABLES
    
    # parallel devices (GPUs/ CPU cores) to run on
    DEVICES = ["cpu"]*23
    
    
    # RUN
    
    for i,(c,_) in enumerate(runs): print(i,c)
    print("%i runs\n"%(len(runs)))
    
    if "local" not in socket.gethostname().lower():
        jobs = [(DEVICES, c, t, i) for i,(c,t) in enumerate(runs)]
        with multiprocess.Pool(processes=len(DEVICES)) as pool:
            pool.starmap(train_models_multiprocess, jobs)
        
        