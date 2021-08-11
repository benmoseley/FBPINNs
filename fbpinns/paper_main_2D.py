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


# below uses 60 points per w


runs = []

plot_lims = (1.1, False)
random = False


# 2D PROBLEMS

# Cos Cos w=15

P = problems.Cos_Cos2D_1(w=15, A=0)
subdomain_xs = get_subdomain_xs([np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]),
                                 np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])], 
                                [2*np.pi, 2*np.pi])
boundary_n = (1/P.w,)
y_n = (0,1/P.w)
batch_size = (900,900)
batch_size_test = (1000,1000)

n_steps = 50000
n_hidden, n_layers = 128, 5
runs.append(run_PINN())

n_hidden, n_layers = 16, 2
width = 0.7
subdomain_ws = get_subdomain_ws(subdomain_xs, width)
for A,args,n_steps in [(AllActiveSchedulerND, (), 50000),
                       (LineActiveSchedulerND, (np.array([0,]),1), 100000),
                       (PointActiveSchedulerND, (np.array([0,0]),), 100000)]:
    runs.append(run_FBPINN())


# Burgers

P = problems.Burgers2D()
subdomain_xs = [np.array([-1, -0.5, 0, 0.5, 1]), np.array([0, 0.5, 1])]
boundary_n = (1,)
y_n = (0,1)
batch_size = (200,200)
batch_size_test = (400,400)

n_steps = 50000
n_hidden, n_layers = 64, 4
runs.append(run_PINN())

n_steps = 50000
n_hidden, n_layers = 16, 2
A, args = AllActiveSchedulerND, ()
for width in [0.1, 0.5, 0.7]:
    subdomain_ws = get_subdomain_ws(subdomain_xs, width)
    runs.append(run_FBPINN())
width="0.5thin"
subdomain_ws = [np.array([0.25, 0.25, 0.05, 0.25, 0.25]), np.array([0.25, 0.25, 0.25])]
runs.append(run_FBPINN())

width="0.7avoid"
subdomain_xs = [np.array([-1, -0.333, 0.333, 1]), np.array([0, 0.5, 1])]
subdomain_ws = [np.array([0.35, 0.35, 0.35, 0.35]), np.array([0.35, 0.35, 0.35])]
runs.append(run_FBPINN())




if __name__ == "__main__":# required for multiprocessing

    import socket
    
    # GLOBAL VARIABLES
    
    # parallel devices (GPUs/ CPU cores) to run on
    DEVICES = [1,2,3]
    
    
    # RUN
    
    for i,(c,_) in enumerate(runs): print(i,c)
    print("%i runs\n"%(len(runs)))
    
    if "local" not in socket.gethostname().lower():
        jobs = [(DEVICES, c, t, i) for i,(c,t) in enumerate(runs)]
        with multiprocess.Pool(processes=len(DEVICES)) as pool:
            pool.starmap(train_models_multiprocess, jobs)
        
        