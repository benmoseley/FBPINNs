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

runs = []

plot_lims = (0.4, True)
random = True


# 3D PROBLEMS

# Wave equation c=1

P = problems.WaveEquation3D(c=1, source_sd=0.2)
subdomain_xs = [np.array([-1, -0.33, 0.33, 1]), np.array([-1, -0.33, 0.33, 1]), np.array([0, 0.5, 1])]
boundary_n = (0.2,)
y_n = (0,1)
batch_size = (30,30,30)
batch_size_test = (100,100,10)

n_steps = 25000
n_hidden, n_layers = 64, 4
runs.append(run_PINN())

n_steps = 25000
A, args = AllActiveSchedulerND, ()
width = 0.9
subdomain_ws = get_subdomain_ws(subdomain_xs, width)
for n_hidden, n_layers in [(16, 2), (32, 3)]:
    runs.append(run_FBPINN())


# Wave equation c=gaussian

P = problems.WaveEquation3D(c="gaussian", source_sd=0.3)
subdomain_xs = [np.array([-10, -3.33, 3.33, 10]), np.array([-10, -3.33, 3.33, 10]), np.array([0, 2.5, 5, 7.5, 10])]
boundary_n = (0.3,)
y_n = (0,1)
batch_size = (58,58,58)
batch_size_test = (100,100,10)

n_steps = 75000
n_hidden, n_layers = 128, 5
runs.append(run_PINN())

n_steps = 150000
A, args = PlaneActiveSchedulerND, (np.array([0,]),[0,1])
width = 0.9
subdomain_ws = get_subdomain_ws(subdomain_xs, width)
for n_hidden, n_layers in [(16, 2), (32, 3), (64, 4)]:
    runs.append(run_FBPINN())




if __name__ == "__main__":# required for multiprocessing

    import socket
    
    # GLOBAL VARIABLES
    
    # parallel devices (GPUs/ CPU cores) to run on
    DEVICES = [2,3]
    
    
    # RUN
    
    for i,(c,_) in enumerate(runs): print(i,c)
    print("%i runs\n"%(len(runs)))
    
    if "local" not in socket.gethostname().lower():
        jobs = [(DEVICES, c, t, i) for i,(c,t) in enumerate(runs)]
        with multiprocess.Pool(processes=len(DEVICES)) as pool:
            pool.starmap(train_models_multiprocess, jobs)
        
        