#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 13:43:14 2018

@author: bmoseley
"""

# This module defines a Constants object which defines and stores a problem setup and all of its hyperparameters,
# for both FBPINN and PINN runs
# The instantiated constants object should be passed to the appropriate trainer classes defined in main.py

# This module is used by main.py and the paper_main_ND.py scripts

import socket

import numpy as np

import models
import problems
import active_schedulers
from constantsBase import ConstantsBase



# helper functions

def get_subdomain_xs(ds, scales):
    xs = []
    for d,scale in zip(ds, scales):
        x = np.cumsum(np.pad(d, (1,0)))
        x = 2*(x-x.min())/(x.max()-x.min())-1# normalise to [-1, 1]
        xs.append(scale*x)
    return xs

def get_subdomain_ws(subdomain_xs, width):
    return [width*np.min(np.diff(x))*np.ones_like(x) for x in subdomain_xs]


class Constants(ConstantsBase):
    
    def __init__(self, **kwargs):
        "Define default parameters"
        
        ######################################
        ##### GLOBAL CONSTANTS FOR MODEL
        ######################################
        
        
        # Define run
        self.RUN = "test"
        
        # Define problem
        w = 1e-10
        #self.P = problems.Cos1D_1(w=w, A=0)
        self.P = problems.Sin1D_2(w=w, A=0, B=-1/w)
        
        # Define domain
        self.SUBDOMAIN_XS = get_subdomain_xs([np.array([2,3,2,4,3])], [2*np.pi/self.P.w])
        self.SUBDOMAIN_WS = get_subdomain_ws(self.SUBDOMAIN_XS, 0.7)
        
        # Define normalisation parameters
        self.BOUNDARY_N = (1/self.P.w,)# sd
        #self.Y_N = (0,1/self.P.w)# mu, sd
        self.Y_N = (0,1/self.P.w**2)# mu, sd
        
        # Define scheduler
        self.ACTIVE_SCHEDULER = active_schedulers.PointActiveSchedulerND
        self.ACTIVE_SCHEDULER_ARGS = (np.array([0,]),)
        
        # GPU parameters
        self.DEVICE = 0# cuda device
        
        # Model parameters
        self.MODEL = models.FCN
        self.N_HIDDEN = 16
        self.N_LAYERS = 2
        
        # Optimisation parameters
        self.BATCH_SIZE = (500,)
        self.RANDOM = False
        self.LRATE = 1e-3
        
        self.N_STEPS = 50000
        
        # seed
        self.SEED = 123
        
        # other
        self.BATCH_SIZE_TEST = (5000,)
        self.PLOT_LIMS = (1, False)
        
        ### summary output frequencies     
        self.SUMMARY_FREQ    = 250
        self.TEST_FREQ       = 5000
        self.MODEL_SAVE_FREQ = 10000
        self.SHOW_FIGURES = True# whether to show figures
        self.SAVE_FIGURES = False# whether to save figures
        self.CLEAR_OUTPUT = False# whether to clear output periodically
        ##########
        
        
        
        # overwrite with input arguments
        for key in kwargs.keys(): self[key] = kwargs[key]# invokes __setitem__ in ConstantsBase
        
        # other calculated variables
        self.SUMMARY_OUT_DIR = "results/summaries/%s/"%(self.RUN)
        self.MODEL_OUT_DIR = "results/models/%s/"%(self.RUN)
        self.HOSTNAME = socket.gethostname().lower()
    



