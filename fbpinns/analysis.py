#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 00:21:36 2021

@author: bmoseley
"""

# This module defines helper functions for loading saved FBPINN / PINN models

# This module is used by Paper plots.ipynb

import os
import pickle

import numpy as np
import torch

import domains

import sys
sys.path.insert(0, '../shared_modules/')
from helper import DictToObj


def _restore_model(file, c):
    "restores an individual model from file"
    
    model = c.MODEL(c.P.d[0], c.P.d[1], c.N_HIDDEN, c.N_LAYERS)# initialise model ! uses current code (pickle only saves name of class)
    cp = torch.load(file, map_location=torch.device('cpu'))# remaps tensors from gpu to cpu if needed
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    
    return model

def load_model(RUN, i=None, rootdir="results/", verbose=False):
    """load a model, its constants object and supplement files from rootdir.
    If i is specified, load the model at that timestep, otherwise
    load the model with the latest timestep."""
    
    # 1. parse MODEL_DIR and SUMMARY_DIR
    MODEL_DIR = rootdir+"models/%s/"%(RUN)
    SUMMARY_DIR = rootdir+"summaries/%s/"%(RUN)
    
    # 2. load constants dictionary
    c_dict = pickle.load(open(SUMMARY_DIR+"constants_%s.pickle"%(RUN), "rb"))
    c = DictToObj(**c_dict, copy=True)# convert to object
    
    # 3. get specific timestep to load
    if i is None:
        last_file = sorted(os.listdir(MODEL_DIR))[-1]
        i = int(os.path.splitext(last_file)[0].split("_")[1])
    
    # 4. get and load model files
    if "FBPINN" in RUN:
        N_MODELS = np.prod([len(x)-1 for x in c.SUBDOMAIN_XS])# number of models in FBPINN
        files = [MODEL_DIR+"model_%.8i_%.8i.torch"%(i, im) for im in range(N_MODELS)]
        if verbose: print("Loading models from:\n%s%s%s"%(files[0], ", ..." if len(files)>2 else "", "\n"+files[-1] if len(files)>1 else ""))
        models = [_restore_model(file, c) for file in files]
        supplement = [np.load(MODEL_DIR+"loss_%.8i.npy"%(i,)), np.load(MODEL_DIR+"active_%.8i.npy"%(i,))]
    elif "PINN" in RUN:
        file = MODEL_DIR+"model_%.8i.torch"%(i)
        if verbose: print("Loading model from:\n%s"%(file))
        models = _restore_model(file, c)
        supplement = [np.load(MODEL_DIR+"loss_%.8i.npy"%(i,))]
    else:
        raise Exception("ERROR: could not recognise run! (%s)"%(RUN))
    
    return models, c, supplement

def load_domain(c, active=None, device=None):
    "Return instantiated problem domain given a constants object"
    
    D = domains.ActiveRectangularDomainND(c.SUBDOMAIN_XS, c.SUBDOMAIN_WS, device=device)
    D.update_sampler(c.BATCH_SIZE, c.RANDOM)
    D.update_active(active)
    
    return D


if __name__ == "__main__":

    models, c, supplement = load_model("t2_FBPINN_Cos1D_1_w15_16h_2l_0.5w_All", rootdir="server/e2/", verbose=True)
    print(len(models))
    print(c)
    
    models, c, supplement = load_model("t2_FBPINN_Cos1D_1_w15_16h_2l_0.5w_All", i=20000, rootdir="server/e2/", verbose=True)
    print(len(models))
    print(c)
    
    model, c, supplement = load_model("t2_PINN_Cos1D_1_w15_32h_3l", rootdir="server/e2/", verbose=True)
    print(c)
    
    model, c, supplement = load_model("t2_PINN_Cos1D_1_w15_32h_3l", i=20000, rootdir="server/e2/", verbose=True)
    print(c)
    