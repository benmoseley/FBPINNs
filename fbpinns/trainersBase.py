#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 00:33:05 2021

@author: bmoseley
"""

# This module defines the base trainer class used by main.py and defines extra helper training functions

# This class is used by main.py

import os
import sys
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
import IPython.display
from tensorboardX import SummaryWriter


class _Trainer:
    "Generic model trainer class"
    
    def __init__(self, c):
        "Initialise torch and output directories"
        
        # set seed
        if c.SEED is None: c.SEED = torch.initial_seed()
        else: torch.manual_seed(c.SEED)# independent of numpy
        np.random.seed(c.SEED)
                       
        # clear directories
        c.get_outdirs()
        c.save_constants_file()
        print(c)
        
        # get device/ set threads
        if c.DEVICE != "cpu" and torch.cuda.is_available():
            device = torch.device("cuda:%i"%(c.DEVICE))
            torch.cuda.set_device(c.DEVICE)# stops weird memory being allocated on cuda:0 even if c.DEVICE != 0
        else: 
            device = torch.device("cpu")
        print("Device: %s"%(device))
        torch.backends.cudnn.benchmark = False#let cudnn find the best algorithm to use for your hardware (not good for dynamic nets)
        torch.set_num_threads(1)# for main inference
        print("Main thread ID: %i"%os.getpid())
        print("Torch seed: ", torch.initial_seed())
        
        # initialise summary writer
        writer = SummaryWriter(c.SUMMARY_OUT_DIR)
        writer.add_text("constants", str(c).replace("\n","  \n"))# uses markdown
        
        self.c, self.device, self.writer = c, device, writer
    
    def _print_summary(self, i, loss, rate, start):
        "Prints training summary"
        
        print('[i: %i/%i] loss: %.4f rate: %.1f elapsed: %.2f hr %s %s\n' % (
               i + 1,
               self.c.N_STEPS,
               loss,
               rate,
               (time.time()-start)/(60*60),
               time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()),
               self.c.RUN,
                ))
        self.writer.add_scalar("rate/", rate, i + 1)
    
    def _save_figs(self, i, fs):
        "Saves figures"
        
        if self.c.CLEAR_OUTPUT: IPython.display.clear_output(wait=True)
        for name,f in fs:
            if self.c.SAVE_FIGURES:
                f.savefig(self.c.SUMMARY_OUT_DIR+"%s_%.8i.png"%(name, i + 1), bbox_inches='tight', pad_inches=0.1, dpi=100)
            self.writer.add_figure(name, f, i + 1, close=False)
        plt.show() if self.c.SHOW_FIGURES else plt.close("all")
    
    def _save_model(self, i, model, im=None):
        "Saves a model"
        
        tag = "model_%.8i_%.8i.torch"%(i + 1, im) if im is not None else "model_%.8i.torch"%(i + 1)
        model.eval()
        model.to(torch.device('cpu'))# put model on cpu before saving to avoid out-of-memory error
        torch.save({'i': i + 1,
                    'model_state_dict': model.state_dict()},
                   self.c.MODEL_OUT_DIR+tag)
        model.to(self.device)
        
        
    def train(self):
        
        raise NotImplementedError


## HELPER FUNCTIONS

def train_models_multiprocess(ip, devices, c, Trainer, wait=0):
    "Helper function for training multiple runs at once (use with multiprocess.Pool)"
    
    time.sleep(wait)# small hack so that tensorboard summaries appear in order
    tag = os.environ["STY"].split(".")[-1] if "STY" in os.environ else "main"# grab socket name if using screen
    logfile = "screenlog.%s.%i.log"%(tag, ip)
    sys.stdout, sys.stderr = open(logfile, "a", buffering=1), open(logfile, "a", buffering=1)# line buffering
    c.DEVICE = devices[ip]# set device to run on, based on process id
    c.SHOW_FIGURES = c.CLEAR_OUTPUT = False# make sure plots are not shown
    run = Trainer(c)
    run.train()




