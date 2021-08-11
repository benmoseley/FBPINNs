#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:32:55 2021

@author: bmoseley
"""

# This module defines standard pytorch NN models

# This module is used by constants.py when defining when defining FBPINN / PINN problems

import torch
import torch.nn as nn

total_params = lambda model: sum(p.numel() for p in model.parameters())


class FCN(nn.Module):
    "Fully connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        
        # define layers

        activation = nn.Tanh
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
        # define helper attributes / methods for analysing computational complexity
        
        d1,d2,h,l = N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS
        self.size =           d1*h + h         + (l-1)*(  h*h + h)        +   h*d2 + d2
        self._single_flop = 2*d1*h + h + 5*h   + (l-1)*(2*h*h + h + 5*h)  + 2*h*d2 + d2# assumes Tanh uses 5 FLOPS
        self.flops = lambda BATCH_SIZE: BATCH_SIZE*self._single_flop
        assert self.size == total_params(self)
        
    def forward(self, x):
                
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        
        return x
    

if __name__ == "__main__":
    
    import numpy as np
    
    x = np.arange(100).reshape((25,4)).astype(np.float32)
    x = torch.from_numpy(x)
    print(x.shape)
    
    model = FCN(4, 2, 8, 3)
    print(model)
    
    y = model(x)
    print(y.shape)
    
    print("Number of parameters:", model.size)
    print("Number of FLOPS:", model.flops(x.shape[0]))
    
    