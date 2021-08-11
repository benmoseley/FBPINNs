#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:40:57 2019

@author: bmoseley
"""

# This module defines the base class inherited by the Constants class and defines helper I/O functions

# This class is used by constants.py

import pickle

import sys
sys.path.insert(0, '../shared_modules/')
import io_utils


class ConstantsBase:
    
    # note can set members freely, below only for index assignment
    def __getitem__(self, key):
        if key not in self.__dict__.keys(): raise Exception('key "%s" not in self.__dict__'%(key))
        return self.__dict__[key]
    def __setitem__(self, key, item):
        if key not in self.__dict__.keys(): raise Exception('key "%s" not in self.__dict__'%(key))
        self.__dict__[key] = item
    
    def __str__(self):
        s = ""
        for k in vars(self): s+="%s: %s\n"%(k,self[k])
        return s
    
    
    # below methods assume RUN, SUMMARY_OUT_DIR and MODEL_OUT_DIR attributes exist
 
    def get_outdirs(self):
        io_utils.get_dir(self.SUMMARY_OUT_DIR)
        io_utils.clear_dir(self.SUMMARY_OUT_DIR)
        io_utils.get_dir(self.MODEL_OUT_DIR)
        io_utils.clear_dir(self.MODEL_OUT_DIR)
        
    def save_constants_file(self):
        "Save a constants to file in self.SUMMARY_OUT_DIR"
        # Note: pickling only saves functions/ classes / modules by name reference so
        # the unpickling environment needs access to the source code
        # https://docs.python.org/3.7/library/pickle.html#what-can-be-pickled-and-unpickled
        with open(self.SUMMARY_OUT_DIR + "constants_%s.txt"%(self.RUN), 'w') as f:
            for k in self.__dict__: f.write("%s: %s\n"%(k,self[k]))
        with open(self.SUMMARY_OUT_DIR + "constants_%s.pickle"%(self.RUN), 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    
    
def print_c_dicts(c_dicts):
    "Pretty print a list of c_dicts"
    
    # get full list of keys
    keys = []
    for c_dict in c_dicts[::-1]:
        for k in c_dict.keys():
            if k not in keys: keys.append(k)
            
    for k in keys:
        print("%s: "%(k),end="")
        for i,c_dict in enumerate(c_dicts):
            if k in c_dict.keys(): item=str(c_dict[k])
            else: item='None'
            if i == len(c_dicts)-1: print("%s"%(item),end="")
            else: print("%s | "%(item),end="")
        print("")