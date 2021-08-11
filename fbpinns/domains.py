#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:49:24 2021

@author: bmoseley
"""

# This module inherits from domainsBase.py and defines a ND FBPINN domain with overlapping 
# rectangular subdomains, its active subdomains, the segments-in-model and models-in-segments 
# index maps required when running and sharing NN outputs across neighbouring subdomains 
# during training, the normalisation values for each subdomain, the window functions for each
# subdomain and a sampler which samples torch points within the domain
# You can call update_active to update the active/fixed/inactive subdomains during training
# You can call update_sampler to choose between a random or regular sampler

# This module is used by main.py

import itertools

import numpy as np
import torch

import domainsBase
import windows



#####
## CONVENTIONS

# i for i in range(nd)
# ii for grid index in nm
# iii for grid index in onm

# im for ravelled index in nm
# iseg for ravelled index in onm

# i1/i2 for dynamic offsets in D.m map

# ioa for segment order number

#####


itergrid = lambda shape: enumerate(itertools.product(*[np.arange(d) for d in shape]))


class ActiveRectangularDomainND(domainsBase._RectangularDomainND):
    "ND domain with hyperrectangular subdomains, where the active model grid can be controlled"
    
    
    # INIT FUNCTIONS
    
    def __init__(self, subdomain_xs, subdomain_ws, scale=0.05, device=None):
        super().__init__(subdomain_xs, subdomain_ws)
        
        self.N_MODELS = np.product(self.nm)
        self.N_ORDERS = len(self.segments)
        self.onm = (self.N_ORDERS,)+self.nm
        self.N_SEGMENTS = np.product(self.onm)
        
        self.subdomain_xs = subdomain_xs
        self.scale = scale
        
        # get outside filters
        self._set_outside_filters()
        
        # set helpers
        self._set_helpers()
        
        # set torch tensors
        self._set_torch_tensors(device)
    
    def _set_outside_filters(self):# [left over processing required from domainsBase._RectangularDomainND]
        "Set filters for segments in model_segments which fall outside of the segment grid"
        
        cs = []# filters for each segment type
        for ioa, ms in enumerate(self.models_segments):# (ne,nd,nm) ms
            shape = np.array(self.segments[ioa].shape[2:])# (nd) shape of segment grid
            shape = shape.reshape((1,self.nd)+(1,)*self.nd)# (1,nd,1..)
            c = (ms >= 0) & (ms < shape)# (ne,nd,nm) (shape is broadcast)
            c = np.product(c, axis=1).astype(bool)# (ne,nm) collapse along ndim (i.e. has to be true across all nd)
            cs.append(c)
        self.cs = cs

    def _set_helpers(self):
        "Set some useful helper arrays"
        
        # get xmin, xmax of each model
        xmins = self.xx[(slice(None),)+(slice(None,-1),)*self.nd]# (nd, nm)  self.xx (nd,nm+1)
        xmaxs = self.xx[(slice(None),)+(slice(1, None),)*self.nd]# (nd, nm)
        
        # get window widths of each model
        wmins = self.ww[(slice(None),)+(slice(None,-1),)*self.nd]# (nd, nm) self.ww (nd,nm+1)
        wmaxs = self.ww[(slice(None),)+(slice(1, None),)*self.nd]# (nd, nm)
        
        # get mu, sd of each model
        mus = (xmins + xmaxs)/2# (nd, nm)
        sds = (xmaxs - xmins)/2# (nd, nm)
        # place into flattened lists
        n = []
        for im,ii in itergrid(self.nm):
            sl = (slice(None),)+ii# slice at grid location
            mu, sd = mus[sl], sds[sl]# (nd) mu, sd of model at grid location
            n.append((mu, sd))
            
        self.xmins, self.xmaxs, self.wmins, self.wmaxs = xmins, xmaxs, wmins, wmaxs
        self.n = n
        
        
    # UPDATE FUNCTIONS
    
    def _get_neighbours(self, active):
        "Get the neighbours of an active array"
        
        # get the neighbouring models active values
        pad = np.pad(active.copy(), 1, mode="constant", constant_values=0)# (nm+2) pads all dimensions with one zero either side
        ds = []
        for i in range(self.nd):
            sp,sn = [slice(1,-1,None)]*self.nd, [slice(1,-1,None)]*self.nd# cut out model grid
            sp[i] = slice(None,-2,None); sn[i] = slice(2,None,None)# take previous/ next along current dimension
            sp,sn = tuple(sp), tuple(sn)
            dp,dn = pad[sp], pad[sn]# (nm)
            ds.append((dp, dn))
        neighbours = np.array(ds)#(nd, 2, nm) neighbours array   active values
        
        return neighbours
        
    def _get_window_functions(self, neighbours):
        "For each model, get the appropriate window function, given the neighbours array"
        
        # get window functions by model
        w = []
        for im,ii in itergrid(self.nm):
            sl = (slice(None),)+ii# slice at grid location
            
            d = neighbours[(slice(None),)+sl]# (nd,2) neighbour values at grid location
            xmin, xmax = self.xmins[sl], self.xmaxs[sl]# (nd) xmin, xmax of model at grid location
            wmin, wmax = self.wmins[sl], self.wmaxs[sl]# (nd) wmin, wmax of model at grid location
            
            # replace xmin, xmax with None where neighbours are inactive
            xmin, xmax = xmin.astype(object).copy(), xmax.astype(object).copy()
            xmin[d[:,0]==0] = None
            xmax[d[:,1]==0] = None
            
            w.append(windows.construct_window_function_ND(xmin, xmax, self.scale*wmin, self.scale*wmax))
        
        return w
    
    def _get_isegs(self, ioa, ii):
        "Get all the isegs for a model at ii at segment order ioa"
        
        # get appropriate map
        ms = self.models_segments[ioa]# (ne,nd,nm) ms
        
        # get segments in model
        ms = ms[(slice(None),slice(None))+ii]# (ne,nd) ms segments at grid location
        
        # filter segments outside of segment grid
        c = self.cs[ioa]# (ne,nm)
        c = c[(slice(None),)+ii]# (ne) filter
        iis = ms[c]# (nc,nd) passed segments
        
        # unravel segment indices, as if segment was on onm grid
        if len(iis):# nc can equal 0 if ioa > 0 and only one model spans a full subdomain axis
            isegs = np.ravel_multi_index([np.array([ioa for _ in range(len(iis))])] + \
                                         [iis[:,i] for i in range(self.nd)], self.onm)
        else:
            isegs = np.array([], dtype=int)
        
        return isegs, iis
    
    def update_active(self, active=None):
        "Update the domain with the current active array"
        
        if active is None: active = np.ones(self.nm, dtype=int)
        if active.shape != self.nm: raise Exception("ERROR: active shape %s does not equal model grid %s!"%(active.shape, self.nm))
        
        # get neighbours
        neighbours = self._get_neighbours(active)#(nd, 2, nm)
        
        # get window functions
        w = self._get_window_functions(neighbours)
        
        # get filter arrays
        c_active = (active == 1)# (nm) active filter
        c_fixed  = (active == 2)# (nm) fixed filter
        c_active_neighbour = (np.sum(neighbours==1, axis=(0,1))>0)# (nm) true if any neighbours are active
        
        # initiate dynamic lists
        active_fixed_neighbours_ims, active_fixed_ims, active_ims = [], [], []
        
        # initiate empty maps
        m = [[] for im in range(self.N_MODELS)]# isegs for each im
        s = [[] for iseg in range(self.N_SEGMENTS)]# ims for each iseg
        
        # first, update maps with active models
        for im,ii in itergrid(self.nm):
            
            if c_active[ii]:
                
                # add to dynamic lists                
                i1 = len(active_fixed_neighbours_ims)
                active_fixed_neighbours_ims.append((im,i1))
                active_ims.append((im,i1))
                
                # for each segment type
                for ioa in range(self.N_ORDERS):
                    
                    # get segments at model location
                    isegs,iis = self._get_isegs(ioa, ii)
                    
                    # add models / segments to maps
                    for iseg,ii_ in zip(isegs,iis):
                        
                        # get start,end indices of this segment in model input array
                        i2 = m[im][-1][-1] if len(m[im]) else 0
                        i3 = i2 + np.product(self.batch_ns[ioa][(slice(None),)+tuple(ii_)])
                        
                        m[im].append((iseg,i2,i3))# add segment to model map
                        s[iseg].append((im,i1,i2,i3))# add model to segment map
        
        # next, update maps with fixed neighbours
        for im,ii in itergrid(self.nm):
                
            if c_fixed[ii] and c_active_neighbour[ii]:
                
                # add to dynamic lists                
                i1 = len(active_fixed_neighbours_ims)
                active_fixed_neighbours_ims.append((im,i1))
                
                # for each segment type
                for ioa in range(self.N_ORDERS):
                    
                    # get segments at model location
                    isegs,iis = self._get_isegs(ioa, ii)
                    
                    # add models / segments to maps
                    for iseg,ii_ in zip(isegs,iis):
                        if s[iseg]:# ONLY IF SEGMENT ALREADY EXISTS
                            
                            # get start,end indices of this segment in model input array
                            i2 = m[im][-1][-1] if len(m[im]) else 0
                            i3 = i2 + np.product(self.batch_ns[ioa][(slice(None),)+tuple(ii_)])
                            
                            m[im].append((iseg,i2,i3))# add segment to model map
                            s[iseg].append((im,i1,i2,i3))# add model to segment map
            
            # finally, create separate dynamic list for test time
            if c_fixed[ii] or c_active[ii]:
                active_fixed_ims.append(im)
                
        # get rid of unrequired indices in m
        m = [[t[0] for t in ts] for ts in m]
        
        # final checks
        assert len(m) == len(w) == len(self.n) == self.N_MODELS
        assert len(s) == self.N_SEGMENTS
        
        # set self values
        self.active, self.m, self.s, self.w = active, m, s, w
        self.active_fixed_neighbours_ims = active_fixed_neighbours_ims
        self.active_fixed_ims = active_fixed_ims
        self.active_ims = active_ims
    
    
    # SAMPLING / TORCH FUNCTIONS
    
    def _set_torch_tensors(self, device):
        "Make torch copies of tensors supplied by this class which are used during training"
        
        self.device = torch.device("cpu") if device is None else device
        totorch = lambda x: torch.from_numpy(x.copy().astype(np.float32)).to(self.device)
        
        self.segments_torch = [totorch(segment) for segment in self.segments]
        self.n_torch = [(totorch(mu), totorch(sd)) for mu,sd in self.n]
        
    def update_sampler(self, batch_size, random):
        "Set the batch size and type of the domain sampler. Be sure to use update_active to update index maps after this (!)"
        
        if len(batch_size) != self.nd: raise Exception("ERROR: len(batch_size) != self.nd! (%s)"%(batch_size,))
        
        # get global properties
        b = np.array(batch_size)# (nd)
        lim = np.array([[x.min(), x.max()] for x in self.subdomain_xs]).T# (2, nd) # global limits along each dim
        d = (lim[1]-lim[0])/(b-1)# (nd) # global grid spacing along each dim
        
        # reshape for broadcasting
        b_       = b.reshape((self.nd,)+(1,)*self.nd)# (nd,1..)
        d_       = d.reshape((self.nd,)+(1,)*self.nd)# (nd,1..)
        lim_ = lim.reshape((2,self.nd,)+(1,)*self.nd)# (2,nd,1..)
        
        # get individual segment batch sizes
        i1s,ns = [],[]
        for ioa in range(self.N_ORDERS): 
            segment = self.segments[ioa]# (2,nd,nm) segment
            
            if random:
                n = np.round(b_*(segment[1]-segment[0])/(lim_[1]-lim_[0])).astype(int)# (nd,nm) n number of points along each dim
            else:
                i1 =  np.ceil((segment[0]-lim_[0])/d_).astype(int)# (nd,nm) i1 (d_ and lim_ are broadcast)
                i2 = np.floor((segment[1]-lim_[0])/d_).astype(int)# (nd,nm) i2 (d_ and lim_ are broadcast)
                n = i2-i1+1# (nd,nm) n number of points along each dim
                i1s.append(i1)
            
            if not n.all(): raise Exception("ERROR: batch_n has zeros in!")
            ns.append(n)
            
        # set self values
        self.batch_lim, self.batch_d, self.batch_i1s, self.batch_ns = lim, d, i1s, ns
        self.random = random
        
    def sample_segments(self):
        """Sample all the segments in the model, returning torch tensors"""
        
        xs = [None for iseg in range(self.N_SEGMENTS)]
        for iseg in range(self.N_SEGMENTS):# for each segment
            
            if self.s[iseg]:# if segment is in the active/fixed neighbour domain
                
                iii = np.unravel_index(iseg, self.onm)# unravel index on onm
                
                # sample using TORCH
                if self.random:
                    n =       self.batch_ns[iii[0]][(slice(None),)+tuple(iii[1:])]# (nd)
                    s = self.segments_torch[iii[0]][(slice(None),slice(None))+tuple(iii[1:])]# (2, nd)
                    x = s[0]+(s[1]-s[0])*torch.rand(np.product(n), self.nd, device=self.device)
                else:
                    i1 = self.batch_i1s[iii[0]][(slice(None),)+tuple(iii[1:])]# (nd)
                    n  =  self.batch_ns[iii[0]][(slice(None),)+tuple(iii[1:])]# (nd)
                    xs_ = [li+(i1i+torch.arange(ni, dtype=torch.float32, device=self.device))*di for li,i1i,ni,di in zip(self.batch_lim[0],i1,n,self.batch_d)]
                    x = torch.stack(torch.meshgrid(*xs_), -1).view(-1, self.nd)
                    
                xs[iseg] = x# (N_segment, nd)
            
        return xs




if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    import plot_domain
    
    import sys
    sys.path.insert(0, '../shared_modules/')
    from helper import Timer
    
    plot = True
    #random = True
    random = False
    batch_size = 50
    
    
    # 1D test
    
    for subdomain_xs, subdomain_ws in [
                                       [[np.array([-5,0,3,6,10])], 
                                        [0.2*np.array([1,2,3,4,5])]],
                                       
                                       [[np.array([-5,0])], 
                                        [0.2*np.array([1,2])]],
                                       ]:
        
        active = np.ones([len(x)-1 for x in subdomain_xs])
        #if active.shape[0] != 1: active[[1,]] = 0; active[-1] = 2
        
        with Timer("__init__"):
            D = ActiveRectangularDomainND(subdomain_xs, subdomain_ws)
            D.update_sampler((batch_size,), random)
            D.update_active(active)
        print(D)
        print()
        
        # 1D sample segment
        xs = D.sample_segments()
        
        # 1D domain plot in grid space
        if plot:
            plt.figure(figsize=(10,10))
            for im,ii in itergrid(D.nm):
                
                # 1. plot all model grid points
                plt.scatter(*ii, 0, c="k", s=40)
                
                # 2. plot all active segments
                isegs = D.m[im]
                if isegs:
                    iiis = np.stack(np.unravel_index(isegs, D.onm), -1)# grid index of segments
                    ioas = iiis[:,0]# order numbers
                    
                    iis = ii + 0.1*(iiis[:,1:]-ii)# shift segment grid points to be around model grid point
                    for i,e in enumerate(iis):
                        plt.scatter(e[0], 0, c=plot_domain.colors[ioas[i]], s=10*(1+50*(D.N_ORDERS-ioas[i])))
            plt.gca().set_aspect("equal")
            plt.show()
    
        # 1D domain plot in x
        if plot:
            plot_domain.plot_1D(subdomain_xs, D)
            for x in xs:
                if x is not None:
                    plt.scatter(x[:,0], np.zeros_like(x[:,0]), marker="^", s=30)
            plt.show()
        
        
    # 2D test
    
    for subdomain_xs, subdomain_ws in [
                                       [[np.array([-5,0,3,6,10]),np.array([5,15,35])], 
                                        [0.2*np.array([1,2,3,4,5]),0.2*np.array([5,6,7])]],
                                       
                                       [[np.array([-5,0]),np.array([5,15,35])], 
                                        [0.2*np.array([1,2]),0.2*np.array([5,6,7])]],
                                       ]:
        
        active = np.ones([len(x)-1 for x in subdomain_xs])
        if active.shape[0] != 1: active[0,1] = 2; active[2,1] = 0; active[1] = 0
        
        with Timer("__init__"):
            D = ActiveRectangularDomainND(subdomain_xs, subdomain_ws)
            D.update_sampler((batch_size,batch_size//2), random)
            D.update_active(active)
        print(D)
        print()
        
        # 2D sample segment
        xs = D.sample_segments()
        
        # 2D print model/ segment maps
        for x in D.m: print(x)
        print()
        for x in D.s: print(x)
        print()
        for x in D.w: print(x)
        print()
        for x in D.n: print(x)
        print()
            
        # 2D domain plot in grid space
        if plot:
            plt.figure(figsize=(10,10))
            for im,ii in itergrid(D.nm):
                
                # 1. plot all model grid points
                plt.scatter(*ii, c="k", s=40)
                
                # 2. plot all active segments
                isegs = D.m[im]
                if isegs:
                    iiis = np.stack(np.unravel_index(isegs, D.onm), -1)# grid index of segments
                    ioas = iiis[:,0]# order numbers
                    
                    iis = ii + 0.1*(iiis[:,1:]-ii)# shift segment grid points to be around model grid point
                    for i,e in enumerate(iis):
                        plt.scatter(e[0], e[1], c=plot_domain.colors[ioas[i]], s=10*(1+50*(D.N_ORDERS-ioas[i])))
            plt.gca().set_aspect("equal")
            plt.show()
        
        # 2D domain plot in x
        if plot:
            plot_domain.plot_2D(subdomain_xs, D)
            for x in xs:
                if x is not None:
                    plt.scatter(x[:,0], x[:,1], marker="^", s=30)
            plt.show()
    

    # 3D test
    
    for subdomain_xs, subdomain_ws in [
                                       [[np.array([-5,0,3,6,10]),np.array([5,15,35,45]),np.array([-2,5,12])],
                                        [0.2*np.array([1,2,3,4,5]),0.2*np.array([2,3,4,5]),0.2*np.array([4,5,6])]],
                                       
                                       [[np.array([-5,0]),np.array([5,15,35,45]),np.array([-2,5])],
                                        [0.2*np.array([1,2]),0.2*np.array([2,3,4,5]),0.2*np.array([4,5])]],
                                       ]:
        
        active = np.ones([len(x)-1 for x in subdomain_xs])
        if active.shape[0] != 1: active[0,1,0] = 2; active[2,1,0] = 0; active[1] = 0
        
        with Timer("__init__"):
            D = ActiveRectangularDomainND(subdomain_xs, subdomain_ws)
            D.update_sampler((batch_size,batch_size,batch_size), random)
            D.update_active(active)
        print(D)
        print()
        
        # 3D sample segment
        xs = D.sample_segments()
        
        # 3D domain plot in x
        if plot:
            plot_domain.plot_2D_cross_section(subdomain_xs, D, [0,1])
            for x in xs:
                if x is not None:
                    plt.scatter(x[:,0], x[:,1], marker="^", s=30)
            plt.show()
            