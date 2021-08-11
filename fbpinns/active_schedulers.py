#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:24:20 2021

@author: bmoseley
"""

# This module defines various active schedulers, which are iterables which allow us 
# to define which FBPINN subdomains are active/fixed/inactive at each training step

# This module is used by constants.py when defining FBPINN / PINN problems

import itertools

import numpy as np


class _ActiveScheduler:
    "Helper class for scheduling updates to the active array"
    
    name = None
    
    def __init__(self, N_STEPS, D):
        
        self.N_STEPS = N_STEPS
        self.nd = D.nd
        self.nm = D.nm
        self.xx = D.xx.copy()
        
    def __len__(self):
        
        return self.N_STEPS
    
    def __iter__(self):
        
        # returns None if active map not to be changed, otherwise active map
        raise NotImplementedError


# ALL ACTIVE SCHEDULER

class AllActiveSchedulerND(_ActiveScheduler):
    "All models are active all of the time"
    
    name = "All"
    
    def __iter__(self):
        
        for i in range(self.N_STEPS):
            if i == 0: yield np.ones(self.nm, dtype=int)# (nm)
            else: yield None


# POINT-BASED ACTIVE SCHEDULERS

class _SubspacePointActiveSchedulerND(_ActiveScheduler):
    "Slowly expands radially outwards from a point in a subspace of the domain (in x units)"
    
    def __init__(self, N_STEPS, D, point, iaxes):
        super().__init__(N_STEPS, D)
        
        point = np.array(point)# point in constrained axes
        iaxes = list(iaxes)# unconstrained axes
        
        # validation
        if point.ndim != 1: raise Exception("ERROR: point ndim !=1")
        if len(point) > self.nd: raise Exception("ERROR: len point > self.nd")
        if len(iaxes) + len(point) != self.nd: raise Exception("ERROR: len iaxes + len point != nd")
        
        self.point = point
        self.iaxes = iaxes
    
    def _get_radii(self, point, xx):
        "Get the radii from a point in a subspace of xx"
        
        # get subspace dimensions
        nd, nm = xx.shape[0], tuple(s-1 for s in xx.shape[1:])
        assert len(nm) == nd
        assert len(point) == nd# make sure they match with point
        
        # get xmin, xmax of each model
        xmins = xx[(slice(None),)+(slice(None,-1),)*nd]# (nd, nm)  self.xx (nd,nm+1)
        xmaxs = xx[(slice(None),)+(slice(1, None),)*nd]# (nd, nm)
        
        # whether point is inside model
        point = point.copy().reshape((nd,)+(1,)*nd)# (nd, (1,)*nd)
        c_inside = (point >= xmins) & (point < xmaxs)# point is broadcast
        c_inside = np.product(c_inside, axis=0).astype(bool)# (nm)    must be true across all dims
        
        # get bounding corners of each model
        x = np.stack([xmins, xmaxs], axis=0)# (2, nd, nm)
        bb = np.zeros((2**nd, nd)+nm)# (2**nd, nd, nm)
        for ic,offsets in enumerate(itertools.product(*([[0,1]]*nd))):# for each corner     
            for i,o in enumerate(offsets):# for each dimension
                bb[(ic,i)+(slice(None),)*nd] = x[(o,i)+(slice(None),)*nd]
        
        # distance from each corner to point
        point = point.copy().reshape((1, nd)+(1,)*nd)# (1, nd, (1,)*nd)
        r = np.sqrt(np.sum((bb - point)**2, axis=1))# (2**nd, nm)   point is broadcast
        rmin, rmax = np.min(r, axis=0), np.max(r, axis=0)# (nm)
        
        # set rmin=0 where point is inside model
        rmin[c_inside] = 0.
        
        return rmin, rmax
        
    def __iter__(self):
        
        # slice constrained axes
        ic = [i for i in range(self.nd) if i not in self.iaxes]# constrained axes
        sl = tuple([ic, *[slice(None) if i in ic else 0 for i in range(self.nd)]])
        xx = self.xx[sl]# (nd-uc, nm-uc)
        
        # get subspace radii
        rmin, rmax = self._get_radii(self.point, xx)
        
        # insert unconstrained axes back in (for broadcasting below)
        rmin, rmax = np.expand_dims(rmin, axis=self.iaxes), np.expand_dims(rmax, axis=self.iaxes)# (nm with 1s)
        
        # initialise active array, start scheduling
        active = np.zeros(self.nm, dtype=int)# (nm)
        r_min, r_max = np.min(rmin), np.max(rmax)
        for i in range(self.N_STEPS):
            
            # advance radius
            rt = r_min + (r_max-r_min)*(i/(self.N_STEPS))
            
            # get filters
            c_inactive = (active == 0)
            c_active   = (active == 1)# (nm) active filter
            c_radius = (rt >= rmin) & (rt < rmax)# (nm) circle inside box (approximately! (only uses corners))
            c_to_active = c_inactive & c_radius# c_radius is broadcast
            c_to_fixed = c_active & (~c_radius)# c_radius is broadcast
            
            # set values
            if c_to_active.any() or c_to_fixed.any():
                active[c_to_active] = 1
                active[c_to_fixed] = 2
                yield active
            else:
                yield None

class PointActiveSchedulerND(_SubspacePointActiveSchedulerND):
    "Slowly expands outwards from a point in the domain (in x units)"
    
    name = "Point"
    
    def __init__(self, N_STEPS, D, point):
        
        if len(point) != D.nd: raise Exception("ERROR: point incorrect shape %s"%(point.shape,))
        
        super().__init__(N_STEPS, D, point, iaxes=[])

class LineActiveSchedulerND(_SubspacePointActiveSchedulerND):
    "Slowly expands outwards from a line in the domain (in x units)"
    
    name = "Line"
    
    def __init__(self, N_STEPS, D, point, iaxis):
        
        if D.nd < 2: raise Exception("ERROR: requires nd >=2")
        if len(point) != D.nd-1: raise Exception("ERROR: point incorrect shape %s"%(point.shape,))
        
        super().__init__(N_STEPS, D, point, iaxes=[iaxis])
        
class PlaneActiveSchedulerND(_SubspacePointActiveSchedulerND):
    "Slowly expands outwards from a plane in the domain (in x units)"
    
    name = "Plane"
    
    def __init__(self, N_STEPS, D, point, iaxes):
        
        if D.nd < 3: raise Exception("ERROR: requires nd >=3")
        if len(point) != D.nd-2: raise Exception("ERROR: point incorrect shape %s"%(point.shape,))
        
        super().__init__(N_STEPS, D, point, iaxes=iaxes)
        


if __name__ == "__main__":
        
    from domains import ActiveRectangularDomainND
    from constants import get_subdomain_ws
    
    x = np.array([-6,-4,-2,0,2,4,6])
    
    subdomain_xs1 = [x]
    D1 = ActiveRectangularDomainND(subdomain_xs1, get_subdomain_ws(subdomain_xs1, 0.5))
    
    subdomain_xs2 = [x, x]
    D2 = ActiveRectangularDomainND(subdomain_xs2, get_subdomain_ws(subdomain_xs2, 0.5))
    
    subdomain_xs3 = [x, x, x]
    D3 = ActiveRectangularDomainND(subdomain_xs3, get_subdomain_ws(subdomain_xs3, 0.5))
    
    # test point
    for D in [D1, D2, D3]:
        print("Point")
        point = np.array([0]*D.nd)
        A = PointActiveSchedulerND(100, D, point)
        for i, active in enumerate(A):
            if active is not None:
                print(i)
                print(active)
        print()
    
    # test line
    for D in [D2, D3]:
        print("Line")
        point = np.array([0]*(D.nd-1))
        A = LineActiveSchedulerND(100, D, point, 0)
        for i, active in enumerate(A):
            if active is not None:
                print(i)
                print(active)
        print()
    
    # test plane
    for D in [D3]:
        print("Plane")
        point = np.array([0]*(D.nd-2))
        A = PlaneActiveSchedulerND(100, D, point, [0,1])
        for i, active in enumerate(A):
            if active is not None:
                print(i)
                print(active)
        print()
        