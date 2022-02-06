#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:15:00 2017

@author: jd1336 
"""

from matplotlib import pyplot
from scipy.integrate import odeint
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


class Model_AWM(object):
    """Aliev-panfilov implemented in Python
    
    Attributes:
        k   model parameter
        e   model parameter
        t   time sequence
        y0  initial activation
        H   transfer matrix
        dim meshfree nodes   
    """
    
    k  =  8
    e  =  0.01   
    lb = 0.0
    ub = 0.52
    sigmaa=1.0  

    def __init__(self, simu1, obs1, simu2, obs2,
                 parTrue, cormfree, maskidx_12lead=0, 
                 use_cpd = False, correspond = 0):       
        
        self.source1 = simu1.source
        self.source2 = simu2.source
#         self.source3 = simu3.source
        
        self.ts1  = np.require(simu1.s, requirements=['C']);
        self.ts2  = np.require(simu2.s, requirements=['C']);
#         self.ts3  = np.require(simu3.s, requirements=['C']);

        
        self.H   = np.require(simu1.H,requirements=['C'])
        self.dim = simu1.dim      

#         self.y03  = np.zeros((self.dim*2), order='C')
        self.y01  = np.zeros((self.dim*2), order='C')
        self.y02  = np.zeros((self.dim*2), order='C')
        
        self.dataf1 = obs1.bsp
        self.dataf2 = obs2.bsp
#         self.dataf3 = obs3.bsp
        print(self.dataf1.shape)
        print(self.dataf2.shape)
#         print(self.dataf3.shape)
        print(self.H.shape)
        print(self.dim)
        
        self.datat = obs1.time        
        self.simt  = (obs1.time)*155.0/376
        
        self.cormfree = cormfree
        self.parTrue = parTrue
        self.maskidx_12lead = maskidx_12lead 
        self.num_leads = obs1.bsp.shape[0]

        self.use_cpd = use_cpd
        self.correspond = correspond
    
     
    @staticmethod    
    def fp(y, t, k, e, s, dim, source, par):
        """
        aliev panfilov model
        """

        dur = 2
        if(t <= 155.0/376*dur):
            s[np.where(source != 0)[0],:]  = 0   
        else:           
            source = np.zeros(dim)


        u    = np.require(y[0:dim])
        v    = np.require(y[dim:dim*2])  

        dudt = np.inner(s,u) + k*u*(1-u)*(u-par)-u*v + source
        dvdt = -e*(k*u*(u-par-1)+v)  
        dydt = np.r_[dudt,dvdt]
        return dydt   
    
    
    
    def compute_objfunc(self, fullpar, vae=0):     
        print("Compute Obj start")
        if len(fullpar) == 2:
            fullpar = self.mapparam(fullpar, vae) 
            print("fullpar",fullpar.shape)# decode the par overhere
        
        print("Op 1")
        sol1 = odeint(self.fp, self.y01, self.simt, 
                      args = (self.k, self.e, self.ts1, 
                              self.dim, self.source1, fullpar))   
        print("Op 2")
        sol2 = odeint(self.fp, self.y02, self.simt, 
                      args = (self.k, self.e, self.ts2, 
                              self.dim, self.source2, fullpar))             
#         print("Op 3")
#         sol3 = odeint(self.fp, self.y03, self.simt, 
#                       args = (self.k, self.e, self.ts3, 
#                               self.dim, self.source3, fullpar))     

        tmp1 = np.require(sol1[:,0:self.dim].transpose(), 
                          requirements=['A','F'])
        tmp2 = np.require(sol2[:,0:self.dim].transpose(), 
                          requirements=['A','F'])
#         tmp3 = np.require(sol3[:,0:self.dim].transpose(), 
#                           requirements=['A','F'])
        
        bsp1 = self.H.dot(tmp1)
        bsp2 = self.H.dot(tmp2)
#         bsp3 = self.H.dot(tmp3)
        print(bsp1.shape)
        print(bsp2.shape)
#         print(bsp3.shape)

        diff1 = ((self.dataf1-bsp1)**2)
        diff2 = ((self.dataf2-bsp2)**2)
#         diff3 = ((self.dataf3-bsp3)**2)
        
        l_lik= -  (diff1.sum() + diff2.sum()) #+ diff3.sum())      
        print("Compute Obj end")
        return l_lik
        
    def mapparam(self, idpar, vae):        
        z_mu = np.array([idpar]*vae.batch_size)
        x_mean = vae.generate(z_mu)           
        fullpar=x_mean[0]
        if self.use_cpd:
            fullpar = fullpar[self.correspond]               
        return fullpar      
        
    def plotparam(self, fullpar, vae =0, ax=None):   
        if len(fullpar)!=self.dim:
            fullpar = self.mapparam(fullpar, vae)  
        if ax is None:
            fig = pyplot.figure()
            ax = Axes3D(fig)
        ax.scatter(self.cormfree[:,0], self.cormfree[:,1],  self.cormfree[:,2],
                   s=20, c=fullpar, vmin=np.min(fullpar), vmax=np.max(fullpar), cmap=cm.get_cmap('jet'))
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        pyplot.show()              
        
    def plotGT(self,ax=None):        
        fullpar = self.parTrue 
        if ax is None:
            fig = pyplot.figure()
            ax = Axes3D(fig)
        ax.scatter(self.cormfree[:,0], self.cormfree[:,1],
                   self.cormfree[:,2], s=20, c=fullpar,  vmin=0,
                   vmax=0.5,cmap=cm.get_cmap('jet'))
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        pyplot.show()