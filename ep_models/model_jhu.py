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
import time

class Model_Jhu(object):
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
    lb = 0
    ub = 0.5
  

    def __init__(self, simu, obs, parTrue, cormfree,  
                 use_cpd = False, correspond = 0):       
        self.dataf = obs.bsp
        self.datat = obs.time
        self.dim = simu.dim
        
        self.y0  = np.zeros((2*self.dim))
        self.ts  = np.require(simu.s, requirements=['A','F']);

       
        self.simt  = (obs.time-4810.0)*215.0/590
        self.cormfree = cormfree
        self.parTrue = parTrue
        self.sti=simu.sti
        self.idx_nodes=simu.sur_idx
        
        self.use_cpd = use_cpd
        self.correspond = correspond


    @staticmethod
    def fp(y, t, k, e, s, dim, sti, par):
        """
        aliev panfilov model
        """
        
        dur=2;
        if(t<=215/590*dur):
            stim=sti[:,0]
            s[np.where(stim!=0)[0],:]  = 0            
        elif(t>215/590*5 and t<=215/590*(5+dur)):
            stim=sti[:,1]
            s[np.where(stim!=0)[0],:]  = 0
        elif(t>215/590*10 and t<=215/590*(10+dur)):
            stim=sti[:,2]
            s[np.where(stim!=0)[0],:]  = 0
        elif(t>215/590*17 and t<=215/590*(17+dur)):
            stim=sti[:,3]
            s[np.where(stim!=0)[0],:]  = 0
        else:
            stim=sti[:,4]

 
        u    = y[0:dim]
        v    = y[dim:dim*2]   
        dudt=s.dot(u)+k*u*(1-u)*(u-par)-u*v+stim
        dvdt=-e*(k*u*(u-par-1)+v)  
        dydt=np.r_[dudt,dvdt]
        return dydt   
    
    def simulate_ecg(self, fullpar, vae=0):            
        if len(fullpar)==2:
            fullpar = self.mapparam(fullpar, vae)   # mean from generative model


        sol = odeint(self.fp,self.y0,self.simt,args=(self.k,self.e, self.ts,self.dim,self.sti,fullpar))       
        tmp = sol[:,0:self.dim].transpose()
        tmp = tmp[self.idx_nodes,:]
        return self.datat,tmp       
    
    def compute_objfunc(self, fullpar, vae=0):
        
        if len(fullpar)==2:
            fullpar = self.mapparam(fullpar, vae)   # decode the par overhere
        sol = odeint(self.fp,self.y0,self.simt,args=(self.k,self.e, self.ts,self.dim,self.sti,fullpar))       
        tmp = sol[:,0:self.dim].transpose()
        tmp = tmp[self.idx_nodes,:]
        diff = ((self.dataf-tmp)**2)
        l_lik= -1 * diff.sum()
        return l_lik
    
    def compute_objfunc_pdf(self, fullpar, vae = 0):
        t1 = time.time()
        fullpar_z=np.array(fullpar)
        if len(fullpar)<=self.dim:
            fullpar = self.mapparam(fullpar, vae)   # mean of the generative model
        sol = odeint(self.fp,self.y0,self.simt,args=(self.k,self.e, self.ts,self.dim,self.sti,fullpar))       
        tmp = sol[:,0:self.dim].transpose()
        tmp = tmp[self.idx_nodes,:]
        diff = ((self.dataf-tmp)**2)   
        ss = (diff.sum())/100 + np.sum(fullpar_z**2)
        sse= (-1/2)*ss   
        print("sse is ",sse)
        t2 = time.time()
        print("time taken for iteration",t2-t1,'seconds')
        return sse
        
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
        ax.scatter(self.cormfree[:,0],self.cormfree[:,1],self.cormfree[:,2],
                  s=20,c=fullpar,vmin=np.min(fullpar),vmax=np.max(fullpar),cmap=cm.get_cmap('jet'))
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        pyplot.show()              
        
    def plotGT(self,ax=None):        
        fullpar = self.parTrue 
        if ax is None:
            fig = pyplot.figure()
            ax = Axes3D(fig)
        ax.scatter(self.cormfree[:,0],self.cormfree[:,1],
                   self.cormfree[:,2],s=20,c=fullpar,vmin=0,
                   vmax=0.5,cmap=cm.get_cmap('jet'))
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        pyplot.show()