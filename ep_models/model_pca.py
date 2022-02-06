#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:15:00 2017

@author: jd1336
"""
import scipy as sp
from scipy.stats import uniform
from matplotlib import pyplot
from scipy.integrate import odeint
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.decomposition import PCA

class Model_pca(object):
    """Aliev-panfilov model
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

    def __init__(self, simu, obs, parTrue, cormfree, maskidx_12lead = 0, 
                 use_cpd = False, correspond = 0):       
        # Measurement data
        self.dataf = obs.bsp 
        self.datat = obs.time
        
        # Simulation parameters
        self.y0  = np.require(simu.X0,requirements='C') 
        self.ts  = np.require(simu.s, requirements=['C'])
        # Forward matrix
        self.H   = np.require(simu.H,requirements='C')
        
        # Other parameter
        self.dim = simu.H.shape[1] # number of Meshfree nodes      
        self.cormfree = cormfree # 3D coordinates of Meshfree nodes
        self.parTrue = parTrue # ground truth parameters
        
        self.maskidx_12lead = maskidx_12lead # if 12 lead ECG is used
        self.num_leads = obs.bsp.shape[0]
        self.use_cpd = use_cpd
        self.correspond = correspond

    @staticmethod
    def fp(y, t, k, e, s, dim, par):
        """
        aliev panfilov model
        """
        u    = y[0:dim]
        v    = y[dim:dim*2]   
        dudt=s.dot(u)+k*u*(1-u)*(u-par)-u*v
        dvdt=-e*(k*u*(u-par-1)+v)  
        dydt=np.r_[dudt,dvdt]
        return dydt   
    
    def simulate_ecg(self, fullpar, vae=0):
        if len(fullpar)<=self.pdim:
            fullpar = self.mapparam(fullpar, vae)   # mean from generative model
        if self.use_cpd and len(fullpar)!=self.dim:
            fullpar = fullpar[self.correspond]

        # calculate transmural action potential (TMP)
        sol = odeint(self.fp,self.y0,self.datat,args=(self.k,self.e, self.ts,self.dim,fullpar))       
        tmp = sol[:,0:self.dim].transpose()
        # compute the ECG measurement
        bsp = self.H.dot(tmp);
        # if 12-lead ECG is used extract those from full 120 lead
        if (self.num_leads==12):
            bsp = self.get12leads(bsp)
        return self.datat, tmp, bsp
    
    def get_12_leads(self, bsp):
        bsp120 = bsp[self.maskidx_12lead-1,:]
        RA = (bsp120[15-3-1,:] + bsp120[16-3-1,:])/2
        LA = (bsp120[64-3-1,:] + bsp120[65-3-1,:])/2
        LL = bsp120[70-3-1,:]
        I  = LA - RA
        II = LL - RA
        III = LL - LA
        aVR = -(I + II)/2;
        aVL = (I - III)/2;
        aVF = (II + III)/2;
        V1  =  bsp120[25-3-1,:]
        V2  = bsp120[39-3-1,:]
        V3  = (bsp120[46-3-1,:] + bsp120[53-3-1,:] + bsp120[47-3-1,:] 
            + bsp120[54-3-1,:])/4
        V4  = bsp120[61-3-1,:]
        V5  = (bsp120[68-3-1,:] + 2*bsp120[72-3-1,:])/3
        V6  = bsp120[76-3-1,:]
        bsp12 = np.asarray([I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6])
        return bsp12
    
    
    def compute_objfunc(self, fullpar, vae = 0):
        if len(fullpar)==2:
            fullpar = self.mapparam(fullpar, vae)   # mean of the generative model
 #       if self.use_cpd and len(fullpar)!=self.dim:
 #           fullpar = fullpar[self.correspond]
        # calculate transmural action potential (TMP)
        sol = odeint(self.fp,self.y0,self.datat,args=(self.k,self.e, self.ts,self.dim, fullpar))       
        tmp = sol[:,0:self.dim].transpose()
        # compute the ECG measurement
        bsp = self.H.dot(tmp)
        # if 12-lead ECG is used extract those from full 120 lead
        if self.num_leads==12:
            bsp = self.get12leads(bsp)
        # compute squared error    
        diff = ((self.dataf-bsp)**2)
        sse= - diff.sum()    
        return sse
        
    def mapparam(self, idpar, vae):        
        z_mu = np.array([idpar]*1)
        x_mean = vae.inverse_transform(z_mu)           
        fullpar = x_mean[0]
        if self.use_cpd:
            fullpar = fullpar[self.correspond]
        return fullpar      
        
    def plotparam(self, fullpar, vae =0, ax=None):   
        if len(fullpar)==2:
            fullpar = self.mapparam(fullpar, vae)  

        if ax is None:
            fig = pyplot.figure()
            ax = Axes3D(fig)
        ax.scatter(self.cormfree[:,0],self.cormfree[:,1],self.cormfree[:,2]
                       ,s=20,c=fullpar,vmin=self.lb,vmax=self.ub,cmap=cm.get_cmap('jet'))
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
               self.cormfree[:,2],s=20,c=fullpar,vmin=self.lb,
                       vmax=self.ub,cmap=cm.get_cmap('jet'))
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        pyplot.show()