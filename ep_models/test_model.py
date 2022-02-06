import scipy as sp
from scipy.stats import uniform
from matplotlib import pyplot
from scipy.integrate import odeint
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import math
import time
from numpy.linalg import multi_dot

class Model(object):
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
        if len(fullpar)<=self.p_dim:
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
        t1 = time.time()
        fullpar_z=np.array(fullpar)
        if len(fullpar)<=self.dim:
            fullpar = self.mapparam(fullpar, vae)   # mean of the generative model
        sol = odeint(self.fp,self.y0,self.datat,args=(self.k,self.e, self.ts,self.dim, fullpar))       
        tmp = sol[:,0:self.dim].transpose()
        # compute the ECG measurement
        bsp = self.H.dot(tmp)
        # if 12-lead ECG is used extract those from full 120 lead
        if self.num_leads==12:
            bsp = self.get12leads(bsp)
        # compute squared error    
        diff = (self.dataf-bsp)**2
        ss = (diff.sum())/.04 + np.sum(fullpar_z**2)
        sse= (-1/2)*ss   
        print("sse is",sse)
        t2 = time.time()
        print("time taken for iteration",t2-t1,'seconds')
        return sse

    def compute_objfunc_jwala(self, fullpar, vae = 0):
        if len(fullpar)<=self.dim:
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
        diff = (self.dataf-bsp)**2
        #new_diff = np.exp(diff)
#         ss = diff.sum() 
        ss = (diff.sum())
        sse= (-1)*ss   
        return sse
    
    
 
 
#     ######This one is for test purpose#########
    #@staticmethod
    def drange2(self,start, stop, step):
        numelements = int((stop-start)/float(step))
        for i in range(numelements+1):
                yield start + i*step
    
    def compute_objfunc2(self, fullpar, vae = 0):
        loss=[]

        print("test")
#         m = np.median(noise_mu[noise_mu > 0])
#         noise_mu[noise_mu==0] = m
#         cov=noise_mu*np.identity(370)
#         cov_inv=np.linalg.inv(cov)
#         fullpar=np.array([0.3,0.6])
        for i in self.drange2(-4, 4, .1):
            for j in self.drange2(-4, 4, .1):    
                fullpar=np.array([i,j])
                fullpar_z=fullpar
                print(fullpar_z)
        #     #                 z_l2=math.sqrt(sum(np.square(fullpar)))
                #print(norm)
                if len(fullpar)<=self.dim:
                    fullpar = self.mapparam(fullpar, vae)   # mean of the generative model
         #       if self.use_cpd and len(fullpar)!=self.dim:
         #           fullpar = fullpar[self.correspond]
                # calculate transmural action potential (TMP)
        #         print(fullpar.shape)
                t1=time.time()

                sol = odeint(self.fp,self.y0,self.datat,args=(self.k,self.e, self.ts,self.dim, fullpar))       
                tmp = sol[:,0:self.dim].transpose()

                t2=time.time()
        #                 print(t2-t1)
                bsp = self.H.dot(tmp)
                # if 12-lead ECG is used extract those from full 120 lead
                if self.num_leads==12:
                    bsp = self.get12leads(bsp)
                # compute squared error    
                diff = (self.dataf-bsp)**2
                ss = diff.sum() 
#                 lc=self.dataf-bsp
#                 b=np.multiply(self.dataf,self.dataf)
#         #                 ybar=np.mean(self.dataf,axis=1)
#                 mm=np.sum(b,axis=1)/309 
#                 noise_mu=mm/4
#         #                 f=open("/home/mz1482/project/BOVAE (miccai2018)/debug_largebsp.txt","a")
#         #                 f.write(str(ss)+","+str(bsp)+"\n")
#                 tot=[]
#                 index=[]
#                 kk=[]
#                 for k in range(len(noise_mu)):
#                     s2=noise_mu[k]
#                     if s2!=0:
#                         p1=lc[k,:]
#                         aa= (0.5*(1/s2)*multi_dot([p1, p1.T]))*(1/309)
#                         tot.append(aa)
#         #                 index.append(i)
#                 kernel=sum(tot)/120
#                 ss=np.exp(-kernel)
#                 print(ss)
        #                 print(fullpar_z)

        #                 loss.append(ss)
        #                 sse= - aa.sum()    
        #                 print(cov_inv)
            #sse= - new_diff.sum()
                f=open("/home/mz1482/project/BOVAE (miccai2018)/likelihood.txt","a")
                f.write(str(ss)+","+str(fullpar_z[0])+","+str(fullpar_z[1])+"\n")
        #                 f.write(str(ss)+","+str(fullpar_z[0])+","+str(fullpar_z[1])+"\n")
                f.close()
        return bsp
     ####################Test Ends############
        
    #MH sampling for exact posterior
    def post_MH(self,fullpar, vae = 0):

        sigma2=.04
        fullpar_z=fullpar
        # print(fullpar_z)
        #     #                 z_l2=math.sqrt(sum(np.square(fullpar)))
                #print(norm)
        if len(fullpar)<=self.dim:
            fullpar = self.mapparam(fullpar, vae)   # mean of the generative model
                # calculate transmural action potential (TMP)
        #         print(fullpar.shape)
        sol = odeint(self.fp,self.y0,self.datat,args=(self.k,self.e, self.ts,self.dim, fullpar))       
        tmp = sol[:,0:self.dim].transpose()
        bsp = self.H.dot(tmp)
                # if 12-lead ECG is used extract those from full 120 lead
        # if self.num_leads==12:
        #     bsp = self.get12leads(bsp)
                # compute squared error    
        diff = (self.dataf-bsp)**2
        ss = diff.sum() 
        prior = np.exp(-0.5*(fullpar_z[0]**2+fullpar_z[1]**2))
        lik=np.exp(-0.5*ss/sigma2)
        post = lik*prior
        return post



    def MH_sampling(self,iteration,chain,m,v,vae=0):
        #the ground truth we are using, there the proposal was m=0,v=0.5
#         z=np.array([0,0])
        samples = np.empty((0, 2))
#         pdf=[]
        t1=time.time()
        for c in range(chain):
            j=0
            z = np.random.normal(loc = 0, scale = 1,size=2)
            a=self.post_MH(z,vae)
            for i in range(iteration):
                z_star = z + np.random.normal(loc = m, scale = v,size=2)
                r=np.random.rand()
                b=self.post_MH(z_star,vae)
                rho = min(1, b/a)
                if r < rho:
                    z = z_star
                    a = b
                    samples = np.append(samples, [z], axis=0)
#                     pdf=np.append(pdf,b)
                    f=open("/home/mz1482/project/my_work/mh_case3_inf_e4_chain.txt","a")
                    f.write(str(c)+","+str(z[0])+ "," +str(z[1]) + ","+ str(b) +"\n")
                    f.close()
                    print(b)
                    j=j+1
                print(i)
        t2=time.time()
        print((t2-t1)/3600)
        ar = j/iteration
        print(ar)
        return samples

    
    
    def test_print(self):
        
        print("hello 2 stage")

        return 0
        #########test 1 z #######
    def testz(self, fullpar, vae = 0):
        loss=[]

        print("test")
        fullpar=np.array([0.3,0.6])
        fullpar_z=fullpar
        print(fullpar_z)
        if len(fullpar)<=self.dim:
            fullpar = self.mapparam(fullpar, vae)   # mean of the generative model
        t1=time.time()

        sol = odeint(self.fp,self.y0,self.datat,args=(self.k,self.e, self.ts,self.dim, fullpar))       
        tmp = sol[:,0:self.dim].transpose()

        t2=time.time()
        bsp = self.H.dot(tmp)
        # if 12-lead ECG is used extract those from full 120 lead
        if self.num_leads==12:
            bsp = self.get12leads(bsp)
        # compute squared error    
#                 diff = (self.dataf-bsp)**2 
        lc=self.dataf-bsp
        b=np.multiply(self.dataf,self.dataf)
#                 ybar=np.mean(self.dataf,axis=1)
        mm=np.sum(b,axis=1)/309 
        noise_mu=mm/100
#                 f=open("/home/mz1482/project/BOVAE (miccai2018)/debug_largebsp.txt","a")
#                 f.write(str(ss)+","+str(bsp)+"\n")
        tot=[]
        index=[]
        kk=[]
        for k in range(len(noise_mu)):
            s2=noise_mu[k]
            if s2!=0:
                p1=lc[k,:]
                aa= (0.5*(1/s2)*multi_dot([p1, p1.T]))*(1/309)
                tot.append(aa)
#                 index.append(i)
        kernel=sum(tot)/120
        ss=np.exp(-kernel)
        return bsp
        
        
    def mapparam(self, idpar, vae):        
        z_mu = np.array([idpar]*vae.batch_size)
#         print(z_mu.shape, vae.batch_size)
        x_mean = vae.generate(z_mu)
#         print(x_mean.shape)
        fullpar = x_mean[0]
        if self.use_cpd:
            fullpar = fullpar[self.correspond]
        return fullpar
    
    def mapparam2(self, idpar, vae):        
        z_mu = np.array([idpar]*vae.batch_size)
        print(z_mu.shape)
        x_mean = vae.generate(z_mu)
        print(x_mean.shape)
        fullpar = x_mean[0]
        if self.use_cpd:
            fullpar = fullpar[self.correspond]
        return fullpar  
    
    
    def ap_model(self, fullpar):        
        sol = odeint(self.fp,self.y0,self.datat,args=(self.k,self.e, self.ts,self.dim, fullpar))       
        tmp = sol[:,0:self.dim].transpose()
        bsp = self.H.dot(tmp)
        return bsp    
 
    
   
        
    def plotparam(self, fullpar, vae =0, ax=None):   
        if len(fullpar)!=self.dim:
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