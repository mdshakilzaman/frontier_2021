import sys
sys.path.append('/home/mz1482/project/frontier/')
import tensorflow as tf
import model_vae_tf

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import time
import pickle

sys.path.append('/home/mz1482/project/frontier/bayesopt/')
sys.path.append('/home/mz1482/project/frontier/ep_models/')
from bayesopt import BayesianOptimization
# from bayes_opt import testbo
from ep_models import test_model
from skimage.filters import threshold_otsu
import math
from numpy.linalg import multi_dot

#case1: 1239
#case2: 1373
#case3:1230
main_path = '/home/mz1482/project/frontier/'
# filename = 'shakil_ir_1230' 
filename = 'training_ir_ic_1230' 
resultspath = '/home/mz1482/project/frontier/models/'
use_cpd = True
p_dim = 1230 # optimized data dim
latent_dim = 2

if use_cpd:
    if (p_dim == 1373):
        vae_pdim = 1230
        filename = 'training_ir_ic_1230' 
        correspond_mfile = scipy.io.loadmat(main_path+'non_rigid_reg/corr3to2.mat',squeeze_me=True,struct_as_record=False)
        correspondance=correspond_mfile['Correspondence']
    elif (p_dim == 1230):
        vae_pdim = 1239
        filename = 'training_ir_ic_1239' 
        correspond_mfile = scipy.io.loadmat(main_path+'non_rigid_reg/corr1to3.mat',squeeze_me=True,struct_as_record=False)
        correspondance=correspond_mfile['Correspondence']
    else:
        vae_pdim = 1230
        filename = 'training_ir_ic_1230' 
        correspond_mfile = scipy.io.loadmat(main_path+'non_rigid_reg/corr3to1.mat',squeeze_me=True,struct_as_record=False)
        correspondance=correspond_mfile['Correspondence']
else:
    correspondance = 1
    vae_pdim = p_dim
correspondance = correspondance - 1
print("hello")

tf.reset_default_graph() # remove all tensors in default graph
network_architecture = dict(n_hidden_recog_1=512, # 1st layer encoder neurons
         n_hidden_recog_2=512, # 2nd layer encoder neurons
         n_hidden_gener_1=512, # 1st layer decoder neurons
         n_hidden_gener_2=512, # 2nd layer decoder neurons
         n_input=vae_pdim, # data input dimensionality (e.g., MNIST img shape: 28*28)
         n_z=latent_dim)  # dimensionality of the latent space

vae = model_vae_tf.VariationalAutoencoder(network_architecture, 
                             learning_rate=0.001, 
                             batch_size=64)

vae.load(resultspath + filename + "/m_" + str(latent_dim) + "d")

with open(resultspath + filename + "/z_posterior.pkl", 'rb') as input:
        z_mu_1 =  pickle.load(input)
        z_var_1 = pickle.load(input)
        z_mu_5 = pickle.load(input)
        z_var_5 = pickle.load(input)
        z_alpha_5 = pickle.load(input)
        
rpath="./results/"
dpath="./data_sets/synthetic_cases/"
files = ['case3_inf_e4']
rpath =  rpath + 'case3/'
dpath = dpath + 'case3/'

fname = files[0] + '.mat'   
matFiles = scipy.io.loadmat(main_path+dpath+fname,squeeze_me=True,struct_as_record=False)
parTrue=matFiles['parTrue']
obs=matFiles['obs']
simu=matFiles['simu']
corMfree=matFiles['corMfree']
cardiac_model = test_model.Model(simu, obs, parTrue, corMfree, maskidx_12lead = 0
                            ,use_cpd = use_cpd, correspond = correspondance) 

# niter =  100
niter =  100
inipts = 5

acq_list  = ['entropy']

parUnknownId = list(range(1,latent_dim+1))
bounds = [(-4,4) for ij in parUnknownId]
parUnknownId = [str(ij) for ij in parUnknownId]


dict(zip(parUnknownId, bounds))

gp_surr=BayesianOptimization(cardiac_model.compute_objfunc, dict(zip(parUnknownId, bounds)),vae,verbose=0)

gp,Z=gp_surr.gpfit(init_points=inipts,n_iter=100,acq='entropy',xi=0.)
np.savetxt("z_bo_entropy.csv", Z, delimiter=",")


#sampling from exp(gp)
mcmc_chain = 2
z1_new = []
z2_new = []
chain = []
samples = []
for c in range(mcmc_chain):
    zfix = np.random.normal(loc = 0, scale = 1,size=2)
    x_0=np.reshape(zfix,(1,2))
    param = gp.predict(x_0, return_std= True)
    m=param[0].ravel() # mean of gp
    v=param[1].ravel() # sigma of gp
    log_mean_0 = np.exp(m+v**2/2)
    for j in range(10000):      
        z_star = zfix + np.random.normal(loc = 0, scale = 1,size=2)
        z_star2 = np.reshape(z_star,(1,2))
        param2 = gp.predict(z_star2, return_std= True)
        m2=param2[0].ravel() # mean of gp
        v2=param2[1].ravel() # sigma of gp
        log_mean_2 = np.exp(m2+v2**2/2)
        if (-4<z_star[0]<4) and (-4<z_star[1]<4):
            rho = min(1, log_mean_2/log_mean_0)
            r=np.random.rand()
            if r < rho:
                f= open("exp_gp_entropy_sample.txt","a")
                f.write(str(c) + "," + str(z_star[0]) + "," + str(z_star[1]) + "," + str(log_mean_2[0]) +"\n")
                f.close()
                zfix = z_star
                log_mean_0 = log_mean_2
                z1_new = np.append(z1_new,z_star[0])
                z2_new = np.append(z2_new,z_star[1])
                chain = np.append(chain,c)
samples = np.vstack([chain,z1_new,z2_new])
ar=len(z1_new)/(mcmc_chain*10000)
print(ar)