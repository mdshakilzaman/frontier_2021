import sys
sys.path.append('/home/stu10/s15/mz1482/project/frontier/')
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

sys.path.append('/home/stu10/s15/mz1482/project/frontier/bayesopt/')
sys.path.append('/home/stu10/s15/mz1482/project/frontier/ep_models/')
from bayesopt import BayesianOptimization
# from bayes_opt import testbo
from ep_models import test_model
from skimage.filters import threshold_otsu
import math
from numpy.linalg import multi_dot

#case1: 1239
#case2: 1373
#case3:1230
main_path = '/home/stu10/s15/mz1482/project/frontier/'
# filename = 'shakil_ir_1230' 
filename = 'training_ir_ic_1230' 
resultspath = '/home/stu10/s15/mz1482/project/frontier/models/'
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

print(vae_pdim)
print(correspondance)

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
# files = ['case3_inf_e4'] #RPA case
# files = ['case3_inf_e4','case3_large','case3_sep_e2','case3_ant_e2','case3_lat_e2','case3_ant_1','case3_antsp_1','case3_inf_1','case3_inf_lat_2','case3_ap_1','case3_inf_2','case3_inf_lat_1','case3_infsp_1']
# files = ['mysimu']
files = ['case3_inf_e4']
# files = ['case3_coordin','case3_bas_ant','case3_mid_ant','case3_ap_ant'] 
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

def MH_sampling2(cardiac_model,iteration,chain,m,v,vae=0):
    samples = np.empty((0, 2))
    for c in range(chain):
        j=0
        z = np.random.normal(loc = 0, scale = 1,size=2)
        a=cardiac_model.post_MH(z,vae)
        for i in range(iteration):
            start = time.time()
            z_star = z + np.random.normal(loc = m, scale = v,size=2)
            r=np.random.rand()
            b=cardiac_model.post_MH(z_star,vae)
            end = time.time()
            print('time taken',end-start,'seconds')
            rho = min(1, b/a)
            if r < rho:
                z = z_star
                a = b
                samples = np.append(samples, [z], axis=0)
                f=open("/home/stu10/s15/mz1482/project/frontier/results/synthetic/case3/case3_inf_e4/true_mcmc.txt","a")
                f.write(str(c)+","+str(z[0])+ "," +str(z[1]) + ","+ str(b) +"\n")
                f.close()
                print(b)
                j=j+1
            print(i)
    ar = j/iteration
    print("true mcmc of case3_inf_e4 is done")
    print('acceptance ratio',ar)
    return samples
print("sampling starts")
gen = MH_sampling2(cardiac_model,10000,2,0,0.7,vae)