from __future__ import print_function
from __future__ import division
import numpy as np
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import minimize
#import nlopt
from scipy.stats import multivariate_normal
import math


def acq_max(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling 1e5 points at random, and then
    running L-BFGS-B from 250 random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.


    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """
    
    opt_type ='l_bfgs_b'
    num_ini  = 10
    num_ini_exp=1000

    # Warm up with random points  
    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                 size=(num_ini_exp, bounds.shape[0]))        
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()
        
    # Explore the parameter space more throughly
    x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(num_ini, bounds.shape[0]))

    if opt_type=='l_bfgs_b':
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method="L-BFGS-B")    
            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]
         
       # Clip output to make sure it lies within the bounds. Due to floating
       # point technicalities this is not always the case.
        return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    else:
        
#        opt = nlopt.opt(nlopt.LN_BOBYQA,bounds.shape[0])
#        opt.set_min_objective(lambda x,grad: float(-ac(x.reshape(1, -1), gp=gp, y_max=y_max)))
#        opt.set_lower_bounds(bounds[:, 0])
#        opt.set_upper_bounds(bounds[:, 1])
#        opt.set_ftol_rel(1)
#        opt.set_xtol_rel(10^-2)
#        opt.set_maxeval(5000)
#             
#        for x_try in x_seeds:
#            # Find the minimum of minus the acquisition function
#            try:
#                res_x = opt.optimize(x_try)
#                res_f = opt.last_optimum_value()
#                if max_acq is None or -res_f >= max_acq:
#                    x_max = res_x
#                    max_acq = -res_f               
#                
#            except nlopt.RoundoffLimited:
#                continue
#                print("Oops! RoundoffLimited exception occured. Next Entry!")
         # Store it if better than previous minimum(maximum).   

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        return np.clip(x_max, bounds[:, 0], bounds[:, 1])

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, z_a, z_m, z_v, xi, surrogate):
        """
        If UCB is to be used, a constant kappa is needed.
        """

        self.surrogate=surrogate
        self.kappa = kappa
        self.xi = xi
        self.z_m = z_m
        self.z_v = z_v
        self.z_a=z_a

        if kind not in ['ei_prior','ei_post_agg','ei_post_k','var','ucb','ei','poi','ev','entropy','ev2','mode','avg','median','cv']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_max):
        
        if self.surrogate=='deepgp':
             if self.kind == 'ucb':
                return self._ucb_deepgp(x, gp, self.kappa)
        else:            
            if self.kind == 'ucb':
                return self._ucb(x, gp, self.kappa)
            if self.kind == 'ei':
#                 print(y_max)
                return self._ei(x, gp, y_max, self.xi)
            if self.kind == 'ei_prior':
                return self._ei_prior(x, gp, y_max, self.xi)
            if self.kind == 'ei_post_k':
                return self._ei_post_k(x, gp, y_max, self.z_a, self.z_m, self.z_v, self.xi)
            if self.kind == 'ei_post_agg':
                return self._ei_post_agg(x, gp, y_max,self.z_m, self.z_v, self.xi)
            if self.kind == 'poi':
                return self._poi(x, gp, y_max, self.xi)   
            if self.kind == 'ev':
                return self._ev(x, gp, y_max, self.xi)
            if self.kind == 'ev2':
                return self._ev2(x, gp, y_max, self.xi)
            if self.kind == 'entropy':
                return self._entropy(x, gp, y_max, self.xi)
            if self.kind == 'mode':
                return self._mode(x, gp, y_max, self.xi)
            if self.kind == 'avg':
                return self._avg(x, gp, y_max, self.xi)
            if self.kind == 'median':
                return self._median(x, gp, y_max, self.xi)
            if self.kind == 'cv':
                return self._cv(x, gp, y_max, self.xi)

    
    @staticmethod
    def _ucb(x, gp, kappa): ###### mean2 and std2 are for exp(GP)
#        kappa= np.sqrt(0.2*log(D^(2+length(xs)/2)*pi^2/(3*0.1)));
        mean, std = gp.predict(x, return_std=True)
        mean2 = np.exp(mean + std**2/2)
        std2 = np.exp(std**2 - 1) * np.exp(2*mean + std**2)
        # return mean2+ kappa * np.sqrt(std2) #exp(gp)
        return mean+ kappa * std #gp

    @staticmethod
    def _ei(x, gp, y_max, xi):        
        mean, std = gp.predict(x, return_std=True)
        mean2 = np.exp(mean + std**2/2)
        std2 = np.exp(std**2 - 1) * np.exp(2*mean + std**2)
        z = (mean - y_max - xi)/std
#         z = (mean2- y_max - xi)/std2
#         print(xi)
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)     
#         return (mean2- y_max - xi) * norm.cdf(z) + std * norm.pdf(z)
    
    @staticmethod
    def _ei_post_agg(x, gp, y_max, z_m, z_v, xi):       
#        z_ivar  = np.array([1/0.32906416,1/0.33221027])
#        z_mean = np.array([ 0.02829593,  0.12978409])        
#        epi = z_ivar[0]*(x-z_mean)[0]**2+z_ivar[1]*(x-z_mean)[1]**2
#        tau = y_max*(1 - 0.1*epi)  
#         print("z test")
        if x.shape[0]>1:
            epi = np.zeros((x.shape[0]))
            for i in range(x.shape[0]):
                epi_temp = (x[i,:]-z_m)@z_v@(x[i,:]-z_m).reshape(-1,1)
                epi[i] = epi_temp[0]
        else:
            epi = (x-z_m) @ z_v @ (x-z_m).reshape(-1,1)
            epi = epi[0]

        tau = y_max*(1 - xi*epi) 
        
        mean, std = gp.predict(x, return_std=True)
        mean2 = np.exp(mean + std**2/2)
        std2 = np.exp(std**2 - 1) * np.exp(2*mean + std**2)
        z = (mean2- tau)/std2
#         print(z.shape)
        return (mean2- tau) * norm.cdf(z) + std * norm.pdf(z)    

    @staticmethod
    def _ei_post_k(x, gp, y_max, z_a,z_m, z_v, xi):        
        if x.shape[0]>1:
            epi = np.zeros((x.shape[0]))
            for i in range(x.shape[0]):
                epi_t = 0
                for j in range(z_m.shape[1]):
                    epi_t = epi_t+ z_a[:,j]*(x[i,:]-z_m[:,j])@z_v[:,:,j]\
                            @(x[i,:]-z_m[:,j]).reshape(-1,1)
                epi[i] = epi_t[0]
        else:
            epi = 0
            for j in range(z_m.shape[1]):
                epi = epi+z_a[:,j]*(x-z_m[:,j])@z_v[:,:,j]@(x-z_m[:,j]).reshape(-1,1)
            epi = epi[0]
            
        tau = y_max*(1 - xi*epi)         
        mean, std = gp.predict(x, return_std=True)
        mean2 = np.exp(mean + std**2/2)
        std2 = np.exp(std**2 - 1) * np.exp(2*mean + std**2)
        z = (mean2- tau)/std2
        return (mean2- tau) * norm.cdf(z) + std * norm.pdf(z)    


    
    @staticmethod
    def _ei_prior(x, gp, y_max, xi):
        
#        xii = 1*1.58*(1- multivariate_normal.pdf(x,[0,0],1)
#                        /multivariate_normal.pdf([0,0],[0,0],1))
        
        tau = y_max*(1 - xi*np.sum(x**2))       
        mean, std = gp.predict(x, return_std=True)
        mean2 = np.exp(mean + std**2/2)
        std2 = np.exp(std**2 - 1) * np.exp(2*mean + std**2)
        z = (mean2- tau)/std2
        return (mean2- tau) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        mean2 = np.exp(mean + std**2/2)
        std2 = np.exp(std**2 - 1) * np.exp(2*mean + std**2)
        z = (mean2- y_max - xi)/std2
        return norm.cdf(z)
       
    @staticmethod    
    def _ev(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        return np.exp(2*mean+std**2)*(np.exp(std**2)-1)
    
    @staticmethod
    def _ev2(x,gp,y_max,xi):
        mean, std = gp.predict(x, return_std=True)
        bape = np.exp(2*mean+std**2)*(np.exp(std**2)-1)
        return bape

    @staticmethod
    def _entropy(x,gp,y_max,xi):
        mean, std = gp.predict(x, return_std=True)
        return mean + 0.5 + np.log(std*math.sqrt(2*math.pi)) # exp(gp)
        # return 0.5*np.log(2*math.pi*np.exp(std**2)) #based on gp


    @staticmethod
    def _mode(x,gp,y_max,xi):
        mean, std = gp.predict(x, return_std=True)
        # mean2 = np.exp(mean + std**2/2)
        # std2 = np.exp(std**2 - 1) * np.exp(2*mean + std**2)
        return np.exp(mean - std**2)
        # return mean
    

    @staticmethod
    def _avg(x,gp,y_max,xi):
        mean, std = gp.predict(x, return_std=True)
        # mean2 = np.exp(mean + std**2/2)
        # std2 = np.exp(std**2 - 1) * np.exp(2*mean + std**2)
        return np.exp(mean + std**2/2)

    @staticmethod
    def _cv(x,gp,y_max,xi):
        mean, std = gp.predict(x, return_std=True)
        mean2 = np.exp(mean + std**2/2)
        std2 = np.exp(std**2 - 1) * np.exp(2*mean + std**2)
        return std2/mean2

    @staticmethod
    def _median(x,gp,y_max,xi):
        mean, std = gp.predict(x, return_std=True)
        return np.exp(mean)

    @staticmethod
    def _ucb_deepgp(x, gp, kappa):
        mean, var = gp.predict(x)
        std=np.sqrt(var)
        return mean + kappa * std
    



def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog(object):

    def __init__(self, params):
          
       
        self.ymax = None
        self.xmax = None
        self.params = params
        self.ite = 1

        self.start_time = datetime.now()
        self.last_round = datetime.now()

        # sizes of parameters name and all
        self.sizes = [max(len(ps), 7) for ps in params]
     

        # Sorted indexes to access parameters
        self.sorti = sorted(range(len(self.params)),
                            key=self.params.__getitem__)

    def reset_timer(self):
        self.start_time = datetime.now()
        self.last_round = datetime.now()

    def print_header(self, initialization=True):

        if initialization:
            print("{}Initialization{}".format(BColours.RED,
                                              BColours.ENDC))
        else:
            print("{}Bayesian Optimization{}".format(BColours.RED,
                                                     BColours.ENDC))

        print(BColours.BLUE + "-" * (29 + sum([s + 5 for s in self.sizes])) +
            BColours.ENDC)

        print("{0:>{1}}".format("Step", 5), end=" | ")
        print("{0:>{1}}".format("Time", 6), end=" | ")
        print("{0:>{1}}".format("Value", 10), end=" | ")

        for index in self.sorti:
            print("{0:>{1}}".format(self.params[index],
                                    self.sizes[index] + 2),
                  end=" | ")
        print('')

    def print_step(self, x, y, warning=False):

        print("{:>5d}".format(self.ite), end=" | ")

        m, s = divmod((datetime.now() - self.last_round).total_seconds(), 60)
        print("{:>02d}m{:>02d}s".format(int(m), int(s)), end=" | ")

        if self.ymax is None or self.ymax < y:
            self.ymax = y
            self.xmax = x
            print("{0}{2: >10.5f}{1}".format(BColours.MAGENTA,
                                             BColours.ENDC,
                                             y),
                  end=" | ")

            for index in self.sorti:
                print("{0}{2: >{3}.{4}f}{1}".format(
                            BColours.GREEN, BColours.ENDC,
                            x[index],
                            self.sizes[index] + 2,
                            min(self.sizes[index] - 3, 6 - 2)
                        ),
                      end=" | ")
        else:
            print("{: >10.5f}".format(y), end=" | ")
            for index in self.sorti:
                print("{0: >{1}.{2}f}".format(x[index],
                                              self.sizes[index] + 2,
                                              min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")

        if warning:
            print("{}Warning: Test point chose at "
                  "random due to repeated sample.{}".format(BColours.RED,
                                                            BColours.ENDC))

        print()

        self.last_round = datetime.now()
        self.ite += 1

    def print_summary(self):
        pass
