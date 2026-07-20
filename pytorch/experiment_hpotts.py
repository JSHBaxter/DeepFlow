import numpy as np
import time
import torch
from itertools import product
from math import sqrt, log, exp
import datetime
from random import shuffle
from multiprocessing import Pool
import time
import scipy
import pickle
import os

from hmf_deepflow import HMF_LBP2d_PytorchNative, HMF_Mean2d, HMF_MAP2d

def hmf_get_gibbs_sampled_marginals(d,rx,ry,p,c_opt,Q=10000,burn=None,rate=None,verbose=False):
    N = d.shape[1]
    c = d.shape[0]
    cr = rx.shape[0]
    index_order = list(product(range(N),range(N)))
    
    if burn is None:
        burn = N**2
    if rate is None:
        rate = N
    
    def sum_up_tree(conf,p):
        u_tree = np.zeros([cr,conf.shape[-2],conf.shape[-1]])
        u_tree[(-c):,:,:] = conf
        for child,parent in reversed(list(enumerate(p))):
            if parent != -1:
                u_tree[parent,:,:] += u_tree[child,:,:]
        return u_tree
    
    def make_new_configuration(conf):
        shuffle(index_order)
        u_tree = sum_up_tree(conf,p)
        for i1,i2 in index_order:
            energy = np.zeros((cr))
            for reg in range(cr):
                if i1 > 0:
                    energy[reg] += rx[reg,i1-1,i2]*(2*u_tree[reg,i1-1,i2]-1)
                if i2 > 0:
                    energy[reg] += ry[reg,i1,i2-1]*(2*u_tree[reg,i1,i2-1]-1)
                if i1 < N-1:
                    energy[reg] += rx[reg,i1,i2]*(2*u_tree[reg,i1+1,i2]-1)
                if i2 < N-1:
                    energy[reg] += ry[reg,i1,i2]*(2*u_tree[reg,i1,i2+1]-1)
            for child,parent in enumerate(p):
                if parent != -1:
                    energy[child] += energy[parent]
            energy[-c:] += d[:,i1,i2]
            cprob = np.cumsum(np.exp(energy[-c:])/np.sum(np.exp(energy[-c:])))
            sample = np.random.uniform()
            for channel in range(c):
                conf[channel,i1,i2] = 0
                if sample < cprob[channel] and (channel == 0 or sample >= cprob[channel-1]):
                    conf[channel,i1,i2] = 1
            u_tree[:,i1:(i1+1),i2:(i2+1)] = sum_up_tree(conf[:,i1:(i1+1),i2:(i2+1)],p)
        return conf
                    
    
    conf = c_opt.copy()
    marginal_calc = np.zeros((c,N,N),dtype=int)
    marginal_calc_x = np.zeros((cr,N-1,N),dtype=int)
    marginal_calc_y = np.zeros((cr,N,N-1),dtype=int)
    
    #discard the first number of iterations 
    for i in range(burn):
        conf = make_new_configuration(conf)
    if verbose:
        print("Finished initial burn period")
    
    #now do Gibbs sampling, keeping 1 per rate samples
    for i in range(Q*rate-rate+1):
        
        #make new configuration
        conf = make_new_configuration(conf)
        u_tree = sum_up_tree(conf,p)
            
        #accumulate to the marginal
        if i % rate == 0:
            marginal_calc += conf.astype(np.int32)
            marginal_calc_x += (u_tree[:,:-1,:] != u_tree[:,1:,:]).astype(np.int32)
            marginal_calc_y += (u_tree[:,:,:-1] != u_tree[:,:,1:]).astype(np.int32)
            
        if verbose and i % 100 == 0:
            print("Processing configuration " + str(i) + " of " + str(Q*rate) + " (" + str(int(100*i/(Q*rate)+0.5)) + "%)",end="\r")
        
    #get marginal estimate and certainty
    marginal_est = (marginal_calc+1).astype(float) / float(Q+c)
    marginal_std = (marginal_calc+1).astype(float) * (Q-marginal_calc+1).astype(float) / float((Q+2)**2 * (Q+3))
    marginal_est_x = (marginal_calc_x+1).astype(float) / float(Q+2)
    marginal_est_y = (marginal_calc_y+1).astype(float) / float(Q+2)
    
    return marginal_est, marginal_std, marginal_calc, (marginal_est_x, marginal_est_y), (marginal_calc_x, marginal_calc_y)

def run_experiment(filename, iteration, margin_for_error, device,N,Q=None,self_error=None,verbose=False):
    b = 1
    c = 5
    c2 = 8
    
    p = np.array([-1, -1, 1, 0, 0, 1, 2, 2]).astype(np.int32)
    p_torch = torch.tensor(p, device=torch.device(device))
    
    if Q is None:
        Q = (c-1)*int(N**2 / self_error - 2 + 0.5)
        self_error = N**2 / (Q+2)
    elif self_error is None:
        self_error = N**2 / (Q+2)
    else:
        raise Exception("You must specify either a number of Monte Carlo samples or a designed error resolution.")
    
    #make sure I haven't alread done this experiment number
    full_filename = filename + "_" + str(iteration) + ".pkl"
    if os.path.isfile(full_filename):
        return
    
    #get maximum limit on regularisation terms for guaranteed convergence
    if N > 2:
        conv_limit = ((-log((2*4-3)/(2*4-1)))/5) / (1+margin_for_error)
    else:
        conv_limit = 20
        
    #create a model parameter configuration
    data_t = np.random.normal(0,1,size=(b,c,N,N)).astype(np.float32)
    data_t = data_t - np.min(data_t,axis=1)
    data_rx = np.random.uniform(size=(b,c2,N,N)).astype(np.float32) * conv_limit
    data_rx[:,:,:,-1] = 0
    data_ry = np.random.uniform(size=(b,c2,N,N)).astype(np.float32) * conv_limit
    data_ry[:,:,:,-1] = 0
    
    t = torch.tensor(data_t, device=torch.device(device))
    t2 = torch.tensor(data_t.copy(), device=torch.device(device))
    t.requires_grad = True
    t2.requires_grad = True
    rx = torch.tensor(data_rx, device=torch.device(device))
    rx2 = torch.tensor(data_rx.copy(), device=torch.device(device))
    rx.requires_grad = True
    rx2.requires_grad = True
    ry = torch.tensor(data_ry, device=torch.device(device))
    ry2 = torch.tensor(data_ry.copy(), device=torch.device(device))
    ry.requires_grad = True
    ry2.requires_grad = True
    
    om_np = HMF_Mean2d.apply(t,rx,ry, p_torch).detach().cpu().numpy()
    om_np = np.exp(om_np) / np.sum(np.exp(om_np),axis=1)
    ol_np = HMF_LBP2d_PytorchNative.apply(t2,rx2,ry2, p_torch).detach().cpu().numpy()
    ol_np = np.exp(ol_np) / np.sum(np.exp(ol_np),axis=1)
    data_only = np.exp(data_t) / np.sum(np.exp(data_t),axis=1)
    if verbose:
        print("parameters:")
        print(data_t)
        print(data_rx)
        print(data_ry)
        print("\ndata term only:")
        print(data_only)
        print("\nmean-field iteration:")
        print(om_np)
        print("\nparallel loopy belief propagation:")
        print(ol_np,flush=True)
    
    #get the MAP configuration estimate for Gibbs sampling
    map_solution = (data_only == np.max(data_only,axis=1)).astype(np.float32)
    if verbose:
        print("\nMAP configuration:")
        print(map_solution)
        print("\n")
    #map_solution = (np.exp(map_solution)+0.5).astype(int)[0,:,:,:]
    
    gibbs_sampling_res = hmf_get_gibbs_sampled_marginals(data_t[0,:,:,:],data_rx[0,:,:,:],data_ry[0,:,:,:],p,map_solution[0,:,:,:],Q,burn=2*N,rate=N,verbose=verbose)
    
    if verbose:
        print("Gibbs marginal estimate:")
        print(gibbs_sampling_res[0])
        print("\n  ... and their standard deviations ...")
        print(np.sqrt(gibbs_sampling_res[1]))
    
    
    retvals = {"data mag":np.sum(np.abs(data_t)),
               "reg_mag": np.sum(data_rx[:,:,:-1,:])+np.sum(data_ry[:,:,:,:-1]),
               "Q":Q,
              }
    
    if verbose:
        print("")
        print("\ndata term only:")
        print(data_only)
        print("\nmean-field iteration:")
        print(om_np)
        print("\nparallel loopy belief propagation:")
        print(ol_np,flush=True)

    #add data and regularisation terms (and processed marginals) to the files to save
    retvals["data_t"] = data_t
    retvals["data_rx"] = data_rx
    retvals["data_ry"] = data_ry
    retvals["data_only"] = data_only
    retvals["om_np"] = om_np
    retvals["ol_np"] = ol_np
    retvals["gibbs_np"] = gibbs_sampling_res[0]
    retvals["gibbs_calc"] = gibbs_sampling_res[2]
        
    #write out work to file for later analysis
    with open(full_filename, 'wb') as handle:
        pickle.dump(retvals, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return
    
    
def run_func(x,size,margin):
    print("Running iteration " + str(x) + " with size " + str(size) + " and margin " + str(margin))
    filename = "../experiments/h_exp_"+str(size)+"x"+str(size)+"_0" + str(margin)
    margin = -margin/100
    run_experiment(filename,x,margin,"cpu",size,self_error=0.01,verbose=False)
    
def run_all_funcs(x, size):
    for margin in [-50, -25, 0, 25, 50, 75]:
        run_func(x,size,margin)
    return None

def run_func_5(x):
    run_all_funcs(x, 5)
def run_func_10(x):
    run_all_funcs(x, 10)
def run_func_20(x):
    run_all_funcs(x, 20)
    
with Pool(10) as p:
    res_list = p.map(run_func_5, range(100))
with Pool(10) as p:
    res_list = p.map(run_func_10, range(100))
with Pool(10) as p:
    res_list = p.map(run_func_20, range(100))