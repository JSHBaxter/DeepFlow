import unittest

import numpy as np
import math
import random

import datetime
import itertools

import torch
from potts_deepflow import PottsDeepFlow1d, PottsDeepFlow2d, PottsDeepFlow3d
from potts_deepflow_function import Potts_MAP1d,Potts_MAP2d,Potts_MAP3d
from binary_deepflow import BinaryDeepFlow1d, BinaryDeepFlow2d, BinaryDeepFlow3d
from binary_deepflow_function import Binary_MAP1d,Binary_MAP2d,Binary_MAP3d

b=1
c=int(4*random.uniform(0,1))+3
x=2**6
epsilon = 10**-5
verbosity = True

def get_size_info(d):
    x_used = int(x**(1/d)+0.5)
    return tuple( [b,c]+[x_used for i in range(d)] ), [tuple([b,c]+[x_used - (1 if i==i2 else 0) for i in range(d)]) for i2 in range(d)], tuple( [1,c]+[x_used for i in range(d)] ), tuple([i+2 for i in range(d)])

def run_all_solvers(t,t2,r,r2):
    d = len(r)
    if d == 1:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = Potts_MAP1d.apply(t,r[0])
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omj = PottsDeepFlow1d(solve_type="MFE",test_type=None)(t,r[0])
        omj = [omj]
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, olj = PottsDeepFlow1d(solve_type="LBP",test_type=None)(t2,r2[0])
        olj = [olj]
        print(4,datetime.datetime.now())
    elif d == 2:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = Potts_MAP2d.apply(t,r[0],r[1])
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omjx, omjy = PottsDeepFlow2d(solve_type="MFE",test_type=None)(t,r[0],r[1])
        omj = [omjx, omjy]
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, oljx, oljy = PottsDeepFlow2d(solve_type="LBP",test_type=None)(t2,r2[0],r2[1])
        olj = [oljx, oljy]
        print(4,datetime.datetime.now())
    elif d == 3:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = Potts_MAP3d.apply(t,r[0],r[1],r[2])
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omjx, omjy, omjz = PottsDeepFlow3d(solve_type="MFE",test_type=None)(t,r[0],r[1],r[2])
        omj = [omjx, omjy, omjz]
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, oljx, oljy, oljz = PottsDeepFlow3d(solve_type="LBP",test_type=None)(t2,r2[0],r2[1],r2[2])
        olj = [oljx, oljy, oljz]
        print(4,datetime.datetime.now())
    return oa, om, omj, ol, olj

def nan_check(m_np, name, verbose=False):
    for m,n in zip(m_np,name):
        if verbose :
            print("\tNAN check on ", n)
        if np.any(np.isnan(m)):
            raise Exception("NaN found in buffer " + n)

def buffer_equality_check(estimate1, name1, estimate2=None, name2=None, epsilon_d=epsilon, asserter=None):
    exception_list = []
    if estimate2 is not None:
        diff = abs(estimate1-estimate2)
    else:
        diff = abs(estimate1)
    loc = np.argmax(diff.flatten())
    max_diff_val = diff.flatten()[loc]
    if max_diff_val > epsilon_d:
        if estimate2 is None:
            exception_list.append(name1 +" buffer ("+str(estimate1.flatten()[loc])+") differs from zero at location " + str(loc))
        elif type(estimate2) is float:
            exception_list.append(name1 +" buffer ("+str(estimate1.flatten()[loc])+") differs from " + str(estimate2) + " at location " + str(loc))
        else:
            exception_list.append(name1 +" buffer ("+str(estimate1.flatten()[loc])+") differs from " + name2 + " buffer ("+str(estimate2.flatten()[loc])+") at location " + str(loc))
    if verbosity:
        print("\tDone for",name1,name2 if name2 else "", "(biggest dif: "+ str(max_diff_val)+")")
    if exception_list:
        for e in exception_list:
            print(e)
        if asserter is not None:
            asserter.assertTrue(False)
            
def test_no_smoothness(d,full_device,asserter):
    print("Testing (no smoothness) \tDim: " +str(d)+ " \tDev: " + full_device + "\t Start Time:\t" + str(datetime.datetime.now()))
    device = "cuda" if "cuda" in full_device else "cpu"

    size_info, size_info_j, size_red_info, axes = get_size_info(d)

    #create data terms for testing (smoothness terms should be zero)
    data_t = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_rx = np.zeros(shape=size_info).astype(np.float32)
    data_wx = np.random.normal(0,1,size=size_info_j[0]).astype(np.float32)
    if d > 1:
        data_ry = np.zeros(shape=size_info).astype(np.float32)
        data_wy = np.random.normal(0,1,size=size_info_j[1]).astype(np.float32)
    if d > 2:
        data_rz = np.zeros(shape=size_info).astype(np.float32)
        data_wz = np.random.normal(0,1,size=size_info_j[2]).astype(np.float32)

    #cast to tensors and make duplicates for the other solver mechanism
    t = torch.tensor(data_t, device=torch.device(device))
    t2 = torch.tensor(data_t.copy(), device=torch.device(device))
    t.requires_grad = True
    t2.requires_grad = True
    w = torch.tensor(data_w, device=torch.device(device))
    wx = torch.tensor(data_wx, device=torch.device(device))
    rx = torch.tensor(data_rx, device=torch.device(device))
    rx2 = torch.tensor(data_rx.copy(), device=torch.device(device))
    rx.requires_grad = True
    rx2.requires_grad = True
    r = [rx]
    r2 = [rx2]
    wl = [wx]
    if d > 1:
        wy = torch.tensor(data_wy, device=torch.device(device))
        ry = torch.tensor(data_ry, device=torch.device(device))
        ry.requires_grad = True
        ry2 = torch.tensor(data_ry, device=torch.device(device))
        ry2.requires_grad = True
        wl.append(wy)
        r.append(ry)
        r2.append(ry2)
    if d > 2:
        wz = torch.tensor(data_wz, device=torch.device(device))
        rz = torch.tensor(data_rz, device=torch.device(device))
        rz.requires_grad = True
        rz2 = torch.tensor(data_rz, device=torch.device(device))
        rz2.requires_grad = True
        wl.append(wz)
        r.append(rz)
        r2.append(rz2)

    #run the solvers
    oa, om, omj, ol, olj = run_all_solvers(t,t2,r,r2)
    
    #run the gradient solvers on random losses
    loss = torch.sum(w*om) + sum([torch.sum(wj*j) for wj,j in zip(wl,omj)])
    print(5,datetime.datetime.now())
    loss.backward()
    print(6,datetime.datetime.now())
    loss = torch.sum(w*ol) + sum([torch.sum(wj*j) for wj,j in zip(wl,olj)])
    print(7,datetime.datetime.now())
    loss.backward()
    print(8,datetime.datetime.now())
    
    #convert to numpy for validation
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    ol_np = ol.detach().cpu().numpy()
    omj_np = [omjn.detach().cpu().numpy() for omjn in omj]
    olj_np = [oljn.detach().cpu().numpy() for oljn in olj]
    ot_np = t.grad.detach().cpu().numpy()
    ot2_np = t2.grad.detach().cpu().numpy()
    ogmj_np = [rx.grad.detach().cpu().numpy()]
    oglj_np = [rx2.grad.detach().cpu().numpy()]
    if d > 1:
        ogmj_np.append(ry.grad.detach().cpu().numpy())
        oglj_np.append(ry2.grad.detach().cpu().numpy())
    if d > 2:
        ogmj_np.append(rz.grad.detach().cpu().numpy())
        oglj_np.append(rz2.grad.detach().cpu().numpy())
    
    #do initial NaN checks
    print(9,"Running NaN checks")
    nan_check([oa_np,om_np,ol_np,ot_np,ot2_np]+omj_np+olj_np+ogmj_np+oglj_np,
              ["oa","om","ol","ot","ot2"]+["omj"+str(i) for i in range(d)]+["olj"+str(i) for i in range(d)]+["ogmj"+str(i) for i in range(d)]+["oglj"+str(i) for i in range(d)], verbosity)
        
    #ensure mean estimate is equivalent to the softmax of the data terms
    print(10,"Checking correctness of marginals")
    true_val_marg = np.exp(data_t) / np.sum(np.exp(data_t),1,keepdims=True)
    for estimate, name in zip([om_np, ol_np], ["MFE","LBP"]):
        buffer_equality_check(estimate, name, true_val_marg, "TRUE VALUE", asserter=asserter)
        
    #ensure mean estimate of joint marginals is correct
    print(11,"Checking correctness of joint marginals")
    for i in range(d):
        if i == 0:
            true_val = true_val_marg[:,:,:-1]*(1-true_val_marg[:,:,1:,...])+(1-true_val_marg[:,:,:-1,...])*true_val_marg[:,:,1:,...]
        elif i == 1:
            true_val = true_val_marg[:,:,:,:-1]*(1-true_val_marg[:,:,:,1:,...])+(1-true_val_marg[:,:,:,:-1,...])*true_val_marg[:,:,:,1:,...]
        elif i == 2:
            true_val = true_val_marg[:,:,:,:,:-1]*(1-true_val_marg[:,:,:,:,1:,...])+(1-true_val_marg[:,:,:,:,:-1,...])*true_val_marg[:,:,:,:,1:,...]
        for estimate, name in zip([omj_np[i], olj_np[i]], ["MFE","LBP"]):
            buffer_equality_check(estimate, name, true_val, "TRUE VALUE", asserter=asserter)
    
    #ensure gradient wrt data terms are equivalent to just softmax (need to update for Potts)
    print(12,"Checking correctness of gradient calculation")
    grad_marg = data_w
    for i in range(d):
        if i == 0:
            grad_marg[:,:,:-1,...] += data_wx * (1 - 2 * true_val_marg[:,:,1:,...])
            grad_marg[:,:,1:,...] += data_wx * (1 - 2 * true_val_marg[:,:,:-1,...])
        elif i == 1:
            grad_marg[:,:,:,:-1,...] += data_wy * (1 - 2 * true_val_marg[:,:,:,1:,...])
            grad_marg[:,:,:,1:,...] += data_wy * (1 - 2 * true_val_marg[:,:,:,:-1,...])
        elif i == 2:
            grad_marg[:,:,:,:,:-1,...] += data_wz * (1 - 2 * true_val_marg[:,:,:,:,1:,...])
            grad_marg[:,:,:,:,1:,...] += data_wz * (1 - 2 * true_val_marg[:,:,:,:,:-1,...])
    true_val = true_val_marg * (grad_marg - np.sum(grad_marg * true_val_marg, 1, keepdims=True))
    for estimate, name in zip([ot2_np, ot_np], ["LBP","MFE"]):
        buffer_equality_check(estimate, name, true_val, "TRUE VALUE", asserter=asserter)
    
    #ensure MAP assigns 1 to highest terms
    print(13,"Check MAP correctness")
    max_channel = np.ascontiguousarray(np.repeat(np.max(data_t,1,keepdims=True),data_t.shape[1],axis=1))
    for val_df, val_d, val_m in zip(oa_np.flatten(),data_t.flatten(),max_channel.flatten()):
        if (val_d == val_m) and (val_df < 1-epsilon):
            print("oa",val_df, val_d, val_m)
            asserter.assertTrue(False)
        elif (val_d != val_m) and (val_df > epsilon):
            print("oa",val_df, val_d, val_m)
            asserter.assertTrue(False)

            
            
            
#This test is the equivalent of the only smoothness one for binary, except simplified for a Potts
#If the smoothness terms are all the same and the data terms are zero then:
# - the pointwise marginals are equal across labels
# - the gradient w.r.t. to the smoothness terms sums to zero (they may not be equal to zero but if
#   the smoothness terms stay equal to each other, the marginals shouldn't change) provided that the
#   only gradient is coming in from the pointwise marginals
def test_no_data(d,full_device,asserter):
    print("Testing (no data term) \tDim: " +str(d)+ " \tDev: " + full_device + "\t Start Time:\t" + str(datetime.datetime.now()))
    device = "cuda" if "cuda" in full_device else "cpu"
    
    size_info, size_info_j, size_red_info, axes = get_size_info(d)
    size_nochannel_info = tuple([size_info[0],1]+list(size_info[2:]))

    #create data terms for testing (smoothness terms should be zero)
    rlimit = 0.5 / d
    data_t = np.zeros(shape=size_info).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_rx = rlimit*np.random.uniform(0,1,size=size_info).astype(np.float32)
    data_rx = np.repeat(rlimit*np.random.uniform(0,1,size=size_nochannel_info).astype(np.float32),size_info[1],axis=1)
    if d > 1:
        data_ry = np.repeat(rlimit*np.random.uniform(0,1,size=size_nochannel_info).astype(np.float32),size_info[1],axis=1)
    if d > 2:
        data_rz = np.repeat(rlimit*np.random.uniform(0,1,size=size_nochannel_info).astype(np.float32),size_info[1],axis=1)

    #cast to tensors and make duplicates for the other solver mechanism
    t = torch.tensor(data_t, device=torch.device(device))
    t2 = torch.tensor(data_t.copy(), device=torch.device(device))
    t.requires_grad = True
    t2.requires_grad = True
    w = torch.tensor(data_w, device=torch.device(device))
    rx = torch.tensor(data_rx, device=torch.device(device))
    rx2 = torch.tensor(data_rx.copy(), device=torch.device(device))
    rx.requires_grad = True
    rx2.requires_grad = True
    r = [rx]
    r2 = [rx2]
    if d > 1:
        ry = torch.tensor(data_ry, device=torch.device(device))
        ry.requires_grad = True
        ry2 = torch.tensor(data_ry, device=torch.device(device))
        ry2.requires_grad = True
        r.append(ry)
        r2.append(ry2)
    if d > 2:
        rz = torch.tensor(data_rz, device=torch.device(device))
        rz.requires_grad = True
        rz2 = torch.tensor(data_rz, device=torch.device(device))
        rz2.requires_grad = True
        r.append(rz)
        r2.append(rz2)

    #run the solvers
    oa, om, omj, ol, olj = run_all_solvers(t,t2,r,r2)
    
    #run the gradient solvers on random losses
    loss = torch.sum(w*om)
    print(5,datetime.datetime.now())
    loss.backward()
    print(6,datetime.datetime.now())
    loss = torch.sum(w*ol)
    print(7,datetime.datetime.now())
    loss.backward()
    print(8,datetime.datetime.now())
    
    #convert to numpy for validation
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    ol_np = ol.detach().cpu().numpy()
    omj_np = [omjn.detach().cpu().numpy() for omjn in omj]
    olj_np = [oljn.detach().cpu().numpy() for oljn in olj]
    ot_np = t.grad.detach().cpu().numpy()
    ot2_np = t2.grad.detach().cpu().numpy()
    ogmj_np = [ri.grad.detach().cpu().numpy() for ri in r]
    oglj_np = [ri.grad.detach().cpu().numpy() for ri in r2]
    
    #do initial NaN checks
    print(9,"Running NaN checks")
    nan_check([oa_np,om_np,ol_np,ot_np,ot2_np]+omj_np+olj_np+ogmj_np+oglj_np,
              ["oa","om","ol","ot","ot2"]+["omj"+str(i) for i in range(d)]+["olj"+str(i) for i in range(d)]+["ogmj"+str(i) for i in range(d)]+["oglj"+str(i) for i in range(d)], verbosity)
    
    #ensure mean estimate is 50/50
    print(10,"Checking correctness of marginals")
    exception_list = []
    val_d = 1/data_t.shape[1]
    for estimate, name in zip([om_np, ol_np], ["MFE","LBP"]):
        buffer_equality_check(estimate, name, val_d, asserter=asserter)
            
    #ensure gradient wrt smoothness terms sum to zero in the channels dimension
    exception_list = []
    print(11,"Checking correctness of gradient calculation")
    val_d = 0.0
    for estimate, name in zip(ogmj_np+oglj_np, ["MFE"+str(i) for i in range(d)]+["LBP"+str(i) for i in range(d)]):
        grad_sum = np.sum(estimate,axis=1)
        buffer_equality_check(grad_sum,"GRADSUM",asserter=asserter)

def test_smoothness_dom(d,full_device,asserter):
    print("Testing (smoothness dom) \tDim: " +str(d)+ " \tDev: " + full_device + "\t Start Time:\t" + str(datetime.datetime.now()))
    device = "cuda" if "cuda" in full_device else "cpu"
    alg_type = "pytorch" if "pytorch" in full_device else "cpp"

    size_info, size_info_j, size_red_info, axes = get_size_info(d)

    data_t = 0.01/(d*size_info[1])*np.random.normal(0,1,size=size_info).astype(np.float32)
    sums = [[np.sum(data_t[b,c,...]) for c in range(size_info[1])] for b in range(size_info[0])]
    c_preferred = [sums[b].index(max(sums[b])) for b in range(size_info[0])]
    for b in range(size_info[0]):
        data_t[b,c_preferred[b],...] += 10/math.prod(size_info[2:])
    data_rx = 0.25/(d*size_info[1])*(3+np.random.uniform(size=size_info).astype(np.float32))
    if d > 1:
        data_ry = 0.25/(d*size_info[1])*(3+np.random.uniform(size=size_info).astype(np.float32))
    if d > 2:
        data_rz = 0.25/(d*size_info[1])*(3+np.random.uniform(size=size_info).astype(np.float32))

    t = torch.tensor(data_t, device=torch.device(device))
    rx = torch.tensor(data_rx, device=torch.device(device))
    r = [rx]
    if d > 1:
        ry = torch.tensor(data_ry, device=torch.device(device))
        r.append(ry)
    if d > 2:
        rz = torch.tensor(data_rz, device=torch.device(device))
        r.append(rz)

    #run the solvers
    oa, om, omj, ol, olj = run_all_solvers(t,t,r,r)
    
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    ol_np = ol.detach().cpu().numpy()
    omj_np = [omjn.detach().cpu().numpy() for omjn in omj]
    olj_np = [oljn.detach().cpu().numpy() for oljn in olj]
    
    #do initial NaN checks
    print(9,"Running NaN checks")
    nan_check([oa_np,om_np,ol_np]+omj_np+olj_np,
              ["oa","om","ol"]+["omj"+str(i) for i in range(d)]+["olj"+str(i) for i in range(d)], verbosity)
    
    #ensure MAP assigns 1 to the preferred c and 0 otherwise
    print(10,"Check MAP correctness")
    oa_epsilon = 0.01
    totals = np.sum(data_t,axis=axes)
    for b,c in itertools.product(range(size_info[0]),range(size_info[1])):
        if c == c_preferred[b]:
            for val_df in oa_np[b,c,...].flatten():
                if val_df < 1-oa_epsilon:
                    print("oa",val_df,1)
                    asserter.assertTrue(False)
        else:
            for val_df in oa_np[b,c,...].flatten():
                if val_df > oa_epsilon:
                    print("oa",val_df,0)
                    asserter.assertTrue(False)


def test_mimic_binary(d,full_device,asserter):
    print("Testing (mimic binary) \tDim: " +str(d)+ " \tDev: " + full_device + "\t Start Time:\t" + str(datetime.datetime.now()))
    device = "cuda" if "cuda" in full_device else "cpu"
    alg_type = "pytorch" if "pytorch" in full_device else "cpp"
    epsilon_light = 0.02
    
    size_info, size_info_j, size_red_info, axes = get_size_info(d)
    size_info_potts = tuple([size_info[0],2]+list(size_info[2:]))
    size_info_j_potts = [tuple([s[0],2]+list(s[2:])) for s in size_info_j]
    size_info_binary = tuple([size_info[0],1]+list(size_info[2:]))
    size_info_j_binary = [tuple([s[0],1]+list(s[2:])) for s in size_info_j]
    
    numels = math.prod(size_info[2:])

    #create data and smoothness terms for testing (label 0 should have 0 everything)
    data_t = np.random.normal(0,1,size=size_info_potts).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info_potts).astype(np.float32)
    data_rx = 0.25/(d*size_info[1])*(np.random.uniform(size=size_info_potts).astype(np.float32))
    data_wx = np.random.normal(0,1,size=size_info_j_potts[0]).astype(np.float32)
    data_t_bin = np.expand_dims(data_t[:,1,...]-data_t[:,0,...], 1)
    data_w_bin = np.expand_dims(data_w[:,1,...]-data_w[:,0,...], 1)
    data_rx_bin = np.sum(data_rx,1,keepdims=True)
    data_wx_bin = np.sum(data_wx,1,keepdims=True)
    if d > 1:
        data_ry = 0.25/(d*size_info[1])*(np.random.uniform(size=size_info_potts).astype(np.float32))
        data_wy = np.random.normal(0,1,size=size_info_j_potts[1]).astype(np.float32)
        data_ry_bin = np.sum(data_ry,1,keepdims=True)
        data_wy_bin = np.sum(data_wy,1,keepdims=True)
    if d > 2:
        data_rz = 0.25/(d*size_info[1])*(np.random.uniform(size=size_info_potts).astype(np.float32))
        data_wz = np.random.normal(0,1,size=size_info_j_potts[2]).astype(np.float32)
        data_rz_bin = np.sum(data_rz,1,keepdims=True)
        data_wz_bin = np.sum(data_wz,1,keepdims=True)

    #cast to tensors and make duplicates for the other solver mechanism
    t = torch.tensor(data_t, device=torch.device(device))
    t2 = torch.tensor(data_t.copy(), device=torch.device(device))
    t.requires_grad = True
    t2.requires_grad = True
    t_bin = torch.tensor(data_t_bin, device=torch.device(device))
    t2_bin = torch.tensor(data_t_bin.copy(), device=torch.device(device))
    t_bin.requires_grad = True
    t2_bin.requires_grad = True
    w = torch.tensor(data_w, device=torch.device(device))
    w_bin = torch.tensor(data_w_bin, device=torch.device(device))
    wx = torch.tensor(data_wx, device=torch.device(device))
    rx = torch.tensor(data_rx, device=torch.device(device))
    rx2 = torch.tensor(data_rx.copy(), device=torch.device(device))
    rx.requires_grad = True
    rx2.requires_grad = True
    wx_bin = torch.tensor(data_wx_bin, device=torch.device(device))
    rx_bin = torch.tensor(data_rx_bin, device=torch.device(device))
    rx2_bin = torch.tensor(data_rx_bin.copy(), device=torch.device(device))
    rx_bin.requires_grad = True
    rx2_bin.requires_grad = True
    r = [rx]
    r2 = [rx2]
    wl = [wx]
    r_bin = [rx_bin]
    r2_bin = [rx2_bin]
    wl_bin = [wx_bin]
    if d > 1:
        ry = torch.tensor(data_ry, device=torch.device(device))
        ry.requires_grad = True
        ry2 = torch.tensor(data_ry.copy(), device=torch.device(device))
        ry2.requires_grad = True
        ry_bin = torch.tensor(data_ry_bin, device=torch.device(device))
        ry_bin.requires_grad = True
        ry2_bin = torch.tensor(data_ry_bin.copy(), device=torch.device(device))
        ry2_bin.requires_grad = True
        wy = torch.tensor(data_wy, device=torch.device(device))
        wy_bin = torch.tensor(data_wy_bin, device=torch.device(device))
        r.append(ry)
        r2.append(ry2)
        wl.append(wy)
        r_bin.append(ry_bin)
        r2_bin.append(ry2_bin)
        wl_bin.append(wy_bin)
    if d > 2:
        rz = torch.tensor(data_rz, device=torch.device(device))
        rz.requires_grad = True
        rz2 = torch.tensor(data_rz.copy(), device=torch.device(device))
        rz2.requires_grad = True
        rz_bin = torch.tensor(data_rz_bin, device=torch.device(device))
        rz_bin.requires_grad = True
        rz2_bin = torch.tensor(data_rz_bin.copy(), device=torch.device(device))
        rz2_bin.requires_grad = True
        wz = torch.tensor(data_wz, device=torch.device(device))
        wz_bin = torch.tensor(data_wz_bin, device=torch.device(device))
        r.append(rz)
        r2.append(rz2)
        wl.append(wz)
        r_bin.append(rz_bin)
        r2_bin.append(rz2_bin)
        wl_bin.append(wz_bin)
        
    d = len(r)
    if d == 1:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = Potts_MAP1d.apply(t,r[0])
        oa_bin = Binary_MAP1d.apply(t_bin,r_bin[0])
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omj = PottsDeepFlow1d(solve_type="MFE",test_type=None)(t,r[0])
        om_bin, omj_bin = BinaryDeepFlow1d(solve_type="MFE",test_type=None)(t_bin,r_bin[0])
        omj = [omj]
        omj_bin = [omj_bin]
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, olj = PottsDeepFlow1d(solve_type="LBP",test_type=None)(t2,r2[0])
        ol_bin, olj_bin = BinaryDeepFlow1d(solve_type="LBP",test_type=None)(t2_bin,r2_bin[0])
        olj = [olj]
        olj_bin = [olj_bin]
        print(4,datetime.datetime.now())
    elif d == 2:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = Potts_MAP2d.apply(t,r[0],r[1])
        oa_bin = Potts_MAP2d.apply(t,r[0],r[1])
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omjx, omjy = PottsDeepFlow2d(solve_type="MFE",test_type=None)(t,r[0],r[1])
        om_bin, omjx_bin, omjy_bin = BinaryDeepFlow2d(solve_type="MFE",test_type=None)(t_bin,r_bin[0],r_bin[1])
        omj = [omjx, omjy]
        omj_bin = [omjx_bin, omjy_bin]
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, oljx, oljy = PottsDeepFlow2d(solve_type="LBP",test_type=None)(t2,r2[0],r2[1])
        ol_bin, oljx_bin, oljy_bin = BinaryDeepFlow2d(solve_type="LBP",test_type=None)(t2_bin,r2_bin[0],r2_bin[1])
        olj = [oljx, oljy]
        olj_bin = [oljx_bin, oljy_bin]
        print(4,datetime.datetime.now())
    elif d == 3:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = Potts_MAP3d.apply(t,r[0],r[1],r[2])
        oa_bin = Potts_MAP3d.apply(t_bin,r_bin[0],r_bin[1],r_bin[2])
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omjx, omjy, omjz = PottsDeepFlow3d(solve_type="MFE",test_type=None)(t,r[0],r[1],r[2])
        om_bin, omjx_bin, omjy_bin, omjz_bin = BinaryDeepFlow3d(solve_type="MFE",test_type=None)(t_bin,r_bin[0],r_bin[1],r_bin[2])
        omj = [omjx, omjy, omjz]
        omj_bin = [omjx_bin, omjy_bin, omjz_bin]
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, oljx, oljy, oljz = PottsDeepFlow3d(solve_type="LBP",test_type=None)(t2,r2[0],r2[1],r2[2])
        ol_bin, oljx_bin, oljy_bin, oljz_bin = BinaryDeepFlow3d(solve_type="LBP",test_type=None)(t2_bin,r2_bin[0],r2_bin[1],r2_bin[2])
        olj = [oljx, oljy, oljz]
        olj_bin = [oljx_bin, oljy_bin, oljz_bin]
        print(4,datetime.datetime.now())
        
    #run the gradient solvers on random losses
    all_losses = []
    loss = torch.sum(w*om)# + sum([torch.sum(wj*j) for wj,j in zip(wl,omj)])
    loss_bin = torch.sum(w_bin*om_bin) + torch.sum(w[:,0,...])# + sum([torch.sum(wj*j) for wj,j in zip(wl_bin,omj_bin)])
    all_losses.append(loss.cpu().item())
    all_losses.append(loss_bin.cpu().item())
    print(5,datetime.datetime.now())
    loss.backward()
    loss_bin.backward()
    print(6,datetime.datetime.now())
    loss = torch.sum(w*ol)# + sum([torch.sum(wj*j) for wj,j in zip(wl,olj)])
    loss_bin = torch.sum(w_bin*ol_bin) + torch.sum(w[:,0,...])# + sum([torch.sum(wj*j) for wj,j in zip(wl_bin,olj_bin)])
    all_losses.append(loss.cpu().item())
    all_losses.append(loss_bin.cpu().item())
    print(7,datetime.datetime.now())
    loss.backward()
    loss_bin.backward()
    print(8,datetime.datetime.now())
    
    #convert to numpy for validation
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    ol_np = ol.detach().cpu().numpy()
    omj_np = [omjn.detach().cpu().numpy() for omjn in omj]
    olj_np = [oljn.detach().cpu().numpy() for oljn in olj]
    ot_np = t.grad.detach().cpu().numpy()
    ot2_np = t2.grad.detach().cpu().numpy()
    ogmj_np = [ri.grad.detach().cpu().numpy() for ri in r]
    oglj_np = [ri.grad.detach().cpu().numpy() for ri in r2]
    oa_np_bin = oa_bin.detach().cpu().numpy()
    om_np_bin = om_bin.detach().cpu().numpy()
    ol_np_bin = ol_bin.detach().cpu().numpy()
    omj_np_bin = [omjn.detach().cpu().numpy() for omjn in omj_bin]
    olj_np_bin = [oljn.detach().cpu().numpy() for oljn in olj_bin]
    ot_np_bin = t_bin.grad.detach().cpu().numpy()
    ot2_np_bin = t2_bin.grad.detach().cpu().numpy()
    ogmj_np_bin = [ri.grad.detach().cpu().numpy() for ri in r_bin]
    oglj_np_bin = [ri.grad.detach().cpu().numpy() for ri in r2_bin]
    
    #do initial NaN checks
    print(9,"Running NaN checks")
    nan_check([oa_np,om_np,ol_np,ot_np,ot2_np]+omj_np+olj_np+ogmj_np+oglj_np,
              ["oa","om","ol","ot","ot2"]+["omj"+str(i) for i in range(d)]+["olj"+str(i) for i in range(d)]+["ogmj"+str(i) for i in range(d)]+["oglj"+str(i) for i in range(d)], verbosity)
    nan_check([oa_np_bin,om_np_bin,ol_np_bin,ot_np_bin,ot2_np_bin]+omj_np_bin+olj_np_bin+ogmj_np_bin+oglj_np_bin,
              [n+"_bin" for n in ["oa","om","ol","ot","ot2"]+["omj"+str(i) for i in range(d)]+["olj"+str(i) for i in range(d)]+["ogmj"+str(i) for i in range(d)]+["oglj"+str(i) for i in range(d)]], verbosity)
    
    #check marginals
    print(10,"Ensure equivalent marginals")
    exception_list = []
    for estimate, name in zip([(om_np,om_np_bin), (ol_np,ol_np_bin)], ["MFE","LBP"]):
        buffer_equality_check(estimate[0][:,1,...],name+"_POTTS",estimate[1][:,0,...],name+"_BINARY",epsilon_light,asserter)
        
    print(11,"Ensure equivalent joint marginals")
    for potts,binary, name in zip(omj_np+olj_np,omj_np_bin+olj_np_bin, ["MFE"+str(i) for i in range(d)]+["LBP"+str(i) for i in range(d)]):
        buffer_equality_check(potts[:,0,...],name+"_POTTS0",binary[:,0,...],name+"_BINARY",epsilon_light,asserter)
        buffer_equality_check(potts[:,1,...],name+"_POTTS1",binary[:,0,...],name+"_BINARY",epsilon_light,asserter)
    
    print(12,"Ensure same losses")
    if abs(all_losses[0] - all_losses[1]) > epsilon*math.sqrt(numels) or abs(all_losses[2] - all_losses[3]) > epsilon*math.sqrt(numels):
        print("Losses do not match exactly",all_losses)
        #asserter.assertTrue(False)
    
    #check gradients
    print(13,"Ensure equivalent gradients with respect to the data terms")
    exception_list = []
    for estimate, name in zip([(ot_np,ot_np_bin), (ot2_np,ot2_np_bin)], ["MFE","LBP"]):
        buffer_equality_check(-estimate[0][:,0,...],name+"_POTTS0",estimate[1][:,0,...],name+"_BINARY",epsilon_light,asserter)
        buffer_equality_check( estimate[1][:,0,...],name+"_POTTS1",estimate[1][:,0,...],name+"_BINARY",epsilon_light,asserter)
    
    #check gradients
    print(14,"Ensure equivalent gradients with respect to the smoothness terms for LBP")
    exception_list = []
    for potts,binary, name in zip(oglj_np,oglj_np_bin, ["LBP"+str(i) for i in range(d)]):
        buffer_equality_check(potts[:,0,...],name+"_POTTS0",binary[:,0,...],name+"_BINARY",epsilon_light,asserter)
        buffer_equality_check(potts[:,1,...],name+"_POTTS1",binary[:,0,...],name+"_BINARY",epsilon_light,asserter)
    
    #check gradients
    print(15,"Ensure weak equivalence (i.e. the average of the Potts model gives you the binary model) for gradients of smoothness terms")
    #Mathematically, this is a bit odd. I think it has to do with the fact that normally a Potts model has more than two labels so it doesn't "know"
    #that edges from label 0 must also be edges from label 1.
    exception_list = []
    for potts,binary, name in zip(ogmj_np,ogmj_np_bin, ["MFE"+str(i) for i in range(d)]):
        potts_estimate = 0.5*(potts[:,0,...]+potts[:,1,...])
        buffer_equality_check(potts_estimate,name+"_POTTS_AVG",binary[:,0,...],name+"_BINARY",epsilon_light,asserter)
    
    
class Test_Extreme(unittest.TestCase):
    
    def test_no_smoothness_1D(self):
        print("")
        test_no_smoothness(1,"pytorch_cpu",self)
        #print("")
        #test_no_smoothness(1,"pytorch_cuda",self)
        
    def test_no_smoothness_2D(self):
        print("")
        test_no_smoothness(2,"pytorch_cpu",self)
        #print("")
        #test_no_smoothness(2,"pytorch_cuda",self)

    def test_no_smoothness_3D(self):
        print("")
        test_no_smoothness(3,"pytorch_cpu",self)
        #print("")
        #test_no_smoothness(3,"pytorch_cuda",self)
        
    def test_no_data_1D(self):
        print("")
        test_no_data(1,"pytorch_cpu",self)
        #print("")
        #test_no_data(1,"pytorch_cuda",self)
        
    def test_no_data_2D(self):
        print("")
        test_no_data(2,"pytorch_cpu",self)
        #print("")
        #test_no_data(2,"pytorch_cuda",self)

    def test_no_data_3D(self):
        print("")
        test_no_data(3,"pytorch_cpu",self)    
        #print("")
        #test_no_data(3,"pytorch_cuda",self)

    def test_smoothness_1D(self):
        print("")
        test_smoothness_dom(1,"pytorch_cpu",self)
        #print("")
        #test_smoothness_dom(1,"pytorch_cuda",self)

    def test_smoothness_2D(self):
        print("")
        test_smoothness_dom(2,"pytorch_cpu",self)
        #print("")
        #test_smoothness_dom(2,"pytorch_cuda",self)

    def test_smoothness_3D(self):
        print("")
        test_smoothness_dom(3,"pytorch_cpu",self)
        #print("")
        #test_smoothness_dom(3,"pytorch_cuda",self)
        
    def test_zmimic_binary_1D(self):
        print("")
        test_mimic_binary(1,"pytorch_cpu",self)
        #print("")
        #test_mimic_binary(1,"pytorch_cuda",self)
    
    def test_zmimic_binary_2D(self):
        print("")
        test_mimic_binary(2,"pytorch_cpu",self)
        #print("")
        #test_mimic_binary(2,"pytorch_cuda",self)
    
    def test_zmimic_binary_3D(self):
        print("")
        test_mimic_binary(3,"pytorch_cpu",self)
    #    print("")
    #    test_mimic_binary(2,"pytorch_cuda",self)
        
if __name__ == '__main__':
    unittest.main()
    
    
