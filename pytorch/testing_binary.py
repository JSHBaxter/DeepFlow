import unittest

import numpy as np
import math
import random

import datetime
import itertools

import torch
from binary_deepflow import BinaryDeepFlow1d, BinaryDeepFlow2d, BinaryDeepFlow3d
from binary_deepflow_function import Binary_MAP1d,Binary_MAP2d,Binary_MAP3d
from graph_deepflow import BinaryDeepFlow
from graph_deepflow_function import DenseGraph_MAP_Binary_PytorchNative

b=1
c=int(4*random.uniform(0,1))+3
x=2**15
x_graph = 4
epsilon = 10**-6
verbosity = True

size_info_graph_d = [b,c,x_graph]
size_info_graph_r = [b,c,x_graph,x_graph]

def get_size_info(d):
    x_used = int(x**(1/d)+0.5)
    return tuple( [b,c]+[x_used for i in range(d)] ), [tuple([b,c]+[x_used - (1 if i==i2 else 0) for i in range(d)]) for i2 in range(d)], tuple( [1,c]+[x_used for i in range(d)] ), tuple([i+2 for i in range(d)])

def BINARY_run_all_solvers(t,t2,r,r2):
    if type(r) is list:
        d = len(r)
    else:
        d = 0
    if d == 0:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = DenseGraph_MAP_Binary_PytorchNative.apply(t,r)
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omj = BinaryDeepFlow(solve_type="MFE",test_type=None)(t,r)
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, olj = BinaryDeepFlow(solve_type="LBP",test_type=None)(t2,r2)
        print(4,datetime.datetime.now())
    elif d == 1:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = Binary_MAP1d.apply(t,r[0])
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omj = BinaryDeepFlow1d(solve_type="MFE",test_type=None)(t,r[0])
        omj = [omj]
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, olj = BinaryDeepFlow1d(solve_type="LBP",test_type=None)(t2,r2[0])
        olj = [olj]
        print(4,datetime.datetime.now())
    elif d == 2:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = Binary_MAP2d.apply(t,r[0],r[1])
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omjx, omjy = BinaryDeepFlow2d(solve_type="MFE",test_type=None)(t,r[0],r[1])
        omj = [omjx, omjy]
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, oljx, oljy = BinaryDeepFlow2d(solve_type="LBP",test_type=None)(t2,r2[0],r2[1])
        olj = [oljx, oljy]
        print(4,datetime.datetime.now())
    elif d == 3:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = Binary_MAP3d.apply(t,r[0],r[1],r[2])
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omjx, omjy, omjz = BinaryDeepFlow3d(solve_type="MFE",test_type=None)(t,r[0],r[1],r[2])
        omj = [omjx, omjy, omjz]
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, oljx, oljy, oljz = BinaryDeepFlow3d(solve_type="LBP",test_type=None)(t2,r2[0],r2[1],r2[2])
        olj = [oljx, oljy, oljz]
        print(4,datetime.datetime.now())
    return oa, om, omj, ol, olj

def nan_check(m_np, name, verbose=False):
    for m,n in zip(m_np,name):
        if verbose :
            print("\tNAN check on ", n)
        if np.any(np.isnan(m)):
            raise Exception("NaN found in buffer " + n)

def test_no_smoothness_graph(full_device,asserter):
    print("Testing (no smoothness) \tDim: GRAPH \tDev: " + full_device + "\t Start Time:\t" + str(datetime.datetime.now()))
    device = "cuda" if "cuda" in full_device else "cpu"
    
    #create data terms for testing (smoothness terms should be zero)
    data_t = 2*np.random.normal(0,1,size=size_info_graph_d).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info_graph_d).astype(np.float32)
    data_r = np.zeros(shape=size_info_graph_r).astype(np.float32)
    data_wr = np.random.normal(0,1,size=size_info_graph_r).astype(np.float32)
    
    #cast to tensors and make duplicates for the other solver mechanism
    t = torch.tensor(data_t, device=torch.device(device))
    t2 = torch.tensor(data_t.copy(), device=torch.device(device))
    t.requires_grad = True
    t2.requires_grad = True
    w = torch.tensor(data_w, device=torch.device(device))
    wr = torch.tensor(data_wr, device=torch.device(device))
    r = torch.tensor(data_r, device=torch.device(device))
    r2 = torch.tensor(data_r.copy(), device=torch.device(device))
    r.requires_grad = True
    r2.requires_grad = True
    
    #run the solvers
    oa, om, omj, ol, olj = BINARY_run_all_solvers(t,t2,r,r2)
    
    #run the gradient solvers on random losses
    loss = torch.sum(w*om) + torch.sum(wr*omj)
    print(5,datetime.datetime.now())
    loss.backward()
    print(6,datetime.datetime.now())
    loss = torch.sum(w*ol) + torch.sum(wr*olj)
    print(7,datetime.datetime.now())
    loss.backward()
    print(8,datetime.datetime.now())
    
    #convert to numpy for validation
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    ol_np = ol.detach().cpu().numpy()
    omj_np = omj.detach().cpu().numpy()
    olj_np = olj.detach().cpu().numpy()
    ot_np = t.grad.detach().cpu().numpy()
    ot2_np = t2.grad.detach().cpu().numpy()
    ogmj_np = r.grad.detach().cpu().numpy()
    oglj_np = r2.grad.detach().cpu().numpy()
    
    #do initial NaN checks
    print(9,"Running NaN checks")
    nan_check([oa_np,om_np,ol_np,ot_np,ot2_np,omj_np,olj_np,ogmj_np,oglj_np],
              ["oa","om","ol","ot","ot2","omj","olj","ogmj","oglj"], verbosity)
        
    #ensure mean estimate is equivalent to the sigmoid of the data terms
    print(10,"Checking correctness of marginals")
    true_val_marg = 1 / (1 + np.exp(-data_t))
    for estimate, name in zip([om_np, ol_np], ["MFE","LBP"]):
        for loc, (val_df, val_d) in enumerate(zip(estimate.flatten(),true_val_marg.flatten())):
            if not (abs(val_df-val_d) < epsilon):
                print(name +" predicted marginal ("+str(val_df)+") differs from expected marginal ("+str(val_d)+") at location " + str(loc) + " (diff " + str(val_df-val_d) + ")")
                asserter.assertTrue(False)
        if verbosity:
            print("\tGood for "+name)
        
    #ensure mean estimate of joint marginals is correct
    print(11,"Checking correctness of joint marginals")
    inv_eye = (1-np.eye(size_info_graph_d[2]).reshape(1,1,size_info_graph_d[2],size_info_graph_d[2]))
    true_val_joint = (np.expand_dims(true_val_marg,-1)*(1-np.expand_dims(true_val_marg,-2)) + np.expand_dims(true_val_marg,-2)*(1-np.expand_dims(true_val_marg,-1)))*inv_eye
    for estimate, name in zip([omj_np, olj_np], ["MFE","LBP"]):
        for loc, (val_df, val_d) in enumerate(zip(estimate.flatten(),true_val_joint.flatten())):
            if not (abs(val_df-val_d) < epsilon):
                print(name +" predicted joint marginal ("+str(val_df)+") differs from expected marginal ("+str(val_d)+") at location " + str(loc) + " (diff " + str(val_df-val_d) + ")")
                asserter.assertTrue(False)
        if verbosity:
            print("\tGood for "+name)
            
    #ensure gradient wrt data terms are equivalent to just sigmoid
    print(12,"Checking correctness of gradient calculation")
    true_val = data_w * true_val_marg * (1-true_val_marg)
    true_val += np.sum(data_wr * np.expand_dims(true_val_marg * (1-true_val_marg),-2) * (1-2*np.expand_dims(true_val_marg,-1)),-1)
    print(true_val)
    print(ot_np)
    print(ot2_np)
            
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
    oa, om, omj, ol, olj = BINARY_run_all_solvers(t,t2,r,r2)
    
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
        
    #ensure mean estimate is equivalent to the sigmoid of the data terms
    print(10,"Checking correctness of marginals")
    true_val_marg = 1 / (1 + np.exp(-data_t))
    for estimate, name in zip([om_np, ol_np], ["MFE","LBP"]):
        for loc, (val_df, val_d) in enumerate(zip(estimate.flatten(),true_val_marg.flatten())):
            if not (abs(val_df-val_d) < epsilon):
                print(name +" predicted marginal ("+str(val_df)+") differs from expected marginal ("+str(val_d)+") at location " + str(loc) + " (diff " + str(val_df-val_d) + ")")
                asserter.assertTrue(False)
        if verbosity:
            print("\tGood for "+name)
        
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
            for loc, (val_df, val_d) in enumerate(zip(estimate.flatten(),true_val.flatten())):
                if not (abs(val_df-val_d) < epsilon):
                    print(name +" (direction " + str(i) + ") predicted joint marginal (" + str(val_df) + ") differs from expected joint marginal (" + str(val_d) + ") at location " + str(loc))
                    asserter.assertTrue(False)
            if verbosity:
                print("\tGood for " + name + " in direction " + str(i))
    
    #ensure gradient wrt data terms are equivalent to just sigmoid
    print(12,"Checking correctness of gradient calculation")
    true_val = data_w * true_val_marg * (1-true_val_marg)
    for i in range(d):
        if i == 0:
            true_val[:,:,:-1,...] += data_wx * true_val_marg[:,:,:-1,...] * (1-true_val_marg[:,:,:-1,...]) * (1-2*true_val_marg[:,:,1:,...])
            true_val[:,:,1:,...] += data_wx * true_val_marg[:,:,1:,...] * (1-true_val_marg[:,:,1:,...]) * (1-2*true_val_marg[:,:,:-1,...])
        elif i == 1:
            true_val[:,:,:,:-1,...] += data_wy * true_val_marg[:,:,:,:-1,...] * (1-true_val_marg[:,:,:,:-1,...]) * (1-2*true_val_marg[:,:,:,1:,...])
            true_val[:,:,:,1:,...] += data_wy * true_val_marg[:,:,:,1:,...] * (1-true_val_marg[:,:,:,1:,...]) * (1-2*true_val_marg[:,:,:,:-1,...])
        elif i == 2:
            true_val[:,:,:,:,:-1,...] += data_wz * true_val_marg[:,:,:,:,:-1,...] * (1-true_val_marg[:,:,:,:,:-1,...]) * (1-2*true_val_marg[:,:,:,:,1:,...])
            true_val[:,:,:,:,1:,...] += data_wz * true_val_marg[:,:,:,:,1:,...] * (1-true_val_marg[:,:,:,:,1:,...]) * (1-2*true_val_marg[:,:,:,:,:-1,...])
    for estimate, name in zip([ot2_np, ot_np], ["LBP","MFE"]):
        for loc, (val_df, val_d) in enumerate(zip(estimate.flatten(),true_val.flatten())):
            if not (abs(val_df-val_d) < epsilon):
                print(name +" predicted gradient ("+str(val_df)+") differs from expected gradient ("+str(val_d)+") at location " + str(loc))
                asserter.assertTrue(False)
        if verbosity:
            print("\tGood for "+name)
    
    #ensure MAP assigns 1 to positive terms
    print(13,"Check MAP correctness")
    for val_df, val_d in zip(oa_np.flatten(),data_t.flatten()):
        if(val_d > epsilon and val_df < 1-epsilon):
            print("oa",val_df, val_d, epsilon)
            asserter.assertFalse(val_d > epsilon and val_df < 1-epsilon)
        if(val_d < -epsilon and val_df > epsilon):
            print("oa",val_df, val_d, epsilon)
            asserter.assertFalse(val_d < -epsilon and val_df > epsilon)

            
            
            
            
def test_no_data(d,full_device,asserter):
    print("Testing (no data term) \tDim: " +str(d)+ " \tDev: " + full_device + "\t Start Time:\t" + str(datetime.datetime.now()))
    device = "cuda" if "cuda" in full_device else "cpu"
    
    size_info, size_info_j, size_red_info, axes = get_size_info(d)

    #create data terms for testing (smoothness terms should be zero)
    rlimit = 0.5 / d
    data_t = np.zeros(shape=size_info).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_rx = rlimit*np.random.uniform(0,1,size=size_info).astype(np.float32)
    if d > 1:
        data_ry = rlimit*np.random.uniform(0,1,size=size_info).astype(np.float32)
    if d > 2:
        data_rz = rlimit*np.random.uniform(0,1,size=size_info).astype(np.float32)

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
    oa, om, omj, ol, olj = BINARY_run_all_solvers(t,t2,r,r2)
    
    #run the gradient solvers on random losses (only for marginals
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
    
    #ensure mean estimate is 50/50
    print(10,"Checking correctness of marginals")
    exception_list = []
    val_d = 0.5
    for estimate, name in zip([om_np, ol_np], ["MFE","LBP"]):
        for loc, val_df in enumerate(estimate.flatten()):
            if not (abs(val_df-val_d) < epsilon):
                exception_list.append(name +" predicted marginal ("+str(val_df)+") differs from expected marginal ("+str(val_d)+") at location " + str(loc))
                break
        if verbosity:
            print("\tDone for "+name)
    if exception_list:
        for e in exception_list:
            print(e)
        asserter.assertTrue(False)
            
    #ensure gradient wrt smoothness terms are zero
    exception_list = []
    print(11,"Checking correctness of gradient calculation")
    val_d = 0.0
    for estimate, name in zip(ogmj_np+oglj_np, ["MFE"+str(i) for i in range(d)]+["LBP"+str(i) for i in range(d)]):
        for loc, val_df in enumerate(estimate.flatten()):
            if not (abs(val_df-val_d) < epsilon):
                exception_list.append(name +" predicted gradient ("+str(val_df)+") differs from expected gradient ("+str(val_d)+") at location " + str(loc))
                break
        if verbosity:
            print("\tDone for "+name)
    if exception_list:
        for e in exception_list:
            print(e)
        asserter.assertTrue(False)

def test_smoothness_dom(d,full_device,asserter):
    print("Testing (smoothness dom) \tDim: " +str(d)+ " \tDev: " + full_device + "\t Start Time:\t" + str(datetime.datetime.now()))
    device = "cuda" if "cuda" in full_device else "cpu"
    alg_type = "pytorch" if "pytorch" in full_device else "cpp"

    size_info, size_info_j, size_red_info, axes = get_size_info(d)

    data_t = 0.001/d*np.random.normal(0,1,size=size_info).astype(np.float32)
    for b,c in itertools.product(range(size_info[0]),range(size_info[1])):
        data_t[b,c,...] += 10*np.sign(np.sum(data_t[b,c,...]))/math.prod(size_info[2:])
    data_rx = 0.5/d+0.5/d*np.random.uniform(size=size_info).astype(np.float32)
    if d > 1:
        data_ry = 0.5/d+0.5/d*np.random.uniform(size=size_info).astype(np.float32)
    if d > 2:
        data_rz = 0.5/d+0.5/d*np.random.uniform(size=size_info).astype(np.float32)

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
    oa, om, omj, ol, olj = BINARY_run_all_solvers(t,t,r,r)
    
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    ol_np = ol.detach().cpu().numpy()
    omj_np = [omjn.detach().cpu().numpy() for omjn in omj]
    olj_np = [oljn.detach().cpu().numpy() for oljn in olj]
    
    #do initial NaN checks
    print(9,"Running NaN checks")
    nan_check([oa_np,om_np,ol_np]+omj_np+olj_np,
              ["oa","om","ol"]+["omj"+str(i) for i in range(d)]+["olj"+str(i) for i in range(d)], verbosity)
    
    #ensure MAP assigns 1 to positive terms and mean shares sign with the sum of the data terms
    print(10,"Check MAP correctness")
    for b,c in itertools.product(range(size_info[0]),range(size_info[1])):
        sign = np.sign(np.sum(data_t[b,c,...]))
        oa_to_check = oa_np[b,c,...]
        if sign == 1:
            for loc,val in enumerate(oa_to_check.flatten()):
                if val < 1-epsilon:
                    print("Sign is",sign,"but value is",val,"at location",loc)
                    assert(False)
        elif sign == -1:
            for loc,val in enumerate(oa_to_check.flatten()):
                if val > epsilon:
                    print("Sign is",sign,"but value is",val,"at location",loc)
                    assert(False)

                
class Test_Extreme(unittest.TestCase):
    
    def test_ano_smoothness_graph(self):
        print("")
        test_no_smoothness_graph("pytorch_cpu",self)
        print("")
        test_no_smoothness_graph("pytorch_cuda",self)
    
    def test_no_smoothness_1D(self):
        print("")
        test_no_smoothness(1,"pytorch_cpu",self)
        print("")
        test_no_smoothness(1,"pytorch_cuda",self)
        
    def test_no_smoothness_2D(self):
        print("")
        test_no_smoothness(2,"pytorch_cpu",self)
        print("")
        test_no_smoothness(2,"pytorch_cuda",self)

    def test_no_smoothness_3D(self):
        print("")
        test_no_smoothness(3,"pytorch_cpu",self)
        print("")
        test_no_smoothness(3,"pytorch_cuda",self)
        
    def test_no_data_1D(self):
        print("")
        test_no_data(1,"pytorch_cpu",self)
        print("")
        test_no_data(1,"pytorch_cuda",self)
        
    def test_no_data_2D(self):
        print("")
        test_no_data(2,"pytorch_cpu",self)
        print("")
        test_no_data(2,"pytorch_cuda",self)

    def test_no_data_3D(self):
        print("")
        test_no_data(3,"pytorch_cpu",self)    
        print("")
        test_no_data(3,"pytorch_cuda",self)     

    def test_smoothness_1D(self):
        print("")
        test_smoothness_dom(1,"pytorch_cpu",self)
        print("")
        test_smoothness_dom(1,"pytorch_cuda",self)

    def test_smoothness_2D(self):
        print("")
        test_smoothness_dom(2,"pytorch_cpu",self)
        print("")
        test_smoothness_dom(2,"pytorch_cuda",self)

    def test_smoothness_3D(self):
        print("")
        test_smoothness_dom(3,"pytorch_cpu",self)
        print("")
        test_smoothness_dom(3,"pytorch_cuda",self)
        
if __name__ == '__main__':
    unittest.main()
    
    
