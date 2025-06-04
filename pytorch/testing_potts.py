import unittest

import numpy as np
from itertools import product
import math

import time

import torch
from potts_deepflow import Potts_MAP1d,Potts_MAP2d,Potts_MAP3d
from potts_deepflow import Potts_Mean1d,Potts_Mean2d,Potts_Mean3d
from potts_deepflow import Potts_Mean1d_PytorchNative,Potts_Mean2d_PytorchNative,Potts_Mean3d_PytorchNative
from potts_deepflow import Potts_LBP1d_PytorchNative, Potts_LBP2d_PytorchNative, Potts_LBP3d_PytorchNative

b=2
c=4
x=2**12
epsilon = 0.1

def APD(v1,v2):
    num = np.sum(np.abs(v1-v2))
    den = 0.5*np.sum(np.abs(v1)+np.abs(v2))
    
    if den > epsilon/100:
        return 100*num/den
    return 100*num

def get_size_into(d):
    x_used = int(x**(2/(d+1))+0.5)
    return tuple( [b,c]+[x_used for i in range(d)] ), tuple( [1,c]+[x_used for i in range(d)] ), tuple([i+2 for i in range(d)])

def test_no_smoothness(d,full_device,asserter):
    print("Testing (no smoothness) \t Dim: " +str(d)+ " \t Dev: " + full_device)
    device = "cuda" if "cuda" in full_device else "cpu"
    alg_type = "pytorch" if "pytorch" in full_device else "cpp"

    size_info, size_red_info, axes = get_size_into(d)

    data_t = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_rx = np.zeros(shape=size_info).astype(np.float32)
    if d > 1:
        data_ry = np.zeros(shape=size_info).astype(np.float32)
    if d > 2:
        data_rz = np.zeros(shape=size_info).astype(np.float32)

    t = torch.tensor(data_t, device=torch.device(device))
    t2 = torch.tensor(data_t.copy(), device=torch.device(device))
    t.requires_grad = True
    t2.requires_grad = True
    w = torch.tensor(data_w, device=torch.device(device))
    rx = torch.tensor(data_rx, device=torch.device(device))
    rx2 = torch.tensor(data_rx.copy(), device=torch.device(device))
    rx.requires_grad = True
    rx2.requires_grad = True
    if d > 1:
        ry = torch.tensor(data_ry, device=torch.device(device))
        ry.requires_grad = True
        ry2 = torch.tensor(data_ry, device=torch.device(device))
        ry2.requires_grad = True
    if d > 2:
        rz = torch.tensor(data_rz, device=torch.device(device))
        rz.requires_grad = True
        rz2 = torch.tensor(data_rz, device=torch.device(device))
        rz2.requires_grad = True
        
    if d == 1:
        oa = torch.exp(Potts_MAP1d.apply(t,rx))
        if alg_type == "cpp":
            om = Potts_Mean1d.apply(t,rx)
        else:
            om = Potts_Mean1d_PytorchNative.apply(t,rx)
        ol = Potts_LBP1d_PytorchNative.apply(t2,rx2)
    elif d == 2:
        oa = torch.exp(Potts_MAP2d.apply(t,rx,ry))
        if alg_type == "cpp":
            om = Potts_Mean2d.apply(t,rx,ry)
        else:
            om = Potts_Mean2d_PytorchNative.apply(t,rx,ry)
        ol = Potts_LBP2d_PytorchNative.apply(t2,rx2,ry2)
    elif d == 3:
        oa = torch.exp(Potts_MAP3d.apply(t,rx,ry,rz))
        if alg_type == "cpp":
            om = Potts_Mean3d.apply(t,rx,ry,rz)
        else:
            om = Potts_Mean3d_PytorchNative.apply(t,rx,ry,rz)
        ol = Potts_LBP3d_PytorchNative.apply(t2,rx2,ry2,rz2)
    
    loss = torch.sum(w*om)
    loss.backward()
    ot_np = t.grad.detach().cpu().numpy()
    
    loss = torch.sum(w*ol)
    loss.backward()
    ot2_np = t2.grad.detach().cpu().numpy()
    
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    ol_np = ol.detach().cpu().numpy()
    
    #make sure not nan
    asserter.assertFalse(np.any(np.isnan(oa_np)))
    asserter.assertFalse(np.any(np.isnan(om_np)))
    asserter.assertFalse(np.any(np.isnan(ot_np)))
    asserter.assertFalse(np.any(np.isnan(ot2_np)))
    
    #resize into more usable form
    dt_np_l = {(ib,ic):data_t[ib,ic,...].flatten() for ib,ic in product(range(b),range(c))}
    dw_np_l = {(ib,ic):data_w[ib,ic,...].flatten() for ib,ic in product(range(b),range(c))}
    oa_np_l = {(ib,ic):oa_np[ib,ic,...].flatten()  for ib,ic in product(range(b),range(c))}
    om_np_l = {(ib,ic):om_np[ib,ic,...].flatten()  for ib,ic in product(range(b),range(c))}
    ol_np_l = {(ib,ic):ol_np[ib,ic,...].flatten()  for ib,ic in product(range(b),range(c))}
    ot_np_l = {(ib,ic):ot_np[ib,ic,...].flatten()  for ib,ic in product(range(b),range(c))}
    ot2_np_l = {(ib,ic):ot2_np[ib,ic,...].flatten()  for ib,ic in product(range(b),range(c))}
    x_space = len(dt_np_l[(0,0)])
    
    #ensure MAP assigns 1 to highest term and 0 to everything else
    for i in range(x_space):
        for ib in range(b):
            highest = max([dt_np_l[(ib,ic)][i] for ic in range(c)])
            for ic in range(c):
                if(dt_np_l[(ib,ic)][i] == highest and oa_np_l[(ib,ic)][i] < 0.5):
                    raise Exception(str(dt_np_l[(ib,ic)][i])+"\t"+str([o[i] for o in dt_np_l])+"\t"+str(highest)+"\t"+str(oa_np_l[(ib,ic)][i])+"\t"+str([o[i] for o in oa_np_l]))
                if(dt_np_l[(ib,ic)][i] < highest - epsilon/100  and oa_np_l[(ib,ic)][i] > 0.5):
                    raise Exception(str(dt_np_l[(ib,ic)][i])+"\t"+str([o[i] for o in dt_np_l])+"\t"+str(highest)+"\t"+str(oa_np_l[(ib,ic)][i])+"\t"+str([o[i] for o in oa_np_l]))
        
    #ensure mean pass is equivalent to the data terms only
    for ib,ic in product(range(b),range(c)): 
        for val_df, val_d in zip(om_np_l[(ib,ic)],dt_np_l[(ib,ic)]):
            if(APD(val_df,val_d) > epsilon):
                raise Exception(str(val_df) + "\t" + str(val_d))
    for ib,ic in product(range(b),range(c)): 
        for val_df, val_d in zip(ol_np_l[(ib,ic)],dt_np_l[(ib,ic)]):
            if(APD(val_df,val_d) > epsilon):
                raise Exception(str(val_df) + "\t" + str(val_d))
    
    #ensure gradient wrt data terms are passed immediately through
    for ib,ic in product(range(b),range(c)): 
        for val_df, val_d in zip(ot_np_l[(ib,ic)],dw_np_l[(ib,ic)]):
            if(APD(val_df,val_d) > epsilon):
                raise Exception(str(val_df) + "\t" + str(val_d))
    for ib,ic in product(range(b),range(c)): 
        for val_df, val_d in zip(ot2_np_l[(ib,ic)],dw_np_l[(ib,ic)]):
            if(APD(val_df,val_d) > epsilon):
                raise Exception(str(val_df) + "\t" + str(val_d))
                

def test_smoothness_dom(d,full_device,asserter):
    print("Testing (no smoothness) \t Dim: " +str(d)+ " \t Dev: " + full_device)
    device = "cuda" if "cuda" in full_device else "cpu"
    alg_type = "pytorch" if "pytorch" in full_device else "cpp"

    size_info, size_red_info, axes = get_size_into(d)

    data_t = 0.1*np.random.uniform(0,1,size=size_info).astype(np.float32)
    for ib in range(b):
        winner = np.argmax(np.sum(data_t[ib,...],tuple([a-1 for a in axes])))
        data_t[ib,winner,...] += 0.05/d
    data_rx = size_info[0]*np.random.uniform(size=size_info).astype(np.float32)
    if d > 1:
        data_ry = size_info[0]*size_info[1]*np.random.uniform(size=size_info).astype(np.float32)
        data_rx *= size_info[1]
    if d > 2:
        data_rz = size_info[0]*size_info[1]*size_info[2]*np.random.uniform(size=size_info).astype(np.float32)
        data_ry *= size_info[2]
        data_rx *= size_info[2]

    t = torch.tensor(data_t, device=torch.device(device))
    rx = torch.tensor(data_rx, device=torch.device(device))
    if d > 1:
        ry = torch.tensor(data_ry, device=torch.device(device))
    if d > 2:
        rz = torch.tensor(data_rz, device=torch.device(device))

    if d == 1:
        oa = torch.exp(Potts_MAP1d.apply(t,rx))
        if alg_type == "cpp":
            om = Potts_Mean1d.apply(t,rx)
        else:
            om = Potts_Mean1d_PytorchNative.apply(t,rx)
        ol = Potts_LBP1d_PytorchNative.apply(t,rx)
    elif d == 2:
        oa = torch.exp(Potts_MAP2d.apply(t,rx,ry))
        if alg_type == "cpp":
            om = Potts_Mean2d.apply(t,rx,ry)
        else:
            om = Potts_Mean2d_PytorchNative.apply(t,rx,ry)
        ol = Potts_LBP2d_PytorchNative.apply(t,rx,ry)
    elif d == 3:
        oa = torch.exp(Potts_MAP3d.apply(t,rx,ry,rz))
        if alg_type == "cpp":
            om = Potts_Mean3d.apply(t,rx,ry,rz)
        else:
            om = Potts_Mean3d_PytorchNative.apply(t,rx,ry,rz)
        ol = Potts_LBP3d_PytorchNative.apply(t,rx,ry,rz)
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    ol_np = om.detach().cpu().numpy()
    
    #make sure not nan
    asserter.assertFalse(np.any(np.isnan(oa_np)))
    asserter.assertFalse(np.any(np.isnan(om_np)))
    asserter.assertFalse(np.any(np.isnan(ol_np)))
    
    #resize into more usable form
    dt_np_l = {(ib,ic):data_t[ib,ic,...].flatten() for ib,ic in product(range(b),range(c))}
    oa_np_l = {(ib,ic):oa_np[ib,ic,...].flatten()  for ib,ic in product(range(b),range(c))}
    om_np_l = {(ib,ic):om_np[ib,ic,...].flatten()  for ib,ic in product(range(b),range(c))}
    ol_np_l = {(ib,ic):ol_np[ib,ic,...].flatten()  for ib,ic in product(range(b),range(c))}
    x_space = len(dt_np_l[(0,0)])
    
    #ensure MAP assigns 1 to highest terms
    for ib in range(b):
        sums = [np.sum(dt_np_l[(ib,ic)]) for ic in range(c)]
        highest = max(sums)
        for ic in range(c):
            for val_df in oa_np_l[(ib,ic)]:
                if(sums[ic] == highest and val_df < 0.5):
                    raise Exception(str(sums[ic])+ "\t" + str(sums) + "\t" + str(val_df))
                if(sums[ic] < highest  and val_df > 0.5):
                    raise Exception(str(sums[ic])+ "\t" + str(sums) + "\t" + str(val_df))

def test_device_equivalence(d,device_list,asserter):
    print("Testing (dev equiv.) \t Dim: " +str(d)+ " \t Dev:",device_list)

    size_info, size_red_info, axes = get_size_into(d)

    data_t = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_rx = (np.random.uniform(size=size_info)).astype(np.float32) * (1/2**d)*0.75 # (math.log((2*2**d-1)/(2*2**d-3))*0.75)
    if d > 1:
        data_ry = (np.random.uniform(size=size_info)).astype(np.float32) * (1/2**d)*0.75 # (math.log((2*2**d-1)/(2*2**d-3))*0.75)
    if d > 2:
        data_rz= (np.random.uniform(size=size_info)).astype(np.float32) * (1/2**d)*0.75 # (math.log((2*2**d-1)/(2*2**d-3))*0.75)

    res = {}
    for full_device in device_list:
        print("\tRunning device " + full_device)
        device = "cuda" if "cuda" in full_device else "cpu"
        alg_type = "pytorch" if "pytorch" in full_device else "cpp"
        
        t = torch.tensor(data_t, device=torch.device(device))
        t2 = torch.tensor(data_t.copy(), device=torch.device(device))
        t.requires_grad = True
        t2.requires_grad = True
        w = torch.tensor(data_w, device=torch.device(device))
        rx = torch.tensor(data_rx, device=torch.device(device))
        rx2 = torch.tensor(data_rx.copy(), device=torch.device(device))
        rx.requires_grad = True
        rx2.requires_grad = True
        if d > 1:
            ry = torch.tensor(data_ry, device=torch.device(device))
            ry.requires_grad = True
            ry2 = torch.tensor(data_ry, device=torch.device(device))
            ry2.requires_grad = True
        if d > 2:
            rz = torch.tensor(data_rz, device=torch.device(device))
            rz.requires_grad = True
            rz2 = torch.tensor(data_rz, device=torch.device(device))
            rz2.requires_grad = True

        if d == 1:
            oa = torch.exp(Potts_MAP1d.apply(t,rx))
            if alg_type == "cpp":
                om = Potts_Mean1d.apply(t,rx)
            else:
                om = Potts_Mean1d_PytorchNative.apply(t,rx)
            ol = Potts_LBP1d_PytorchNative.apply(t2,rx2)
        elif d == 2:
            oa = torch.exp(Potts_MAP2d.apply(t,rx,ry))
            if alg_type == "cpp":
                om = Potts_Mean2d.apply(t,rx,ry)
            else:
                om = Potts_Mean2d_PytorchNative.apply(t,rx,ry)
            ol = Potts_LBP2d_PytorchNative.apply(t2,rx2,ry2)
        elif d == 3:
            oa = torch.exp(Potts_MAP3d.apply(t,rx,ry,rz))
            if alg_type == "cpp":
                om = Potts_Mean3d.apply(t,rx,ry,rz)
            else:
                om = Potts_Mean3d_PytorchNative.apply(t,rx,ry,rz)
            ol = Potts_LBP3d_PytorchNative.apply(t2,rx2,ry2,rz2)
            
        loss = torch.sum(w*om)
        loss.backward()
        loss = torch.sum(w*ol)
        loss.backward()

        oa_np = oa.detach().cpu().numpy()
        om_np = om.detach().cpu().numpy()
        ol_np = ol.detach().cpu().numpy()
        ot_np = t.grad.detach().cpu().numpy()
        ot2_np = t2.grad.detach().cpu().numpy()
        
        #make sure not nan
        asserter.assertFalse(np.any(np.isnan(oa_np)))
        asserter.assertFalse(np.any(np.isnan(om_np)))
        asserter.assertFalse(np.any(np.isnan(ol_np)))
        asserter.assertFalse(np.any(np.isnan(ot_np)))
        asserter.assertFalse(np.any(np.isnan(ot2_np)))
        res[full_device] = [om_np,ot_np,oa_np,ol_np,ot2_np]
        
        orx_np = rx.grad.detach().cpu().numpy()
        asserter.assertFalse(np.any(np.isnan(orx_np)))
        orx_np = rx.grad.detach().cpu().numpy()
        res[full_device].append(orx_np)
        if d > 1:
            ory_np = ry.grad.detach().cpu().numpy()
            asserter.assertFalse(np.any(np.isnan(ory_np)))
            ory_np = ry.grad.detach().cpu().numpy()
            res[full_device].append(ory_np)
        if d > 2:
            orz_np = rz.grad.detach().cpu().numpy() 
            asserter.assertFalse(np.any(np.isnan(orz_np)))
            orz_np = rz.grad.detach().cpu().numpy()
            res[full_device].append(orz_np)
                           

    name = ['om_np','ot_np','oa_np','orx_np','ol_np,ot2_np']
    for i,dev1 in enumerate(device_list):
        for dev2 in device_list[(i+1):]:
            diffs = [APD(res[dev1][j],res[dev2][j]) for j in range(len(name))]
            for var in range(len(name)):
                if(diffs[var] > epsilon or np.isnan(diffs[var])):
                    print(dev1,"\t\t",dev2)
                    for i1,i2 in zip(res[dev1][var].flatten(),res[dev2][var].flatten()):
                        print(i1,"\t",i2)
                    raise Exception(name[var]+"\t"+str(diffs[var]))

class Test_Extreme(unittest.TestCase):

    def test_no_smoothness_1D(self):
        print("")
        test_no_smoothness(1,"cpu",self)
        test_no_smoothness(1,"python_cpu",self)
        
    def test_no_smoothness_1D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():  
            test_no_smoothness(1,"cuda",self)
            test_no_smoothness(1,"python_cuda",self)
            
    def test_no_smoothness_2D(self):
        print("")
        test_no_smoothness(2,"cpu",self)
        test_no_smoothness(2,"python_cpu",self)

    def test_no_smoothness_2D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_no_smoothness(2,"cuda",self)
            test_no_smoothness(2,"python_cuda",self)

    def test_no_smoothness_3D(self):
        print("")
        test_no_smoothness(3,"cpu",self)
        test_no_smoothness(3,"python_cpu",self)
        
    def test_no_smoothness_3D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_no_smoothness(3,"cuda",self)
            test_no_smoothness(3,"python_cuda",self)

    def test_smoothness_dom_1D(self):
        print("")
        test_smoothness_dom(1,"cpu",self)
        test_smoothness_dom(1,"python_cpu",self)

    def test_smoothness_dom_1D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_smoothness_dom(1,"cuda",self)
            test_smoothness_dom(1,"python_cuda",self)

    def test_smoothness_dom_2D(self):
        print("")
        test_smoothness_dom(2,"cpu",self)
        test_smoothness_dom(2,"python_cpu",self)

    def test_smoothness_dom_2D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_smoothness_dom(2,"cuda",self)
            test_smoothness_dom(2,"python_cuda",self)

    def test_smoothness_dom_3D(self):
        print("")
        test_smoothness_dom(3,"cpu",self)
        test_smoothness_dom(3,"python_cpu",self)

    def test_smoothness_dom_3D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_smoothness_dom(3,"cuda",self)
            test_smoothness_dom(3,"python_cuda",self)
            
    def test_equivalence_1d(self):
        print("")
        if torch.backends.cuda.is_built():
            test_device_equivalence(1,["cpu","cuda","python_cpu","python_cuda"],self)
            
    def test_equivalence_2d(self):
        print("")
        if torch.backends.cuda.is_built():
            test_device_equivalence(2,["cpu","cuda","python_cpu","python_cuda"],self)
            
    def test_equivalence_3d(self):
        print("")
        if torch.backends.cuda.is_built():
            test_device_equivalence(3,["cpu","cuda","python_cpu","python_cuda"],self)
        
        


if __name__ == '__main__':
    unittest.main()
    
    
