import unittest

import numpy as np

import time

import torch
from potts_deepflow import Potts_MAP1d,Potts_MAP2d,Potts_MAP3d
from potts_deepflow import Potts_Mean1d,Potts_Mean2d,Potts_Mean3d

b=1
c=3
x=2**12
epsilon = 0.01

def get_size_into(d):
    x_used = int(x**(1/d)+0.001)
    return tuple( [b,c]+[x_used for i in range(d)] ), tuple( [1,c]+[x_used for i in range(d)] ), tuple([i+2 for i in range(d)])

def test_no_smoothness(d,device,asserter):
    print("Testing (no smoothness) \t Dim: " +str(d)+ " \t Dev: " + device)

    size_info, size_red_info, axes = get_size_into(d)

    data_t = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_rx = np.zeros(shape=size_info).astype(np.float32)
    if d > 1:
        data_ry = np.zeros(shape=size_info).astype(np.float32)
    if d > 2:
        data_rz = np.zeros(shape=size_info).astype(np.float32)

    t = torch.tensor(data_t, device=torch.device(device))
    t.requires_grad = True
    w = torch.tensor(data_w, device=torch.device(device))
    rx = torch.tensor(data_rx, device=torch.device(device))
    rx.requires_grad = True
    if d > 1:
        ry = torch.tensor(data_ry, device=torch.device(device))
        ry.requires_grad = True
    if d > 2:
        rz = torch.tensor(data_rz, device=torch.device(device))
        rz.requires_grad = True

    if d == 1:
        oa = torch.exp(Potts_MAP1d.apply(t,rx))
        om = Potts_Mean1d.apply(t,rx)
    elif d == 2:
        oa = torch.exp(Potts_MAP2d.apply(t,rx,ry))
        om = Potts_Mean2d.apply(t,rx,ry)
    elif d == 3:
        oa = torch.exp(Potts_MAP3d.apply(t,rx,ry,rz))
        om = Potts_Mean3d.apply(t,rx,ry,rz)
    loss = torch.sum(w*om)
    loss.backward()
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    ot_np = t.grad.detach().cpu().numpy()
    
    #make sure not nan
    asserter.assertFalse(np.any(np.isnan(oa_np)))
    asserter.assertFalse(np.any(np.isnan(om_np)))
    asserter.assertFalse(np.any(np.isnan(ot_np)))
    
    #resize into more usable form
    dt_np_l = [data_t[0,i,...].flatten() for i in range(c)]
    dw_np_l = [data_w[0,i,...].flatten() for i in range(c)]
    oa_np_l = [oa_np[0,i,...].flatten()  for i in range(c)]
    om_np_l = [om_np[0,i,...].flatten()  for i in range(c)]
    ot_np_l = [ot_np[0,i,...].flatten()  for i in range(c)]
    x_space = len(dt_np_l[0])
    
    #ensure MAP assigns 1 to highest term and 0 to everything else
    for i in range(x_space):
        highest = max([o[i] for o in dt_np_l])
        for ic in range(c):
            if(dt_np_l[ic][i] == highest and oa_np_l[ic][i] < 0.5):
                raise Exception(str(dt_np_l[ic][i])+"\t"+str([o[i] for o in dt_np_l])+"\t"+str(highest)+"\t"+str(oa_np_l[ic][i])+"\t"+str([o[i] for o in oa_np_l]))
            if(dt_np_l[ic][i] < highest - epsilon  and oa_np_l[ic][i] > 0.5):
                raise Exception(str(dt_np_l[ic][i])+"\t"+str([o[i] for o in dt_np_l])+"\t"+str(highest)+"\t"+str(oa_np_l[ic][i])+"\t"+str([o[i] for o in oa_np_l]))
        
    #ensure mean pass is equivalent to the data terms only
    for i in range(c): 
        for val_df, val_d in zip(om_np_l[i],dt_np_l[i]):
            if(abs(val_df-val_d) > epsilon):
                raise Exception(str(val_df) + "\t" + str(val_d))
    
    #ensure gradient wrt data terms are passed immediately through
    for i in range(c): 
        for val_df, val_d in zip(ot_np_l[i],dw_np_l[i]):
            if(abs(val_df-val_d) > epsilon):
                raise Exception(str(val_df) + "\t" + str(val_d))

def test_smoothness_dom(d,device,asserter):
    print("Testing (smoothness dom.) \t Dim: " +str(d)+ " \t Dev: " + device)

    size_info, size_red_info, axes = get_size_into(d)

    winner = int(np.random.uniform()*c)
    data_t = 1*np.random.uniform(0,1,size=size_info).astype(np.float32)
    data_t[:,winner,...] = 0.75
    data_r = 100+0*np.random.uniform(size=size_info).astype(np.float32)

    t = torch.tensor(data_t, device=torch.device(device))
    r = torch.tensor(data_r, device=torch.device(device))
    
    if d == 1:
        oa = torch.exp(Potts_MAP1d.apply(t,r))
        om = Potts_Mean1d.apply(t,r)
    elif d == 2:
        oa = torch.exp(Potts_MAP2d.apply(t,r,r))
        om = Potts_Mean2d.apply(t,r,r)
    elif d == 3:
        oa = torch.exp(Potts_MAP3d.apply(t,r,r,r))
        om = Potts_Mean3d.apply(t,r,r,r)
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    
    #make sure not nan
    asserter.assertFalse(np.any(np.isnan(oa_np)))
    asserter.assertFalse(np.any(np.isnan(om_np)))
    
    #resize into more usable form
    dt_np_l = [data_t[0,i,...].flatten() for i in range(c)]
    oa_np_l = [oa_np[0,i,...].flatten()  for i in range(c)]
    om_np_l = [om_np[0,i,...].flatten()  for i in range(c)]
    x_space = len(dt_np_l[0])
    
    #ensure MAP assigns 1 to highest terms
    sums = [np.sum(o) for o in dt_np_l]
    highest = max(sums)
    for i in range(c):
        for val_df in oa_np_l[i]:
            if(sums[i] == highest and val_df < 0.5):
                raise Exception(str(sums[i])+ "\t" + str(sums) + "\t" + str(val_df))
            if(sums[i] < highest  and val_df > 0.5):
                raise Exception(str(sums[i])+ "\t" + str(sums) + "\t" + str(val_df))

def test_device_equivalence(d,device_list,asserter):
    print("Testing (dev equiv.) \t Dim: " +str(d)+ " \t Dev:",device_list)

    size_info, size_red_info, axes = get_size_into(d)

    data_t = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_rx = (0.25*np.random.uniform(size=size_info)+0.00001).astype(np.float32)
    if d > 1:
        data_ry = (0.25*np.random.uniform(size=size_info)+0.00001).astype(np.float32)
    if d > 2:
        data_rz= (0.25*np.random.uniform(size=size_info)+0.00001).astype(np.float32)
        
    res = {}
    for device in device_list:
        print("Running device \t"+device)
        t = torch.tensor(data_t, device=torch.device(device))
        w = torch.tensor(data_w, device=torch.device(device))
        t.requires_grad = True
        
        rx = torch.tensor(data_rx, device=torch.device(device))
        rx.requires_grad = True
        if d > 1:
            ry = torch.tensor(data_ry, device=torch.device(device))
            ry.requires_grad = True
        if d > 2:
            rz = torch.tensor(data_rz, device=torch.device(device))
            rz.requires_grad = True
            
        if d == 1:
            oa = torch.exp(Potts_MAP1d.apply(t,rx))
            om = Potts_Mean1d.apply(t,rx)
        elif d == 2:
            oa = torch.exp(Potts_MAP2d.apply(t,rx,ry))
            om = Potts_Mean2d.apply(t,rx,ry)
        elif d == 3:
            oa = torch.exp(Potts_MAP3d.apply(t,rx,ry,rz))
            om = Potts_Mean3d.apply(t,rx,ry,rz)
        loss = torch.sum(w*om)
        loss.backward()

        oa_np = oa.detach().cpu().numpy()
        om_np = om.detach().cpu().numpy()
        ot_np = t.grad.detach().cpu().numpy()
        
        #make sure not nan
        asserter.assertFalse(np.any(np.isnan(oa_np)))
        asserter.assertFalse(np.any(np.isnan(om_np)))
        asserter.assertFalse(np.any(np.isnan(ot_np)))
        res[device] = [om_np,ot_np,oa_np]
        
        orx_np = rx.grad.detach().cpu().numpy()
        asserter.assertFalse(np.any(np.isnan(orx_np)))
        orx_np = rx.grad.detach().cpu().numpy()
        res[device].append(orx_np)
        if d > 1:
            ory_np = ry.grad.detach().cpu().numpy()
            asserter.assertFalse(np.any(np.isnan(ory_np)))
            ory_np = ry.grad.detach().cpu().numpy()
            res[device].append(ory_np)
        if d > 2:
            orz_np = rz.grad.detach().cpu().numpy() 
            asserter.assertFalse(np.any(np.isnan(orz_np)))
            orz_np = rz.grad.detach().cpu().numpy()
            res[device].append(orz_np)
                           

    name = ['om_np','ot_np','oa_np','orx_np']
    epsilon_l = [0.1*epsilon,0.1*epsilon,0.1,0.1*epsilon]
    if d > 1:
        name.append('ory_np')
        epsilon_l.append(0.1*epsilon)
    if d > 2:
        name.append('orz_np')
        epsilon_l.append(0.1*epsilon)
    for i,dev1 in enumerate(device_list):
        for dev2 in device_list[(i+1):]:
            diffs = [np.max(abs(res[dev1][j]-res[dev2][j])) for j in range(len(name))]
            for var in range(len(name)):
                if(diffs[var] > epsilon_l[var] or np.isnan(diffs[var])):
                    raise Exception(name[var]+"\t"+str(diffs[var]))

class Test_Extreme(unittest.TestCase):

    def test_no_smoothness_1D(self):
        print("")
        test_no_smoothness(1,"cpu",self)
        if torch.has_cuda:  
            test_no_smoothness(1,"cuda",self)

    def test_no_smoothness_2D(self):
        print("")
        if torch.has_cuda:
            test_no_smoothness(2,"cuda",self)
        test_no_smoothness(2,"cpu",self)

    def test_no_smoothness_3D(self):
        print("")
        if torch.has_cuda:
            test_no_smoothness(3,"cuda",self)
        test_no_smoothness(3,"cpu",self)

    def test_smoothness_dom_1D(self):
        print("")
        if torch.has_cuda:
            test_smoothness_dom(1,"cuda",self)
        test_smoothness_dom(1,"cpu",self)

    def test_smoothness_dom_2D(self):
        print("")
        if torch.has_cuda:
            test_smoothness_dom(2,"cuda",self)
        test_smoothness_dom(2,"cpu",self)

    def test_smoothness_dom_3D(self):
        print("")
        #if torch.has_cuda:
        #    test_smoothness_dom(3,"cuda",self)
        #test_smoothness_dom(3,"cpu",self)
            
    def test_equivalence_1d(self):
        print("")
        if torch.has_cuda:
            test_device_equivalence(1,["cpu","cuda"],self)
            
    def test_equivalence_2d(self):
        print("")
        if torch.has_cuda:
            test_device_equivalence(2,["cpu","cuda"],self)
            
    def test_equivalence_3d(self):
        print("")
        if torch.has_cuda:
            test_device_equivalence(3,["cpu","cuda"],self)
        
        


if __name__ == '__main__':
    unittest.main()
    
    
