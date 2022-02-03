import unittest

import numpy as np

import time

import torch
from binary_deepflow import Binary_MAP1d,Binary_MAP2d,Binary_MAP3d
from binary_deepflow import Binary_Mean1d,Binary_Mean2d,Binary_Mean3d

b=1
c=4
x=4096
epsilon = 0.01

def test_no_smoothness(d,device,asserter):
    print("Testing (no smoothness) \t Dim: " +str(d)+ " \t Dev: " + device)
    x_used = int(x**(1/d))

    size_info = tuple( [b,c]+[x_used for i in range(d)] )
    size_red_info = tuple( [1,c]+[x_used for i in range(d)] )

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
        oa = torch.exp(Binary_MAP1d.apply(t,rx))
        om = Binary_Mean1d.apply(t,rx)
    elif d == 2:
        oa = torch.exp(Binary_MAP2d.apply(t,rx,ry))
        om = Binary_Mean2d.apply(t,rx,ry)
    elif d == 3:
        oa = torch.exp(Binary_MAP3d.apply(t,rx,ry,rz))
        om = Binary_Mean3d.apply(t,rx,ry,rz)
    loss = torch.sum(w*om)
    loss.backward()
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    ot_np = t.grad.detach().cpu().numpy()
    
    #make sure not nan
    asserter.assertFalse(np.any(np.isnan(oa_np)))
    asserter.assertFalse(np.any(np.isnan(om_np)))
    asserter.assertFalse(np.any(np.isnan(ot_np)))
    
    #ensure MAP assigns 1 to positive terms
    for val_df, val_d in zip(oa_np.flatten(),data_t.flatten()):
        asserter.assertFalse(val_d > epsilon and val_df < 1-epsilon)
        asserter.assertFalse(val_d < -epsilon and val_df > epsilon)
        
    #ensure mean pass is equivalent to the data terms only
    for val_df, val_d in zip(om_np.flatten(),data_t.flatten()):
        asserter.assertTrue(abs(val_df-val_d) < epsilon)
    
    #ensure gradient wrt data terms are passed immediately through
    for val_df, val_d in zip(ot_np.flatten(),data_w.flatten()):
        asserter.assertTrue(abs(val_df-val_d) < epsilon)

def test_smoothness_dom(d,device,asserter):
    print("Testing (smoothness dom.) \t Dim: " +str(d)+ " \t Dev: " + device)
    x_used = int(x**(1/d))

    size_info = tuple( [b,c]+[x_used for i in range(d)] )
    size_red_info = tuple( [1,c]+[x_used for i in range(d)] )
    axes = tuple([i+2 for i in range(d)])

    data_t = 0.1*np.random.normal(0,1,size=size_info).astype(np.float32)
    data_t += epsilon / np.sum(data_t,axis=axes,keepdims=True)
    data_rx = 250+25*np.random.uniform(size=size_info).astype(np.float32)
    if d > 1:
        data_ry = 250+25*np.random.uniform(size=size_info).astype(np.float32)
    if d > 2:
        data_rz = 250+25*np.random.uniform(size=size_info).astype(np.float32)

    t = torch.tensor(data_t, device=torch.device(device))
    rx = torch.tensor(data_rx, device=torch.device(device))
    if d > 1:
        ry = torch.tensor(data_ry, device=torch.device(device))
    if d > 2:
        rz = torch.tensor(data_rz, device=torch.device(device))
        
    
    if d == 1:
        oa = torch.exp(Binary_MAP1d.apply(t,rx))
        om = Binary_Mean1d.apply(t,rx)
    elif d == 2:
        oa = torch.exp(Binary_MAP2d.apply(t,rx,ry))
        om = Binary_Mean2d.apply(t,rx,ry)
    elif d == 3:
        oa = torch.exp(Binary_MAP3d.apply(t,rx,ry,rz))
        om = Binary_Mean3d.apply(t,rx,ry,rz)
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    
    #make sure not nan
    asserter.assertFalse(np.any(np.isnan(oa_np)))
    asserter.assertFalse(np.any(np.isnan(om_np)))
    
    #ensure MAP assigns 1 to positive terms and mean shares sign with the sum of the data terms
    totals = np.sum(data_t,axis=axes)
    for i in range(c):
        for val_df in oa_np[0,i,...].flatten():
            asserter.assertFalse(totals[0,i] > epsilon and val_df < 1-epsilon)
            asserter.assertFalse(totals[0,i] < -epsilon and val_df > epsilon)
        for val_df in om_np[0,i,...].flatten():
            asserter.assertFalse(totals[0,i] > epsilon and val_df < -epsilon)
            asserter.assertFalse(totals[0,i] < -epsilon and val_df > epsilon)

def test_device_equivalence(d,device_list,asserter):
    print("Testing (dev equiv.) \t Dim: " +str(d)+ " \t Dev:",device_list)
    x_used = int(x**(1/d))

    size_info = tuple( [b,c]+[x_used for i in range(d)] )
    size_red_info = tuple( [1,c]+[x_used for i in range(d)] )

    data_t = 0*np.random.normal(0,1,size=size_info).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_rx = (0.25*np.random.uniform(size=size_info)+0.00001).astype(np.float32)
    if d > 1:
        data_ry = (0.25*np.random.uniform(size=size_info)+0.00001).astype(np.float32)
    if d > 2:
        data_rz= (0.25*np.random.uniform(size=size_info)+0.00001).astype(np.float32)

    res = {}
    for device in device_list:
        print("\tRunning device " + device)
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
            oa = torch.exp(Binary_MAP1d.apply(t,rx))
            om = Binary_Mean1d.apply(t,rx)
        elif d == 2:
            oa = torch.exp(Binary_MAP2d.apply(t,rx,ry))
            om = Binary_Mean2d.apply(t,rx,ry)
        elif d == 3:
            oa = torch.exp(Binary_MAP3d.apply(t,rx,ry,rz))
            om = Binary_Mean3d.apply(t,rx,ry,rz)
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
        if torch.has_cuda:
            test_no_smoothness(1,"cuda",self)
        test_no_smoothness(1,"cpu",self)

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
        if torch.has_cuda:
            test_smoothness_dom(3,"cuda",self)
        test_smoothness_dom(3,"cpu",self)
            
    def test_equivalence_1d(self):
        print("")
        if torch.has_cuda:
            test_device_equivalence(1,["cuda","cpu"],self)
            
    def test_equivalence_2d(self):
        print("")
        if torch.has_cuda:
            test_device_equivalence(2,["cuda","cpu"],self)
            
    def test_equivalence_3d(self):
        print("")
        if torch.has_cuda:
            test_device_equivalence(3,["cuda","cpu"],self)
        
        


if __name__ == '__main__':
    unittest.main()
    
    
