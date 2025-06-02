import unittest

import numpy as np

import time

import torch
from binary_deepflow import Binary_MAP1d,Binary_MAP2d,Binary_MAP3d
from binary_deepflow import Binary_Mean1d,Binary_Mean2d,Binary_Mean3d
from binary_deepflow import Binary_Mean1d_PytorchNative,Binary_Mean2d_PytorchNative,Binary_Mean3d_PytorchNative
from binary_deepflow import Binary_LBP1d_PytorchNative, Binary_LBP2d_PytorchNative, Binary_LBP3d_PytorchNative

b=1
c=1
x=2**12
epsilon = 0.01

def APD(v1,v2):
    num = np.sum(np.abs(v1-v2))
    den = 0.5*np.sum(np.abs(v1)+np.abs(v2))
    return 0 if den == 0 else 100*num/den

def get_size_into(d):
    x_used = int(x**(1/d)+0.5)
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
        oa = torch.exp(Binary_MAP1d.apply(t,rx))
        if alg_type == "cpp":
            om = Binary_Mean1d.apply(t,rx)
        else:
            om = Binary_Mean1d_PytorchNative.apply(t,rx)
        ol = Binary_LBP1d_PytorchNative.apply(t2,rx2)
    elif d == 2:
        oa = torch.exp(Binary_MAP2d.apply(t,rx,ry))
        if alg_type == "cpp":
            om = Binary_Mean2d.apply(t,rx,ry)
        else:
            om = Binary_Mean2d_PytorchNative.apply(t,rx,ry)
        ol = Binary_LBP2d_PytorchNative.apply(t2,rx2,ry2)
    elif d == 3:
        oa = torch.exp(Binary_MAP3d.apply(t,rx,ry,rz))
        if alg_type == "cpp":
            om = Binary_Mean3d.apply(t,rx,ry,rz)
        else:
            om = Binary_Mean3d_PytorchNative.apply(t,rx,ry,rz)
        ol = Binary_LBP3d_PytorchNative.apply(t2,rx2,ry2,rz2)
    
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
        
    #ensure mean pass is equivalent to the data terms only
    for val_df, val_d in zip(om_np.flatten(),data_t.flatten()):
        if not (abs(val_df-val_d) < epsilon):
            print(data_t)
            print(om_np)
            asserter.assertTrue(APD(val_df,val_d) < epsilon)
    
    #ensure gradient wrt data terms are passed immediately through
    for val_df, val_d in zip(ot_np.flatten(),data_w.flatten()):
        if not (abs(val_df-val_d) < epsilon):
            print(data_w)
            print(ot_np)
            asserter.assertTrue(APD(val_df,val_d) < epsilon)
    for val_df, val_d in zip(ot2_np.flatten(),data_w.flatten()):
        if not (abs(val_df-val_d) < epsilon):
            print(data_w)
            print(ot_np)
            asserter.assertTrue(APD(val_df,val_d) < epsilon)
    
    #ensure MAP assigns 1 to positive terms
    for val_df, val_d in zip(oa_np.flatten(),data_t.flatten()):
        if(val_d > epsilon and val_df < 1-epsilon):
            print("oa",val_df, val_d, epsilon)
            print(data_t)
            print(oa_np)
            print(om_np)
            asserter.assertFalse(val_d > epsilon and val_df < 1-epsilon)
        if(val_d < -epsilon and val_df > epsilon):
            print("oa",val_df, val_d, epsilon)
            print(data_t)
            print(oa_np)
            print(om_np)
            asserter.assertFalse(val_d < -epsilon and val_df > epsilon)

def test_smoothness_dom(d,full_device,asserter):
    print("Testing (smoothness dom.) \t Dim: " +str(d)+ " \t Dev: " + full_device)
    device = "cuda" if "cuda" in full_device else "cpu"
    alg_type = "pytorch" if "pytorch" in full_device else "cpp"

    size_info, size_red_info, axes = get_size_into(d)

    data_t = 0.01*np.random.normal(0,1,size=size_info).astype(np.float32)
    data_t += epsilon / np.sum(data_t,axis=axes,keepdims=True)
    data_rx = 20+2*np.random.uniform(size=size_info).astype(np.float32)
    if d > 1:
        data_ry = 20+2*np.random.uniform(size=size_info).astype(np.float32)
    if d > 2:
        data_rz = 20+2*np.random.uniform(size=size_info).astype(np.float32)

    t = torch.tensor(data_t, device=torch.device(device))
    rx = torch.tensor(data_rx, device=torch.device(device))
    if d > 1:
        ry = torch.tensor(data_ry, device=torch.device(device))
    if d > 2:
        rz = torch.tensor(data_rz, device=torch.device(device))

    if d == 1:
        oa = torch.exp(Binary_MAP1d.apply(t,rx))
        if alg_type == "cpp":
            om = Binary_Mean1d.apply(t,rx)
        else:
            om = Binary_Mean1d_PytorchNative.apply(t,rx)
        ol = Binary_LBP1d_PytorchNative.apply(t,rx)
    elif d == 2:
        oa = torch.exp(Binary_MAP2d.apply(t,rx,ry))
        if alg_type == "cpp":
            om = Binary_Mean2d.apply(t,rx,ry)
        else:
            om = Binary_Mean2d_PytorchNative.apply(t,rx,ry)
        ol = Binary_LBP2d_PytorchNative.apply(t,rx,ry)
    elif d == 3:
        oa = torch.exp(Binary_MAP3d.apply(t,rx,ry,rz))
        if alg_type == "cpp":
            om = Binary_Mean3d.apply(t,rx,ry,rz)
        else:
            om = Binary_Mean3d_PytorchNative.apply(t,rx,ry,rz)
        ol = Binary_LBP3d_PytorchNative.apply(t,rx,ry,rz)
    oa_np = oa.detach().cpu().numpy()
    om_np = om.detach().cpu().numpy()
    ol_np = ol.detach().cpu().numpy()
    
    #make sure not nan
    asserter.assertFalse(np.any(np.isnan(oa_np)))
    asserter.assertFalse(np.any(np.isnan(om_np)))
    asserter.assertFalse(np.any(np.isnan(ol_np)))
    
    #ensure MAP assigns 1 to positive terms and mean shares sign with the sum of the data terms
    totals = np.sum(data_t,axis=axes)
    for i in range(c):
        for val_df in oa_np[0,i,...].flatten():
            if(totals[0,i] > x*epsilon/10 and val_df < 1-epsilon):
                print("oa",val_df, totals[0,i], x*epsilon/10)
                asserter.assertFalse(totals[0,i] > x*epsilon/10 and val_df < 1-epsilon)
            if(totals[0,i] < -x*epsilon/10 and val_df > epsilon):
                print("oa",val_df, totals[0,i], x*epsilon/10)
                asserter.assertFalse(totals[0,i] < -x*epsilon/10 and val_df > epsilon)
        for val_df in om_np[0,i,...].flatten():
            if(totals[0,i] > x*epsilon/10 and val_df < 1-epsilon):
                print("om",val_df, totals[0,i], x*epsilon/10)
                print(data_t)
                print(totals)
                print(oa_np)
                print(om_np)
                asserter.assertFalse(totals[0,i] > x*epsilon/10 and val_df < 1-epsilon)
            if(totals[0,i] < -x*epsilon/10 and val_df > epsilon):
                print("om",val_df, totals[0,i], x*epsilon/10)
                print(data_t)
                print(totals)
                print(oa_np)
                print(om_np)
                asserter.assertFalse(totals[0,i] < -x*epsilon/10 and val_df > epsilon)
        for val_df in ol_np[0,i,...].flatten():
            if(totals[0,i] > x*epsilon/10 and val_df < 1-epsilon):
                print("ol",val_df, totals[0,i], x*epsilon/10)
                print(data_t)
                print(totals)
                print(oa_np)
                print(ol_np)
                asserter.assertFalse(totals[0,i] > x*epsilon/10 and val_df < 1-epsilon)
            if(totals[0,i] < -x*epsilon/10 and val_df > epsilon):
                print("ol",val_df, totals[0,i], x*epsilon/10)
                print(data_t)
                print(totals)
                print(oa_np)
                print(ol_np)
                asserter.assertFalse(totals[0,i] < -x*epsilon/10 and val_df > epsilon)

def test_device_equivalence(d,device_list,asserter):
    print("Testing (dev equiv.) \t Dim: " +str(d)+ " \t Dev:",device_list)

    size_info, size_red_info, axes = get_size_into(d)

    data_t = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_rx = (np.random.uniform(size=size_info)+0.00001).astype(np.float32)/(2**(d+1))
    if d > 1:
        data_ry = (np.random.uniform(size=size_info)+0.00001).astype(np.float32)/(2**(d+1))
    if d > 2:
        data_rz= (np.random.uniform(size=size_info)+0.00001).astype(np.float32)/(2**(d+1))

    res = {}
    for full_device in device_list:
        print("\tRunning device " + full_device)
        device = "cuda" if "cuda" in full_device else "cpu"
        alg_type = "pytorch" if "pytorch" in full_device else "cpp"
        t = torch.tensor(data_t, device=torch.device(device))
        t2 = torch.tensor(data_t, device=torch.device(device))
        w = torch.tensor(data_w, device=torch.device(device))
        w2 = torch.tensor(data_w, device=torch.device(device))
        t.requires_grad = True
        t2.requires_grad = True
        
        rx = torch.tensor(data_rx, device=torch.device(device))
        rx.requires_grad = True
        rx2 = torch.tensor(data_rx, device=torch.device(device))
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
            oa = torch.exp(Binary_MAP1d.apply(t,rx))
            if alg_type == "cpp":
                om = Binary_Mean1d.apply(t,rx)
            else:
                om = Binary_Mean1d_PytorchNative.apply(t,rx)
            ol = Binary_LBP1d_PytorchNative.apply(t2,rx2)
        elif d == 2:
            oa = torch.exp(Binary_MAP2d.apply(t,rx,ry))
            if alg_type == "cpp":
                om = Binary_Mean2d.apply(t,rx,ry)
            else:
                om = Binary_Mean2d_PytorchNative.apply(t,rx,ry)
            ol = Binary_LBP2d_PytorchNative.apply(t2,rx2,ry2)
        elif d == 3:
            oa = torch.exp(Binary_MAP3d.apply(t,rx,ry,rz))
            if alg_type == "cpp":
                om = Binary_Mean3d.apply(t,rx,ry,rz)
            else:
                om = Binary_Mean3d_PytorchNative.apply(t,rx,ry,rz)
            ol = Binary_LBP3d_PytorchNative.apply(t2,rx2,ry2,rz2)
        loss = torch.sum(w*om)
        loss.backward()
        loss2 = torch.sum(w2*ol)
        loss2.backward()

        oa_np = oa.detach().cpu().numpy()
        om_np = om.detach().cpu().numpy()
        ol_np = ol.detach().cpu().numpy()
        ot_np = t.grad.detach().cpu().numpy()
        ot2_np = t2.grad.detach().cpu().numpy()
        
        #make sure not nan
        asserter.assertFalse(np.any(np.isnan(oa_np)))
        asserter.assertFalse(np.any(np.isnan(om_np)))
        asserter.assertFalse(np.any(np.isnan(ot_np)))
        asserter.assertFalse(np.any(np.isnan(ot2_np)))
        res[full_device] = [om_np,ot_np,ol_np,ot2_np,oa_np]
        
        orx_np = rx.grad.detach().cpu().numpy()
        asserter.assertFalse(np.any(np.isnan(orx_np)))
        res[full_device].append(orx_np)
        orx2_np = rx2.grad.detach().cpu().numpy()
        asserter.assertFalse(np.any(np.isnan(orx2_np)))
        res[full_device].append(orx2_np)
        if d > 1:
            ory_np = ry.grad.detach().cpu().numpy()
            asserter.assertFalse(np.any(np.isnan(ory_np)))
            res[full_device].append(ory_np)
            ory2_np = ry2.grad.detach().cpu().numpy()
            asserter.assertFalse(np.any(np.isnan(ory2_np)))
            res[full_device].append(ory2_np)
        if d > 2:
            orz_np = rz.grad.detach().cpu().numpy() 
            asserter.assertFalse(np.any(np.isnan(orz_np)))
            res[full_device].append(orz_np)
            orz2_np = rz2.grad.detach().cpu().numpy() 
            asserter.assertFalse(np.any(np.isnan(orz2_np)))
            res[full_device].append(orz2_np)
                           

    name = ['om_np','ot_np','ol_np','ot2_np','oa_np','orx_np']
    if d > 1:
        name += ['ory_np','ory2_np']
    if d > 2:
        name += ['orz_np','orz2_np']
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
        
    def test_no_smoothness_2D(self):
        print("")
        test_no_smoothness(2,"cpu",self)
        test_no_smoothness(2,"python_cpu",self)

    def test_no_smoothness_3D(self):
        print("")
        test_no_smoothness(3,"cpu",self)
        test_no_smoothness(3,"python_cpu",self)

    def test_smoothness_1D(self):
        print("")
        test_smoothness_dom(1,"cpu",self)
        test_smoothness_dom(1,"python_cpu",self)

    def test_smoothness_2D(self):
        print("")
        test_smoothness_dom(2,"cpu",self)
        test_smoothness_dom(2,"python_cpu",self)

    def test_smoothness_3D(self):
        print("")
        test_smoothness_dom(3,"cpu",self)
        test_smoothness_dom(3,"python_cpu",self)
        
    def test_no_smoothness_1D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_no_smoothness(1,"cuda",self)
            test_no_smoothness(1,"python_cuda",self)

    def test_no_smoothness_2D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_no_smoothness(2,"cuda",self)
            test_no_smoothness(2,"python_cuda",self)

    def test_no_smoothness_3D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_no_smoothness(3,"cuda",self)
            test_no_smoothness(3,"python_cuda",self)

    def test_smoothness_1D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_smoothness_dom(1,"cuda",self)
            test_smoothness_dom(1,"python_cuda",self)

    def test_smoothness_2D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_smoothness_dom(2,"cuda",self)
            test_smoothness_dom(2,"python_cuda",self)

    def test_smoothness_3D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_smoothness_dom(3,"cuda",self)
            test_smoothness_dom(3,"python_cuda",self)
    
    def test_equivalence_1d(self):
        print("")
        test_device_equivalence(1,["cuda","cpu","python_cuda","python_cpu"],self);
        
    def test_equivalence_2d(self):
        print("")
        test_device_equivalence(2,["cuda","cpu","python_cuda","python_cpu"],self);
        
    def test_equivalence_3d(self):
        print("")
        test_device_equivalence(3,["cuda","cpu","python_cuda","python_cpu"],self);

if __name__ == '__main__':
    unittest.main()
    
    
