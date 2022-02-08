import unittest

import numpy as np

import time

import torch
from hmf_deepflow import HMF_MAP1d,HMF_MAP2d,HMF_MAP3d
from hmf_deepflow import HMF_Mean1d,HMF_Mean2d,HMF_Mean3d

b=1
l=3
c=2**l
br=2**(l+1)-2
x=2**18
epsilon = 0.000000001



def get_size_into(d, device):
    x_used = int(x**(1/d)+0.001)
    
    parentage = np.zeros(shape=(br,)).astype(np.int32)
    for i in range(br):
        parentage[i] = i//2-1
    parentage = torch.tensor(parentage, device=torch.device(device))
    
    return tuple( [b,c]+[x_used for i in range(d)] ), tuple( [b,br]+[x_used for i in range(d)] ), \
           tuple( [1,c]+[x_used for i in range(d)] ), tuple( [1,br]+[x_used for i in range(d)] ), \
           tuple([i+2 for i in range(d)]), parentage,
            

def test_no_smoothness(d,device,asserter):
    print("Testing (no smoothness) \t Dim: " +str(d)+ " \t Dev: " + device)

    size_info_d, size_info_r, size_red_info_d, size_red_info_r, axes, parentage = get_size_into(d, device)

    data_t = np.random.normal(0,1,size=size_info_d).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info_d).astype(np.float32)
    data_rx = np.zeros(shape=size_info_r).astype(np.float32)
    if d > 1:
        data_ry = np.zeros(shape=size_info_r).astype(np.float32)
    if d > 2:
        data_rz = np.zeros(shape=size_info_r).astype(np.float32)

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
        oa = torch.exp(HMF_MAP1d.apply(t,rx,parentage))
        print("Finished MAP")
        om = HMF_Mean1d.apply(t,rx,parentage)
        print("Finished Mean")
    elif d == 2:
        oa = torch.exp(HMF_MAP2d.apply(t,rx,ry,parentage))
        om = HMF_Mean2d.apply(t,rx,ry,parentage)
    elif d == 3:
        oa = torch.exp(HMF_MAP3d.apply(t,rx,ry,rz,parentage))
        om = HMF_Mean3d.apply(t,rx,ry,rz,parentage)
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














class Test_Extreme(unittest.TestCase):

    def test_no_smoothness_1D(self):
        print("")
        test_no_smoothness(1,"cpu",self)
        if torch.has_cuda:  
            test_no_smoothness(1,"cuda",self)

    def test_no_smoothness_2D(self):
        print("")
        test_no_smoothness(2,"cpu",self)
        if torch.has_cuda:
            test_no_smoothness(2,"cuda",self)

    def test_no_smoothness_3D(self):
        print("")
        test_no_smoothness(3,"cpu",self)
        if torch.has_cuda:
            test_no_smoothness(3,"cuda",self)

    #def test_smoothness_dom_1D(self):
    #    print("")
    #    if torch.has_cuda:
    #        test_smoothness_dom(1,"cuda",self)
    #    test_smoothness_dom(1,"cpu",self)

    #def test_smoothness_dom_2D(self):
    #    print("")
    #    if torch.has_cuda:
    #        test_smoothness_dom(2,"cuda",self)
    #    test_smoothness_dom(2,"cpu",self)

    #def test_smoothness_dom_3D(self):
    #    print("")
    #    if torch.has_cuda:
    #        test_smoothness_dom(3,"cuda",self)
    #    test_smoothness_dom(3,"cpu",self)
            
    #def test_equivalence_1d(self):
    #    print("")
    #    if torch.has_cuda:
    #        test_device_equivalence(1,["cpu","cuda"],self)
            
    #def test_equivalence_2d(self):
    #    print("")
    #    if torch.has_cuda:
    #        test_device_equivalence(2,["cpu","cuda"],self)
            
    #def test_equivalence_3d(self):
    #    print("")
    #    if torch.has_cuda:
    #        test_device_equivalence(3,["cpu","cuda"],self)
        
        


if __name__ == '__main__':
    unittest.main()
    
    

