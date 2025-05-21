import unittest

import numpy as np

import time

import itertools
from itertools import product

import torch
from hmf_deepflow import HMF_MAP1d,HMF_MAP2d,HMF_MAP3d
from hmf_deepflow import HMF_Mean1d,HMF_Mean2d,HMF_Mean3d

b=1
l=3
c=2**l
br=2**(l+1)-2
x1 = 256
x2 = 256
x3 = 64
epsilon = 0.01



def get_size_into(d, device):
    if d == 1:
        x_used = x1
    elif d == 2:
        x_used = x2
    else:
        x_used = x3
    
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
        om = HMF_Mean1d.apply(t,rx,parentage)
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


def test_smoothness_dom(d,device,asserter):
    print("Testing (smoothness dom.) \t Dim: " +str(d)+ " \t Dev: " + device)

    size_info_d, size_info_r, size_red_info_d, size_red_info_r, axes, parentage = get_size_into(d, device)

    data_t = 1*np.random.uniform(0,1,size=size_info_d).astype(np.float32)
    sums = [np.sum(data_t[:,i,...]) for i in range(c)]
    winner = sums.index(max(sums))
    data_t[:,winner,...] += 0.05*np.random.uniform(0,1,size=data_t[:,winner,...].shape)
    data_rx = (20*max(size_info_d[2:(2+d)])*x1+np.zeros(shape=size_info_r)).astype(np.float32)
    if d > 1:
        data_ry = (20*max(size_info_d[2:(2+d)])*x1+np.zeros(shape=size_info_r)).astype(np.float32)
    if d > 2:
        data_rz = (20*max(size_info_d[2:(2+d)])*x1+np.zeros(shape=size_info_r)).astype(np.float32)

    t = torch.tensor(data_t, device=torch.device(device))
    t.requires_grad = True
    rx = torch.tensor(data_rx, device=torch.device(device))
    rx.requires_grad = True
    if d > 1:
        ry = torch.tensor(data_ry, device=torch.device(device))
    if d > 2:
        rz = torch.tensor(data_rz, device=torch.device(device))

    if d == 1:
        oa = torch.exp(HMF_MAP1d.apply(t,rx,parentage))
        om = HMF_Mean1d.apply(t,rx,parentage)
    elif d == 2:
        oa = torch.exp(HMF_MAP2d.apply(t,rx,ry,parentage))
        om = HMF_Mean2d.apply(t,rx,ry,parentage)
    elif d == 3:
        oa = torch.exp(HMF_MAP3d.apply(t,rx,ry,rz,parentage))
        om = HMF_Mean3d.apply(t,rx,ry,rz,parentage)
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


# Has high smoothness for upper levels and no smoothness for lower ones
def test_mixed_smoothness(levels,d,device,asserter):
    print("Testing (mixed smoothness) \t Dim: " +str(d)+ "\t Level: " + str(levels) + " \t Dev: " + device)

    size_info_d, size_info_r, size_red_info_d, size_red_info_r, axes, parentage = get_size_into(d, device)

    #first crack at making data terms
    data_t = np.random.uniform(0,1,size=size_info_d).astype(np.float32)
    
    # find the 'winning' subtree
    highest_idx = [data_t[0,i,...].flatten()  for i in range(c)]
    sums = [[np.sum(o) for o in highest_idx]]
    while(len(sums[0]) > 2):
        length = len(sums[0])//2
        highest_idx = [np.maximum(highest_idx[2*i],highest_idx[2*i+1]) for i in range(length)]
        sums = [[np.sum(o) for o in highest_idx]] + sums
    highest = max(sums[levels-1])
    winner_subtree = (sums[levels-1].index(highest))
    winner_subtree = list(range(winner_subtree*(2**(l-levels)),(winner_subtree+1)*(2**(l-levels))))
    
    #make data terms have a distinctive winner in the winner subtree
    data_t[:,winner_subtree,...] += 0.5*np.random.uniform(0,1,size=data_t[:,winner_subtree,...].shape)
    boost = 0.1
    if d == 1:
        for i_b,i_x in product(range(b),range(size_info_d[2])):
            winner = winner_subtree[np.argmax(data_t[i_b,winner_subtree,i_x])]
            #winner_subtree[int(np.random.uniform()*len(winner_subtree))]
            #data_t[i_b,winner_subtree,i_x] -= boost/len(winner_subtree)
            data_t[i_b,winner,i_x] += boost
    elif d == 2:
        for i_b,i_x,i_y in product(range(b),range(size_info_d[2]),range(size_info_d[3])):
            winner = winner_subtree[np.argmax(data_t[i_b,winner_subtree,i_x,i_y])]
            #winner = winner_subtree[int(np.random.uniform()*len(winner_subtree))]
            #data_t[i_b,winner_subtree,i_x,i_y] -= boost/len(winner_subtree)
            data_t[i_b,winner,i_x,i_y] += boost
    elif d == 3:
        for i_b,i_x,i_y,i_z in product(range(b),range(size_info_d[2]),range(size_info_d[3]),range(size_info_d[4])):
            winner = winner_subtree[np.argmax(data_t[i_b,winner_subtree,i_x,i_y,i_z])]
            #winner = winner_subtree[int(np.random.uniform()*len(winner_subtree))]
            #data_t[i_b,winner_subtree,i_x,i_y,i_z] -= boost/len(winner_subtree)
            data_t[i_b,winner,i_x,i_y,i_z] += boost
        
    #add in high smoothness terms
    data_rx = np.zeros(shape=size_info_r).astype(np.float32)
    if d > 1:
        data_ry = np.zeros(shape=size_info_r).astype(np.float32)
    if d > 2:
        data_rz = np.zeros(shape=size_info_r).astype(np.float32)
    if l > 0:
        start_from_term = 0
        up_to_term = 2**(levels+1)-2
        data_rx[:,start_from_term:up_to_term,...] += 20*max(size_info_d[2:(2+d)])
        if d > 1:
            data_ry[:,start_from_term:up_to_term,...] += 20*max(size_info_d[2:(2+d)])
        if d > 2:
            data_ry[:,start_from_term:up_to_term,...] += 20*max(size_info_d[2:(2+d)])
    
    t = torch.tensor(data_t, device=torch.device(device))
    rx = torch.tensor(data_rx, device=torch.device(device))
    if d > 1:
        ry = torch.tensor(data_ry, device=torch.device(device))
    if d > 2:
        rz = torch.tensor(data_rz, device=torch.device(device))

    if d == 1:
        oa = torch.exp(HMF_MAP1d.apply(t,rx,parentage))
        om = HMF_Mean1d.apply(t,rx,parentage)
    elif d == 2:
        oa = torch.exp(HMF_MAP2d.apply(t,rx,ry,parentage))
        om = HMF_Mean2d.apply(t,rx,ry,parentage)
    elif d == 3:
        oa = torch.exp(HMF_MAP3d.apply(t,rx,ry,rz,parentage))
        om = HMF_Mean3d.apply(t,rx,ry,rz,parentage)
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
    
    #ensure MAP assigns 0 to terms not in the 'best' tree
    highest_idx = [data_t[0,i,...].flatten()  for i in range(c)]
    sums = [[np.sum(o) for o in highest_idx]]
    while(len(sums[0]) > 2):
        length = len(sums[0])//2
        highest_idx = [np.maximum(highest_idx[2*i],highest_idx[2*i+1]) for i in range(length)]
        sums = [[np.sum(o) for o in highest_idx]] + sums
    highest = max(sums[levels-1])
    subtree = (sums[levels-1].index(highest))
    subtree = list(range(subtree*(2**(l-levels)),(subtree+1)*(2**(l-levels))))
    
    #find best performer of valid sub-trees
    best_performer = dt_np_l[subtree[0]].copy()
    for i in subtree[1:]:
        best_performer = np.maximum(best_performer,dt_np_l[i])
    
    for i in range(c):
        if i in subtree:
            for e, (val_t,val_df,val_bf) in enumerate(zip(dt_np_l[i],oa_np_l[i],best_performer)):
                if(val_t < val_bf-epsilon and val_df > 0.5):
                    raise Exception("Item (" + str(i) + ") in chosen subtree ("+str(winner_subtree)+"), definitely not best item, u is non-neglibile \n" + str([o[e] for o in oa_np_l])+ "\n" + str([o[e] for o in dt_np_l])+ "\n" + str(sums))
                elif(val_t >= val_bf and val_df < 0.5 ):
                    raise Exception("Item (" + str(i) + ") in chosen subtree ("+str(winner_subtree)+") and is best item ("+ str(val_t) +"), but u is too low \n" + str([o[e] for o in oa_np_l]) + "\n" + str([o[e] for o in dt_np_l])+ "\n" + str(sums))
        else:
            for e, val_df in enumerate(oa_np_l[i]):
                if(val_df > 0.5):
                    raise Exception("Item (" + str(i) + ") not in chosen subtree ("+str(winner_subtree)+") but has non-negligible u \n" + str([o[e] for o in oa_np_l])+ "\n" + str([o[e] for o in dt_np_l])+ "\n" + str(sums))



                    
                    
def test_device_equivalence(d,device_list,asserter):
    print("Testing (dev equiv.) \t Dim: " +str(d)+ " \t Dev:",device_list)

    size_info_d, size_info_r, size_red_info_d, size_red_info_r, axes, p = get_size_into(d, 'cpu')

    data_t = np.random.uniform(0,1,size=size_info_d).astype(np.float32)
    data_w = np.random.uniform(0,1,size=size_info_d).astype(np.float32)
    data_rx = ((0.12495/(2*d*l))*np.random.uniform(size=size_info_r)+0.000001).astype(np.float32)
    if d > 1:
        data_ry = ((0.12495/(2*d*l))*np.random.uniform(size=size_info_r)+0.000001).astype(np.float32)
    if d > 2:
        data_rz= ((0.12495/(2*d*l))*np.random.uniform(size=size_info_r)+0.000001).astype(np.float32)
        
    res = {}
    for device in device_list:
        print("\tRunning device " + device)
        t = torch.tensor(data_t, device=torch.device(device))
        w = torch.tensor(data_w, device=torch.device(device))
        p = p.to(t.device)
        t.requires_grad = True
        w.requires_grad = False
        p.requires_grad = False
        
        rx = torch.tensor(data_rx, device=torch.device(device))
        rx.requires_grad = True
        if d > 1:
            ry = torch.tensor(data_ry, device=torch.device(device))
            ry.requires_grad = True
        if d > 2:
            rz = torch.tensor(data_rz, device=torch.device(device))
            rz.requires_grad = True
            
        if d == 1:
            oa = torch.exp(HMF_MAP1d.apply(t,rx,p))
        elif d == 2:
            oa = torch.exp(HMF_MAP2d.apply(t,rx,ry,p))
        elif d == 3:
            oa = torch.exp(HMF_MAP3d.apply(t,rx,ry,rz,p))
        
        if d == 1:
            om = HMF_Mean1d.apply(t,rx,p)
        elif d == 2:
            om = HMF_Mean2d.apply(t,rx,ry,p)
        elif d == 3:
            om = HMF_Mean3d.apply(t,rx,ry,rz,p)
        loss = torch.mean(w*om)
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
                    print(dev1,"\t\t",dev2)
                    for i1,i2 in zip(res[dev1][var].flatten(),res[dev2][var].flatten()):
                        print(i1,"\t",i2)
                    raise Exception(name[var]+"\t"+str(diffs[var]))








class Test_Extreme(unittest.TestCase):

    """
    def test_no_smoothness_1D(self):
        print("")
        test_no_smoothness(1,"cpu",self)
        
    def test_no_smoothness_1D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():  
            test_no_smoothness(1,"cuda",self)

    def test_no_smoothness_2D(self):
        print("")
        test_no_smoothness(2,"cpu",self)

    def test_no_smoothness_2D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_no_smoothness(2,"cuda",self)

    def test_no_smoothness_3D(self):
        print("")
        test_no_smoothness(3,"cpu",self)

    def test_no_smoothness_3D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_no_smoothness(3,"cuda",self)

    def test_smoothness_dom_1D(self):
        print("")
        test_smoothness_dom(1,"cpu",self)

    def test_smoothness_dom_1D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_smoothness_dom(1,"cuda",self)

    def test_smoothness_dom_2D(self):
        print("")
        test_smoothness_dom(2,"cpu",self)

    def test_smoothness_dom_2D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_smoothness_dom(2,"cuda",self)

    def test_smoothness_dom_3D(self):
        print("")
        test_smoothness_dom(3,"cpu",self)

    def test_smoothness_dom_3D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            test_smoothness_dom(3,"cuda",self)

    def test_mixed_smoothness_1D(self):
        print("")
        for i in range(1,l):    
            test_mixed_smoothness(i,1,"cpu",self)

    def test_mixed_smoothness_1D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            for i in range(1,l):    
                test_mixed_smoothness(i,1,"cuda",self)

    def test_mixed_smoothness_2D(self):
        print("")
        for i in range(1,l):    
            test_mixed_smoothness(i,2,"cpu",self)
    
    def test_mixed_smoothness_2D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            for i in range(1,l):    
                test_mixed_smoothness(i,2,"cuda",self)

    def test_mixed_smoothness_3D(self):
        print("")
        for i in range(1,l):    
            test_mixed_smoothness(i,3,"cpu",self)

    def test_mixed_smoothness_3D_cuda(self):
        print("")
        if torch.backends.cuda.is_built():
            for i in range(1,l):    
                test_mixed_smoothness(i,3,"cuda",self)
"""

    def test_equivalence_1d(self):
        print("")
        if torch.backends.cuda.is_built():
            test_device_equivalence(1,["cpu","cuda"],self)
"""      
    def test_equivalence_2d(self):
        print("")
        if torch.backends.cuda.is_built():
            test_device_equivalence(2,["cpu","cuda"],self)
            
    def test_equivalence_3d(self):
        print("")
        if torch.backends.cuda.is_built():
            test_device_equivalence(3,["cpu","cuda"],self)
"""   


if __name__ == '__main__':
    unittest.main()
    
    

