import unittest
import numpy as np
import time
import torch
from itertools import product
from math import sqrt, log
import datetime


from graph_deepflow import BinaryDeepFlow, PottsDeepFlow

epsilon = 0.5

def APD(v1,v2):
    num = np.sum(np.abs(v1-v2))
    den = 0.5*np.sum(np.abs(v1)+np.abs(v2))
    
    if den > epsilon/100:
        return 100*num/den
    return 100*num

def test_no_smoothness(device,model_type,b,c,N,asserter):
    
    data_t = np.random.normal(0,1,size=(b,c,N)).astype(np.float32)
    data_w = np.random.normal(0,1,size=(b,c,N)).astype(np.float32)
    data_r = np.zeros(shape=(b,c,N,N)).astype(np.float32)
    
    if model_type == "Potts":
        operator_LBP = PottsDeepFlow(solve_type="LBP",activation=False,test_type="MAP")
        operator_MFE = PottsDeepFlow(solve_type="MFE",activation=False,test_type="MAP")
    else: #model_type == "Binary"
        operator_LBP = BinaryDeepFlow(solve_type="LBP",activation=False,test_type="MAP")
        operator_MFE = BinaryDeepFlow(solve_type="MFE",activation=False,test_type="MAP")
    
    t = torch.tensor(data_t, device=torch.device(device))
    t2 = torch.tensor(data_t.copy(), device=torch.device(device))
    t.requires_grad = True
    t2.requires_grad = True
    w = torch.tensor(data_w, device=torch.device(device))
    r = torch.tensor(data_r, device=torch.device(device))
    r2 = torch.tensor(data_r.copy(), device=torch.device(device))
    r.requires_grad = True
    r2.requires_grad = True
    
    om = operator_MFE(t,r)
    om_np = om.detach().cpu().numpy()
    ol = operator_LBP(t2,r2)
    ol_np = ol.detach().cpu().numpy()
    
    loss = torch.sum(w*om)
    loss.backward()
    ot_np = t.grad.detach().cpu().numpy()
    
    loss = torch.sum(w*ol)
    loss.backward()
    ot2_np = t2.grad.detach().cpu().numpy()
    
    operator_LBP.eval()
    oa = operator_LBP(t,r)
    oa_np = oa.detach().cpu().numpy()

    #make sure not nan
    asserter.assertFalse(np.any(np.isnan(oa_np)))
    asserter.assertFalse(np.any(np.isnan(om_np)))
    asserter.assertFalse(np.any(np.isnan(ol_np)))
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
        
    #ensure mean pass is equivalent to the data terms only
    for ib,ic in product(range(b),range(c)): 
        for val_df, val_d in zip(om_np_l[(ib,ic)],dt_np_l[(ib,ic)]):
            if(APD(val_df,val_d) > epsilon):
                raise Exception(str(val_df) + "\t" + str(val_d))
    for ib,ic in product(range(b),range(c)): 
        for val_df, val_d in zip(ol_np_l[(ib,ic)],dt_np_l[(ib,ic)]):
            if(APD(val_df,val_d) > epsilon):
                raise Exception(str(val_df) + "\t" + str(val_d)+"\t" + str(APD(val_df,val_d)))
    
    #ensure gradient wrt data terms are passed immediately through
    for ib,ic in product(range(b),range(c)): 
        for val_df, val_d in zip(ot_np_l[(ib,ic)],dw_np_l[(ib,ic)]):
            if(APD(val_df,val_d) > epsilon):
                raise Exception(str(val_df) + "\t" + str(val_d))
    for ib,ic in product(range(b),range(c)): 
        for val_df, val_d in zip(ot2_np_l[(ib,ic)],dw_np_l[(ib,ic)]):
            if(APD(val_df,val_d) > epsilon):
                raise Exception(str(val_df) + "\t" + str(val_d))
    
    #if it is a Potts model, ensure MAP assigns 1 to highest term and 0 to everything else
    if model_type == "Potts":
        for i in range(N):
            for ib in range(b):
                highest = max([dt_np_l[(ib,ic)][i] for ic in range(c)])
                for ic in range(c):
                    if(dt_np_l[(ib,ic)][i] == highest and oa_np_l[(ib,ic)][i] < 0.75):
                        raise Exception("")
                    if(dt_np_l[(ib,ic)][i] < highest - epsilon/100  and oa_np_l[(ib,ic)][i] > 0.25):
                        raise Exception("")
    
    #if it is a Binary model, ensure MAP assigns 1 to positive terms and 0 to negative terms
    else: #if model_type == "Binary"
        for i in range(N):
            for ib in range(b):
                for ic in range(c):
                    if(dt_np_l[(ib,ic)][i] > 0 and oa_np_l[(ib,ic)][i] < 0.75):
                        raise Exception(str(dt_np_l[(ib,ic)][i])+"\t"+str(oa_np_l[(ib,ic)][i]))
                    if(dt_np_l[(ib,ic)][i] < 0  and oa_np_l[(ib,ic)][i] > 0.25):
                        raise Exception(str(dt_np_l[(ib,ic)][i])+"\t"+str(oa_np_l[(ib,ic)][i]))


def test_middling_smoothness(device,model_type,b,c,N,asserter):
    
    #get maximum limit for guaranteed convergence
    margin_for_error = 0.05
    if model_type == "Potts":
        conv_limit = (-log((2*(N-1)-3)/(2*(N-1)-1)))/2 / (1+margin_for_error)
    else: #model_type == "Binary"
        conv_limit = (-log((2*(N-1)-3)/(2*(N-1)-1))) / (1+margin_for_error)
        
    data_t = np.random.normal(0,1,size=(b,c,N)).astype(np.float32)
    data_w = np.random.normal(0,1,size=(b,c,N)).astype(np.float32)
    data_r = np.random.uniform(size=(b,c,N,N)).astype(np.float32) * conv_limit
    for i in range(N):
        for j in range(i,N):
            data_r[:,:,j,i] = data_r[:,:,i,j]
    
    if model_type == "Potts":
        operator_LBP = PottsDeepFlow(solve_type="LBP",activation=False,test_type="MAP")
        operator_MFE = PottsDeepFlow(solve_type="MFE",activation=False,test_type="MAP")
    else: #model_type == "Binary"
        operator_LBP = BinaryDeepFlow(solve_type="LBP",activation=False,test_type="MAP")
        operator_MFE = BinaryDeepFlow(solve_type="MFE",activation=False,test_type="MAP")
    
    t = torch.tensor(data_t, device=torch.device(device))
    t2 = torch.tensor(data_t.copy(), device=torch.device(device))
    t.requires_grad = True
    t2.requires_grad = True
    w = torch.tensor(data_w, device=torch.device(device))
    r = torch.tensor(data_r, device=torch.device(device))
    r2 = torch.tensor(data_r.copy(), device=torch.device(device))
    r.requires_grad = True
    r2.requires_grad = True
    
    om = operator_MFE(t,r)
    om_np = om.detach().cpu().numpy()
    ol = operator_LBP(t2,r2)
    ol_np = ol.detach().cpu().numpy()
    
    loss = torch.sum(w*om)
    loss.backward()
    ot_np = t.grad.detach().cpu().numpy()
    
    loss = torch.sum(w*ol)
    loss.backward()
    ot2_np = t2.grad.detach().cpu().numpy()
    
    operator_LBP.eval()
    oa = operator_LBP(t,r)
    oa_np = oa.detach().cpu().numpy()

    #make sure not nan
    asserter.assertFalse(np.any(np.isnan(oa_np)))
    asserter.assertFalse(np.any(np.isnan(om_np)))
    asserter.assertFalse(np.any(np.isnan(ol_np)))
    asserter.assertFalse(np.any(np.isnan(ot_np)))
    asserter.assertFalse(np.any(np.isnan(ot2_np)))
    

def test_high_smoothness(device,model_type,b,c,N,asserter):
    
    data_t = 0.5*np.random.normal(0,1,size=(b,c,N)).astype(np.float32)
    data_r = 2 + np.random.uniform(size=(b,c,N,N)).astype(np.float32)/N
    for i in range(N):
        for j in range(i,N):
            data_r[:,:,j,i] = data_r[:,:,i,j]
            
    #if we are a Potts model, give a slight bias to highest data term to prevent close cases
    if model_type == "Potts":
        highest = np.max(np.sum(data_t,axis=2,keepdims=True),axis=1,keepdims=True)
        data_t += 1/sqrt(N) * (np.sum(data_t,axis=2,keepdims=True)==highest) - 1/(2*sqrt(N))
        
    #if we are a binary model, make data terms more extreme
    else: #model_type == "Binary"
        data_t += np.sign(np.sum(data_t,axis=-1,keepdims=True))/(2*sqrt(N))
    
    if model_type == "Potts":
        operator_LBP = PottsDeepFlow(solve_type="LBP",activation=False,test_type="MAP")
        operator_MFE = PottsDeepFlow(solve_type="MFE",activation=False,test_type="MAP")
    else: #model_type == "Binary"
        operator_LBP = BinaryDeepFlow(solve_type="LBP",activation=False,test_type="MAP")
        operator_MFE = BinaryDeepFlow(solve_type="MFE",activation=False,test_type="MAP")
    
    t = torch.tensor(data_t, device=torch.device(device))
    t2 = torch.tensor(data_t.copy(), device=torch.device(device))
    t.requires_grad = True
    t2.requires_grad = True
    r = torch.tensor(data_r, device=torch.device(device))
    r2 = torch.tensor(data_r.copy(), device=torch.device(device))
    r.requires_grad = True
    r2.requires_grad = True
    
    om = operator_MFE(t,r)
    om_np = om.detach().cpu().numpy()
    ol = operator_LBP(t2,r2)
    ol_np = ol.detach().cpu().numpy()
    
    operator_LBP.eval()
    oa = operator_LBP(t,r)
    oa_np = oa.detach().cpu().numpy()

    #make sure not nan
    asserter.assertFalse(np.any(np.isnan(oa_np)))
    asserter.assertFalse(np.any(np.isnan(om_np)))
    asserter.assertFalse(np.any(np.isnan(ol_np)))
    
    #resize into more usable form
    dt_np_l = {(ib,ic):data_t[ib,ic,...].flatten() for ib,ic in product(range(b),range(c))}
    oa_np_l = {(ib,ic):oa_np[ib,ic,...].flatten()  for ib,ic in product(range(b),range(c))}
    om_np_l = {(ib,ic):om_np[ib,ic,...].flatten()  for ib,ic in product(range(b),range(c))}
    ol_np_l = {(ib,ic):ol_np[ib,ic,...].flatten()  for ib,ic in product(range(b),range(c))}
    
    #if it is a Potts model, ensure MAP assigns 1 to highest term and 0 to everything else
    if model_type == "Potts":
        for ib in range(b):
            agg_d_values = [np.sum(dt_np_l[(ib,ic)]) for ic in range(c)]
            highest_c = max(range(len(agg_d_values)), key=agg_d_values.__getitem__)
            
            for i in range(N):
                for ic in range(c):
                    if(ic == highest_c and oa_np_l[(ib,ic)][i] < 0.75):
                        raise Exception(str(agg_d_values)+"\t"+str(ic)+"\t"+str(highest_c)+"\t"+str([oa_np_l[(ib,q)][i] for q in range(c)])+"\t"+str([dt_np_l[(ib,q)][i] for q in range(c)]))
                    if(ic != highest_c  and oa_np_l[(ib,ic)][i] > 0.25):
                        print(str(agg_d_values)+"\t"+str(ic)+"\t"+str(highest_c)+"\t"+str(oa_np_l[(ib,ic)][i]))
                        raise Exception(str(agg_d_values)+"\t"+str(ic)+"\t"+str(highest_c)+"\t"+str([oa_np_l[(ib,q)][i] for q in range(c)])+"\t"+str([dt_np_l[(ib,q)][i] for q in range(c)]))
    
    #if it is a Binary model, ensure MAP assigns 1 when the aggregate term is positive and 0 if negative
    else: #if model_type == "Binary"
        for ib in range(b):
            agg_d_values = [np.sum(dt_np_l[(ib,ic)]) for ic in range(c)]
            for i in range(N):
                for ic in range(c):
                    if(agg_d_values[ic] > 0 and oa_np_l[(ib,ic)][i] < 0.75):
                        raise Exception(str(agg_d_values[ic])+"\t"+str(oa_np_l[(ib,ic)][i]))
                    if(agg_d_values[ic] < 0  and oa_np_l[(ib,ic)][i] > 0.25):
                        raise Exception(str(agg_d_values[ic])+"\t"+str(oa_np_l[(ib,ic)][i]))

class Test_Extreme(unittest.TestCase):
    
    bcN_list = [(1,3,5),(1,5,12),(2,3,6),(1,3,256)]
    
    def test_no_smoothness(self):
        print("")
        for b,c,N in Test_Extreme.bcN_list:
            print("Potts no smoothness:\t",b,c,N,end="")
            start_time = datetime.datetime.now()
            test_no_smoothness("cpu","Potts",b,c,N,self)
            end_time = datetime.datetime.now()
            print("\t",end_time-start_time)
        for b,c,N in Test_Extreme.bcN_list:
            print("Binary no smoothness:\t",b,c,N,end="")
            start_time = datetime.datetime.now()
            test_no_smoothness("cpu","Binary",b,c,N,self)
            end_time = datetime.datetime.now()
            print("\t",end_time-start_time)
    
    def test_high_smoothness(self):
        print("")
        for b,c,N in Test_Extreme.bcN_list:
            print("Potts high smoothness:\t",b,c,N,end="")
            start_time = datetime.datetime.now()
            test_high_smoothness("cpu","Potts",b,c,N,self)
            end_time = datetime.datetime.now()
            print("\t",end_time-start_time)
        for b,c,N in Test_Extreme.bcN_list:
            print("Binary high smoothness:\t",b,c,N,end="")
            start_time = datetime.datetime.now()
            test_high_smoothness("cpu","Binary",b,c,N,self)
            end_time = datetime.datetime.now()
            print("\t",end_time-start_time)
    
    def test_middling_smoothness(self):
        print("")
        for b,c,N in Test_Extreme.bcN_list:
            print("Potts middling smoothness:\t",b,c,N,end="")
            start_time = datetime.datetime.now()
            test_middling_smoothness("cpu","Potts",b,c,N,self)
            end_time = datetime.datetime.now()
            print("\t",end_time-start_time)
        for b,c,N in Test_Extreme.bcN_list:
            print("Binary middling smoothness:\t",b,c,N,end="")
            start_time = datetime.datetime.now()
            test_middling_smoothness("cpu","Binary",b,c,N,self)
            end_time = datetime.datetime.now()
            print("\t",end_time-start_time)
    

if __name__ == '__main__':
    unittest.main()