import unittest

import numpy as np
import math
import random

import datetime
import itertools

import torch
from hmf_deepflow import HMFDeepFlow1d, HMFDeepFlow2d, HMFDeepFlow3d
from hmf_deepflow_function import HMF_MAP1d_PytorchNative,HMF_MAP2d_PytorchNative,HMF_MAP3d_PytorchNative

b=1
c=int(random.uniform(3,8))
br=2*c-2
x=2**12
epsilon = 10**-5
verbosity = True

def get_size_info(d,device="cuda"):
    x_used = int(x**(1/d)+0.5)
    
    parentage = np.zeros(shape=(br,)).astype(np.int32)
    for i in range(br):
        parentage[i] = i//2 - 1
    parentage = torch.tensor(parentage, device=torch.device(device))
    
    return parentage, \
           tuple( [b,c]+[x_used for i in range(d)] ), \
           tuple( [b,br]+[x_used for i in range(d)] ), \
           [tuple([b,br]+[x_used - (1 if i==i2 else 0) for i in range(d)]) for i2 in range(d)], \
           tuple([i+2 for i in range(d)])

def HMF_run_all_solvers(t,t2,r,r2,p):
    d = len(r)
    if d == 1:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = HMF_MAP1d_PytorchNative.apply(t,r[0],p.detach())
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omj = HMFDeepFlow1d(solve_type="MFE",test_type=None)(t,r[0],p.detach())
        omj = [omj]
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, olj = HMFDeepFlow1d(solve_type="LBP",test_type=None)(t2,r2[0],p.detach())
        olj = [olj]
        print(4,datetime.datetime.now())
    elif d == 2:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = HMF_MAP2d_PytorchNative.apply(t,r[0],r[1],p.detach())
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omjx, omjy = HMFDeepFlow2d(solve_type="MFE",test_type=None)(t,r[0],r[1],p.detach())
        omj = [omjx, omjy]
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, oljx, oljy = HMFDeepFlow2d(solve_type="LBP",test_type=None)(t2,r2[0],r2[1],p.detach())
        olj = [oljx, oljy]
        print(4,datetime.datetime.now())
    elif d == 3:
        print(1,"Running MAP estimation",datetime.datetime.now())
        oa = HMF_MAP3d_PytorchNative.apply(t,r[0],r[1],r[2],p.detach())
        print(2,"Running MFE estimation",datetime.datetime.now())
        om, omjx, omjy, omjz = HMFDeepFlow3d(solve_type="MFE",test_type=None)(t,r[0],r[1],r[2],p.detach())
        omj = [omjx, omjy, omjz]
        print(3,"Running LBP estimation",datetime.datetime.now())
        ol, oljx, oljy, oljz = HMFDeepFlow3d(solve_type="LBP",test_type=None)(t2,r2[0],r2[1],r2[2],p.detach())
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

    parentage, size_info, size_info_r, size_info_j, axes = get_size_info(d,device)

    #create data terms for testing (smoothness terms should be zero)
    data_t = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_rx = np.zeros(shape=size_info_r).astype(np.float32)
    data_wx = np.random.normal(0,1,size=size_info_j[0]).astype(np.float32)
    if d > 1:
        data_ry = np.zeros(shape=size_info_r).astype(np.float32)
        data_wy = np.random.normal(0,1,size=size_info_j[1]).astype(np.float32)
    if d > 2:
        data_rz = np.zeros(shape=size_info_r).astype(np.float32)
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
    oa, om, omj, ol, olj = HMF_run_all_solvers(t,t2,r,r2,parentage)
    
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
    true_val_marg_ext = np.zeros(size_info_r)
    true_val_marg_ext[:,-size_info[1]:,...] = true_val_marg
    for l,p in reversed(list(enumerate(parentage.cpu().detach().numpy()))):
        if p != -1:
            true_val_marg_ext[:,p,...] += true_val_marg_ext[:,l,...]
    for i in range(d):
        if i == 0:
            true_val = true_val_marg_ext[:,:,:-1]*(1-true_val_marg_ext[:,:,1:,...])+(1-true_val_marg_ext[:,:,:-1,...])*true_val_marg_ext[:,:,1:,...]
        elif i == 1:
            true_val = true_val_marg_ext[:,:,:,:-1]*(1-true_val_marg_ext[:,:,:,1:,...])+(1-true_val_marg_ext[:,:,:,:-1,...])*true_val_marg_ext[:,:,:,1:,...]
        elif i == 2:
            true_val = true_val_marg_ext[:,:,:,:,:-1]*(1-true_val_marg_ext[:,:,:,:,1:,...])+(1-true_val_marg_ext[:,:,:,:,:-1,...])*true_val_marg_ext[:,:,:,:,1:,...]
        for estimate, name in zip([omj_np[i], olj_np[i]], ["MFE","LBP"]):
            buffer_equality_check(estimate, name, true_val, "TRUE VALUE", asserter=asserter)
    
    #ensure gradient wrt data terms are equivalent to just softmax (need to update for HMF)
    print(12,"Checking correctness of gradient calculation")
    grad_marg = data_w
    for i in range(d):
        if i == 0:
            grad_marg_all = data_wx * (1 - 2 * true_val_marg_ext[:,:,1:,...])
        elif i==1:
            grad_marg_all = data_wy * (1 - 2 * true_val_marg_ext[:,:,:,1:,...])
        elif i==2:
            grad_marg_all = data_wz * (1 - 2 * true_val_marg_ext[:,:,:,:,1:,...])
        for l,p in enumerate(parentage.cpu().detach().numpy()):
            if p != -1:
                grad_marg_all[:,l,...] += grad_marg_all[:,p,...]
        if i == 0:
            grad_marg[:,:,:-1,...] += grad_marg_all[:,-size_info[1]:,...]
        elif i == 1:
            grad_marg[:,:,:,:-1,...] += grad_marg_all[:,-size_info[1]:,...]
        elif i == 2:
            grad_marg[:,:,:,:,:-1,...] += grad_marg_all[:,-size_info[1]:,...]
            
            
        if i == 0:
            grad_marg_all = data_wx * (1 - 2 * true_val_marg_ext[:,:,:-1,...])
        elif i==1:
            grad_marg_all = data_wy * (1 - 2 * true_val_marg_ext[:,:,:,:-1,...])
        elif i==2:
            grad_marg_all = data_wz * (1 - 2 * true_val_marg_ext[:,:,:,:,:-1,...])
        for l,p in enumerate(parentage.cpu().detach().numpy()):
            if p != -1:
                grad_marg_all[:,l,...] += grad_marg_all[:,p,...]
        if i == 0:
            grad_marg[:,:,1:,...] += grad_marg_all[:,-size_info[1]:,...]
        elif i == 1:
            grad_marg[:,:,:,1:,...] += grad_marg_all[:,-size_info[1]:,...]
        elif i == 2:
            grad_marg[:,:,:,:,1:,...] += grad_marg_all[:,-size_info[1]:,...]
            
        
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

            
            
            
#This test is the equivalent of the only smoothness one for binary, except simplified for a HMF
#If the smoothness terms for the leaf nodes are all the same (with the others being zero) and the data terms are zero then:
# - the pointwise marginals are equal across labels
# - the gradient w.r.t. to the leaf smoothness terms sums to zero (they may not be equal to zero but if
#   the smoothness terms stay equal to each other, the marginals shouldn't change) provided that the
#   only gradient is coming in from the pointwise marginals
def test_no_data(d,full_device,asserter):
    print("Testing (no data term) \tDim: " +str(d)+ " \tDev: " + full_device + "\t Start Time:\t" + str(datetime.datetime.now()))
    device = "cuda" if "cuda" in full_device else "cpu"
    
    parentage, size_info, size_info_r, size_info_j, axes = get_size_info(d,device)
    size_nochannel_info = tuple([size_info[0],1]+list(size_info[2:]))

    #create data terms for testing (smoothness terms should be zero)
    rlimit = 0.5 / d
    data_t = np.zeros(shape=size_info).astype(np.float32)
    data_w = np.random.normal(0,1,size=size_info).astype(np.float32)
    data_rx = np.repeat(rlimit*np.random.uniform(0,1,size=size_nochannel_info).astype(np.float32),size_info_r[1],axis=1)
    data_rx[:,:-c,...] *= 0
    if d > 1:
        data_ry = np.repeat(rlimit*np.random.uniform(0,1,size=size_nochannel_info).astype(np.float32),size_info_r[1],axis=1)
        data_ry[:,:-c,...] *= 0
    if d > 2:
        data_rz = np.repeat(rlimit*np.random.uniform(0,1,size=size_nochannel_info).astype(np.float32),size_info_r[1],axis=1)
        data_rz[:,:-c,...] *= 0

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
    oa, om, omj, ol, olj = HMF_run_all_solvers(t,t2,r,r2,parentage)
    
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
        grad_sum = np.sum(estimate[:,-c:,...],axis=1)
        buffer_equality_check(grad_sum,"GRADSUM",asserter=asserter)

def test_smoothness_dom(d,full_device,asserter):
    print("Testing (smoothness dom) \tDim: " +str(d)+ " \tDev: " + full_device + "\t Start Time:\t" + str(datetime.datetime.now()))
    device = "cuda" if "cuda" in full_device else "cpu"
    alg_type = "pytorch" if "pytorch" in full_device else "cpp"

    parentage, size_info, size_info_r, size_info_j, axes = get_size_info(d,device)

    data_t = 0.1/(d*size_info[1])*np.random.normal(0,1,size=size_info).astype(np.float32)
    sums = [[np.sum(data_t[b,c,...]) for c in range(size_info[1])] for b in range(size_info[0])]
    c_preferred = [sums[b].index(max(sums[b])) for b in range(size_info[0])]
    for b in range(size_info[0]):
        data_t[b,c_preferred[b],...] += 0.0025/np.sqrt(math.prod(size_info[2:]))
    data_rx = 1.25/(d*size_info_r[1])*(10+np.random.uniform(size=size_info_r).astype(np.float32))
    if d > 1:
        data_ry = 1.25/(d*size_info_r[1])*(10+np.random.uniform(size=size_info_r).astype(np.float32))
    if d > 2:
        data_rz = 1.25/(d*size_info_r[1])*(10+np.random.uniform(size=size_info_r).astype(np.float32))

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
    oa, om, omj, ol, olj = HMF_run_all_solvers(t,t,r,r,parentage)
    
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
    num_wrong = {}
    for b,c in itertools.product(range(size_info[0]),range(size_info[1])):
        num_wrong[b,c] = 0
        if c == c_preferred[b]:
            for val_df in oa_np[b,c,...].flatten():
                if val_df < 1-oa_epsilon:
                    num_wrong[b,c] += 1
        else:
            for val_df in oa_np[b,c,...].flatten():
                if val_df > oa_epsilon:
                    num_wrong[b,c] += 1
    if sum(num_wrong.values()) > 0:
        print("num_wrong",num_wrong)
        asserter.assertTrue(False)
    
class Test_Extreme(unittest.TestCase):
    '''def test_no_smoothness_1D(self):
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
'''
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
        
if __name__ == '__main__':
    unittest.main()
    
    
