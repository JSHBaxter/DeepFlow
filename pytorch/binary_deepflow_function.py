import torch
import deepflow
from deepflow_function import DeepFlowFunction
from math import log, prod
from itertools import product

class BinaryDeepFlow1dLoss_functional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, rx, marginals_d, marginals_rx, y):
        epsilon = 10**-12
        ctx.save_for_backward(marginals_d, marginals_rx, y)
        loss = torch.mean(-y*torch.log(marginals_d+epsilon)-(1-y)*torch.log(1-marginals_d+epsilon))
        return loss
            
    @staticmethod
    def backward(ctx, grad_l):
        marginals_d, marginals_rx, y = ctx.saved_tensors

        #get gradient for negative log-likelihood loss
        N = y.shape[0]*y.shape[1]*y.shape[2]
        yrx = torch.abs(y[:,:,1:]-y[:,:,:-1])
        
        return grad_l*(marginals_d-y)/N, \
               grad_l*(torch.nn.functional.pad(yrx-marginals_rx,(0,1),value=0))/N, \
               torch.zeros_like(marginals_d), \
               torch.zeros_like(marginals_rx), \
               torch.zeros_like(y)
    
class BinaryDeepFlow2dLoss_functional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, rx, ry, marginals_d, marginals_rx, marginals_ry, y):
        epsilon = 10**-12
        ctx.save_for_backward(marginals_d, marginals_rx, marginals_ry, y)
        loss = torch.mean(-y*torch.log(marginals_d+epsilon)-(1-y)*torch.log(1-marginals_d+epsilon))
        return loss
            
    @staticmethod
    def backward(ctx, grad_l):
        marginals_d, marginals_rx, marginals_ry, y = ctx.saved_tensors
        
        #get gradient for negative log-likelihood loss
        N = y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]
        yrx = torch.abs(y[:,:,1:,:]-y[:,:,:-1,:])
        yry = torch.abs(y[:,:,:,1:]-y[:,:,:,:-1])
        
        return grad_l*(marginals_d-y)/N, \
               grad_l*(torch.nn.functional.pad(yrx-marginals_rx,(0,0,0,1),value=0))/N, \
               grad_l*(torch.nn.functional.pad(yry-marginals_ry,(0,1,0,0),value=0))/N, \
               torch.zeros_like(marginals_d), \
               torch.zeros_like(marginals_rx), \
               torch.zeros_like(marginals_ry), \
               torch.zeros_like(y)
    
            
class BinaryDeepFlow3dLoss_functional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, rx, ry, rz, marginals_d, marginals_rx, marginals_ry, marginals_rz, y):
        epsilon = 10**-12
        ctx.save_for_backward(marginals_d, marginals_rx, marginals_ry, marginals_rz, y)
        loss = torch.mean(-y*torch.log(marginals_d+epsilon)-(1-y)*torch.log(1-marginals_d+epsilon))
        return loss
            
    @staticmethod
    def backward(ctx, grad_l):
        marginals_d, marginals_rx, marginals_ry, marginals_rz, y = ctx.saved_tensors
        
        #get gradient for negative log-likelihood loss
        N = y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]*y.shape[4]
        yrx = torch.abs(y[:,:,1:,:,:]-y[:,:,:-1,:,:])
        yry = torch.abs(y[:,:,:,1:,:]-y[:,:,:,:-1,:])
        yrz = torch.abs(y[:,:,:,:,1:]-y[:,:,:,:,:-1])
        
        return grad_l*(marginals_d-y)/N, \
               grad_l*(torch.nn.functional.pad(yrx-marginals_rx,(0,0,0,0,0,1),value=0))/N, \
               grad_l*(torch.nn.functional.pad(yry-marginals_ry,(0,0,0,1,0,0),value=0))/N, \
               grad_l*(torch.nn.functional.pad(yrz-marginals_rz,(0,1,0,0,0,0),value=0))/N, \
               torch.zeros_like(marginals_d), \
               torch.zeros_like(marginals_rx), \
               torch.zeros_like(marginals_ry), \
               torch.zeros_like(marginals_rz), \
               torch.zeros_like(y)
    
def Binary_LBPnD_PytorchNative_forward_compute_vertex_messages(ed0,ed1,er,meu0,med0,meu1,med1):
    dims = len(er)
    mvu0 = [None for i in range(dims)]
    mvd0 = [None for i in range(dims)]
    mvu1 = [None for i in range(dims)]
    mvd1 = [None for i in range(dims)]
    
    #compute vertex to edge messages    
    for di1 in range(dims):
        #put in contribution from data term
        if di1 == 0:
            mvd1[di1] = ed1[:,:,1:,...].clone()
            mvu1[di1] = ed1[:,:,:-1,...].clone()
            mvd0[di1] = ed0[:,:,1:,...].clone()
            mvu0[di1] = ed0[:,:,:-1,...].clone()
        elif di1 == 1:
            mvd1[di1] = ed1[:,:,:,1:,...].clone()
            mvu1[di1] = ed1[:,:,:,:-1,...].clone()
            mvd0[di1] = ed0[:,:,:,1:,...].clone()
            mvu0[di1] = ed0[:,:,:,:-1,...].clone()
        elif di1 == 2:
            mvd1[di1] = ed1[:,:,:,:,1:,...].clone()
            mvu1[di1] = ed1[:,:,:,:,:-1,...].clone()
            mvd0[di1] = ed0[:,:,:,:,1:,...].clone()
            mvu0[di1] = ed0[:,:,:,:,:-1,...].clone()

        #put in contribution from own direction
        if di1 == 0:
            mvd1[di1][:,:,:-1,...] *= med1[di1][:,:,1:,...]
            mvu1[di1][:,:,1:,...]  *= meu1[di1][:,:,:-1,...]
            mvd0[di1][:,:,:-1,...] *= med0[di1][:,:,1:,...]
            mvu0[di1][:,:,1:,...]  *= meu0[di1][:,:,:-1,...]
        elif di1 == 1:
            mvd1[di1][:,:,:,:-1,...] *= med1[di1][:,:,:,1:,...]
            mvu1[di1][:,:,:,1:,...]  *= meu1[di1][:,:,:,:-1,...]
            mvd0[di1][:,:,:,:-1,...] *= med0[di1][:,:,:,1:,...]
            mvu0[di1][:,:,:,1:,...]  *= meu0[di1][:,:,:,:-1,...]
        elif di1 == 2:
            mvd1[di1][:,:,:,:,:-1,...] *= med1[di1][:,:,:,:,1:,...]
            mvu1[di1][:,:,:,:,1:,...]  *= meu1[di1][:,:,:,:,:-1,...]
            mvd0[di1][:,:,:,:,:-1,...] *= med0[di1][:,:,:,:,1:,...]
            mvu0[di1][:,:,:,:,1:,...]  *= meu0[di1][:,:,:,:,:-1,...]

        #put in contributions from other directions
        for di2 in range(dims):
            if di1 == di2:
                continue
            if di1 == 0:
                contribution_1_d_u = med1[di2][:,:,:-1,...]
                contribution_1_u_u = meu1[di2][:,:,:-1,...]
                contribution_1_d_d = med1[di2][:,:,1:,...]
                contribution_1_u_d = meu1[di2][:,:,1:,...]
                contribution_0_d_u = med0[di2][:,:,:-1,...]
                contribution_0_u_u = meu0[di2][:,:,:-1,...]
                contribution_0_d_d = med0[di2][:,:,1:,...]
                contribution_0_u_d = meu0[di2][:,:,1:,...]
            elif di1 == 1:
                contribution_1_d_u = med1[di2][:,:,:,:-1,...]
                contribution_1_u_u = meu1[di2][:,:,:,:-1,...]
                contribution_1_d_d = med1[di2][:,:,:,1:,...]
                contribution_1_u_d = meu1[di2][:,:,:,1:,...]
                contribution_0_d_u = med0[di2][:,:,:,:-1,...]
                contribution_0_u_u = meu0[di2][:,:,:,:-1,...]
                contribution_0_d_d = med0[di2][:,:,:,1:,...]
                contribution_0_u_d = meu0[di2][:,:,:,1:,...]
            elif di1 == 2:
                contribution_1_d_u = med1[di2][:,:,:,:,:-1,...]
                contribution_1_u_u = meu1[di2][:,:,:,:,:-1,...]
                contribution_1_d_d = med1[di2][:,:,:,:,1:,...]
                contribution_1_u_d = meu1[di2][:,:,:,:,1:,...]
                contribution_0_d_u = med0[di2][:,:,:,:,:-1,...]
                contribution_0_u_u = meu0[di2][:,:,:,:,:-1,...]
                contribution_0_d_d = med0[di2][:,:,:,:,1:,...]
                contribution_0_u_d = meu0[di2][:,:,:,:,1:,...]

            if di2 == 0:
                mvd1[di1][:,:,:-1,...] *= contribution_1_d_d
                mvd1[di1][:,:,1:,...]  *= contribution_1_u_d
                mvu1[di1][:,:,:-1,...] *= contribution_1_d_u
                mvu1[di1][:,:,1:,...]  *= contribution_1_u_u
                mvd0[di1][:,:,:-1,...] *= contribution_0_d_d
                mvd0[di1][:,:,1:,...]  *= contribution_0_u_d
                mvu0[di1][:,:,:-1,...] *= contribution_0_d_u
                mvu0[di1][:,:,1:,...]  *= contribution_0_u_u
            elif di2 == 1:
                mvd1[di1][:,:,:,:-1,...] *= contribution_1_d_d
                mvd1[di1][:,:,:,1:,...]  *= contribution_1_u_d
                mvu1[di1][:,:,:,:-1,...] *= contribution_1_d_u
                mvu1[di1][:,:,:,1:,...]  *= contribution_1_u_u
                mvd0[di1][:,:,:,:-1,...] *= contribution_0_d_d
                mvd0[di1][:,:,:,1:,...]  *= contribution_0_u_d
                mvu0[di1][:,:,:,:-1,...] *= contribution_0_d_u
                mvu0[di1][:,:,:,1:,...]  *= contribution_0_u_u
            elif di2 == 3:
                mvd1[di1][:,:,:,:,:-1,...] *= contribution_1_d_d
                mvd1[di1][:,:,:,:,1:,...]  *= contribution_1_u_d
                mvu1[di1][:,:,:,:,:-1,...] *= contribution_1_d_u
                mvu1[di1][:,:,:,:,1:,...]  *= contribution_1_u_u
                mvd0[di1][:,:,:,:,:-1,...] *= contribution_0_d_d
                mvd0[di1][:,:,:,:,1:,...]  *= contribution_0_u_d
                mvu0[di1][:,:,:,:,:-1,...] *= contribution_0_d_u
                mvu0[di1][:,:,:,:,1:,...]  *= contribution_0_u_u

    #and return messages
    return mvu0, mvd0, mvu1, mvd1
    
def Binary_LBPnD_PytorchNative_forward(d,r):
    dims = len(r)
    DeepFlowFunction.check_var_dims([d,]+r,dims)
    b,c,s = d.shape[0],d.shape[1],d.shape[2:]
    
    #create containers for edge messages
    meu0 = []
    med0 = []
    meu1 = []
    med1 = []
    err_meu0 = []
    err_med0 = []
    err_meu1 = []
    err_med1 = []
    for i in range(dims):
        s_prime = tuple([b,c,]+[(si-1 if isi==i else si) for isi,si in enumerate(s)])
        meu0.append(torch.ones(s_prime,device=d.device))
        med0.append(torch.ones(s_prime,device=d.device))
        meu1.append(torch.ones(s_prime,device=d.device))
        med1.append(torch.ones(s_prime,device=d.device))
        err_meu0.append(torch.zeros(s_prime,device=d.device))
        err_med0.append(torch.zeros(s_prime,device=d.device))
        err_meu1.append(torch.zeros(s_prime,device=d.device))
        err_med1.append(torch.zeros(s_prime,device=d.device))
    
    #calculate theoretical worst decay rate
    epsilon = 10**-6
    max_reg = max([torch.max(ri) for ri in r])
    min_decay_const = (2*dims-1)*(1-torch.exp(-max_reg/2))/(1+torch.exp(-max_reg/2))
    
    #get a maximum number of iterations to perform
    max_num_iters = None
    if min_decay_const <= epsilon:
        max_num_iters = 1
    if min_decay_const > epsilon and min_decay_const < 1-epsilon:
        max_num_iters = int((log(epsilon)) / log(min_decay_const)) + 1
    if max_num_iters is None or max_num_iters > 20*sum(s):
        max_num_iters = 20*sum(s)
    max_num_iters += 10
    
    #adjust terms according to temperature
    ed1 = torch.exp(d * (d<0))
    ed0 = torch.exp(-d * (d>0))
    er = [torch.exp(-ri) for ri in r]
    
    iter = 0
    while True:
        
        mvu0, mvd0, mvu1, mvd1 = Binary_LBPnD_PytorchNative_forward_compute_vertex_messages(ed0, ed1, er,meu0,med0,meu1,med1)
        
        #normalised messages
        avu = [mvu0[i]+mvu1[i] for i in range(dims)]
        avd = [mvd0[i]+mvd1[i] for i in range(dims)]
        for i in range(dims):
            mvu0[i] = mvu0[i] / avu[i]
            mvu1[i] = mvu1[i] / avu[i]
            mvd0[i] = mvd0[i] / avd[i]
            mvd1[i] = mvd1[i] / avd[i]
        
        #compute edge to vertex messages
        for i in range(dims):
            if i == 0:
                new_med0 = mvd0[i] + er[i][:,:,:-1,...] * mvd1[i]
                new_med1 = mvd1[i] + er[i][:,:,:-1,...] * mvd0[i]
                new_meu0 = mvu0[i] + er[i][:,:,:-1,...] * mvu1[i]
                new_meu1 = mvu1[i] + er[i][:,:,:-1,...] * mvu0[i]
            elif i == 1:
                new_med0 = mvd0[i] + er[i][:,:,:,:-1,...] * mvd1[i]
                new_med1 = mvd1[i] + er[i][:,:,:,:-1,...] * mvd0[i]
                new_meu0 = mvu0[i] + er[i][:,:,:,:-1,...] * mvu1[i]
                new_meu1 = mvu1[i] + er[i][:,:,:,:-1,...] * mvu0[i]
            elif i == 2:
                new_med0 = mvd0[i] + er[i][:,:,:,:,:-1,...] * mvd1[i]
                new_med1 = mvd1[i] + er[i][:,:,:,:,:-1,...] * mvd0[i]
                new_meu0 = mvu0[i] + er[i][:,:,:,:,:-1,...] * mvu1[i]
                new_meu1 = mvu1[i] + er[i][:,:,:,:,:-1,...] * mvu0[i]
            err_med0[i] = med0[i] - new_med0
            err_med1[i] = med1[i] - new_med1
            err_meu0[i] = meu0[i] - new_meu0
            err_meu1[i] = meu1[i] - new_meu1
            med0[i] = new_med0
            med1[i] = new_med1
            meu0[i] = new_meu0
            meu1[i] = new_meu1

        #calculate early convergence criteria
        max_change = max([torch.sum(torch.abs(em)) for em in err_med0] + 
                         [torch.sum(torch.abs(em)) for em in err_med1] + 
                         [torch.sum(torch.abs(em)) for em in err_meu0] + 
                         [torch.sum(torch.abs(em)) for em in err_meu1] )
        
                
        iter+=1
        if iter > max_num_iters or max_change < 0*epsilon:
            break
    
    #finally return the edge messages (all else can be discarded / easily reconstructed)
    #print(min_decay_const,max_num_iters,iter,max_change)
    return meu0, med0, meu1, med1, iter
    
    
def Binary_LBPnD_PytorchNative_backward(grad_meu0, grad_med0, grad_meu1, grad_med1, 
                                meu0, med0, meu1, med1, d, r, max_num_iters):
    dims = len(r)
    numels = prod(d.shape[2:])
    sumels = sum(d.shape[2:])
    grad_d0 =  torch.zeros_like(d) #(will multiply out the ed at the end, keeps everything in exponential domain)
    grad_d1 =  torch.zeros_like(d) #(will multiply out the ed at the end, keeps everything in exponential domain)
    grad_r = [torch.zeros_like(ri) for ri in r]

    #see if you can exit early
    max_grad = max([torch.max(torch.abs(gm)) for gm in grad_meu0] + 
                   [torch.max(torch.abs(gm)) for gm in grad_med0] + 
                   [torch.max(torch.abs(gm)) for gm in grad_meu1] + 
                   [torch.max(torch.abs(gm)) for gm in grad_med1] )
    if max_grad == 0:
        return grad_d, grad_r
    epsilon = (10**-8)*max_grad
    
    #calculate theoretical worst decay rate
    max_reg = max([torch.max(ri) for ri in r])
    min_decay_const = (2*dims-1)*(1-torch.exp(-max_reg/2))/(1+torch.exp(-max_reg/2))
    
    ed1 = torch.exp(d * (d<0))
    ed0 = torch.exp(-d * (d>0))
    er = [torch.exp(-ri) for ri in r]
        
    #get vertex messages and make containers for their gradients
    #(save unnormalised version to make gradient computation wrt edge messages easier and a "doubly normalised" version for backprop through normalisation)
    mvu0, mvd0, mvu1, mvd1 = Binary_LBPnD_PytorchNative_forward_compute_vertex_messages(ed0,ed1,er,meu0,med0,meu1,med1)
    avu = [mvu0[i]+mvu1[i] for i in range(dims)]
    avd = [mvd0[i]+mvd1[i] for i in range(dims)]
    dnmvu0, dnmvu1 = [mvu0[i]/avu[i]**2 for i in range(dims)], [mvu1[i]/avu[i]**2 for i in range(dims)]
    dnmvd0, dnmvd1 = [mvd0[i]/avd[i]**2 for i in range(dims)], [mvd1[i]/avd[i]**2 for i in range(dims)]
    #also save the normalised versions
    nmvu0, nmvu1 = [mvu0[i]/avu[i] for i in range(dims)], [mvu1[i]/avu[i] for i in range(dims)]
    nmvd0, nmvd1 = [mvd0[i]/avd[i] for i in range(dims)], [mvd1[i]/avd[i] for i in range(dims)]
    #print("avu/d",[(torch.min(torch.abs(avu[i])),torch.min(torch.abs(avd[i]))) for i in range(dims)])
    #print("mvu/d0",[(torch.min(torch.abs(mvu0[i])),torch.min(torch.abs(mvd0[i]))) for i in range(dims)])
    #print("mvu/d1`",[(torch.min(torch.abs(mvu1[i])),torch.min(torch.abs(mvd1[i]))) for i in range(dims)])
    del avu, avd
    grad_mvu0 = [None for i in range(dims)]
    grad_mvd0 = [None for i in range(dims)]
    grad_mvu1 = [None for i in range(dims)]
    grad_mvd1 = [None for i in range(dims)]
    
    iter = 0
    while True:

        #backprop back to vertex messages
        
        for i in range(dims):
            if i == 0:
                grad_mvu0[i] = grad_meu0[i] + er[i][:,:,:-1,...]*grad_meu1[i]
                grad_mvu1[i] = grad_meu1[i] + er[i][:,:,:-1,...]*grad_meu0[i]
                grad_mvd0[i] = grad_med0[i] + er[i][:,:,:-1,...]*grad_med1[i]
                grad_mvd1[i] = grad_med1[i] + er[i][:,:,:-1,...]*grad_med0[i]
                grad_r[i][:,:,:-1,...] += grad_meu1[i]*nmvu0[i] + grad_meu0[i]*nmvu1[i] + grad_med1[i]*nmvd0[i] + grad_med0[i]*nmvd1[i]
            if i == 1:
                grad_mvu0[i] = grad_meu0[i] + er[i][:,:,:,:-1,...]*grad_meu1[i]
                grad_mvu1[i] = grad_meu1[i] + er[i][:,:,:,:-1,...]*grad_meu0[i]
                grad_mvd0[i] = grad_med0[i] + er[i][:,:,:,:-1,...]*grad_med1[i]
                grad_mvd1[i] = grad_med1[i] + er[i][:,:,:,:-1,...]*grad_med0[i]
                grad_r[i][:,:,:,:-1,...] += grad_meu1[i]*nmvu0[i] + grad_meu0[i]*nmvu1[i] + grad_med1[i]*nmvd0[i] + grad_med0[i]*nmvd1[i]
            if i == 2:
                grad_mvu0[i] = grad_meu0[i] + er[i][:,:,:,:,:-1,...]*grad_meu1[i]
                grad_mvu1[i] = grad_meu1[i] + er[i][:,:,:,:,:-1,...]*grad_meu0[i]
                grad_mvd0[i] = grad_med0[i] + er[i][:,:,:,:,:-1,...]*grad_med1[i]
                grad_mvd1[i] = grad_med1[i] + er[i][:,:,:,:,:-1,...]*grad_med0[i]
                grad_r[i][:,:,:,:,:-1,...] += grad_meu1[i]*nmvu0[i] + grad_meu0[i]*nmvu1[i] + grad_med1[i]*nmvd0[i] + grad_med0[i]*nmvd1[i]
        
        
        #backprop through renormalisation
        for i in range(dims):
            grad_mvu0[i], grad_mvu1[i] = dnmvu1[i]*(grad_mvu0[i] - grad_mvu1[i]), dnmvu0[i]*(grad_mvu1[i] - grad_mvu0[i])
            grad_mvd0[i], grad_mvd1[i] = dnmvd1[i]*(grad_mvd0[i] - grad_mvd1[i]), dnmvd0[i]*(grad_mvd1[i] - grad_mvd0[i])

        #clear gradient buffers for the edge messages (about to compute updated ones
        for i in range(dims):
            grad_med0[i] *= 0
            grad_meu0[i] *= 0
            grad_med1[i] *= 0
            grad_meu1[i] *= 0

        #handle data terms
        for ni in range(dims):
            #print("grad_mvu1",grad_mvu1[ni])
            #print("mvu1",mvu1[ni])
            if ni == 0:
                grad_d1[:,:,:-1,...] += (grad_mvu1[ni] * mvu1[ni])*(d[:,:,:-1,...]>0)
                grad_d1[:,:,1:,...] += (grad_mvd1[ni] * mvd1[ni])*(d[:,:,1:,...]>0)
                grad_d0[:,:,:-1,...] += (grad_mvu0[ni] * mvu0[ni])*(d[:,:,:-1,...]<0)
                grad_d0[:,:,1:,...] += (grad_mvd0[ni] * mvd0[ni])*(d[:,:,1:,...]<0)
            elif ni == 1:
                grad_d1[:,:,:,:-1,...] += (grad_mvu1[ni] * mvu1[ni])*(d[:,:,:,:-1,...]>0)
                grad_d1[:,:,:,1:,...] += (grad_mvd1[ni] * mvd1[ni])*(d[:,:,:,1:,...]>0)
                grad_d0[:,:,:,:-1,...] += (grad_mvu0[ni] * mvu0[ni])*(d[:,:,:,:-1,...]<0)
                grad_d0[:,:,:,1:,...] += (grad_mvd0[ni] * mvd0[ni])*(d[:,:,:,1:,...]<0)
            elif ni == 2:
                grad_d1[:,:,:,:,:-1,...] += (grad_mvu1[ni] * mvu1[ni])*(d[:,:,:,:,:-1,...]>0)
                grad_d1[:,:,:,:,1:,...] += (grad_mvd1[ni] * mvd1[ni])*(d[:,:,:,:,1:,...]>0)
                grad_d0[:,:,:,:,:-1,...] += (grad_mvu0[ni] * mvu0[ni])*(d[:,:,:,:,:-1,...]<0)
                grad_d0[:,:,:,:,1:,...] += (grad_mvd0[ni] * mvd0[ni])*(d[:,:,:,:,1:,...]<0)
            
        #handle colinear edges
        for ni in range(dims):
            if ni == 0:
                grad_meu1[ni][:,:,:-1,...] += grad_mvu1[ni][:,:,1:,...] * mvu1[ni][:,:,1:,...] / meu1[ni][:,:,:-1,...]
                grad_meu0[ni][:,:,:-1,...] += grad_mvu0[ni][:,:,1:,...] * mvu0[ni][:,:,1:,...] / meu0[ni][:,:,:-1,...]
                grad_med1[ni][:,:,1:,...] += grad_mvd1[ni][:,:,:-1,...] * mvd1[ni][:,:,:-1,...] / med1[ni][:,:,1:,...]
                grad_med0[ni][:,:,1:,...] += grad_mvd0[ni][:,:,:-1,...] * mvd0[ni][:,:,:-1,...] / med0[ni][:,:,1:,...]
            elif ni == 1:
                grad_meu1[ni][:,:,:,:-1,...] += grad_mvu1[ni][:,:,:,1:,...] * mvu1[ni][:,:,:,1:,...] / meu1[ni][:,:,:,:-1,...]
                grad_meu0[ni][:,:,:,:-1,...] += grad_mvu0[ni][:,:,:,1:,...] * mvu0[ni][:,:,:,1:,...] / meu0[ni][:,:,:,:-1,...]
                grad_med1[ni][:,:,:,1:,...] += grad_mvd1[ni][:,:,:,:-1,...] * mvd1[ni][:,:,:,:-1,...] / med1[ni][:,:,:,1:,...]
                grad_med0[ni][:,:,:,1:,...] += grad_mvd0[ni][:,:,:,:-1,...] * mvd0[ni][:,:,:,:-1,...] / med0[ni][:,:,:,1:,...]
            elif ni == 2:
                grad_meu1[ni][:,:,:,:,:-1,...] += grad_mvu1[ni][:,:,:,:,1:,...] * mvu1[ni][:,:,:,:,1:,...] / meu1[ni][:,:,:,:,:-1,...]
                grad_meu0[ni][:,:,:,:,:-1,...] += grad_mvu0[ni][:,:,:,:,1:,...] * mvu0[ni][:,:,:,:,1:,...] / meu0[ni][:,:,:,:,:-1,...]
                grad_med1[ni][:,:,:,:,1:,...] += grad_mvd1[ni][:,:,:,:,:-1,...] * mvd1[ni][:,:,:,:,:-1,...] / med1[ni][:,:,:,:,1:,...]
                grad_med0[ni][:,:,:,:,1:,...] += grad_mvd0[ni][:,:,:,:,:-1,...] * mvd0[ni][:,:,:,:,:-1,...] / med0[ni][:,:,:,:,1:,...]
        
        
        
        
        
        
        #put in contributions from other directions
        for di2 in range(dims):
            if ni == di2:
                continue

            if di2 == 0:
                grad_contribution_1_d_d = grad_mvd1[ni][:,:,:-1,...] * mvd1[ni][:,:,:-1,...]
                grad_contribution_1_u_d = grad_mvd1[ni][:,:,1:,...] * mvd1[ni][:,:,1:,...]
                grad_contribution_1_d_u = grad_mvu1[ni][:,:,:-1,...] * mvu1[ni][:,:,:-1,...]
                grad_contribution_1_u_u = grad_mvu1[ni][:,:,1:,...] * mvu1[ni][:,:,1:,...]
                grad_contribution_0_d_d = grad_mvd0[ni][:,:,:-1,...] * mvd0[ni][:,:,:-1,...]
                grad_contribution_0_u_d = grad_mvd0[ni][:,:,1:,...] * mvd0[ni][:,:,1:,...]
                grad_contribution_0_d_u = grad_mvu0[ni][:,:,:-1,...] * mvu0[ni][:,:,:-1,...]
                grad_contribution_0_u_u = grad_mvu0[ni][:,:,1:,...] * mvu0[ni][:,:,1:,...]
            elif di2 == 1:
                grad_contribution_1_d_d = grad_mvd1[ni][:,:,:,:-1,...] * mvd1[ni][:,:,:,:-1,...]
                grad_contribution_1_u_d = grad_mvd1[ni][:,:,:,1:,...] * mvd1[ni][:,:,:,1:,...]
                grad_contribution_1_d_u = grad_mvu1[ni][:,:,:,:-1,...] * mvu1[ni][:,:,:,:-1,...]
                grad_contribution_1_u_u = grad_mvu1[ni][:,:,:,1:,...] * mvu1[ni][:,:,:,1:,...]
                grad_contribution_0_d_d = grad_mvd0[ni][:,:,:,:-1,...] * mvd0[ni][:,:,:,:-1,...]
                grad_contribution_0_u_d = grad_mvd0[ni][:,:,:,1:,...] * mvd0[ni][:,:,:,1:,...]
                grad_contribution_0_d_u = grad_mvu0[ni][:,:,:,:-1,...] * mvu0[ni][:,:,:,:-1,...]
                grad_contribution_0_u_u = grad_mvu0[ni][:,:,:,1:,...] * mvu0[ni][:,:,:,1:,...]
            elif di2 == 3:
                grad_contribution_1_d_d = grad_mvd1[ni][:,:,:,:,:-1,...] * mvd1[ni][:,:,:,:,:-1,...]
                grad_contribution_1_u_d = grad_mvd1[ni][:,:,:,:,1:,...] * mvd1[ni][:,:,:,:,1:,...]
                grad_contribution_1_d_u = grad_mvu1[ni][:,:,:,:,:-1,...] * mvu1[ni][:,:,:,:,:-1,...]
                grad_contribution_1_u_u = grad_mvu1[ni][:,:,:,:,1:,...] * mvu1[ni][:,:,:,:,1:,...]
                grad_contribution_0_d_d = grad_mvd0[ni][:,:,:,:,:-1,...] * mvd0[ni][:,:,:,:,:-1,...]
                grad_contribution_0_u_d = grad_mvd0[ni][:,:,:,:,1:,...] * mvd0[ni][:,:,:,:,1:,...]
                grad_contribution_0_d_u = grad_mvu0[ni][:,:,:,:,:-1,...] * mvu0[ni][:,:,:,:,:-1,...]
                grad_contribution_0_u_u = grad_mvu0[ni][:,:,:,:,1:,...] * mvu0[ni][:,:,:,:,1:,...]
            
            if ni == 0:
                grad_med1[di2][:,:,:-1,...] = grad_contribution_1_d_u / med1[di2][:,:,:-1,...]
                grad_meu1[di2][:,:,:-1,...] = grad_contribution_1_u_u / meu1[di2][:,:,:-1,...]
                grad_med1[di2][:,:,1:,...] = grad_contribution_1_d_d / med1[di2][:,:,1:,...]
                grad_meu1[di2][:,:,1:,...] = grad_contribution_1_u_d / meu1[di2][:,:,1:,...]
                grad_med0[di2][:,:,:-1,...] = grad_contribution_0_d_u / med0[di2][:,:,:-1,...]
                grad_meu0[di2][:,:,:-1,...] = grad_contribution_0_u_u / meu0[di2][:,:,:-1,...]
                grad_med0[di2][:,:,1:,...] = grad_contribution_0_d_d / med0[di2][:,:,1:,...]
                grad_meu0[di2][:,:,1:,...] = grad_contribution_0_u_d / meu0[di2][:,:,1:,...]
            elif ni == 1:
                grad_med1[di2][:,:,:,:-1,...] = grad_contribution_1_d_u / med1[di2][:,:,:,:-1,...]
                grad_meu1[di2][:,:,:,:-1,...] = grad_contribution_1_u_u / meu1[di2][:,:,:,:-1,...]
                grad_med1[di2][:,:,:,1:,...] = grad_contribution_1_d_d / med1[di2][:,:,:,1:,...]
                grad_meu1[di2][:,:,:,1:,...] = grad_contribution_1_u_d / meu1[di2][:,:,:,1:,...]
                grad_med0[di2][:,:,:,:-1,...] = grad_contribution_0_d_u / med0[di2][:,:,:,:-1,...]
                grad_meu0[di2][:,:,:,:-1,...] = grad_contribution_0_u_u / meu0[di2][:,:,:,:-1,...]
                grad_med0[di2][:,:,:,1:,...] = grad_contribution_0_d_d / med0[di2][:,:,:,1:,...]
                grad_meu0[di2][:,:,:,1:,...] = grad_contribution_0_u_d / meu0[di2][:,:,:,1:,...]
            elif ni == 2:
                grad_med1[di2][:,:,:,:,:-1,...] = grad_contribution_1_d_u / med1[di2][:,:,:,:,:-1,...]
                grad_meu1[di2][:,:,:,:,:-1,...] = grad_contribution_1_u_u / meu1[di2][:,:,:,:,:-1,...]
                grad_med1[di2][:,:,:,:,1:,...] = grad_contribution_1_d_d / med1[di2][:,:,:,:,1:,...]
                grad_meu1[di2][:,:,:,:,1:,...] = grad_contribution_1_u_d / meu1[di2][:,:,:,:,1:,...]
                grad_med0[di2][:,:,:,:,:-1,...] = grad_contribution_0_d_u / med0[di2][:,:,:,:,:-1,...]
                grad_meu0[di2][:,:,:,:,:-1,...] = grad_contribution_0_u_u / meu0[di2][:,:,:,:,:-1,...]
                grad_med0[di2][:,:,:,:,1:,...] = grad_contribution_0_d_d / med0[di2][:,:,:,:,1:,...]
                grad_meu0[di2][:,:,:,:,1:,...] = grad_contribution_0_u_d / meu0[di2][:,:,:,:,1:,...]
        
        
        
        

        iter+=1
        max_grad = max([torch.max(gm) for gm in grad_med0] +
                       [torch.max(gm) for gm in grad_med1] +
                       [torch.max(gm) for gm in grad_meu0] +
                       [torch.max(gm) for gm in grad_meu1])
        #print("Binary_LBPnD_PytorchNative_backward",iter,":",max_grad,end="\n")
        if max_grad <= epsilon or iter > 10*sumels:
            break

    #correct to no longer be in the exponential domain
    for i in range(dims):
        grad_r[i] *= -er[i]
    grad_d = grad_d1 - grad_d0
    
    return grad_d, grad_r
    
def Binary_MFEnD_forward(d,r):
    epsilon = 10**-4
    dims = len(r)
    numels = prod(d.shape[2:])
    sumels = sum(d.shape[2:])
    DeepFlowFunction.check_var_dims([d,]+r,dims)
    u = torch.sigmoid(d)
    iters = 0
    while True:

        #initialise the new energy estimate with the data terms and then add in the regularisation components
        energy = d.clone()
        for i in range(dims):
            if i == 0:
                energy[:,:,:-1,...] += r[i][:,:,:-1,...]*(2*u[:,:,1:,...]-1)
                energy[:,:,1:,...] += r[i][:,:,:-1,...]*(2*u[:,:,:-1,...]-1)
            elif i == 1:
                energy[:,:,:,:-1,...] += r[i][:,:,:,:-1,...]*(2*u[:,:,:,1:,...]-1)
                energy[:,:,:,1:,...] += r[i][:,:,:,:-1,...]*(2*u[:,:,:,:-1,...]-1)
            elif i == 2:
                energy[:,:,:,:,:-1,...] += r[i][:,:,:,:,:-1,...]*(2*u[:,:,:,:,1:,...]-1)
                energy[:,:,:,:,1:,...] += r[i][:,:,:,:,:-1,...]*(2*u[:,:,:,:,:-1,...]-1)
                
        #create the new marginals and determine if we need to break
        u_new = torch.sigmoid(energy)
        change = torch.max(torch.abs(u_new-u))
        u = 0.25*u+0.75*u_new
        iters += 1
        #print("Binary_MFEnD_forward",iters,":",change,end="\n")
        if change < epsilon or iters > 10*sumels:
            break

    #estimate joint marginals
    j = []
    for i in range(dims):
        if i == 0:
            e01 = energy[:,:,1:,...] - 2*r[i][:,:,:-1,...]*u[:,:,:-1,...]
            e10 = energy[:,:,:-1,...] - 2*r[i][:,:,:-1,...]*u[:,:,1:,...]
            e11 = energy[:,:,1:,...] + energy[:,:,:-1,...] - 2*r[i][:,:,:-1,...]*(u[:,:,:-1,...]+u[:,:,1:,...]-1) 
        elif i == 1:
            e01 = energy[:,:,:,1:,...] - 2*r[i][:,:,:,:-1,...]*u[:,:,:,:-1,...]
            e10 = energy[:,:,:,:-1,...] - 2*r[i][:,:,:,:-1,...]*u[:,:,:,1:,...]
            e11 = energy[:,:,:,1:,...] + energy[:,:,:,:-1,...] - 2*r[i][:,:,:,:-1,...]*(u[:,:,:,:-1,...]+u[:,:,:,1:,...]-1)
        elif i == 2:
            e01 = energy[:,:,:,:,1:,...] - 2*r[i][:,:,:,:,:-1,...]*u[:,:,:,:,:-1,...]
            e10 = energy[:,:,:,:,:-1,...] - 2*r[i][:,:,:,:,:-1,...]*u[:,:,:,:,1:,...]
            e11 = energy[:,:,:,:,1:,...] + energy[:,:,:,:,:-1,...] - 2*r[i][:,:,:,:,:-1,...]*(u[:,:,:,:,:-1,...]+u[:,:,:,:,1:,...]-1)
        e00 = -torch.clip(torch.maximum(torch.maximum(e01,e10),e11),min=0)
        e01 += e00
        e10 += e00
        e11 += e00
        u00 = torch.exp(e00)
        u01 = torch.exp(e01)
        u10 = torch.exp(e10)
        u11 = torch.exp(e11)
        j.append( (u10+u01)/(u00+u01+u10+u11) )
    
    return u, j
    
def Binary_MFEnD_backward(grad_u, grad_j, u, d, r):
    dims = len(r)
    numels = prod(d.shape[2:])
    sumels = sum(d.shape[2:])
    
    #use an easy exit if there are no non-zero gradients
    max_grad = max([torch.max(torch.abs(grad_u)),]+[torch.max(torch.abs(gj)) for gj in grad_j])
    #print(max_grad)
    if max_grad == 0:
        return torch.zeros_like(d), [torch.zeros_like(ri) for ri in r]
    
    grad_u = grad_u.clone()
    grad_d = torch.zeros_like(d)
    grad_e = torch.zeros_like(u)
    grad_r = [torch.zeros_like(r[i]) for i in range(dims)]

    #recalculate the energy
    energy = d.clone()
    for i in range(dims):
        if i == 0:
            energy[:,:,:-1,...] += r[i][:,:,:-1,...]*(2*u[:,:,1:,...]-1)
            energy[:,:,1:,...] += r[i][:,:,:-1,...]*(2*u[:,:,:-1,...]-1)
        elif i == 1:
            energy[:,:,:,:-1,...] += r[i][:,:,:,:-1,...]*(2*u[:,:,:,1:,...]-1)
            energy[:,:,:,1:,...] += r[i][:,:,:,:-1,...]*(2*u[:,:,:,:-1,...]-1)
        elif i == 2:
            energy[:,:,:,:,:-1,...] += r[i][:,:,:,:,:-1,...]*(2*u[:,:,:,:,1:,...]-1)
            energy[:,:,:,:,1:,...] += r[i][:,:,:,:,:-1,...]*(2*u[:,:,:,:,:-1,...]-1)
    
    #backprop through final joint marginal estimates
    for i in range(dims):
        if i == 0:
            e01 = energy[:,:,1:,...] - 2*r[i][:,:,:-1,...]*u[:,:,:-1,...]
            e10 = energy[:,:,:-1,...] - 2*r[i][:,:,:-1,...]*u[:,:,1:,...]
            e11 = energy[:,:,1:,...] + energy[:,:,:-1,...] - 2*r[i][:,:,:-1,...]*(u[:,:,:-1,...]+u[:,:,1:,...]-1)
        elif i == 1:
            e01 = energy[:,:,:,1:,...] - 2*r[i][:,:,:,:-1,...]*u[:,:,:,:-1,...]
            e10 = energy[:,:,:,:-1,...] - 2*r[i][:,:,:,:-1,...]*u[:,:,:,1:,...]
            e11 = energy[:,:,:,1:,...] + energy[:,:,:,:-1,...] - 2*r[i][:,:,:,:-1,...]*(u[:,:,:,:-1,...]+u[:,:,:,1:,...]-1)
        elif i == 2:
            e01 = energy[:,:,:,:,1:,...] - 2*r[i][:,:,:,:,:-1,...]*u[:,:,:,:,:-1,...]
            e10 = energy[:,:,:,:,:-1,...] - 2*r[i][:,:,:,:,:-1,...]*u[:,:,:,:,1:,...]
            e11 = energy[:,:,:,:,1:,...] + energy[:,:,:,:,:-1,...] - 2*r[i][:,:,:,:,:-1,...]*(u[:,:,:,:,:-1,...]+u[:,:,:,:,1:,...]-1)
        
        #changing the 00 baseline away from 0 prevents some instability when the energies grow too large
        e00 = -torch.clip(torch.maximum(torch.maximum(e01,e10),e11),min=0)
        u00 = torch.exp(e00)
        u01 = torch.exp(e01+e00)
        u10 = torch.exp(e10+e00)
        u11 = torch.exp(e11+e00)
        
        #don't have to worry about the change to u00 as it turns into a constant (squared) for both num and denom, so we can pretend it never happened
        den = (u00+u01+u10+u11)**2
        grad_e01 = grad_j[i] * u01 * (u00+u11) / den
        grad_e10 = grad_j[i] * u10 * (u00+u11) / den
        grad_e11 = -grad_j[i] * u11 * (u01+u10) / den
        
        if i == 0:
            grad_e[:,:,1:,...] += grad_e01 + grad_e11
            grad_e[:,:,:-1,...] += grad_e10 + grad_e11
            
            grad_r[i][:,:,:-1,...] -= 2 * (grad_e01 * u[:,:,:-1,...] + grad_e10 * u[:,:,1:,...] + grad_e11 * (u[:,:,:-1,...]+u[:,:,1:,...]-1))
            
            grad_u[:,:,:-1,...] -= 2 * (grad_e01 + grad_e11) * r[i][:,:,:-1,...]
            grad_u[:,:,1:,...] -= 2 * (grad_e10 + grad_e11) * r[i][:,:,:-1,...]
            
        elif i == 1:
            grad_e[:,:,:,1:,...] += grad_e01 + grad_e11
            grad_e[:,:,:,:-1,...] += grad_e10 + grad_e11
            
            grad_r[i][:,:,:,:-1,...] -= 2 * (grad_e01 * u[:,:,:,:-1,...] + grad_e10 * u[:,:,:,1:,...] + grad_e11 * (u[:,:,:,:-1,...]+u[:,:,:,1:,...]-1))
            
            grad_u[:,:,:,:-1,...] -= 2 * (grad_e01 + grad_e11) * r[i][:,:,:,:-1,...]
            grad_u[:,:,:,1:,...] -= 2 * (grad_e10 + grad_e11) * r[i][:,:,:,:-1,...]
            
        elif i == 2:
            grad_e[:,:,:,:,1:,...] += grad_e01 + grad_e11
            grad_e[:,:,:,:,:-1,...] += grad_e10 + grad_e11
            
            grad_r[i][:,:,:,:,:-1,...] -= 2 * (grad_e01 * u[:,:,:,:,:-1,...] + grad_e10 * u[:,:,:,:,1:,...] + grad_e11 * (u[:,:,:,:,:-1,...]+u[:,:,:,:,1:,...]-1))
            
            grad_u[:,:,:,:,:-1,...] -= 2 * (grad_e01 + grad_e11) * r[i][:,:,:,:,:-1,...]
            grad_u[:,:,:,:,1:,...] -= 2 * (grad_e10 + grad_e11) * r[i][:,:,:,:,:-1,...]

    #backprop through final sigmoid layer (for the marginal estimates)
    grad_e += grad_u * u * (1-u)
    change=torch.max(torch.abs(grad_e))
    epsilon = 10**-4*change
    
    #now backprop through the regular layers
    iter = 0
    while True:

        grad_u = torch.zeros_like(u)
        grad_d += grad_e
        for i in range(dims):
            if i == 0:
                grad_r[i][:,:,:-1,...] += (2*u[:,:,1:,...]-1)* grad_e[:,:,:-1,...]
                grad_r[i][:,:,:-1,...] += (2*u[:,:,:-1,...]-1)* grad_e[:,:,1:,...]
                grad_u[:,:,1:,...] += 2*grad_e[:,:,:-1,...]*r[i][:,:,:-1,...]
                grad_u[:,:,:-1,...] += 2*grad_e[:,:,1:,...]*r[i][:,:,:-1,...]
            elif i == 1:
                grad_r[i][:,:,:,:-1,...] += (2*u[:,:,:,1:,...]-1)* grad_e[:,:,:,:-1,...]
                grad_r[i][:,:,:,:-1,...] += (2*u[:,:,:,:-1,...]-1)* grad_e[:,:,:,1:,...]
                grad_u[:,:,:,1:,...] += 2*grad_e[:,:,:,:-1,...]*r[i][:,:,:,:-1,...]
                grad_u[:,:,:,:-1,...] += 2*grad_e[:,:,:,1:,...]*r[i][:,:,:,:-1,...]
            elif i == 2:
                grad_r[i][:,:,:,:,:-1,...] += (2*u[:,:,:,:,1:,...]-1)* grad_e[:,:,:,:,:-1,...]
                grad_r[i][:,:,:,:,:-1,...] += (2*u[:,:,:,:,:-1,...]-1)* grad_e[:,:,:,:,1:,...]
                grad_u[:,:,:,:,1:,...] += 2*grad_e[:,:,:,:,:-1,...]*r[i][:,:,:,:,:-1,...]
                grad_u[:,:,:,:,:-1,...] += 2*grad_e[:,:,:,:,1:,...]*r[i][:,:,:,:,:-1,...]

        grad_e = grad_u * u * (1-u)
        change=torch.max(torch.abs(grad_e))

        iter+=1
        #print("Binary_MFEnD_backward",iter,":",change,end="\n")
        if change < epsilon or iter > 10*sumels:
            break
            
    return grad_d, grad_r
    
    
    
class Binary_MAP1d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx):
        DeepFlowFunction.check_var_dims([d,rx],1)
        d, rx = d.contiguous(), rx.contiguous()
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.binary_auglag_1d_gpu_forward(d,rx, output)
        else:
            deepflow.binary_auglag_1d_cpu_forward(d,rx, output)
        return (output > 0.5).float()
            
    #For the optimisers, there is no well defined backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input
    
class Binary_MAP2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry):
        DeepFlowFunction.check_var_dims([d,rx,ry],2)
        d, rx, ry = d.contiguous(), rx.contiguous(), ry.contiguous()
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.binary_auglag_2d_gpu_forward(d,rx, ry, output)
        else:
            deepflow.binary_auglag_2d_cpu_forward(d,rx, ry, output)
        return (output > 0.5).float()
            
    #For the optimisers, there is no well defined backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input
            
class Binary_MAP3d(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3)
        d, rx, ry, rz = d.contiguous(), rx.contiguous(), ry.contiguous(), rz.contiguous()
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.binary_auglag_3d_gpu_forward(d, rx, ry, rz, output)
        else:
            deepflow.binary_auglag_3d_cpu_forward(d, rx, ry, rz, output)
        return (output > 0.5).float()
        
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input

class Binary_Mean1d_PytorchNative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, rx):
        u, j = Binary_MFEnD_forward(d,[rx,])
        ctx.save_for_backward(u,d,rx)
        return u, j[0]
            
    @staticmethod
    def backward(ctx, grad_u, grad_jx):
        u,d,rx = ctx.saved_tensors
        grad_d, grad_r = Binary_MFEnD_backward(grad_u, [grad_jx], u, d, [rx])
        return grad_d, grad_r[0]

    
class Binary_Mean2d_PytorchNative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, rx, ry):
        u, j = Binary_MFEnD_forward(d,[rx,ry,])
        ctx.save_for_backward(u,d,rx,ry)
        return u, j[0], j[1]
            
    @staticmethod
    def backward(ctx, grad_u, grad_jx, grad_jy):
        u,d,rx,ry = ctx.saved_tensors
        grad_d, grad_r = Binary_MFEnD_backward(grad_u, [grad_jx,grad_jy,], u, d, [rx,ry,])
        return grad_d, grad_r[0], grad_r[1]
    

class Binary_Mean3d_PytorchNative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        u, j = Binary_MFEnD_forward(d,[rx,ry,rz,])
        ctx.save_for_backward(u,d,rx,ry,rz)
        return u, j[0], j[1], j[2]
            
    @staticmethod
    def backward(ctx, grad_u, grad_jx, grad_jy, grad_jz):
        u,d,rx,ry,rz = ctx.saved_tensors
        grad_d, grad_r = Binary_MFEnD_backward(grad_u, [grad_jx,grad_jy,grad_jz,], u, d, [rx,ry,rz,])
        return grad_d, grad_r[0], grad_r[1], grad_r[2]
    
    
class Binary_LBP1d_PytorchNative(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx):
        meu0, med0, meu1, med1, num_iters = Binary_LBPnD_PytorchNative_forward(d,[rx,])
        mexu0,mexu1,mexd0,mexd1 = meu0[0],meu1[0],med0[0],med1[0]
        ctx.save_for_backward(d,rx,mexu0,mexu1,mexd0,mexd1,torch.tensor(num_iters))
        return mexu0,mexu1,mexd0,mexd1
    
            
    @staticmethod
    def backward(ctx, grad_mexu0, grad_mexu1, grad_mexd0, grad_mexd1):
        d,rx,mexu0,mexu1,mexd0,mexd1,max_num_iters = ctx.saved_tensors
        
        grad_d, grad_r = Binary_LBPnD_PytorchNative_backward([grad_mexu0,], [grad_mexd0,],
                                                      [grad_mexu1,], [grad_mexd1,],
                                                      [mexu0,], [mexd0,],
                                                      [mexu1,], [mexd1,],
                                                      d, [rx,], 
                                                      int(max_num_iters.cpu().item()))
        return grad_d, grad_r[0]

class Binary_LBP2d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-4
    
    @staticmethod
    def forward(ctx, d, rx, ry):
        meu0, med0, meu1, med1, num_iters = Binary_LBPnD_PytorchNative_forward(d,[rx,ry,])
        mexu0,mexu1,mexd0,mexd1 = meu0[0],meu1[0],med0[0],med1[0]
        meyu0,meyu1,meyd0,meyd1 = meu0[1],meu1[1],med0[1],med1[1]
        ctx.save_for_backward(d,rx,ry,mexu0,mexu1,mexd0,mexd1,
                                      meyu0,meyu1,meyd0,meyd1,
                                      torch.tensor(num_iters))
        return mexu0,mexu1,mexd0,mexd1,meyu0,meyu1,meyd0,meyd1
    
    @staticmethod 
    def backward(ctx, grad_mexu0,grad_mexu1,grad_mexd0,grad_mexd1,grad_meyu0,grad_meyu1,grad_meyd0,grad_meyd1):
        d,rx,ry,mexu0,mexu1,mexd0,mexd1,meyu0,meyu1,meyd0,meyd1,max_num_iters = ctx.saved_tensors
        
        grad_d, grad_r = Binary_LBPnD_PytorchNative_backward([grad_mexu0,grad_meyu0,], [grad_mexd0,grad_meyd0,],
                                                      [grad_mexu1,grad_meyu1,], [grad_mexd1,grad_meyd1,],
                                                      [mexu0,meyu0,], [mexd0,meyd0,],
                                                      [mexu1,meyu1,], [mexd1,meyd1,],
                                                      d, [rx,ry,], 
                                                      int(max_num_iters.cpu().item()))
                        
        return grad_d, grad_r[0], grad_r[1]
     
    
class Binary_LBP3d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-4
    
    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        meu0, med0, meu1, med1, num_iters = Binary_LBPnD_PytorchNative_forward(d,[rx,ry,rz,])
        mexu0,mexu1,mexd0,mexd1 = meu0[0],meu1[0],med0[0],med1[0]
        meyu0,meyu1,meyd0,meyd1 = meu0[1],meu1[1],med0[1],med1[1]
        mezu0,mezu1,mezd0,mezd1 = meu0[2],meu1[2],med0[2],med1[2]
        ctx.save_for_backward(d,rx,ry,rz,mexu0,mexu1,mexd0,mexd1,
                                         meyu0,meyu1,meyd0,meyd1,
                                         mezu0,mezu1,mezd0,mezd1,
                                         torch.tensor(num_iters))
        
        return mexu0,mexu1,mexd0,mexd1,meyu0,meyu1,meyd0,meyd1,mezu0,mezu1,mezd0,mezd1
    
    @staticmethod 
    def backward(ctx, grad_mexu0, grad_mexu1, grad_mexd0, grad_mexd1,
                      grad_meyu0, grad_meyu1, grad_meyd0, grad_meyd1,
                      grad_mezu0, grad_mezu1, grad_mezd0, grad_mezd1):
        
        d,rx,ry,rz,mexu0,mexu1,mexd0,mexd1,meyu0,meyu1,meyd0,meyd1,mezu0,mezu1,mezd0,mezd1,max_num_iters = ctx.saved_tensors
        
        grad_d, grad_r = Binary_LBPnD_PytorchNative_backward([grad_mexu0,grad_meyu0,grad_mezu0,], [grad_mexd0,grad_meyd0,grad_mezd0,],
                                                      [grad_mexu1,grad_meyu1,grad_mezu1,], [grad_mexd1,grad_meyd1,grad_mezd1,],
                                                      [mexu0,meyu0,mezu0,], [mexd0,meyd0,mezd0,],
                                                      [mexu1,meyu1,mezu1,], [mexd1,meyd1,mezd1,],
                                                      d, [rx,ry,rz,], 
                                                      int(max_num_iters.cpu().item()))
        return grad_d, grad_r[0], grad_r[1], grad_r[2]

    
        
"""        
class Binary_Mean1d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx):
        DeepFlowFunction.check_var_dims([d,rx],1)
        d, rx = d.contiguous(), rx.contiguous()
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.binary_meanpass_1d_gpu_forward(d,rx, output)
        else:
            deepflow.binary_meanpass_1d_cpu_forward(d,rx, output)
        ctx.save_for_backward(output,rx)
        return output
            
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        u,rx, = ctx.saved_tensors
        grad_output = grad_output.clone()
        u = u.clone()
        rx = rx.clone()
        grad_d =  torch.zeros_like(u)
        grad_rx = torch.zeros_like(rx)
        if u.is_cuda:
            deepflow.binary_meanpass_1d_gpu_backward(grad_output, u, rx, grad_d, grad_rx)
        else:
            deepflow.binary_meanpass_1d_cpu_backward(grad_output, u, rx, grad_d, grad_rx)
        return grad_d, grad_rx
"""
"""
class Binary_Mean2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry):
        DeepFlowFunction.check_var_dims([d,rx,ry],2)
        d, rx, ry = d.contiguous(), rx.contiguous(), ry.contiguous()
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.binary_meanpass_2d_gpu_forward(d,rx, ry, output)
        else:
            deepflow.binary_meanpass_2d_cpu_forward(d,rx, ry, output)
        ctx.save_for_backward(output,rx,ry)
        return output
            
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        u,rx,ry, = ctx.saved_tensors
        grad_output = grad_output.clone()
        u = u.clone()
        rx = rx.clone()
        ry = ry.clone()
        grad_d =  torch.zeros_like(u)
        grad_rx = torch.zeros_like(rx)
        grad_ry = torch.zeros_like(ry)
        if u.is_cuda:
            deepflow.binary_meanpass_2d_gpu_backward(grad_output, u, rx, ry, grad_d, grad_rx, grad_ry)
        else:
            deepflow.binary_meanpass_2d_cpu_backward(grad_output, u, rx, ry, grad_d, grad_rx, grad_ry)
        return grad_d, grad_rx, grad_ry
"""   

"""
class Binary_Mean3d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3)
        d, rx, ry, rz = d.contiguous(), rx.contiguous(), ry.contiguous(), rz.contiguous()
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.binary_meanpass_3d_gpu_forward(d,rx, ry, rz, output)
        else:
            deepflow.binary_meanpass_3d_cpu_forward(d,rx, ry, rz, output)
        ctx.save_for_backward(output,rx,ry,rz)
        return output
            
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        u,rx,ry,rz, = ctx.saved_tensors
        grad_output = grad_output.clone()
        u = u.clone()
        rx = rx.clone()
        ry = ry.clone()
        rz = rz.clone()
        grad_d =  torch.zeros_like(u)
        grad_rx = torch.zeros_like(rx)
        grad_ry = torch.zeros_like(ry)
        grad_rz = torch.zeros_like(rz)
        if u.is_cuda:
            deepflow.binary_meanpass_3d_gpu_backward(grad_output, u, rx, ry, rz, grad_d, grad_rx, grad_ry, grad_rz)
        else:
            deepflow.binary_meanpass_3d_cpu_backward(grad_output, u, rx, ry, rz, grad_d, grad_rx, grad_ry, grad_rz)
        return grad_d, grad_rx, grad_ry, grad_rz
"""