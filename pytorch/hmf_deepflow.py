import torch
import numpy as np
import deepflow
from deepflow_function import DeepFlowFunction
from unhmf_deepflow import UNHMF_LBP1d_PytorchNative, UNHMF_LBP2d_PytorchNative, UNHMF_LBP3d_PytorchNative

def check_ordering(p,b,l):
    p = p.cpu().numpy().copy()
    is_leaf = np.ones(b,dtype=int)
    if len(p) != b:
        raise Exception("Length of the ordering is not correct")
    for i,pspec in enumerate(p):
        if i < pspec:
            raise Exception("Parents must occur before children in the tree",i,pspec)
        if pspec > -1:
            is_leaf[pspec] = 0
    if np.sum(is_leaf) != l:
        raise Exception("Incorrect number of leaves")

def mix_regularisation_terms(p,b,l,device,rx,sx,ry=None,sy=None,rz=None,sz=None):
    p = p.detach().cpu().numpy().astype(int)
    if sy is not None:
        if sz is not None:
            rx_eff = torch.zeros((b,l,l,sx,sy,sz),device=device)
            ry_eff = torch.zeros((b,l,l,sx,sy,sz),device=device)
            rz_eff = torch.zeros((b,l,l,sx,sy,sz),device=device)
        else:
            rx_eff = torch.zeros((b,l,l,sx,sy),device=device)
            ry_eff = torch.zeros((b,l,l,sx,sy),device=device)
    else:
        rx_eff = torch.zeros((b,l,l,sx),device=device)
        
    #find the leaves
    leafs = []
    for label,parent in enumerate(p):
        if parent in leafs:
            leafs.remove(parent)
        leafs.append(label)
        
    #go through every pair and add reg terms for non-shared parents
    for i in range(l):
        for j in range(i+1,l):
            pi = leafs[i]
            pj = leafs[j]
            while pi != pj:
                if pj > pi:
                    rx_eff[:,i,j,...] += rx[:,pj,...]
                    if sy is not None:
                        ry_eff[:,i,j,...] += ry[:,pj,...]
                    if sz is not None:
                        rz_eff[:,i,j,...] += rz[:,pj,...]
                    pj = p[pj]
                else: #pi < pj
                    rx_eff[:,i,j,...] += rx[:,pi,...]
                    if sy is not None:
                        ry_eff[:,i,j,...] += ry[:,pi,...]
                    if sz is not None:
                        rz_eff[:,i,j,...] += rz[:,pi,...]
                    pi = p[pi]
            rx_eff[:,j,i,...] = rx_eff[:,i,j,...]
            if sy is not None:
                ry_eff[:,j,i,...] = ry_eff[:,i,j,...]
            if sz is not None:
                rz_eff[:,j,i,...] = rz_eff[:,i,j,...]
                
    #return the containers for the grouped terms
    if sz is not None:
        return rx_eff, ry_eff, rz_eff
    if sy is not None:
        return rx_eff, ry_eff
    return rx_eff

        
class HMF_MAP1d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, p):
        DeepFlowFunction.check_var_dims([d,rx],1,True)
        check_ordering(p,rx.shape[1],d.shape[1])
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.hmf_auglag_1d_gpu_forward(d,rx, output, p)
        else:
            deepflow.hmf_auglag_1d_cpu_forward(d,rx, output, p)
        return output
            
    #For the optimisers, there is no well defined backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input
        
class HMF_MAP2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry, p):
        DeepFlowFunction.check_var_dims([d,rx,ry],2,True)
        check_ordering(p,rx.shape[1],d.shape[1])
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.hmf_auglag_2d_gpu_forward(d,rx, ry, output, p)
        else:
            deepflow.hmf_auglag_2d_cpu_forward(d,rx, ry, output, p)
        return output
            
    #For the optimisers, there is no well defined backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input
            
class HMF_MAP3d(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, d, rx, ry, rz, p):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3,True)
        check_ordering(p,rx.shape[1],d.shape[1])
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.hmf_auglag_3d_gpu_forward(d, rx, ry, rz, output, p)
        else:
            deepflow.hmf_auglag_3d_cpu_forward(d, rx, ry, rz, output, p)
        return output
        
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input
        
        
class HMF_Mean1d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, p):
        DeepFlowFunction.check_var_dims([d,rx],1,True)
        check_ordering(p,rx.shape[1],d.shape[1])
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.hmf_meanpass_1d_gpu_forward(d,rx, output, p)
        else:
            deepflow.hmf_meanpass_1d_cpu_forward(d,rx, output, p)
        ctx.save_for_backward(output,rx,p)
        return output
            
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        u,rx,p, = ctx.saved_tensors
        grad_output = grad_output.clone()
        u = u.clone()
        rx = rx.clone()
        grad_d =  torch.zeros_like(u)
        grad_rx = torch.zeros_like(rx)
        if u.is_cuda:
            deepflow.hmf_meanpass_1d_gpu_backward(grad_output, u, rx, grad_d, grad_rx, p)
        else:
            deepflow.hmf_meanpass_1d_cpu_backward(grad_output, u, rx, grad_d, grad_rx, p)
        return grad_d, grad_rx, torch.zeros_like(p)

class HMF_Mean2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry, p):
        DeepFlowFunction.check_var_dims([d,rx,ry],2,True)
        check_ordering(p,rx.shape[1],d.shape[1])
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.hmf_meanpass_2d_gpu_forward(d,rx, ry, output, p)
        else:
            deepflow.hmf_meanpass_2d_cpu_forward(d,rx, ry, output, p)
        ctx.save_for_backward(output,rx,ry,p)
        return output
            
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        u,rx,ry,p, = ctx.saved_tensors
        grad_output = grad_output.clone()
        u = u.clone()
        rx = rx.clone()
        ry = ry.clone()
        grad_d =  torch.zeros_like(u)
        grad_rx = torch.zeros_like(rx)
        grad_ry = torch.zeros_like(ry)
        if u.is_cuda:
            deepflow.hmf_meanpass_2d_gpu_backward(grad_output, u, rx, ry, grad_d, grad_rx, grad_ry, p)
        else:
            deepflow.hmf_meanpass_2d_cpu_backward(grad_output, u, rx, ry, grad_d, grad_rx, grad_ry, p)
        return grad_d, grad_rx, grad_ry, torch.zeros_like(p)
        
class HMF_Mean3d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry, rz, p):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3,True)
        check_ordering(p,rx.shape[1],d.shape[1])
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.hmf_meanpass_3d_gpu_forward(d,rx, ry, rz, output, p)
        else:
            deepflow.hmf_meanpass_3d_cpu_forward(d,rx, ry, rz, output, p)
        ctx.save_for_backward(output,rx,ry,rz,p)
        return output
            
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        u,rx,ry,rz,p, = ctx.saved_tensors
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
            deepflow.hmf_meanpass_3d_gpu_backward(grad_output, u, rx, ry, rz, grad_d, grad_rx, grad_ry, grad_rz, p)
        else:
            deepflow.hmf_meanpass_3d_cpu_backward(grad_output, u, rx, ry, rz, grad_d, grad_rx, grad_ry, grad_rz, p)
        return grad_d, grad_rx, grad_ry, grad_rz, torch.zeros_like(p)
        
        
class HMF_LBP1d_PytorchNative:

    @staticmethod
    def apply(d, rx, p):
        DeepFlowFunction.check_var_dims([d,rx],1,True)
        b,c,sx = d.shape[0],d.shape[1],d.shape[2]
        check_ordering(p,rx.shape[1],d.shape[1])
        rx_eff = mix_regularisation_terms(p,b,c,d.device,rx,sx)
        return UNHMF_LBP1d_PytorchNative.apply(d,rx_eff)
    
class HMF_LBP2d_PytorchNative:

    @staticmethod
    def apply(d, rx, ry, p):
        DeepFlowFunction.check_var_dims([d,rx,ry],2,True)
        b,c,sx,sy = d.shape[0],d.shape[1],d.shape[2],d.shape[3]
        check_ordering(p,rx.shape[1],d.shape[1])
        rx_eff, ry_eff = mix_regularisation_terms(p,b,c,d.device,rx,sx,ry,sy)
        return UNHMF_LBP2d_PytorchNative.apply(d,rx_eff,ry_eff)
        
class HMF_LBP3d_PytorchNative:

    @staticmethod
    def apply(d, rx, ry, rz, p):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3,True)
        b,c,sx,sy,sz = d.shape[0],d.shape[1],d.shape[2],d.shape[3],d.shape[4]
        check_ordering(p,rx.shape[1],d.shape[1])
        rx_eff, ry_eff, rz_eff = mix_regularisation_terms(p,b,c,d.device,rx,sx,ry,sy,rz,sz)
        return UNHMF_LBP3d_PytorchNative.apply(d,rx_eff,ry_eff,rz_eff)
            