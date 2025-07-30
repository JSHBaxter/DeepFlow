import torch
import deepflow
from deepflow_function import DeepFlowFunction
import math
import numpy as np

def check_reg_channel_number(regs,c):
    for r in regs:
        if r.shape[1] != c or r.shape[2] != c:
            raise Exception("Regularisation terms do not have correct structure\n")
            return

class UNHMF_LBP1d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-4
    
    @staticmethod
    def forward(ctx, d, rx):
        b,c,sx = d.shape[0],d.shape[1],d.shape[2]
        check_reg_channel_number([rx,],c)
        DeepFlowFunction.check_var_dims([d,torch.flatten(rx,1,2)],1,True)
        ed = torch.exp(d)
        erx = torch.exp(-rx)
        mvxu = torch.ones((b,c,sx-1),device=d.device)
        mvxd = torch.ones((b,c,sx-1),device=d.device)
        mexu = torch.ones((b,c,sx-1),device=d.device)
        mexd = torch.ones((b,c,sx-1),device=d.device)
        
        min_er = torch.min(erx).item()
        decay_const = 2 * (2-1) * (1-math.sqrt(min_er)) / (1+math.sqrt(min_er))
        if decay_const >= 1-UNHMF_LBP1d_PytorchNative.epsilon:
            #not guaranteed to converge, should throw error
            max_num_iters = 10*sx
        elif decay_const <= UNHMF_LBP1d_PytorchNative.epsilon:
            max_num_iters = 1
        else:
            max_num_iters = int(math.log(UNHMF_LBP1d_PytorchNative.epsilon) / math.log(decay_const))
        
        iter = 0
        while True:
            
            #compute vertex to edge messages
            #messages in x
            mvxd = ed[:,:,1:].clone()
            mvxd[:,:,:-1] *= mexd[:,:,1:]
            mvxu = ed[:,:,:-1].clone()
            mvxu[:,:,1:] *= mexu[:,:,:-1]
            
            #renormalise messages
            avxu = torch.sum(mvxu,1,keepdim=True)
            mvxu /= avxu
            avxd = torch.sum(mvxd,1,keepdim=True)
            mvxd /= avxd
            
            #compute edge to vertex messages
            mexd = torch.sum(erx[:,:,:,:-1] * torch.unsqueeze(mvxd,1), dim=2)
            mexu = torch.sum(erx[:,:,:,:-1] * torch.unsqueeze(mvxu,1), dim=2)
                
            iter+=1
            if iter >= max_num_iters:
                break
        
        #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
        marginal = d.clone()
        marginal[:,:,:-1] += torch.log(mexd)
        marginal[:,:,1:] += torch.log(mexu)
        
        #save context for backwards pass and return the marginal
        ctx.save_for_backward(ed,erx,mvxu,mvxd,mexu,mexd,
                                     avxu,avxd,torch.tensor(max_num_iters))
        return marginal
    
    @staticmethod 
    def backward(ctx, grad_output):
        ed,erx,mvxu,mvxd,mexu,mexd,avxu,avxd,max_num_iters = ctx.saved_tensors
        b,c,sx = ed.shape[0],ed.shape[1],ed.shape[2]
        max_num_iters = int(max_num_iters.cpu().item())
        grad_d =  torch.zeros_like(ed) #(will multiply out the ed at the end, keeps everything in exponential domain)
        grad_rx = torch.zeros_like(erx)
        
        #backprop through final layer
        grad_mexu = grad_output[:,:,1:] / mexu
        grad_mexd = grad_output[:,:,:-1] / mexd
        
        iter = 0
        while True:
            
            #backprop back to vertex messages
            grad_mvxd = torch.sum(erx[:,:,:,:-1]*torch.unsqueeze(grad_mexd,2),1)
            grad_mvxu = torch.sum(erx[:,:,:,:-1]*torch.unsqueeze(grad_mexu,2),1)
            grad_rx[:,:,:,:-1] += torch.unsqueeze(mvxu,1)*torch.unsqueeze(grad_mexu,2) + torch.unsqueeze(mvxd,1)*torch.unsqueeze(grad_mexd,2)
                        
            #backprop through renormalisation
            grad_mvxd = (grad_mvxd - torch.sum(grad_mvxd*mvxd,1,keepdim=True)) / avxd
            grad_mvxu = (grad_mvxu - torch.sum(grad_mvxu*mvxu,1,keepdim=True)) / avxu
            
            grad_mexd *= 0
            grad_mexu *= 0
            
            #backprop back to edge messages in x direction TODO Check correctness
            to_add = grad_mvxd.clone() #ed part for mvxd equation
            to_add[:,:,:-1] *= mexd[:,:,1:]
            grad_d[:,:,1:] += to_add 
            to_add = grad_mvxd * ed[:,:,1:] #mexd part for mvxd equation
            grad_mexd[:,:,1:] += to_add[:,:,:-1]
            
            to_add = grad_mvxu.clone() #ed part for mvxu equation
            to_add[:,:,1:] *= mexu[:,:,:-1]
            grad_d[:,:,:-1] += to_add 
            to_add = grad_mvxu * ed[:,:,:-1] #mexu part for mvxu equation
            grad_mexu[:,:,:-1] += to_add[:,:,1:]
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #correct to no longer be in the exponential domain
        grad_rx *= erx
        grad_d *= ed
        grad_d += grad_output
        
        return grad_d, grad_rx
        
class UNHMF_LBP2d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-4
    
    @staticmethod
    def forward(ctx, d, rx, ry):
        b,c,sx,sy = d.shape[0],d.shape[1],d.shape[2],d.shape[3]
        check_reg_channel_number([rx,ry],c)
        DeepFlowFunction.check_var_dims([d,torch.flatten(rx,1,2),
                                           torch.flatten(ry,1,2)],2,True)
        ed = torch.exp(d)
        erx = torch.exp(-rx)
        ery = torch.exp(-ry)
        mvxu = torch.ones((b,c,sx-1,sy),device=d.device)
        mvxd = torch.ones((b,c,sx-1,sy),device=d.device)
        mexu = torch.ones((b,c,sx-1,sy),device=d.device)
        mexd = torch.ones((b,c,sx-1,sy),device=d.device)
        mvyu = torch.ones((b,c,sx,sy-1),device=d.device)
        mvyd = torch.ones((b,c,sx,sy-1),device=d.device)
        meyu = torch.ones((b,c,sx,sy-1),device=d.device)
        meyd = torch.ones((b,c,sx,sy-1),device=d.device)
        
        
        min_er = min(torch.min(erx).item(),
                     torch.min(ery).item())
        decay_const = 2 * (4-1) * (1-math.sqrt(min_er)) / (1+math.sqrt(min_er))
        
        if decay_const >= 1-UNHMF_LBP2d_PytorchNative.epsilon:
            #not guaranteed to converge, should throw error
            max_num_iters = 10*(sx+sy)
        elif decay_const <= UNHMF_LBP2d_PytorchNative.epsilon:
            max_num_iters = 1
        else:
            max_num_iters = int(math.log(UNHMF_LBP2d_PytorchNative.epsilon) / math.log(decay_const))
        
        iter = 0
        while True:
            
            #compute vertex to edge messages
            #messages in x
            mvxd = ed[:,:,1:,:].clone()
            mvxd[:,:,:-1,:] *= mexd[:,:,1:,:]
            mvxd[:,:,:,:-1] *= meyd[:,:,:-1,:]
            mvxd[:,:,:,1:] *= meyu[:,:,:-1,:]
            mvxu = ed[:,:,:-1,:].clone()
            mvxu[:,:,1:,:] *= mexu[:,:,:-1,:]
            mvxu[:,:,:,:-1] *= meyd[:,:,:-1,:]
            mvxu[:,:,:,1:] *= meyu[:,:,:-1,:]
            
            #messages in y
            mvyd = ed[:,:,:,1:].clone()
            mvyd[:,:,:,:-1] *= meyd[:,:,:,1:]
            mvyd[:,:,:-1,:] *= mexd[:,:,:,:-1]
            mvyd[:,:,1:,:] *= mexu[:,:,:,:-1]
            mvyu = ed[:,:,:,:-1].clone()
            mvyu[:,:,:,1:] *= meyu[:,:,:,:-1]
            mvyu[:,:,:-1,:] *= mexd[:,:,:,:-1]
            mvyu[:,:,1:,:] *= mexu[:,:,:,:-1]
            
            #renormalise messages
            avxu = torch.sum(mvxu,1,keepdim=True)
            mvxu /= avxu
            avxd = torch.sum(mvxd,1,keepdim=True)
            mvxd /= avxd
            avyu = torch.sum(mvyu,1,keepdim=True)
            mvyu /= avyu
            avyd = torch.sum(mvyd,1,keepdim=True)
            mvyd /= avyd
            
            #compute edge to vertex messages
            mexd = torch.sum(erx[:,:,:,:-1,:] * torch.unsqueeze(mvxd,1), dim=2)
            mexu = torch.sum(erx[:,:,:,:-1,:] * torch.unsqueeze(mvxu,1), dim=2)
            meyd = torch.sum(ery[:,:,:,:,:-1] * torch.unsqueeze(mvyd,1), dim=2)
            meyu = torch.sum(ery[:,:,:,:,:-1] * torch.unsqueeze(mvyu,1), dim=2)
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
        marginal = d.clone()
        marginal[:,:,:-1,:] += torch.log(mexd)
        marginal[:,:,1:,:] += torch.log(mexu)
        marginal[:,:,:,:-1] += torch.log(meyd)
        marginal[:,:,:,1:] += torch.log(meyu)
        
        #save context for backwards pass and return the marginal
        ctx.save_for_backward(ed,erx,ery,mvxu,mvxd,mexu,mexd,
                                         mvyu,mvyd,meyu,meyd,
                                         avxu,avxd,avyu,avyd,torch.tensor(max_num_iters))
        return marginal
    
    @staticmethod 
    def backward(ctx, grad_output):
        ed,erx,ery,mvxu,mvxd,mexu,mexd,mvyu,mvyd,meyu,meyd,avxu,avxd,avyu,avyd,max_num_iters = ctx.saved_tensors
        b,c,sx,sy = ed.shape[0],ed.shape[1],ed.shape[2],ed.shape[3]
        max_num_iters = int(max_num_iters.cpu().item())
        grad_d =  torch.zeros_like(ed) #(will multiply out the ed at the end, keeps everything in exponential domain)
        grad_rx = torch.zeros_like(erx)
        grad_ry = torch.zeros_like(ery)
        
        #backprop through final layer
        grad_mexu = grad_output[:,:,1:,:] / mexu
        grad_mexd = grad_output[:,:,:-1,:] / mexd
        grad_meyu = grad_output[:,:,:,1:] / meyu
        grad_meyd = grad_output[:,:,:,:-1] / meyd
        
        iter = 0
        while True:
            
            #backprop back to vertex messages
            grad_mvxd = torch.sum(erx[:,:,:,:-1,:]*torch.unsqueeze(grad_mexd,2),1)
            grad_mvxu = torch.sum(erx[:,:,:,:-1,:]*torch.unsqueeze(grad_mexu,2),1)
            grad_rx[:,:,:,:-1,:] += torch.unsqueeze(mvxu,1)*torch.unsqueeze(grad_mexu,2) + torch.unsqueeze(mvxd,1)*torch.unsqueeze(grad_mexd,2)
            grad_mvyd = torch.sum(ery[:,:,:,:,:-1]*torch.unsqueeze(grad_meyd,2),1)
            grad_mvyu = torch.sum(ery[:,:,:,:,:-1]*torch.unsqueeze(grad_meyu,2),1)
            grad_ry[:,:,:,:,:-1] += torch.unsqueeze(mvyu,1)*torch.unsqueeze(grad_meyu,2) + torch.unsqueeze(mvyd,1)*torch.unsqueeze(grad_meyd,2)
                        
            #backprop through renormalisation
            grad_mvxd = (grad_mvxd - torch.sum(grad_mvxd*mvxd,1,keepdim=True)) / avxd
            grad_mvxu = (grad_mvxu - torch.sum(grad_mvxu*mvxu,1,keepdim=True)) / avxu
            grad_mvyd = (grad_mvyd - torch.sum(grad_mvyd*mvyd,1,keepdim=True)) / avyd
            grad_mvyu = (grad_mvyu - torch.sum(grad_mvyu*mvyu,1,keepdim=True)) / avyu
            
            grad_mexd *= 0
            grad_mexu *= 0
            grad_meyd *= 0
            grad_meyu *= 0
            
            #backprop back to edge messages in x direction TODO Check correctness
            to_add = grad_mvxd.clone() #ed part for mvxd equation
            to_add[:,:,:-1,:] *= mexd[:,:,1:,:]
            to_add[:,:,:,:-1] *= meyd[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu[:,:,:-1,:]
            grad_d[:,:,1:,:] += to_add 
            to_add = grad_mvxd * ed[:,:,1:,:] #mexd part for mvxd equation
            to_add[:,:,:,:-1] *= meyd[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu[:,:,:-1,:]
            grad_mexd[:,:,1:,:] += to_add[:,:,:-1]
            to_add = grad_mvxd * ed[:,:,1:,:] #meyu part for mvxd equation
            to_add[:,:,:-1,:] *= mexd[:,:,1:,:]
            to_add[:,:,:,:-1] *= meyd[:,:,:-1,:]
            grad_meyu[:,:,:-1,:] += to_add[:,:,:,1:]
            to_add = grad_mvxd * ed[:,:,1:,:] #meyd part for mvxd equation
            to_add[:,:,:-1,:] *= mexd[:,:,1:,:]
            to_add[:,:,:,1:] *= meyu[:,:,:-1,:]
            grad_meyd[:,:,:-1,:] += to_add[:,:,:,:-1]
            
            to_add = grad_mvxu.clone() #ed part for mvxu equation
            to_add[:,:,1:,:] *= mexu[:,:,:-1,:]
            to_add[:,:,:,:-1] *= meyd[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu[:,:,:-1,:]
            grad_d[:,:,:-1,:] += to_add 
            to_add = grad_mvxu * ed[:,:,:-1,:] #mexu part for mvxu equation
            to_add[:,:,:,:-1] *= meyd[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu[:,:,:-1,:]
            grad_mexu[:,:,:-1,:] += to_add[:,:,1:,:]
            to_add = grad_mvxu * ed[:,:,:-1,:] #meyu part for mvxu equation
            to_add[:,:,1:,:] *= mexu[:,:,:-1,:]
            to_add[:,:,:,:-1] *= meyd[:,:,:-1,:]
            grad_meyu[:,:,:-1,:] += to_add[:,:,:,1:]
            to_add = grad_mvxu * ed[:,:,:-1,:] #meyd part for mvxu equation
            to_add[:,:,1:,:] *= mexu[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu[:,:,:-1,:]
            grad_meyd[:,:,:-1,:] += to_add[:,:,:,:-1]
            
            
            #backprop back to edge messages in y direction TODO Check correctness            
            to_add = grad_mvyd.clone() #ed part for mvyd equation
            to_add[:,:,:,:-1] *= meyd[:,:,:,1:]
            to_add[:,:,:-1,:] *= mexd[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu[:,:,:,:-1]
            grad_d[:,:,:,1:] += to_add 
            to_add = grad_mvyd * ed[:,:,:,1:] # meyd part for mvyd equation
            to_add[:,:,:-1,:] *= mexd[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu[:,:,:,:-1]
            grad_meyd[:,:,:,1:] += to_add[:,:,:,:-1]
            to_add = grad_mvyd * ed[:,:,:,1:] # mexd part for mvyd equation
            to_add[:,:,:,:-1] *= meyd[:,:,:,1:]
            to_add[:,:,1:,:] *= mexu[:,:,:,:-1]
            grad_mexd[:,:,:,:-1] += to_add[:,:,:-1,:]
            to_add = grad_mvyd * ed[:,:,:,1:] # mexu part for mvyd equation
            to_add[:,:,:,:-1] *= meyd[:,:,:,1:]
            to_add[:,:,:-1,:] *= mexd[:,:,:,:-1]
            grad_mexu[:,:,:,:-1] += to_add[:,:,1:,:]
            
            to_add = grad_mvyu.clone() #ed part for mvyu equation
            to_add[:,:,:,1:] *= meyu[:,:,:,:-1]
            to_add[:,:,:-1,:] *= mexd[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu[:,:,:,:-1]
            grad_d[:,:,:,:-1] += to_add
            to_add = grad_mvyu * ed[:,:,:,:-1] #meyu part for mvyu equation
            to_add[:,:,:-1,:] *= mexd[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu[:,:,:,:-1]
            grad_meyu[:,:,:,:-1] += to_add[:,:,:,1:]
            to_add = grad_mvyu * ed[:,:,:,:-1] #mexd part for mvyu equation
            to_add[:,:,:,1:] *= meyu[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu[:,:,:,:-1]
            grad_mexd[:,:,:,:-1] += to_add[:,:,:-1,:]
            to_add = grad_mvyu * ed[:,:,:,:-1] #mexu part for mvyu equation
            to_add[:,:,:,1:] *= meyu[:,:,:,:-1]
            to_add[:,:,:-1,:] *= mexd[:,:,:,:-1]
            grad_mexu[:,:,:,:-1] += to_add[:,:,1:,:]
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #correct to no longer be in the exponential domain
        grad_rx *= erx
        grad_ry *= ery
        grad_d *= ed
        grad_d += grad_output
        
        return grad_d, grad_rx, grad_ry
        
class UNHMF_LBP3d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-4
    
    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        b,c,sx,sy,sz = d.shape[0],d.shape[1],d.shape[2],d.shape[3],d.shape[4]
        check_reg_channel_number([rx,ry,rz],c)
        DeepFlowFunction.check_var_dims([d,torch.flatten(rx,1,2),
                                           torch.flatten(ry,1,2),
                                           torch.flatten(rz,1,2)],3,True)
        ed = torch.exp(d)
        erx = torch.exp(-rx)
        ery = torch.exp(-ry)
        erz = torch.exp(-rz)
        mvxu = torch.ones((b,c,sx-1,sy,sz),device=d.device)
        mvxd = torch.ones((b,c,sx-1,sy,sz),device=d.device)
        mexu = torch.ones((b,c,sx-1,sy,sz),device=d.device)
        mexd = torch.ones((b,c,sx-1,sy,sz),device=d.device)
        mvyu = torch.ones((b,c,sx,sy-1,sz),device=d.device)
        mvyd = torch.ones((b,c,sx,sy-1,sz),device=d.device)
        meyu = torch.ones((b,c,sx,sy-1,sz),device=d.device)
        meyd = torch.ones((b,c,sx,sy-1,sz),device=d.device)
        mvzu = torch.ones((b,c,sx,sy,sz-1),device=d.device)
        mvzd = torch.ones((b,c,sx,sy,sz-1),device=d.device)
        mezu = torch.ones((b,c,sx,sy,sz-1),device=d.device)
        mezd = torch.ones((b,c,sx,sy,sz-1),device=d.device)
        
        min_er = min([torch.min(erx).item(),
                      torch.min(ery).item(),
                      torch.min(erz).item()])
        decay_const = 2 * (6-1) * (1-math.sqrt(min_er)) / (1+math.sqrt(min_er))
        if decay_const >= 1-UNHMF_LBP3d_PytorchNative.epsilon:
            #not guaranteed to converge, should throw error
            max_num_iters = 10*(sx+sy+sz)
        elif decay_const <= UNHMF_LBP3d_PytorchNative.epsilon:
            max_num_iters = 1
        else:
            max_num_iters = int(math.log(UNHMF_LBP3d_PytorchNative.epsilon) / math.log(decay_const))
        
        iter = 0
        while True:
            
            #compute vertex to edge messages
            #messages in x
            mvxd = ed[:,:,1:,:,:].clone()
            mvxd[:,:,:-1,:,:] *= mexd[:,:,1:,:,:]
            mvxd[:,:,:,:-1,:] *= meyd[:,:,:-1,:,:]
            mvxd[:,:,:,1:,:] *= meyu[:,:,:-1,:,:]
            mvxd[:,:,:,:,:-1] *= mezd[:,:,:-1,:,:]
            mvxd[:,:,:,:,1:] *= mezu[:,:,:-1,:,:]
            mvxu = ed[:,:,:-1,:,:].clone()
            mvxu[:,:,1:,:,:] *= mexu[:,:,:-1,:,:]
            mvxu[:,:,:,:-1,:] *= meyd[:,:,:-1,:,:]
            mvxu[:,:,:,1:,:] *= meyu[:,:,:-1,:,:]
            mvxu[:,:,:,:,:-1] *= mezd[:,:,:-1,:,:]
            mvxu[:,:,:,:,1:] *= mezu[:,:,:-1,:,:]
            
            #messages in y
            mvyd = ed[:,:,:,1:,:].clone()
            mvyd[:,:,:,:-1,:] *= meyd[:,:,:,1:,:]
            mvyd[:,:,:-1,:,:] *= mexd[:,:,:,:-1,:]
            mvyd[:,:,1:,:,:] *= mexu[:,:,:,:-1,:]
            mvyd[:,:,:,:,:-1] *= mezd[:,:,:,:-1,:]
            mvyd[:,:,:,:,1:] *= mezu[:,:,:,:-1,:]
            mvyu = ed[:,:,:,:-1,:].clone()
            mvyu[:,:,:,1:,:] *= meyu[:,:,:,:-1,:]
            mvyu[:,:,:-1,:,:] *= mexd[:,:,:,:-1,:]
            mvyu[:,:,1:,:,:] *= mexu[:,:,:,:-1,:]
            mvyu[:,:,:,:,:-1] *= mezd[:,:,:,:-1,:]
            mvyu[:,:,:,:,1:] *= mezu[:,:,:,:-1,:]
            
            #messages in z
            mvzd = ed[:,:,:,:,1:].clone()
            mvzd[:,:,:,:,:-1] *= mezd[:,:,:,:,1:]
            mvzd[:,:,:-1,:,:] *= mexd[:,:,:,:,:-1]
            mvzd[:,:,1:,:,:] *= mexu[:,:,:,:,:-1]
            mvzd[:,:,:,:-1,:] *= meyd[:,:,:,:,:-1]
            mvzd[:,:,:,1:,:] *= meyu[:,:,:,:,:-1]
            mvzu = ed[:,:,:,:,:-1].clone()
            mvzu[:,:,:,:,1:] *= mezu[:,:,:,:,:-1]
            mvzu[:,:,:-1,:,:] *= mexd[:,:,:,:,:-1]
            mvzu[:,:,1:,:,:] *= mexu[:,:,:,:,:-1]
            mvzu[:,:,:,:-1,:] *= meyd[:,:,:,:,:-1]
            mvzu[:,:,:,1:,:] *= meyu[:,:,:,:,:-1]
            
            #renormalise messages
            avxu = torch.sum(mvxu,1,keepdim=True)
            mvxu /= avxu
            avxd = torch.sum(mvxd,1,keepdim=True)
            mvxd /= avxd
            avyu = torch.sum(mvyu,1,keepdim=True)
            mvyu /= avyu
            avyd = torch.sum(mvyd,1,keepdim=True)
            mvyd /= avyd
            avzu = torch.sum(mvzu,1,keepdim=True)
            mvzu /= avzu
            avzd = torch.sum(mvzd,1,keepdim=True)
            mvzd /= avzd
            
            #compute edge to vertex messages
            mexd = torch.sum(erx[:,:,:,:-1,:,:] * torch.unsqueeze(mvxd,1), dim=2)
            mexu = torch.sum(erx[:,:,:,:-1,:,:] * torch.unsqueeze(mvxu,1), dim=2)
            meyd = torch.sum(ery[:,:,:,:,:-1,:] * torch.unsqueeze(mvyd,1), dim=2)
            meyu = torch.sum(ery[:,:,:,:,:-1,:] * torch.unsqueeze(mvyu,1), dim=2)
            mezd = torch.sum(erz[:,:,:,:,:,:-1] * torch.unsqueeze(mvzd,1), dim=2)
            mezu = torch.sum(erz[:,:,:,:,:,:-1] * torch.unsqueeze(mvzu,1), dim=2)
                
            iter+=1
            if iter >= max_num_iters:
                break
        
        #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
        marginal = d.clone()
        marginal[:,:,:-1,:,:] += torch.log(mexd)
        marginal[:,:,1:,:,:] += torch.log(mexu)
        marginal[:,:,:,:-1,:] += torch.log(meyd)
        marginal[:,:,:,1:,:] += torch.log(meyu)
        marginal[:,:,:,:,:-1] += torch.log(mezd)
        marginal[:,:,:,:,1:] += torch.log(mezu)
        
        #save context for backwards pass and return the marginal
        ctx.save_for_backward(ed,erx,ery,erz,mvxu,mvxd,mexu,mexd,
                                             mvyu,mvyd,meyu,meyd,
                                             mvzu,mvzd,mezu,mezd,
                                             avxu,avxd,avyu,avyd,avzu,avzd,torch.tensor(max_num_iters))
        return marginal
    
    @staticmethod 
    def backward(ctx, grad_output):
        ed,erx,ery,erz,mvxu,mvxd,mexu,mexd,mvyu,mvyd,meyu,meyd,mvzu,mvzd,mezu,mezd,avxu,avxd,avyu,avyd,avzu,avzd,max_num_iters = ctx.saved_tensors
        b,c,sx,sy,sz = ed.shape[0],ed.shape[1],ed.shape[2],ed.shape[3],ed.shape[4]
        max_num_iters = int(max_num_iters.cpu().item())
        grad_d =  torch.zeros_like(ed) #(will multiply out the ed at the end, keeps everything in exponential domain)
        grad_rx = torch.zeros_like(erx)
        grad_ry = torch.zeros_like(ery)
        grad_rz = torch.zeros_like(erz)
        
        #backprop through final layer
        grad_mexu = grad_output[:,:,1:,:,:] / mexu
        grad_mexd = grad_output[:,:,:-1,:,:] / mexd
        grad_meyu = grad_output[:,:,:,1:,:] / meyu
        grad_meyd = grad_output[:,:,:,:-1,:] / meyd
        grad_mezu = grad_output[:,:,:,:,1:] / mezu
        grad_mezd = grad_output[:,:,:,:,:-1] / mezd
        
        iter = 0
        while True:
            
            #backprop back to vertex messages
            grad_mvxd = torch.sum(erx[:,:,:,:-1,:,:]*torch.unsqueeze(grad_mexd,2),1)
            grad_mvxu = torch.sum(erx[:,:,:,:-1,:,:]*torch.unsqueeze(grad_mexu,2),1)
            grad_rx[:,:,:,:-1,:,:] += torch.unsqueeze(mvxu,1)*torch.unsqueeze(grad_mexu,2) + torch.unsqueeze(mvxd,1)*torch.unsqueeze(grad_mexd,2)
            grad_mvyd = torch.sum(ery[:,:,:,:,:-1,:]*torch.unsqueeze(grad_meyd,2),1)
            grad_mvyu = torch.sum(ery[:,:,:,:,:-1,:]*torch.unsqueeze(grad_meyu,2),1)
            grad_ry[:,:,:,:,:-1,:] += torch.unsqueeze(mvyu,1)*torch.unsqueeze(grad_meyu,2) + torch.unsqueeze(mvyd,1)*torch.unsqueeze(grad_meyd,2)
            grad_mvzd = torch.sum(erz[:,:,:,:,:,:-1]*torch.unsqueeze(grad_mezd,2),1)
            grad_mvzu = torch.sum(erz[:,:,:,:,:,:-1]*torch.unsqueeze(grad_mezu,2),1)
            grad_rz[:,:,:,:,:,:-1] += torch.unsqueeze(mvzu,1)*torch.unsqueeze(grad_mezu,2) + torch.unsqueeze(mvzd,1)*torch.unsqueeze(grad_mezd,2)
                        
            #backprop through renormalisation
            grad_mvxd = (grad_mvxd - torch.sum(grad_mvxd*mvxd,1,keepdim=True)) / avxd
            grad_mvxu = (grad_mvxu - torch.sum(grad_mvxu*mvxu,1,keepdim=True)) / avxu
            grad_mvyd = (grad_mvyd - torch.sum(grad_mvyd*mvyd,1,keepdim=True)) / avyd
            grad_mvyu = (grad_mvyu - torch.sum(grad_mvyu*mvyu,1,keepdim=True)) / avyu
            grad_mvzd = (grad_mvzd - torch.sum(grad_mvzd*mvzd,1,keepdim=True)) / avzd
            grad_mvzu = (grad_mvzu - torch.sum(grad_mvzu*mvzu,1,keepdim=True)) / avzu
            
            grad_mexd *= 0
            grad_mexu *= 0
            grad_meyd *= 0
            grad_meyu *= 0
            grad_mezd *= 0
            grad_mezu *= 0
            
            #backprop back to edge messages in x direction TODO Check correctness
            to_add = grad_mvxd.clone() #ed part for mvxd equation
            to_add[:,:,:-1,:,:] *= mexd[:,:,1:,:,:]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:-1,:,:]
            grad_d[:,:,1:,:,:] += to_add 
            to_add = grad_mvxd * ed[:,:,1:,:,:] #mexd part for mvxd equation
            to_add[:,:,:,:-1,:] *= meyd[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:-1,:,:]
            grad_mexd[:,:,1:,:,:] += to_add[:,:,:-1,:]
            to_add = grad_mvxd * ed[:,:,1:,:,:] #meyu part for mvxd equation
            to_add[:,:,:-1,:,:] *= mexd[:,:,1:,:,:]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:-1,:,:]
            grad_meyu[:,:,:-1,:,:] += to_add[:,:,:,1:,:]
            to_add = grad_mvxd * ed[:,:,1:,:,:] #meyd part for mvxd equation
            to_add[:,:,:-1,:,:] *= mexd[:,:,1:,:,:]
            to_add[:,:,:,1:,:] *= meyu[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:-1,:,:]
            grad_meyd[:,:,:-1,:,:] += to_add[:,:,:,:-1,:]
            to_add = grad_mvxd * ed[:,:,1:,:,:] #mezu part for mvxd equation
            to_add[:,:,:-1,:,:] *= mexd[:,:,1:,:,:]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:-1,:,:]
            grad_mezu[:,:,:-1,:,:] += to_add[:,:,:,:,1:]
            to_add = grad_mvxd * ed[:,:,1:,:,:] #mezd part for mvxd equation
            to_add[:,:,:-1,:,:] *= mexd[:,:,1:,:,:]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:-1,:,:]
            grad_mezd[:,:,:-1,:,:] += to_add[:,:,:,:,:-1]
            
            to_add = grad_mvxu.clone() #ed part for mvxu equation
            to_add[:,:,1:,:,:] *= mexu[:,:,:-1,:,:]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:-1,:,:]
            grad_d[:,:,:-1,:,:] += to_add 
            to_add = grad_mvxu * ed[:,:,:-1,:,:] #mexu part for mvxu equation
            to_add[:,:,:,:-1,:] *= meyd[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:-1,:,:]
            grad_mexu[:,:,:-1,:,:] += to_add[:,:,1:,:,:]
            to_add = grad_mvxu * ed[:,:,:-1,:,:] #meyu part for mvxu equation
            to_add[:,:,1:,:,:] *= mexu[:,:,:-1,:,:]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:-1,:,:]
            grad_meyu[:,:,:-1,:,:] += to_add[:,:,:,1:,:]
            to_add = grad_mvxu * ed[:,:,:-1,:,:] #meyd part for mvxu equation
            to_add[:,:,1:,:,:] *= mexu[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:-1,:,:]
            grad_meyd[:,:,:-1,:,:] += to_add[:,:,:,:-1,:]
            to_add = grad_mvxu * ed[:,:,:-1,:,:] #mezu part for mvxd equation
            to_add[:,:,1:,:,:] *= mexu[:,:,:-1,:,:]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:-1,:,:]
            grad_mezd[:,:,:-1,:,:] += to_add[:,:,:,:,:-1]
            to_add = grad_mvxu * ed[:,:,:-1,:,:] #mezu part for mvxu equation
            to_add[:,:,1:,:,:] *= mexu[:,:,:-1,:,:]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:-1,:,:]
            grad_mezu[:,:,:-1,:,:] += to_add[:,:,:,:,1:]
            
            
            #backprop back to edge messages in y direction TODO Check correctness            
            to_add = grad_mvyd.clone() #ed part for mvyd equation
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,1:,:]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:-1,:]
            grad_d[:,:,:,1:,:] += to_add 
            to_add = grad_mvyd * ed[:,:,:,1:,:] # meyd part for mvyd equation
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:-1,:]
            grad_meyd[:,:,:,1:,:] += to_add[:,:,:,:-1,:]
            to_add = grad_mvyd * ed[:,:,:,1:,:] # mexd part for mvyd equation
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,1:,:]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:-1,:]
            grad_mexd[:,:,:,:-1,:] += to_add[:,:,:-1,:,:]
            to_add = grad_mvyd * ed[:,:,:,1:,:] # mexu part for mvyd equation
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,1:,:]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:-1,:]
            grad_mexu[:,:,:,:-1,:] += to_add[:,:,1:,:,:]           
            to_add = grad_mvyd * ed[:,:,:,1:,:] #mezd part for mvyd equation
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,1:,:]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:-1,:]
            grad_mezd[:,:,:,:-1,:] += to_add[:,:,:,:,:-1]           
            to_add = grad_mvyd * ed[:,:,:,1:,:] #mezu part for mvyd equation
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,1:,:]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:-1,:]
            grad_mezu[:,:,:,:-1,:] += to_add[:,:,:,:,1:]
            
            to_add = grad_mvyu.clone() #ed part for mvyu equation
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:-1,:]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:-1,:]
            grad_d[:,:,:,:-1,:] += to_add
            to_add = grad_mvyu * ed[:,:,:,:-1,:] #meyu part for mvyu equation
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:-1,:]
            grad_meyu[:,:,:,:-1,:] += to_add[:,:,:,1:,:]
            to_add = grad_mvyu * ed[:,:,:,:-1,:] #mexd part for mvyu equation
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:-1,:]
            grad_mexd[:,:,:,:-1,:] += to_add[:,:,:-1,:,:]
            to_add = grad_mvyu * ed[:,:,:,:-1,:] #mexu part for mvyu equation
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:-1,:]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:-1,:]
            grad_mexu[:,:,:,:-1,:] += to_add[:,:,1:,:,:]
            to_add = grad_mvyu * ed[:,:,:,:-1,:] #mezd part for mvyu equation
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:-1,:]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:-1,:]
            grad_mezd[:,:,:,:-1,:] += to_add[:,:,:,:,:-1]
            to_add = grad_mvyu * ed[:,:,:,:-1,:] #mezu part for mvyu equation
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:-1,:]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:-1,:]
            grad_mezu[:,:,:,:-1,:] += to_add[:,:,:,:,1:]
        
        
            #backprop back to edge messages in z direction TODO Check correctness
            to_add = grad_mvzd.clone() #ed part for mvzd equation
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:,1:]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:,:-1]
            grad_d[:,:,:,:,1:] += to_add
            to_add = grad_mvzd * ed[:,:,:,:,1:] #mezd part for mvzd equation
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:,:-1]
            grad_mezd[:,:,:,:,1:] += to_add[:,:,:,:,:-1]
            to_add = grad_mvzd * ed[:,:,:,:,1:] #mexd part for mvzd equation
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:,1:]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:,:-1]
            grad_mexd[:,:,:,:,:-1] += to_add[:,:,:-1,:,:]
            to_add = grad_mvzd * ed[:,:,:,:,1:] #mexu part for mvzd equation
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:,1:]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:,:-1]
            grad_mexu[:,:,:,:,:-1] += to_add[:,:,1:,:,:]
            to_add = grad_mvzd * ed[:,:,:,:,1:] #meyd part for mvzd equation
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:,1:]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:,:-1]
            grad_meyd[:,:,:,:,:-1] += to_add[:,:,:,:-1,:]
            to_add = grad_mvzd * ed[:,:,:,:,1:] #meyu part for mvzd equation
            to_add[:,:,:,:,:-1] *= mezd[:,:,:,:,1:]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,:,:-1]
            grad_meyu[:,:,:,:,:-1] += to_add[:,:,:,1:,:]
            
            to_add = grad_mvzu.clone() #ed part for mvzu equation
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:,:-1]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:,:-1]
            grad_d[:,:,:,:,:-1] += to_add
            to_add = grad_mvzu * ed[:,:,:,:,:-1]  #mezu part for mvzu equation
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:,:-1]
            grad_mezu[:,:,:,:,:-1] += to_add[:,:,:,:,1:]
            to_add = grad_mvzu * ed[:,:,:,:,:-1]  #mexd part for mvzu equation
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:,:-1]
            grad_mexd[:,:,:,:,:-1] += to_add[:,:,:-1,:,:]
            to_add = grad_mvzu * ed[:,:,:,:,:-1]  #mexu part for mvzu equation
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:,:-1]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:,:-1]
            grad_mexu[:,:,:,:,:-1] += to_add[:,:,1:,:,:]
            to_add = grad_mvzu * ed[:,:,:,:,:-1]  #meyd part for mvzu equation
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:,:-1]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu[:,:,:,:,:-1]
            grad_meyd[:,:,:,:,:-1] += to_add[:,:,:,:-1,:]
            to_add = grad_mvzu * ed[:,:,:,:,:-1]  #meyu part for mvzu equation
            to_add[:,:,:,:,1:] *= mezu[:,:,:,:,:-1]
            to_add[:,:,:-1,:,:] *= mexd[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd[:,:,:,:,:-1]
            grad_meyu[:,:,:,:,:-1] += to_add[:,:,:,1:,:]
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #correct to no longer be in the exponential domain
        grad_rx *= erx
        grad_ry *= ery
        grad_rz *= erz
        grad_d *= ed
        grad_d += grad_output
        
        return grad_d, grad_rx, grad_ry, grad_rz