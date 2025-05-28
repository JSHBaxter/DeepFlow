import torch
import deepflow
from deepflow_function import DeepFlowFunction
import math

class Binary_MAP1d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx):
        DeepFlowFunction.check_var_dims([d,rx],1)
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.binary_auglag_1d_gpu_forward(d,rx, output)
        else:
            deepflow.binary_auglag_1d_cpu_forward(d,rx, output)
        return output
            
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
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.binary_auglag_2d_gpu_forward(d,rx, ry, output)
        else:
            deepflow.binary_auglag_2d_cpu_forward(d,rx, ry, output)
        return output
            
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
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.binary_auglag_3d_gpu_forward(d, rx, ry, rz, output)
        else:
            deepflow.binary_auglag_3d_cpu_forward(d, rx, ry, rz, output)
        return output
        
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input
        
        
class Binary_Mean1d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx):
        DeepFlowFunction.check_var_dims([d,rx],1)
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

class Binary_Mean1d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-6
    
    @staticmethod
    def forward(ctx, d, rx):
        DeepFlowFunction.check_var_dims([d,rx],1)
        u = 1/(1+torch.exp(-d))
        iter = 0
        while True:
            
            energy = torch.zeros_like(u)
            energy[:,:,:-1] += rx[:,:,:-1]*(2*u[:,:,1:]-1)
            energy[:,:,1:] += rx[:,:,:-1]*(2*u[:,:,:-1]-1)
            u_new = 1/(1+torch.exp(-d-energy))
            change = torch.max(torch.abs(u_new-u))
            u_new = u
            
            iter+=1
            if change < Binary_Mean1d_PytorchNative.epsilon:
                break
            
        output = d + energy
        ctx.save_for_backward(output,rx)
        return output
            
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        u,rx, = ctx.saved_tensors
        u = 1/(1+torch.exp(-u))
        grad_d =  torch.zeros_like(u)
        grad_rx = torch.zeros_like(rx)
        
        dE = grad_output.clone()
        
        iter = 0
        while True:
            
            grad_d += dE
            grad_rx[:,:,:-1] += (2*u[:,:,1:]-1)* dE[:,:,:-1]
            grad_rx[:,:,:-1] += (2*u[:,:,:-1]-1)* dE[:,:,1:]
            
            new_du = torch.zeros_like(u)
            new_du[:,:,1:] += 2*dE[:,:,:-1]*rx[:,:,:-1]
            new_du[:,:,:-1] += 2*dE[:,:,1:]*rx[:,:,:-1]
            
            new_dE = new_du * u * (1-u)
            
            change=torch.max(torch.abs(new_dE-dE))
            dE = new_dE
            
            iter+=1
            if change < Binary_Mean1d_PytorchNative.epsilon:
                break
        
        return grad_d, grad_rx

class Binary_LBP1d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-6
    
    @staticmethod
    def forward(ctx, d, rx):
        DeepFlowFunction.check_var_dims([d,rx],1)
        b,c,sx = d.shape[0],d.shape[1],d.shape[2]
        ed = torch.exp(d)
        erx = torch.exp(-rx)
        mvxu0 = torch.ones((b,c,sx-1),device=d.device)
        mvxd0 = torch.ones((b,c,sx-1),device=d.device)
        mvxu1 = torch.ones((b,c,sx-1),device=d.device)
        mvxd1 = torch.ones((b,c,sx-1),device=d.device)
        mexu0 = torch.ones((b,c,sx-1),device=d.device)
        mexd0 = torch.ones((b,c,sx-1),device=d.device)
        mexu1 = torch.ones((b,c,sx-1),device=d.device)
        mexd1 = torch.ones((b,c,sx-1),device=d.device)
        
        min_erx = torch.min(erx)
        if min_erx >= 1-Binary_LBP1d_PytorchNative.epsilon:
            max_num_iters = 1
        elif min_erx > 10**-5:
            max_num_iters = int(math.log(Binary_LBP1d_PytorchNative.epsilon) / (math.log(1-min_erx))-math.log(1+min_erx))
            if max_num_iters < sx:
                max_num_iters = sx
        if min_erx > 0 and min_erx <= 10**-5:
            max_num_iters = -int(math.log(Binary_LBP1d_PytorchNative.epsilon) / (2*min_erx))
            if max_num_iters > 10*sx:
                max_num_iters = 10*sx
        else:
            max_num_iters = 1000
        
        iter = 0
        while True:
            
            #compute vertex to edge messages
            mvxd1 = ed[:,:,1:].clone()
            mvxd1[:,:,:-1] *= mexd1[:,:,1:]
            mvxu1 = ed[:,:,:-1].clone()
            mvxu1[:,:,1:] *= mexu1[:,:,:-1]
            mvxd0 = torch.ones_like(mexd0)
            mvxd0[:,:,:-1] *= mexd0[:,:,1:]
            mvxu0 = torch.ones_like(mexu0)
            mvxu0[:,:,1:] *= mexu0[:,:,:-1]
            
            #renormalise messages
            a = mvxu0+mvxu1
            mvxu0 /= a
            mvxu1 /= a
            a = mvxd0+mvxd1
            mvxd0 /= a
            mvxd1 /= a
            
            #compute edge to vertex messages
            mexd0 = mvxd0 + erx[:,:,:-1] * mvxd1
            mexd1 = mvxd1 + erx[:,:,:-1] * mvxd0
            mexu0 = mvxu0 + erx[:,:,:-1] * mvxu1
            mexu1 = mvxu1 + erx[:,:,:-1] * mvxu0
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
        marginal = d.clone()
        marginal[:,:,:-1] -= torch.log(mexd0)
        marginal[:,:,1:] -= torch.log(mexu0)
        marginal[:,:,:-1] += torch.log(mexd1)
        marginal[:,:,1:] += torch.log(mexu1)
        
        #save context for backwards pass and return the marginal
        ctx.save_for_backward(ed,erx,mvxu0,mvxu1,mvxd0,mvxd1,mexu0,mexu1,mexd0,mexd1,torch.tensor(max_num_iters))
        return marginal
            
    @staticmethod
    def backward(ctx, grad_output):
        ed,erx,mvxu0,mvxu1,mvxd0,mvxd1,mexu0,mexu1,mexd0,mexd1,max_num_iters = ctx.saved_tensors
        max_num_iters = int(max_num_iters.cpu().item())
        a_u_2 = (mvxu0+mvxu1)**2
        a_d_2 = (mvxd0+mvxd1)**2
        grad_d =  torch.zeros_like(ed) #(will multiply out the ed at the end, keeps everything in exponential domain)
        grad_rx = torch.zeros_like(erx)
        
        #backprop through final layer
        grad_mexu0 = - grad_output[:,:,1:] / mexu0
        grad_mexd0 = - grad_output[:,:,:-1] / mexd0
        grad_mexu1 = grad_output[:,:,1:] / mexu1
        grad_mexd1 = grad_output[:,:,:-1] / mexd1
        
        iter = 0
        while True:
            
            #backprop back to vertex messages
            grad_mvxu0 = grad_mexu0 + erx[:,:,:-1]*grad_mexu1
            grad_mvxu1 = grad_mexu1 + erx[:,:,:-1]*grad_mexu0
            grad_mvxd0 = grad_mexd0 + erx[:,:,:-1]*grad_mexd1
            grad_mvxd1 = grad_mexd1 + erx[:,:,:-1]*grad_mexd0
            grad_rx[:,:,:-1] += grad_mexu1 + grad_mexu0 + grad_mexd1 + grad_mexd0
            
            #backprop through renormalisation
            grad_mvxu0, grad_mvxu1 = (mvxu1*grad_mvxu0 - mvxu0*grad_mvxu1) / a_u_2, (mvxu0*grad_mvxu1 - mvxu1*grad_mvxu0) / a_u_2
            grad_mvxd0, grad_mvxd1 = (mvxd1*grad_mvxd0 - mvxd0*grad_mvxd1) / a_d_2, (mvxd0*grad_mvxd1 - mvxd1*grad_mvxd0) / a_d_2
            
            #backprop back to edge messages (REDO - check correctness)
            grad_d[:,:,-1] += grad_mvxd1[:,:,-1] #ed contribution to mvxd1 and mvxu1
            grad_d[:,:,0] += grad_mvxu1[:,:,0]
            grad_d[:,:,1:-1] += grad_mvxd1[:,:,:-1] * mexd1[:,:,1:] + grad_mvxu1[:,:,1:] * mexu1[:,:,:-1]
            grad_mexd1[:,:,0] = 0
            grad_mexd1[:,:,1:] = grad_mvxd1[:,:,:-1]*ed[:,:,1:-1] #mexd1 contribution to mvxd1
            grad_mexu1[:,:,-1] = 0
            grad_mexu1[:,:,:-1] = grad_mvxd1[:,:,1:]*ed[:,:,1:-1] #mexu1 contribution to mvxu1
            grad_mexd0[:,:,0] = 0
            grad_mexd0[:,:,1:] = grad_mvxd0[:,:,:-1].clone() #mexd0 contribution to mvxd0
            grad_mexu0[:,:,-1] = 0
            grad_mexu0[:,:,:-1] = grad_mvxd1[:,:,1:].clone() #mexu0 contribution to mvxu0
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #correct to no longer be in the exponential domain
        grad_rx *= erx
        grad_d *= ed
        grad_d += grad_output
        
        return grad_d, grad_rx
    
class Binary_Mean2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry):
        DeepFlowFunction.check_var_dims([d,rx,ry],2)
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
    

class Binary_Mean2d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-6
    
    @staticmethod
    def forward(ctx, d, rx, ry):
        DeepFlowFunction.check_var_dims([d,rx,ry],2)
        u = 1/(1+torch.exp(-d))
        iter = 0
        while True:
            
            energy = torch.zeros_like(u)
            energy[:,:,:-1,:] += rx[:,:,:-1,:]*(2*u[:,:,1:,:]-1)
            energy[:,:,1:,:] += rx[:,:,:-1,:]*(2*u[:,:,:-1,:]-1)
            energy[:,:,:,:-1] += ry[:,:,:,:-1]*(2*u[:,:,:,1:]-1)
            energy[:,:,:,1:] += ry[:,:,:,:-1]*(2*u[:,:,:,:-1]-1)
            u_new = 1/(1+torch.exp(-d-energy))
            change = torch.max(torch.abs(u_new-u))
            u_new = u
            
            iter+=1
            if change < epsilon:
                break
            
        output = d + energy
        ctx.save_for_backward(output,rx,ry)
        return output
            
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        u,rx,ry, = ctx.saved_tensors
        u = 1/(1+torch.exp(-u))
        grad_d =  torch.zeros_like(u)
        grad_rx = torch.zeros_like(rx)
        grad_ry = torch.zeros_like(ry)
        
        dE = grad_output.clone()
        
        iter = 0
        while True:
            
            grad_d += dE
            grad_rx[:,:,:-1,:] += (2*u[:,:,1:,:]-1)* dE[:,:,:-1,:]
            grad_rx[:,:,:-1,:] += (2*u[:,:,:-1,:]-1)* dE[:,:,1:,:]
            grad_ry[:,:,:,:-1] += (2*u[:,:,:,1:]-1)* dE[:,:,:,:-1]
            grad_ry[:,:,:,:-1] += (2*u[:,:,:,:-1]-1)* dE[:,:,:,1:]
            
            new_du = torch.zeros_like(u)
            new_du[:,:,1:,:] += 2*dE[:,:,:-1,:]*rx[:,:,:-1,:]
            new_du[:,:,:-1,:] += 2*dE[:,:,1:,:]*rx[:,:,:-1,:]
            new_du[:,:,:,1:] += 2*dE[:,:,:,:-1]*ry[:,:,:,:-1]
            new_du[:,:,:,:-1] += 2*dE[:,:,:,1:]*ry[:,:,:,:-1]
            
            new_dE = new_du * u * (1-u)
            
            change=torch.max(torch.abs(new_dE-dE))
            dE = new_dE
            
            iter+=1
            if change < epsilon:
                break
        
        return grad_d, grad_rx, grad_ry
        
class Binary_LBP2d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-6
    
    @staticmethod
    def forward(ctx, d, rx, ry):
        DeepFlowFunction.check_var_dims([d,rx,ry],2)
        b,c,sx,sy = d.shape[0],d.shape[1],d.shape[2],d.shape[3]
        ed = torch.exp(d)
        erx = torch.exp(-rx)
        ery = torch.exp(-ry)
        mvxu0 = torch.ones((b,c,sx-1,sy),device=d.device)
        mvxd0 = torch.ones((b,c,sx-1,sy),device=d.device)
        mvxu1 = torch.ones((b,c,sx-1,sy),device=d.device)
        mvxd1 = torch.ones((b,c,sx-1,sy),device=d.device)
        mexu0 = torch.ones((b,c,sx-1,sy),device=d.device)
        mexd0 = torch.ones((b,c,sx-1,sy),device=d.device)
        mexu1 = torch.ones((b,c,sx-1,sy),device=d.device)
        mexd1 = torch.ones((b,c,sx-1,sy),device=d.device)
        mvyu0 = torch.ones((b,c,sx,sy-1),device=d.device)
        mvyd0 = torch.ones((b,c,sx,sy-1),device=d.device)
        mvyu1 = torch.ones((b,c,sx,sy-1),device=d.device)
        mvyd1 = torch.ones((b,c,sx,sy-1),device=d.device)
        meyu0 = torch.ones((b,c,sx,sy-1),device=d.device)
        meyd0 = torch.ones((b,c,sx,sy-1),device=d.device)
        meyu1 = torch.ones((b,c,sx,sy-1),device=d.device)
        meyd1 = torch.ones((b,c,sx,sy-1),device=d.device)
        
        min_decay_const = 3*max(torch.max((1-torch.sqrt(erx))/(1+torch.sqrt(erx))),torch.max((1-torch.sqrt(ery))/(1+torch.sqrt(ery))))
        if min_decay_const > 0 and min_decay_const < 1:
            max_num_iters = int(math.log(Binary_LBP1d_PytorchNative.epsilon) / math.log(min_decay_const))
            if max_num_iters < sx + sy:
                max_num_iters = sx+sy
            if max_num_iters < 10:
                max_num_iters = 10
        elif min_decay_const == 0.0:
            max_num_iters = 1
        else:
            max_num_iters = 10000
        
        iter = 0
        while True:
            
            #compute vertex to edge messages
            #messages in x
            mvxd1 = ed[:,:,1:,:].clone()
            mvxd1[:,:,:-1,:] *= mexd1[:,:,1:,:]
            mvxd1[:,:,:,:-1] *= meyd1[:,:,:-1,:]
            mvxd1[:,:,:,1:] *= meyu1[:,:,:-1,:]
            mvxu1 = ed[:,:,:-1,:].clone()
            mvxu1[:,:,1:,:] *= mexu1[:,:,:-1,:]
            mvxu1[:,:,:,:-1] *= meyd1[:,:,:-1,:]
            mvxu1[:,:,:,1:] *= meyu1[:,:,:-1,:]
            mvxd0 = torch.ones_like(mexd0)
            mvxd0[:,:,:-1,:] *= mexd0[:,:,1:,:]
            mvxd0[:,:,:,:-1] *= meyd0[:,:,:-1,:]
            mvxd0[:,:,:,1:] *= meyu0[:,:,:-1,:]
            mvxu0 = torch.ones_like(mexu0)
            mvxu0[:,:,1:,:] *= mexu0[:,:,:-1,:]
            mvxu0[:,:,:,:-1] *= meyd0[:,:,:-1,:]
            mvxu0[:,:,:,1:] *= meyu0[:,:,:-1,:]
            
            #messages in y
            mvyd1 = ed[:,:,:,1:].clone()
            mvyd1[:,:,:,:-1] *= meyd1[:,:,:,1:]
            mvyd1[:,:,:-1,:] *= mexd1[:,:,:,:-1]
            mvyd1[:,:,1:,:] *= mexu1[:,:,:,:-1]
            mvyu1 = ed[:,:,:,:-1].clone()
            mvyu1[:,:,:,1:] *= meyu1[:,:,:,:-1]
            mvyu1[:,:,:-1,:] *= mexd1[:,:,:,:-1]
            mvyu1[:,:,1:,:] *= mexu1[:,:,:,:-1]
            mvyd0 = torch.ones_like(meyd0)
            mvyd0[:,:,:,:-1] *= meyd0[:,:,:,1:]
            mvyd0[:,:,:-1,:] *= mexd0[:,:,:,:-1]
            mvyd0[:,:,1:,:] *= mexu0[:,:,:,:-1]
            mvyu0 = torch.ones_like(meyu0)
            mvyu0[:,:,:,1:] *= meyu0[:,:,:,:-1]
            mvyu0[:,:,:-1,:] *= mexd0[:,:,:,:-1]
            mvyu0[:,:,1:,:] *= mexu0[:,:,:,:-1]
            
            #renormalise messages
            a = mvxu0+mvxu1
            mvxu0 /= a
            mvxu1 /= a
            a = mvxd0+mvxd1
            mvxd0 /= a
            mvxd1 /= a
            a = mvyu0+mvyu1
            mvyu0 /= a
            mvyu1 /= a
            a = mvyd0+mvyd1
            mvyd0 /= a
            mvyd1 /= a
            
            #compute edge to vertex messages
            mexd0 = mvxd0 + erx[:,:,:-1,:] * mvxd1
            mexd1 = mvxd1 + erx[:,:,:-1,:] * mvxd0
            mexu0 = mvxu0 + erx[:,:,:-1,:] * mvxu1
            mexu1 = mvxu1 + erx[:,:,:-1,:] * mvxu0
            meyd0 = mvyd0 + ery[:,:,:,:-1] * mvyd1
            meyd1 = mvyd1 + ery[:,:,:,:-1] * mvyd0
            meyu0 = mvyu0 + ery[:,:,:,:-1] * mvyu1
            meyu1 = mvyu1 + ery[:,:,:,:-1] * mvyu0
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
        marginal = d.clone()
        marginal[:,:,:-1,:] -= torch.log(mexd0)
        marginal[:,:,1:,:] -= torch.log(mexu0)
        marginal[:,:,:-1,:] += torch.log(mexd1)
        marginal[:,:,1:,:] += torch.log(mexu1)
        marginal[:,:,:,:-1] -= torch.log(meyd0)
        marginal[:,:,:,1:] -= torch.log(meyu0)
        marginal[:,:,:,:-1] += torch.log(meyd1)
        marginal[:,:,:,1:] += torch.log(meyu1)
        
        #save context for backwards pass and return the marginal
        ctx.save_for_backward(ed,erx,ery,mvxu0,mvxu1,mvxd0,mvxd1,mexu0,mexu1,mexd0,mexd1,
                                         mvyu0,mvyu1,mvyd0,mvyd1,meyu0,meyu1,meyd0,meyd1,torch.tensor(max_num_iters))
        return marginal
    
    @staticmethod 
    def backward(ctx, grad_output):
        ed,erx,ery,mvxu0,mvxu1,mvxd0,mvxd1,mexu0,mexu1,mexd0,mexd1,mvyu0,mvyu1,mvyd0,mvyd1,meyu0,meyu1,meyd0,meyd1,max_num_iters = ctx.saved_tensors
        max_num_iters = int(max_num_iters.cpu().item())
        a_xu_2 = (mvxu0+mvxu1)**2
        a_xd_2 = (mvxd0+mvxd1)**2
        a_yu_2 = (mvyu0+mvyu1)**2
        a_yd_2 = (mvyd0+mvyd1)**2
        grad_d =  torch.zeros_like(ed) #(will multiply out the ed at the end, keeps everything in exponential domain)
        grad_rx = torch.zeros_like(erx)
        grad_ry = torch.zeros_like(ery)
        
        #backprop through final layer
        grad_mexu0 = - grad_output[:,:,1:,:] / mexu0
        grad_mexd0 = - grad_output[:,:,:-1,:] / mexd0
        grad_mexu1 = grad_output[:,:,1:,:] / mexu1
        grad_mexd1 = grad_output[:,:,:-1,:] / mexd1
        grad_meyu0 = - grad_output[:,:,:,1:] / meyu0
        grad_meyd0 = - grad_output[:,:,:,:-1] / meyd0
        grad_meyu1 = grad_output[:,:,:,1:] / meyu1
        grad_meyd1 = grad_output[:,:,:,:-1] / meyd1
        
        iter = 0
        while True:
            
            #backprop back to vertex messages (REDO)
            grad_mvxu0 = grad_mexu0 + erx[:,:,:-1,:]*grad_mexu1
            grad_mvxu1 = grad_mexu1 + erx[:,:,:-1,:]*grad_mexu0
            grad_mvxd0 = grad_mexd0 + erx[:,:,:-1,:]*grad_mexd1
            grad_mvxd1 = grad_mexd1 + erx[:,:,:-1,:]*grad_mexd0
            grad_mvyu0 = grad_meyu0 + ery[:,:,:,:-1]*grad_meyu1
            grad_mvyu1 = grad_meyu1 + ery[:,:,:,:-1]*grad_meyu0
            grad_mvyd0 = grad_meyd0 + ery[:,:,:,:-1]*grad_meyd1
            grad_mvyd1 = grad_meyd1 + ery[:,:,:,:-1]*grad_meyd0
            grad_rx[:,:,:-1,:] += grad_mexu1 + grad_mexu0 + grad_mexd1 + grad_mexd0
            grad_ry[:,:,:,:-1] += grad_meyu1 + grad_meyu0 + grad_meyd1 + grad_meyd0
            
            #backprop through renormalisation
            grad_mvxu0, grad_mvxu1 = (mvxu1*grad_mvxu0 - mvxu0*grad_mvxu1) / a_xu_2, (mvxu0*grad_mvxu1 - mvxu1*grad_mvxu0) / a_xu_2
            grad_mvxd0, grad_mvxd1 = (mvxd1*grad_mvxd0 - mvxd0*grad_mvxd1) / a_xd_2, (mvxd0*grad_mvxd1 - mvxd1*grad_mvxd0) / a_xd_2
            grad_mvyu0, grad_mvyu1 = (mvyu1*grad_mvyu0 - mvyu0*grad_mvyu1) / a_yu_2, (mvyu0*grad_mvyu1 - mvyu1*grad_mvyu0) / a_yu_2
            grad_mvyd0, grad_mvyd1 = (mvyd1*grad_mvyd0 - mvyd0*grad_mvyd1) / a_yd_2, (mvyd0*grad_mvyd1 - mvyd1*grad_mvyd0) / a_yd_2
            
            grad_mexd0 *= 0
            grad_mexu0 *= 0
            grad_mexd1 *= 0
            grad_mexu1 *= 0
            grad_meyd0 *= 0
            grad_meyu0 *= 0
            grad_meyd1 *= 0
            grad_meyu1 *= 0
            
            #backprop back to edge messages in x direction TODO Check correctness
            to_add = grad_mvxd1.clone() #ed part for mvxd1 equation
            to_add[:,:,:-1,:] *= mexd1[:,:,1:,:]
            to_add[:,:,:,:-1] *= meyd1[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu1[:,:,:-1,:]
            grad_d[:,:,1:,:] += to_add 
            to_add = grad_mvxd1 * ed[:,:,1:,:] #mexd1 part for mvxd1 equation
            to_add[:,:,:,:-1] *= meyd1[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu1[:,:,:-1,:]
            grad_mexd1[:,:,1:,:] += to_add[:,:,:-1,:]
            to_add = grad_mvxd1 * ed[:,:,1:,:] #meyu1 part for mvxd1 equation
            to_add[:,:,:-1,:] *= mexd1[:,:,1:,:]
            to_add[:,:,:,:-1] *= meyd1[:,:,:-1,:]
            grad_meyu1[:,:,:-1,:] += to_add[:,:,:,1:]
            to_add = grad_mvxd1 * ed[:,:,1:,:] #meyd1 part for mvxd1 equation
            to_add[:,:,:-1,:] *= mexd1[:,:,1:,:]
            to_add[:,:,:,1:] *= meyu1[:,:,:-1,:]
            grad_meyd1[:,:,:-1,:] += to_add[:,:,:,:-1]
            
            to_add = grad_mvxd0.clone() #mexd0 part for mvxd0 equation
            to_add[:,:,:,:-1] *= meyd0[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu0[:,:,:-1,:]
            grad_mexd0[:,:,1:,:] += to_add[:,:,:-1,:]
            to_add = grad_mvxd0.clone() #meyu0 part for mvxd0 equation
            to_add[:,:,:-1,:] *= mexd0[:,:,1:,:]
            to_add[:,:,:,:-1] *= meyd0[:,:,:-1,:]
            grad_meyu0[:,:,:-1,:] += to_add[:,:,:,1:]
            to_add = grad_mvxd0.clone() #meyd0 part for mvxd0 equation
            to_add[:,:,:-1,:] *= mexd0[:,:,1:,:]
            to_add[:,:,:,1:] *= meyu0[:,:,:-1,:]
            grad_meyd0[:,:,:-1,:] += to_add[:,:,:,:-1]
            
            to_add = grad_mvxu1.clone() #ed part for mvxu1 equation
            to_add[:,:,1:,:] *= mexu1[:,:,:-1,:]
            to_add[:,:,:,:-1] *= meyd1[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu1[:,:,:-1,:]
            grad_d[:,:,:-1,:] += to_add 
            to_add = grad_mvxu1 * ed[:,:,:-1,:] #mexu1 part for mvxu1 equation
            to_add[:,:,:,:-1] *= meyd1[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu1[:,:,:-1,:]
            grad_mexu1[:,:,:-1,:] += to_add[:,:,1:,:]
            to_add = grad_mvxu1 * ed[:,:,:-1,:] #meyu1 part for mvxu1 equation
            to_add[:,:,1:,:] *= mexu1[:,:,:-1,:]
            to_add[:,:,:,:-1] *= meyd1[:,:,:-1,:]
            grad_meyu1[:,:,:-1,:] += to_add[:,:,:,1:]
            to_add = grad_mvxu1 * ed[:,:,:-1,:] #meyd1 part for mvxu1 equation
            to_add[:,:,1:,:] *= mexu1[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu1[:,:,:-1,:]
            grad_meyd1[:,:,:-1,:] += to_add[:,:,:,:-1]
            
            to_add = grad_mvxu0.clone() #mexu0 part for mvxu0 equation
            to_add[:,:,:,:-1] *= meyd0[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu0[:,:,:-1,:]
            grad_mexu0[:,:,:-1,:] += to_add[:,:,1:,:]
            to_add = grad_mvxu0.clone() #meyu0 part for mvxu0 equation
            to_add[:,:,1:,:] *= mexu0[:,:,:-1,:]
            to_add[:,:,:,:-1] *= meyd0[:,:,:-1,:]
            grad_meyu0[:,:,:-1,:] += to_add[:,:,:,1:]
            to_add = grad_mvxu0.clone() #meyd0 part for mvxu0 equation
            to_add[:,:,1:,:] *= mexu0[:,:,:-1,:]
            to_add[:,:,:,1:] *= meyu0[:,:,:-1,:]
            grad_meyd0[:,:,:-1,:] += to_add[:,:,:,:-1]
            
            #backprop back to edge messages in y direction TODO Check correctness            
            to_add = grad_mvyd1.clone() #ed part for mvyd1 equation
            to_add[:,:,:,:-1] *= meyd1[:,:,:,1:]
            to_add[:,:,:-1,:] *= mexd1[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu1[:,:,:,:-1]
            grad_d[:,:,:,1:] += to_add 
            to_add = grad_mvyd1 * ed[:,:,:,1:] # meyd1 part for mvyd1 equation
            to_add[:,:,:-1,:] *= mexd1[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu1[:,:,:,:-1]
            grad_meyd1[:,:,:,1:] += to_add[:,:,:,:-1]
            to_add = grad_mvyd1 * ed[:,:,:,1:] # mexd1 part for mvyd1 equation
            to_add[:,:,:,:-1] *= meyd1[:,:,:,1:]
            to_add[:,:,1:,:] *= mexu1[:,:,:,:-1]
            grad_mexd1[:,:,:,:-1] += to_add[:,:,:-1,:]
            to_add = grad_mvyd1 * ed[:,:,:,1:] # mexu1 part for mvyd1 equation
            to_add[:,:,:,:-1] *= meyd1[:,:,:,1:]
            to_add[:,:,:-1,:] *= mexd1[:,:,:,:-1]
            grad_mexu1[:,:,:,:-1] += to_add[:,:,1:,:]
            
            to_add = grad_mvyd0.clone() # meyd0 part for mvyd0 equation
            to_add[:,:,:-1,:] *= mexd0[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu0[:,:,:,:-1]
            grad_meyd0[:,:,:,1:] += to_add[:,:,:,:-1]
            to_add = grad_mvyd0.clone() # mexd0 part for mvyd0 equation
            to_add[:,:,:,:-1] *= meyd0[:,:,:,1:]
            to_add[:,:,1:,:] *= mexu0[:,:,:,:-1]
            grad_mexd0[:,:,:,:-1] += to_add[:,:,:-1,:]
            to_add = grad_mvyd0.clone() # mexu0 part for mvyd0 equation
            to_add[:,:,:,:-1] *= meyd0[:,:,:,1:]
            to_add[:,:,:-1,:] *= mexd0[:,:,:,:-1]
            grad_mexu0[:,:,:,:-1] += to_add[:,:,1:,:]
            
            to_add = grad_mvyu1.clone() #ed part for mvyu1 equation
            to_add[:,:,:,1:] *= meyu1[:,:,:,:-1]
            to_add[:,:,:-1,:] *= mexd1[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu1[:,:,:,:-1]
            grad_d[:,:,:,:-1] += to_add
            to_add = grad_mvyu1 * ed[:,:,:,:-1] #meyu1 part for mvyu1 equation
            to_add[:,:,:-1,:] *= mexd1[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu1[:,:,:,:-1]
            grad_meyu1[:,:,:,:-1] += to_add[:,:,:,1:]
            to_add = grad_mvyu1 * ed[:,:,:,:-1] #mexd1 part for mvyu1 equation
            to_add[:,:,:,1:] *= meyu1[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu1[:,:,:,:-1]
            grad_mexd1[:,:,:,:-1] += to_add[:,:,:-1,:]
            to_add = grad_mvyu1 * ed[:,:,:,:-1] #mexu1 part for mvyu1 equation
            to_add[:,:,:,1:] *= meyu1[:,:,:,:-1]
            to_add[:,:,:-1,:] *= mexd1[:,:,:,:-1]
            grad_mexu1[:,:,:,:-1] += to_add[:,:,1:,:]
            
            to_add = grad_mvyu0.clone() #meyu0 part for mvyu0 equation
            to_add[:,:,:-1,:] *= mexd0[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu0[:,:,:,:-1]
            grad_meyu0[:,:,:,:-1] += to_add[:,:,:,1:]
            to_add = grad_mvyu0.clone() #mexd0 part for mvyu0 equation
            to_add[:,:,:,1:] *= meyu0[:,:,:,:-1]
            to_add[:,:,1:,:] *= mexu0[:,:,:,:-1]
            grad_mexd0[:,:,:,:-1] += to_add[:,:,:-1,:]
            to_add = grad_mvyu0.clone() #mexu0 part for mvyu0 equation
            to_add[:,:,:,1:] *= meyu0[:,:,:,:-1]
            to_add[:,:,:-1,:] *= mexd0[:,:,:,:-1]
            grad_mexu0[:,:,:,:-1] += to_add[:,:,1:,:]
        
            iter+=1
            if iter >= max_num_iters:
                break
        
        #correct to no longer be in the exponential domain
        grad_rx *= erx
        grad_ry *= ery
        grad_d *= ed
        grad_d += grad_output
        
        return grad_d, grad_rx, grad_ry
        
        
class Binary_Mean3d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3)
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
    

class Binary_Mean3d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-6
    
    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3)
        u = 1/(1+torch.exp(-d))
        iter = 0
        while True:
            
            energy = torch.zeros_like(u)
            energy[:,:,:-1,:,:] += rx[:,:,:-1,:,:]*(2*u[:,:,1:,:,:]-1)
            energy[:,:,1:,:,:] += rx[:,:,:-1,:,:]*(2*u[:,:,:-1,:,:]-1)
            energy[:,:,:,:-1,:] += ry[:,:,:,:-1,:]*(2*u[:,:,:,1:,:]-1)
            energy[:,:,:,1:,:] += ry[:,:,:,:-1,:]*(2*u[:,:,:,:-1,:]-1)
            energy[:,:,:,:,:-1] += rz[:,:,:,:,:-1]*(2*u[:,:,:,:,1:]-1)
            energy[:,:,:,:,1:] += rz[:,:,:,:,:-1]*(2*u[:,:,:,:,:-1]-1)
            u_new = 1/(1+torch.exp(-d-energy))
            change = torch.max(torch.abs(u_new-u))
            u_new = u
            
            iter+=1
            if change < epsilon:
                break
            
        output = d + energy
        ctx.save_for_backward(output,rx,ry,rz)
        return output
            
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        u,rx,ry,rz, = ctx.saved_tensors
        u = 1/(1+torch.exp(-u))
        grad_d =  torch.zeros_like(u)
        grad_rx = torch.zeros_like(rx)
        grad_ry = torch.zeros_like(ry)
        grad_rz = torch.zeros_like(rz)
        
        dE = grad_output.clone()
        
        iter = 0
        while True:
            
            grad_d += dE
            grad_rx[:,:,:-1,:,:] += (2*u[:,:,1:,:,:]-1)* dE[:,:,:-1,:,:]
            grad_rx[:,:,:-1,:,:] += (2*u[:,:,:-1,:,:]-1)* dE[:,:,1:,:,:]
            grad_ry[:,:,:,:-1,:] += (2*u[:,:,:,1:,:]-1)* dE[:,:,:,:-1,:]
            grad_ry[:,:,:,:-1,:] += (2*u[:,:,:,:-1,:]-1)* dE[:,:,:,1:,:]
            grad_rz[:,:,:,:,:-1] += (2*u[:,:,:,:,1:]-1)* dE[:,:,:,:,:-1]
            grad_rz[:,:,:,:,:-1] += (2*u[:,:,:,:,:-1]-1)* dE[:,:,:,:,1:]
            
            new_du = torch.zeros_like(u)
            new_du[:,:,1:,:,:] += 2*dE[:,:,:-1,:,:]*rx[:,:,:-1,:,:]
            new_du[:,:,:-1,:,:] += 2*dE[:,:,1:,:,:]*rx[:,:,:-1,:,:]
            new_du[:,:,:,1:,:] += 2*dE[:,:,:,:-1,:]*ry[:,:,:,:-1,:]
            new_du[:,:,:,:-1,:] += 2*dE[:,:,:,1:,:]*ry[:,:,:,:-1,:]
            new_du[:,:,:,:,1:] += 2*dE[:,:,:,:,:-1]*rz[:,:,:,:,:-1]
            new_du[:,:,:,:,:-1] += 2*dE[:,:,:,:,1:]*rz[:,:,:,:,:-1]
            
            new_dE = new_du * u * (1-u)
            
            change=torch.max(torch.abs(new_dE-dE))
            dE = new_dE
            
            iter+=1
            if change < epsilon:
                break
        
        return grad_d, grad_rx, grad_ry, grad_rz
    
class Binary_LBP3d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-6
    
    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3)
        b,c,sx,sy, sz = d.shape[0],d.shape[1],d.shape[2],d.shape[3],d.shape[4]
        ed = torch.exp(d)
        erx = torch.exp(-rx)
        ery = torch.exp(-ry)
        erz = torch.exp(-rz)
        mvxu0 = torch.ones((b,c,sx-1,sy,sz),device=d.device)
        mvxd0 = torch.ones((b,c,sx-1,sy,sz),device=d.device)
        mvxu1 = torch.ones((b,c,sx-1,sy,sz),device=d.device)
        mvxd1 = torch.ones((b,c,sx-1,sy,sz),device=d.device)
        mexu0 = torch.ones((b,c,sx-1,sy,sz),device=d.device)
        mexd0 = torch.ones((b,c,sx-1,sy,sz),device=d.device)
        mexu1 = torch.ones((b,c,sx-1,sy,sz),device=d.device)
        mexd1 = torch.ones((b,c,sx-1,sy,sz),device=d.device)
        mvyu0 = torch.ones((b,c,sx,sy-1,sz),device=d.device)
        mvyd0 = torch.ones((b,c,sx,sy-1,sz),device=d.device)
        mvyu1 = torch.ones((b,c,sx,sy-1,sz),device=d.device)
        mvyd1 = torch.ones((b,c,sx,sy-1,sz),device=d.device)
        meyu0 = torch.ones((b,c,sx,sy-1,sz),device=d.device)
        meyd0 = torch.ones((b,c,sx,sy-1,sz),device=d.device)
        meyu1 = torch.ones((b,c,sx,sy-1,sz),device=d.device)
        meyd1 = torch.ones((b,c,sx,sy-1,sz),device=d.device)
        mvzu0 = torch.ones((b,c,sx,sy,sz-1),device=d.device)
        mvzd0 = torch.ones((b,c,sx,sy,sz-1),device=d.device)
        mvzu1 = torch.ones((b,c,sx,sy,sz-1),device=d.device)
        mvzd1 = torch.ones((b,c,sx,sy,sz-1),device=d.device)
        mezu0 = torch.ones((b,c,sx,sy,sz-1),device=d.device)
        mezd0 = torch.ones((b,c,sx,sy,sz-1),device=d.device)
        mezu1 = torch.ones((b,c,sx,sy,sz-1),device=d.device)
        mezd1 = torch.ones((b,c,sx,sy,sz-1),device=d.device)
        
        min_decay_const = 5*max([torch.max((1-torch.sqrt(erx))/(1+torch.sqrt(erx))),
                                 torch.max((1-torch.sqrt(ery))/(1+torch.sqrt(ery))),
                                 torch.max((1-torch.sqrt(ery))/(1+torch.sqrt(ery)))])
        if min_decay_const > 0 and min_decay_const < 1:
            max_num_iters = int(math.log(Binary_LBP1d_PytorchNative.epsilon) / math.log(min_decay_const))
            if max_num_iters < sx+sy+sz:
                max_num_iters = sx+sy+sz
            if max_num_iters < 10:
                max_num_iters = 10
        elif min_decay_const == 0.0:
            max_num_iters = 1
        else:
            max_num_iters = 10000
        
        iter = 0
        while True:
            
            #compute vertex to edge messages
            #messages in x
            mvxd1 = ed[:,:,1:,:,:].clone()
            mvxd1[:,:,:-1,:,:] *= mexd1[:,:,1:,:,:]
            mvxd1[:,:,:,:-1,:] *= meyd1[:,:,:-1,:,:]
            mvxd1[:,:,:,1:,:] *= meyu1[:,:,:-1,:,:]
            mvxd1[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            mvxd1[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            mvxu1 = ed[:,:,:-1,:,:].clone()
            mvxu1[:,:,1:,:,:] *= mexu1[:,:,:-1,:,:]
            mvxu1[:,:,:,:-1,:] *= meyd1[:,:,:-1,:,:]
            mvxu1[:,:,:,1:,:] *= meyu1[:,:,:-1,:,:]
            mvxu1[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            mvxu1[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            mvxd0 = torch.ones_like(mexd0)
            mvxd0[:,:,:-1,:,:] *= mexd0[:,:,1:,:,:]
            mvxd0[:,:,:,:-1,:] *= meyd0[:,:,:-1,:,:]
            mvxd0[:,:,:,1:,:] *= meyu0[:,:,:-1,:,:]
            mvxd0[:,:,:,:,:-1] *= mezd0[:,:,:-1,:,:]
            mvxd0[:,:,:,:,1:] *= mezu0[:,:,:-1,:,:]
            mvxu0 = torch.ones_like(mexu0)
            mvxu0[:,:,1:,:,:] *= mexu0[:,:,:-1,:,:]
            mvxu0[:,:,:,:-1,:] *= meyd0[:,:,:-1,:,:]
            mvxu0[:,:,:,1:,:] *= meyu0[:,:,:-1,:,:]
            mvxu0[:,:,:,:,:-1] *= mezd0[:,:,:-1,:,:]
            mvxu0[:,:,:,:,1:] *= mezu0[:,:,:-1,:,:]
            
            #messages in y
            mvyd1 = ed[:,:,:,1:,:].clone()
            mvyd1[:,:,:,:-1,:] *= meyd1[:,:,:,1:,:]
            mvyd1[:,:,:-1,:,:] *= mexd1[:,:,:,:-1,:]
            mvyd1[:,:,1:,:,:] *= mexu1[:,:,:,:-1,:]
            mvyd1[:,:,:,:,:-1] *= mezd1[:,:,:,:-1,:]
            mvyd1[:,:,:,:,1:] *= mezu1[:,:,:,:-1,:]
            mvyu1 = ed[:,:,:,:-1,:].clone()
            mvyu1[:,:,:,1:,:] *= meyu1[:,:,:,:-1,:]
            mvyu1[:,:,:-1,:,:] *= mexd1[:,:,:,:-1,:]
            mvyu1[:,:,1:,:,:] *= mexu1[:,:,:,:-1,:]
            mvyu1[:,:,:,:,:-1] *= mezd1[:,:,:,:-1,:]
            mvyu1[:,:,:,:,1:] *= mezu1[:,:,:,:-1,:]
            mvyd0 = torch.ones_like(meyd0)
            mvyd0[:,:,:,:-1,:] *= meyd0[:,:,:,1:,:]
            mvyd0[:,:,:-1,:,:] *= mexd0[:,:,:,:-1,:]
            mvyd0[:,:,1:,:,:] *= mexu0[:,:,:,:-1,:]
            mvyd0[:,:,:,:,:-1] *= mezd0[:,:,:,:-1,:]
            mvyd0[:,:,:,:,1:] *= mezu0[:,:,:,:-1,:]
            mvyu0 = torch.ones_like(meyu0)
            mvyu0[:,:,:,1:,:] *= meyu0[:,:,:,:-1,:]
            mvyu0[:,:,:-1,:,:] *= mexd0[:,:,:,:-1,:]
            mvyu0[:,:,1:,:,:] *= mexu0[:,:,:,:-1,:]
            mvyu0[:,:,:,:,:-1] *= mezd0[:,:,:,:-1,:]
            mvyu0[:,:,:,:,1:] *= mezu0[:,:,:,:-1,:]
            
            #messages in z
            mvzd1 = ed[:,:,:,:,1:].clone()
            mvzd1[:,:,:,:,:-1] *= mezd1[:,:,:,:,1:]
            mvzd1[:,:,:-1,:,:] *= mexd1[:,:,:,:,:-1]
            mvzd1[:,:,1:,:,:] *= mexu1[:,:,:,:,:-1]
            mvzd1[:,:,:,:-1,:] *= meyd1[:,:,:,:,:-1]
            mvzd1[:,:,:,1:,:] *= meyu1[:,:,:,:,:-1]
            mvzu1 = ed[:,:,:,:,:-1].clone()
            mvzu1[:,:,:,:,1:] *= mezu1[:,:,:,:,:-1]
            mvzu1[:,:,:-1,:,:] *= mexd1[:,:,:,:,:-1]
            mvzu1[:,:,1:,:,:] *= mexu1[:,:,:,:,:-1]
            mvzu1[:,:,:,:-1,:] *= meyd1[:,:,:,:,:-1]
            mvzu1[:,:,:,1:,:] *= meyu1[:,:,:,:,:-1]
            mvzd0 = torch.ones_like(mezd0)
            mvzd0[:,:,:,:,:-1] *= mezd0[:,:,:,:,1:]
            mvzd0[:,:,:-1,:,:] *= mexd0[:,:,:,:,:-1]
            mvzd0[:,:,1:,:,:] *= mexu0[:,:,:,:,:-1]
            mvzd0[:,:,:,:-1,:] *= meyd0[:,:,:,:,:-1]
            mvzd0[:,:,:,1:,:] *= meyu0[:,:,:,:,:-1]
            mvzu0 = torch.ones_like(mezu0)
            mvzu0[:,:,:,:,1:] *= mezu0[:,:,:,:,:-1]
            mvzu0[:,:,:-1,:,:] *= mexd0[:,:,:,:,:-1]
            mvzu0[:,:,1:,:,:] *= mexu0[:,:,:,:,:-1]
            mvzu0[:,:,:,:-1,:] *= meyd0[:,:,:,:,:-1]
            mvzu0[:,:,:,1:,:] *= meyu0[:,:,:,:,:-1]
            
            #renormalise messages
            a = mvxu0+mvxu1
            mvxu0 /= a
            mvxu1 /= a
            a = mvxd0+mvxd1
            mvxd0 /= a
            mvxd1 /= a
            a = mvyu0+mvyu1
            mvyu0 /= a
            mvyu1 /= a
            a = mvyd0+mvyd1
            mvyd0 /= a
            mvyd1 /= a
            a = mvzu0+mvzu1
            mvzu0 /= a
            mvzu1 /= a
            a = mvzd0+mvzd1
            mvzd0 /= a
            mvzd1 /= a
            
            #compute edge to vertex messages
            mexd0 = mvxd0 + erx[:,:,:-1,:,:] * mvxd1
            mexd1 = mvxd1 + erx[:,:,:-1,:,:] * mvxd0
            mexu0 = mvxu0 + erx[:,:,:-1,:,:] * mvxu1
            mexu1 = mvxu1 + erx[:,:,:-1,:,:] * mvxu0
            meyd0 = mvyd0 + ery[:,:,:,:-1,:] * mvyd1
            meyd1 = mvyd1 + ery[:,:,:,:-1,:] * mvyd0
            meyu0 = mvyu0 + ery[:,:,:,:-1,:] * mvyu1
            meyu1 = mvyu1 + ery[:,:,:,:-1,:] * mvyu0
            mezd0 = mvzd0 + erz[:,:,:,:,:-1] * mvzd1
            mezd1 = mvzd1 + erz[:,:,:,:,:-1] * mvzd0
            mezu0 = mvzu0 + erz[:,:,:,:,:-1] * mvzu1
            mezu1 = mvzu1 + erz[:,:,:,:,:-1] * mvzu0
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
        marginal = d.clone()
        marginal[:,:,:-1,:,:] -= torch.log(mexd0)
        marginal[:,:,1:,:,:] -= torch.log(mexu0)
        marginal[:,:,:-1,:,:] += torch.log(mexd1)
        marginal[:,:,1:,:,:] += torch.log(mexu1)
        marginal[:,:,:,:-1,:] -= torch.log(meyd0)
        marginal[:,:,:,1:,:] -= torch.log(meyu0)
        marginal[:,:,:,:-1,:] += torch.log(meyd1)
        marginal[:,:,:,1:,:] += torch.log(meyu1)
        marginal[:,:,:,:,:-1] -= torch.log(mezd0)
        marginal[:,:,:,:,1:] -= torch.log(mezu0)
        marginal[:,:,:,:,:-1] += torch.log(mezd1)
        marginal[:,:,:,:,1:] += torch.log(mezu1)
        
        #save context for backwards pass and return the marginal
        ctx.save_for_backward(ed,erx,ery,erz,mvxu0,mvxu1,mvxd0,mvxd1,mexu0,mexu1,mexd0,mexd1,
                                             mvyu0,mvyu1,mvyd0,mvyd1,meyu0,meyu1,meyd0,meyd1,
                                             mvzu0,mvzu1,mvzd0,mvzd1,mezu0,mezu1,mezd0,mezd1,torch.tensor(max_num_iters))
        return marginal
    
    @staticmethod 
    def backward(ctx, grad_output):
        ed,erx,ery,erz,mvxu0,mvxu1,mvxd0,mvxd1,mexu0,mexu1,mexd0,mexd1,mvyu0,mvyu1,mvyd0,mvyd1,meyu0,meyu1,meyd0,meyd1,mvzu0,mvzu1,mvzd0,mvzd1,mezu0,mezu1,mezd0,mezd1,max_num_iters = ctx.saved_tensors
        max_num_iters = int(max_num_iters.cpu().item())
        a_xu_2 = (mvxu0+mvxu1)**2
        a_xd_2 = (mvxd0+mvxd1)**2
        a_yu_2 = (mvyu0+mvyu1)**2
        a_yd_2 = (mvyd0+mvyd1)**2
        a_zu_2 = (mvzu0+mvzu1)**2
        a_zd_2 = (mvzd0+mvzd1)**2
        grad_d =  torch.zeros_like(ed) #(will multiply out the ed at the end, keeps everything in exponential domain)
        grad_rx = torch.zeros_like(erx)
        grad_ry = torch.zeros_like(ery)
        grad_rz = torch.zeros_like(erz)
        
        #backprop through final layer
        grad_mexu0 = - grad_output[:,:,1:,:,:] / mexu0
        grad_mexd0 = - grad_output[:,:,:-1,:,:] / mexd0
        grad_mexu1 = grad_output[:,:,1:,:,:] / mexu1
        grad_mexd1 = grad_output[:,:,:-1,:,:] / mexd1
        grad_meyu0 = - grad_output[:,:,:,1:,:] / meyu0
        grad_meyd0 = - grad_output[:,:,:,:-1,:] / meyd0
        grad_meyu1 = grad_output[:,:,:,1:,:] / meyu1
        grad_meyd1 = grad_output[:,:,:,:-1,:] / meyd1
        grad_mezu0 = - grad_output[:,:,:,:,1:] / mezu0
        grad_mezd0 = - grad_output[:,:,:,:,:-1] / mezd0
        grad_mezu1 = grad_output[:,:,:,:,1:] / mezu1
        grad_mezd1 = grad_output[:,:,:,:,:-1] / mezd1
        
        iter = 0
        while True:
            
            #backprop back to vertex messages (REDO)
            grad_mvxu0 = grad_mexu0 + erx[:,:,:-1,:,:]*grad_mexu1
            grad_mvxu1 = grad_mexu1 + erx[:,:,:-1,:,:]*grad_mexu0
            grad_mvxd0 = grad_mexd0 + erx[:,:,:-1,:,:]*grad_mexd1
            grad_mvxd1 = grad_mexd1 + erx[:,:,:-1,:,:]*grad_mexd0
            grad_mvyu0 = grad_meyu0 + ery[:,:,:,:-1,:]*grad_meyu1
            grad_mvyu1 = grad_meyu1 + ery[:,:,:,:-1,:]*grad_meyu0
            grad_mvyd0 = grad_meyd0 + ery[:,:,:,:-1,:]*grad_meyd1
            grad_mvyd1 = grad_meyd1 + ery[:,:,:,:-1,:]*grad_meyd0
            grad_mvzu0 = grad_mezu0 + erz[:,:,:,:,:-1]*grad_mezu1
            grad_mvzu1 = grad_mezu1 + erz[:,:,:,:,:-1]*grad_mezu0
            grad_mvzd0 = grad_mezd0 + erz[:,:,:,:,:-1]*grad_mezd1
            grad_mvzd1 = grad_mezd1 + erz[:,:,:,:,:-1]*grad_mezd0
            grad_rx[:,:,:-1,:,:] += grad_mexu1 + grad_mexu0 + grad_mexd1 + grad_mexd0
            grad_ry[:,:,:,:-1,:] += grad_meyu1 + grad_meyu0 + grad_meyd1 + grad_meyd0
            grad_rz[:,:,:,:,:-1] += grad_mezu1 + grad_mezu0 + grad_mezd1 + grad_mezd0
            
            #backprop through renormalisation
            grad_mvxu0, grad_mvxu1 = (mvxu1*grad_mvxu0 - mvxu0*grad_mvxu1) / a_xu_2, (mvxu0*grad_mvxu1 - mvxu1*grad_mvxu0) / a_xu_2
            grad_mvxd0, grad_mvxd1 = (mvxd1*grad_mvxd0 - mvxd0*grad_mvxd1) / a_xd_2, (mvxd0*grad_mvxd1 - mvxd1*grad_mvxd0) / a_xd_2
            grad_mvyu0, grad_mvyu1 = (mvyu1*grad_mvyu0 - mvyu0*grad_mvyu1) / a_yu_2, (mvyu0*grad_mvyu1 - mvyu1*grad_mvyu0) / a_yu_2
            grad_mvyd0, grad_mvyd1 = (mvyd1*grad_mvyd0 - mvyd0*grad_mvyd1) / a_yd_2, (mvyd0*grad_mvyd1 - mvyd1*grad_mvyd0) / a_yd_2
            grad_mvzu0, grad_mvzu1 = (mvzu1*grad_mvzu0 - mvzu0*grad_mvzu1) / a_zu_2, (mvzu0*grad_mvzu1 - mvzu1*grad_mvzu0) / a_zu_2
            grad_mvzd0, grad_mvzd1 = (mvzd1*grad_mvzd0 - mvzd0*grad_mvzd1) / a_zd_2, (mvzd0*grad_mvzd1 - mvzd1*grad_mvzd0) / a_zd_2
            
            grad_mexd0 *= 0
            grad_mexu0 *= 0
            grad_mexd1 *= 0
            grad_mexu1 *= 0
            grad_meyd0 *= 0
            grad_meyu0 *= 0
            grad_meyd1 *= 0
            grad_meyu1 *= 0
            grad_mezd0 *= 0
            grad_mezu0 *= 0
            grad_mezd1 *= 0
            grad_mezu1 *= 0
            
            #backprop back to edge messages in x direction TODO Check correctness
            to_add = grad_mvxd1.clone() #ed part for mvxd1 equation
            to_add[:,:,:-1,:,:] *= mexd1[:,:,1:,:,:]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            grad_d[:,:,1:,:,:] += to_add 
            to_add = grad_mvxd1 * ed[:,:,1:,:,:] #mexd1 part for mvxd1 equation
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            grad_mexd1[:,:,1:,:,:] += to_add[:,:,:-1,:]
            to_add = grad_mvxd1 * ed[:,:,1:,:,:] #meyu1 part for mvxd1 equation
            to_add[:,:,:-1,:,:] *= mexd1[:,:,1:,:,:]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            grad_meyu1[:,:,:-1,:,:] += to_add[:,:,:,1:,:]
            to_add = grad_mvxd1 * ed[:,:,1:,:,:] #meyd1 part for mvxd1 equation
            to_add[:,:,:-1,:,:] *= mexd1[:,:,1:,:,:]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            grad_meyd1[:,:,:-1,:,:] += to_add[:,:,:,:-1,:]
            to_add = grad_mvxd1 * ed[:,:,1:,:,:] #mezu1 part for mvxd1 equation
            to_add[:,:,:-1,:,:] *= mexd1[:,:,1:,:,:]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            grad_mezu1[:,:,:-1,:,:] += to_add[:,:,:,:,1:]
            to_add = grad_mvxd1 * ed[:,:,1:,:,:] #mezd1 part for mvxd1 equation
            to_add[:,:,:-1,:,:] *= mexd1[:,:,1:,:,:]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            grad_mezd1[:,:,:-1,:,:] += to_add[:,:,:,:,:-1]
            
            to_add = grad_mvxd0.clone() #mexd0 part for mvxd0 equation
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            grad_mexd0[:,:,1:,:,:] += to_add[:,:,:-1,:,:]
            to_add = grad_mvxd0.clone() #meyu0 part for mvxd0 equation
            to_add[:,:,:-1,:,:] *= mexd0[:,:,1:,:,:]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            grad_meyu0[:,:,:-1,:,:] += to_add[:,:,:,1:,:]
            to_add = grad_mvxd0.clone() #meyd0 part for mvxd0 equation
            to_add[:,:,:-1,:,:] *= mexd0[:,:,1:,:,:]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:-1,:,:]
            grad_meyd0[:,:,:-1,:,:] += to_add[:,:,:,:-1,:]
            to_add = grad_mvxd0.clone() #mezu0 part for mvxd0 equation
            to_add[:,:,:-1,:,:] *= mexd0[:,:,1:,:,:]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:-1,:,:]
            grad_mezu0[:,:,:-1,:,:] += to_add[:,:,:,:,1:]
            to_add = grad_mvxd0.clone() #mezd0 part for mvxd0 equation
            to_add[:,:,:-1,:,:] *= mexd0[:,:,1:,:,:]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:-1,:,:]
            grad_mezd0[:,:,:-1,:,:] += to_add[:,:,:,:,:-1]
            
            to_add = grad_mvxu1.clone() #ed part for mvxu1 equation
            to_add[:,:,1:,:,:] *= mexu1[:,:,:-1,:,:]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            grad_d[:,:,:-1,:,:] += to_add 
            to_add = grad_mvxu1 * ed[:,:,:-1,:,:] #mexu1 part for mvxu1 equation
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            grad_mexu1[:,:,:-1,:,:] += to_add[:,:,1:,:,:]
            to_add = grad_mvxu1 * ed[:,:,:-1,:,:] #meyu1 part for mvxu1 equation
            to_add[:,:,1:,:,:] *= mexu1[:,:,:-1,:,:]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            grad_meyu1[:,:,:-1,:,:] += to_add[:,:,:,1:,:]
            to_add = grad_mvxu1 * ed[:,:,:-1,:,:] #meyd1 part for mvxu1 equation
            to_add[:,:,1:,:,:] *= mexu1[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            grad_meyd1[:,:,:-1,:,:] += to_add[:,:,:,:-1,:]
            to_add = grad_mvxu1 * ed[:,:,:-1,:,:] #mezu1 part for mvxd1 equation
            to_add[:,:,1:,:,:] *= mexu1[:,:,:-1,:,:]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:-1,:,:]
            grad_mezd1[:,:,:-1,:,:] += to_add[:,:,:,:,:-1]
            to_add = grad_mvxu1 * ed[:,:,:-1,:,:] #mezu1 part for mvxu1 equation
            to_add[:,:,1:,:,:] *= mexu1[:,:,:-1,:,:]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:-1,:,:]
            grad_mezu1[:,:,:-1,:,:] += to_add[:,:,:,:,1:]
            
            to_add = grad_mvxu0.clone() #mexu0 part for mvxu0 equation
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:-1,:,:]
            grad_mexu0[:,:,:-1,:,:] += to_add[:,:,1:,:,:]
            to_add = grad_mvxu0.clone() #meyu0 part for mvxu0 equation
            to_add[:,:,1:,:,:] *= mexu0[:,:,:-1,:,:]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:-1,:,:]
            grad_meyu0[:,:,:-1,:,:] += to_add[:,:,:,1:,:]
            to_add = grad_mvxu0.clone() #meyd0 part for mvxu0 equation
            to_add[:,:,1:,:,:] *= mexu0[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:-1,:,:]
            grad_meyd0[:,:,:-1,:,:] += to_add[:,:,:,:-1,:]
            to_add = grad_mvxu0.clone() #mezu0 part for mvxd0 equation
            to_add[:,:,1:,:,:] *= mexu0[:,:,:-1,:,:]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:-1,:,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:-1,:,:]
            grad_mezd0[:,:,:-1,:,:] += to_add[:,:,:,:,:-1]
            to_add = grad_mvxu0.clone() #mezu0 part for mvxu0 equation
            to_add[:,:,1:,:,:] *= mexu0[:,:,:-1,:,:]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:-1,:,:]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:-1,:,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:-1,:,:]
            grad_mezu0[:,:,:-1,:,:] += to_add[:,:,:,:,1:]
            
            
            #backprop back to edge messages in y direction TODO Check correctness            
            to_add = grad_mvyd1.clone() #ed part for mvyd1 equation
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,1:,:]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:-1,:]
            grad_d[:,:,:,1:,:] += to_add 
            to_add = grad_mvyd1 * ed[:,:,:,1:,:] # meyd1 part for mvyd1 equation
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:-1,:]
            grad_meyd1[:,:,:,1:,:] += to_add[:,:,:,:-1,:]
            to_add = grad_mvyd1 * ed[:,:,:,1:,:] # mexd1 part for mvyd1 equation
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,1:,:]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:-1,:]
            grad_mexd1[:,:,:,:-1,:] += to_add[:,:,:-1,:,:]
            to_add = grad_mvyd1 * ed[:,:,:,1:,:] # mexu1 part for mvyd1 equation
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,1:,:]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:-1,:]
            grad_mexu1[:,:,:,:-1,:] += to_add[:,:,1:,:,:]           
            to_add = grad_mvyd1 * ed[:,:,:,1:,:] #mezd1 part for mvyd1 equation
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,1:,:]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:-1,:]
            grad_mezd1[:,:,:,:-1,:] += to_add[:,:,:,:,:-1]           
            to_add = grad_mvyd1 * ed[:,:,:,1:,:] #mezu1 part for mvyd1 equation
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,1:,:]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:-1,:]
            grad_mezu1[:,:,:,:-1,:] += to_add[:,:,:,:,1:]
            
            to_add = grad_mvyd0.clone() # meyd0 part for mvyd0 equation
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:,:-1,:]
            grad_meyd0[:,:,:,1:,:] += to_add[:,:,:,:-1,:]
            to_add = grad_mvyd0.clone() # mexd0 part for mvyd0 equation
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:,1:,:]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:,:-1,:]
            grad_mexd0[:,:,:,:-1,:] += to_add[:,:,:-1,:,:]
            to_add = grad_mvyd0.clone() # mexu0 part for mvyd0 equation
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:,1:,:]
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:,:-1,:]
            grad_mexu0[:,:,:,:-1,:] += to_add[:,:,1:,:,:]    
            to_add = grad_mvyd1.clone() #mezd0 part for mvyd0 equation
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:,1:,:]
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:,:-1,:]
            grad_mezd0[:,:,:,:-1,:] += to_add[:,:,:,:,:-1]           
            to_add = grad_mvyd1.clone() #mezu0 part for mvyd0 equation
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:,1:,:]
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:-1,:]
            grad_mezu0[:,:,:,:-1,:] += to_add[:,:,:,:,1:]
            
            to_add = grad_mvyu1.clone() #ed part for mvyu1 equation
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:-1,:]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:-1,:]
            grad_d[:,:,:,:-1,:] += to_add
            to_add = grad_mvyu1 * ed[:,:,:,:-1,:] #meyu1 part for mvyu1 equation
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:-1,:]
            grad_meyu1[:,:,:,:-1,:] += to_add[:,:,:,1:,:]
            to_add = grad_mvyu1 * ed[:,:,:,:-1,:] #mexd1 part for mvyu1 equation
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:-1,:]
            grad_mexd1[:,:,:,:-1,:] += to_add[:,:,:-1,:,:]
            to_add = grad_mvyu1 * ed[:,:,:,:-1,:] #mexu1 part for mvyu1 equation
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:-1,:]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:-1,:]
            grad_mexu1[:,:,:,:-1,:] += to_add[:,:,1:,:,:]
            to_add = grad_mvyu1 * ed[:,:,:,:-1,:] #mezd1 part for mvyu1 equation
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:-1,:]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:-1,:]
            grad_mezd1[:,:,:,:-1,:] += to_add[:,:,:,:,:-1]
            to_add = grad_mvyu1 * ed[:,:,:,:-1,:] #mezu1 part for mvyu1 equation
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:-1,:]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:-1,:]
            grad_mezu1[:,:,:,:-1,:] += to_add[:,:,:,:,1:]
            
            to_add = grad_mvyu0.clone() #meyu0 part for mvyu0 equation
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:,:-1,:]
            grad_meyu0[:,:,:,:-1,:] += to_add[:,:,:,1:,:]
            to_add = grad_mvyu0.clone() #mexd0 part for mvyu0 equation
            to_add[:,:,:,1:,:] *= meyu0[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:,:-1,:]
            grad_mexd0[:,:,:,:-1,:] += to_add[:,:,:-1,:,:]
            to_add = grad_mvyu0.clone() #mexu0 part for mvyu0 equation
            to_add[:,:,:,1:,:] *= meyu0[:,:,:,:-1,:]
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:,:-1,:]
            grad_mexu0[:,:,:,:-1,:] += to_add[:,:,1:,:,:]
            to_add = grad_mvyu0.clone() #mezd0 part for mvyu0 equation
            to_add[:,:,:,1:,:] *= meyu0[:,:,:,:-1,:]
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:-1,:]
            to_add[:,:,:,:,1:] *= mezu0[:,:,:,:-1,:]
            grad_mezd0[:,:,:,:-1,:] += to_add[:,:,:,:,:-1]
            to_add = grad_mvyu0.clone() #mezu0 part for mvyu0 equation
            to_add[:,:,:,1:,:] *= meyu0[:,:,:,:-1,:]
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:-1,:]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:-1,:]
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:-1,:]
            grad_mezu0[:,:,:,:-1,:] += to_add[:,:,:,:,1:]
        
        
            #backprop back to edge messages in z direction TODO Check correctness
            to_add = grad_mvzd1.clone() #ed part for mvzd1 equation
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:,1:]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:,:-1]
            grad_d[:,:,:,:,1:] += to_add
            to_add = grad_mvzd1 * ed[:,:,:,:,1:] #mezd1 part for mvzd1 equation
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:,:-1]
            grad_mezd1[:,:,:,:,1:] += to_add[:,:,:,:,:-1]
            to_add = grad_mvzd1 * ed[:,:,:,:,1:] #mexd1 part for mvzd1 equation
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:,1:]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:,:-1]
            grad_mexd1[:,:,:,:,:-1] += to_add[:,:,:-1,:,:]
            to_add = grad_mvzd1 * ed[:,:,:,:,1:] #mexu1 part for mvzd1 equation
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:,1:]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:,:-1]
            grad_mexu1[:,:,:,:,:-1] += to_add[:,:,1:,:,:]
            to_add = grad_mvzd1 * ed[:,:,:,:,1:] #meyd1 part for mvzd1 equation
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:,1:]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:,:-1]
            grad_meyd1[:,:,:,:,:-1] += to_add[:,:,:,:-1,:]
            to_add = grad_mvzd1 * ed[:,:,:,:,1:] #meyu1 part for mvzd1 equation
            to_add[:,:,:,:,:-1] *= mezd1[:,:,:,:,1:]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,:,:-1]
            grad_meyu1[:,:,:,:,:-1] += to_add[:,:,:,1:,:]
            
            to_add = grad_mvzd0.clone() #mezd0 part for mvzd0 equation
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:,:,:-1]
            grad_mezd0[:,:,:,:,1:] += to_add[:,:,:,:,:-1]
            to_add = grad_mvzd0.clone() #mexd0 part for mvzd0 equation
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:,1:]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:,:,:-1]
            grad_mexd0[:,:,:,:,:-1] += to_add[:,:,:-1,:,:]
            to_add = grad_mvzd0.clone() #mexu0 part for mvzd0 equation
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:,1:]
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:,:,:-1]
            grad_mexu0[:,:,:,:,:-1] += to_add[:,:,1:,:,:]
            to_add = grad_mvzd0.clone() #meyd0 part for mvzd0 equation
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:,1:]
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:,:,:-1]
            grad_meyd0[:,:,:,:,:-1] += to_add[:,:,:,:-1,:]
            to_add = grad_mvzd0.clone() #meyu0 part for mvzd0 equation
            to_add[:,:,:,:,:-1] *= mezd0[:,:,:,:,1:]
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:,:,:-1]
            grad_meyu0[:,:,:,:,:-1] += to_add[:,:,:,1:,:]
            
            to_add = grad_mvzu1.clone() #ed part for mvzu1 equation
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:,:-1]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:,:-1]
            grad_d[:,:,:,:,:-1] += to_add
            to_add = grad_mvzu1 * ed[:,:,:,:,:-1]  #mezu1 part for mvzu1 equation
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:,:-1]
            grad_mezu1[:,:,:,:,:-1] += to_add[:,:,:,:,1:]
            to_add = grad_mvzu1 * ed[:,:,:,:,:-1]  #mexd1 part for mvzu1 equation
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:,:-1]
            grad_mexd1[:,:,:,:,:-1] += to_add[:,:,:-1,:,:]
            to_add = grad_mvzu1 * ed[:,:,:,:,:-1]  #mexu1 part for mvzu1 equation
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:,:-1]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:,:-1]
            grad_mexu1[:,:,:,:,:-1] += to_add[:,:,1:,:,:]
            to_add = grad_mvzu1 * ed[:,:,:,:,:-1]  #meyd1 part for mvzu1 equation
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:,:-1]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu1[:,:,:,:,:-1]
            grad_meyd1[:,:,:,:,:-1] += to_add[:,:,:,:-1,:]
            to_add = grad_mvzu1 * ed[:,:,:,:,:-1]  #meyu1 part for mvzu1 equation
            to_add[:,:,:,:,1:] *= mezu1[:,:,:,:,:-1]
            to_add[:,:,:-1,:,:] *= mexd1[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu1[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd1[:,:,:,:,:-1]
            grad_meyu1[:,:,:,:,:-1] += to_add[:,:,:,1:,:]
        
            to_add = grad_mvzu0.clone()  #mezu0 part for mvzu0 equation
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:,:,:-1]
            grad_mezu0[:,:,:,:,:-1] += to_add[:,:,:,:,1:]
            to_add = grad_mvzu0.clone()  #mexd0 part for mvzu0 equation
            to_add[:,:,:,:,1:] *= mezu0[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:,:,:-1]
            grad_mexd0[:,:,:,:,:-1] += to_add[:,:,:-1,:,:]
            to_add = grad_mvzu0.clone()  #mexu0 part for mvzu0 equation
            to_add[:,:,:,:,1:] *= mezu0[:,:,:,:,:-1]
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:,:,:-1]
            grad_mexu0[:,:,:,:,:-1] += to_add[:,:,1:,:,:]
            to_add = grad_mvzu0.clone()  #meyd0 part for mvzu0 equation
            to_add[:,:,:,:,1:] *= mezu0[:,:,:,:,:-1]
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:,:-1]
            to_add[:,:,:,1:,:] *= meyu0[:,:,:,:,:-1]
            grad_meyd0[:,:,:,:,:-1] += to_add[:,:,:,:-1,:]
            to_add = grad_mvzu0.clone()  #meyu0 part for mvzu0 equation
            to_add[:,:,:,:,1:] *= mezu0[:,:,:,:,:-1]
            to_add[:,:,:-1,:,:] *= mexd0[:,:,:,:,:-1]
            to_add[:,:,1:,:,:] *= mexu0[:,:,:,:,:-1]
            to_add[:,:,:,:-1,:] *= meyd0[:,:,:,:,:-1]
            grad_meyu0[:,:,:,:,:-1] += to_add[:,:,:,1:,:]
            
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