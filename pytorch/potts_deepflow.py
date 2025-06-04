import torch
import deepflow
from deepflow_function import DeepFlowFunction
import math

class Potts_MAP1d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx):
        DeepFlowFunction.check_var_dims([d,rx],1)
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.potts_auglag_1d_gpu_forward(d,rx, output)
        else:
            deepflow.potts_auglag_1d_cpu_forward(d,rx, output)
        return output
            
    #For the optimisers, there is no well defined backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input
        
class Potts_MAP2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry):
        DeepFlowFunction.check_var_dims([d,rx,ry],2)
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.potts_auglag_2d_gpu_forward(d,rx, ry, output)
        else:
            deepflow.potts_auglag_2d_cpu_forward(d,rx, ry, output)
        return output
            
    #For the optimisers, there is no well defined backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input
            
class Potts_MAP3d(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3)
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.potts_auglag_3d_gpu_forward(d, rx, ry, rz, output)
        else:
            deepflow.potts_auglag_3d_cpu_forward(d, rx, ry, rz, output)
        return output
        
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input
        
        
class Potts_Mean1d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx):
        DeepFlowFunction.check_var_dims([d,rx],1)
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.potts_meanpass_1d_gpu_forward(d,rx, output)
        else:
            deepflow.potts_meanpass_1d_cpu_forward(d,rx, output)
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
            deepflow.potts_meanpass_1d_gpu_backward(grad_output, u, rx, grad_d, grad_rx)
        else:
            deepflow.potts_meanpass_1d_cpu_backward(grad_output, u, rx, grad_d, grad_rx)
        return grad_d, grad_rx


class Potts_Mean1d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-6
    
    @staticmethod
    def forward(ctx, d, rx):
        DeepFlowFunction.check_var_dims([d,rx],1)
        u = torch.exp(d) / torch.sum(torch.exp(d),1)
        iter = 0
        while True:
            
            energy = -d
            energy[:,:,:-1] += rx[:,:,:-1]*(2*u[:,:,1:]-1)
            energy[:,:,1:] += rx[:,:,:-1]*(2*u[:,:,:-1]-1)
            u_new = torch.exp(energy) / torch.sum(torch.exp(energy),1)
            change = torch.max(torch.abs(u_new-u))
            u_new = u
            
            iter+=1
            if change < epsilon:
                break
            
        ctx.save_for_backward(energy,rx)
        return energy
            
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
            
            new_dE = new_du * (u - torch.sum(torch.unsqueeze(u,2),torch.unsqueeze(u,1),dim=1))
            
            change=torch.max(torch.abs(new_dE-dE))
            dE = new_dE
            
            iter+=1
            if change < epsilon:
                break
        
        return grad_d, grad_rx
    
class Potts_LBP1d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-4
    
    @staticmethod
    def forward(ctx, d, rx):
        DeepFlowFunction.check_var_dims([d,rx],1)
        b,c,sx = d.shape[0],d.shape[1],d.shape[2]
        ed = torch.exp(d)
        erx = torch.exp(-rx)
        mvxu = torch.ones((b,c,sx-1),device=d.device)
        mvxd = torch.ones((b,c,sx-1),device=d.device)
        mexu = torch.ones((b,c,sx-1),device=d.device)
        mexd = torch.ones((b,c,sx-1),device=d.device)
        
        min_er = torch.min(torch.prod(torch.sort(erx,dim=1)[0][:,0:2,:],dim=1))
        decay_const = 2 * (2-1) * (1-math.sqrt(min_er)) / (1+math.sqrt(min_er))
        if decay_const >= 1-Potts_LBP1d_PytorchNative.epsilon:
            #not guaranteed to converge, should throw error
            max_num_iters = 10*sx
        elif decay_const <= Potts_LBP1d_PytorchNative.epsilon:
            max_num_iters = 1
        else:
            max_num_iters = int(math.log(Potts_LBP1d_PytorchNative.epsilon) / math.log(decay_const))
        
        iter = 0
        while True:
            
            #compute vertex to edge messages
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
            mexd = mvxd.clone()
            a = torch.sum(erx[:,:,:-1] * mvxd, 1, keepdim=True)
            mexd += erx[:,:,:-1] * (a - erx[:,:,:-1]*mvxd)
            mexu = mvxu.clone()
            a = torch.sum(erx[:,:,1:] * mvxu, 1, keepdim=True)
            mexu += erx[:,:,:-1] * (a - erx[:,:,:-1]*mvxu)
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
        marginal = d.clone()
        marginal[:,:,:-1] += torch.log(mexd)
        marginal[:,:,1:] += torch.log(mexu)
        
        #save context for backwards pass and return the marginal
        ctx.save_for_backward(ed,erx,mvxu,mvxd,mexu,mexd,avxu,avxd,torch.tensor(max_num_iters))
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
            grad_mvxd = grad_mexd.clone()
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    grad_mvxd[:,j,:] += erx[:,i,:-1]*erx[:,j,:-1]*grad_mexd[:,i,:]
                    grad_rx[:,j,:-1] += erx[:,i,:-1]*mvxd[:,j,:]*grad_mexd[:,i,:]
            grad_mvxu = grad_mexu.clone()
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    grad_mvxu[:,j,:] += erx[:,i,:-1]*erx[:,j,:-1]*grad_mexu[:,i,:]
                    grad_rx[:,j,:-1] += erx[:,i,:-1]*mvxu[:,j,:]*grad_mexu[:,i,:]
            
            #backprop through renormalisation
            grad_mvxd = (grad_mvxd - torch.sum(grad_mvxd*mvxd,1,keepdim=True)) / avxd
            grad_mvxu = (grad_mvxu - torch.sum(grad_mvxu*mvxu,1,keepdim=True)) / avxu
            
            #backprop back to edge messages TODO check correctness
            grad_d[:,:,-1] += grad_mvxd[:,:,-1] #ed contribution to mvxd and mvxu
            grad_d[:,:,0] += grad_mvxu[:,:,0]
            grad_d[:,:,1:-1] += grad_mvxd[:,:,:-1] * mexd[:,:,1:] + grad_mvxu[:,:,1:] * mexu[:,:,:-1]
            grad_mexd[:,:,0] = 0
            grad_mexd[:,:,1:] = grad_mvxd[:,:,:-1]*ed[:,:,1:-1] #mexd contribution to mvxd
            grad_mexu[:,:,-1] = 0
            grad_mexu[:,:,:-1] = grad_mvxu[:,:,1:]*ed[:,:,1:-1] #mexu contribution to mvxu
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #correct to no longer be in the exponential domain
        grad_rx *= erx
        grad_d *= ed
        grad_d += grad_output
        
        return grad_d, grad_rx
    
class Potts_Mean2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry):
        DeepFlowFunction.check_var_dims([d,rx,ry],2)
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.potts_meanpass_2d_gpu_forward(d,rx, ry, output)
        else:
            deepflow.potts_meanpass_2d_cpu_forward(d,rx, ry, output)
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
            deepflow.potts_meanpass_2d_gpu_backward(grad_output, u, rx, ry, grad_d, grad_rx, grad_ry)
        else:
            deepflow.potts_meanpass_2d_cpu_backward(grad_output, u, rx, ry, grad_d, grad_rx, grad_ry)
        return grad_d, grad_rx, grad_ry
    
class Potts_Mean2d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-6
    
    @staticmethod
    def forward(ctx, d, rx):
        DeepFlowFunction.check_var_dims([d,rx],1)
        u = torch.exp(d) / torch.sum(torch.exp(d),1)
        iter = 0
        while True:
            
            energy = -d
            energy[:,:,:-1,:] += rx[:,:,:-1,:]*(2*u[:,:,1:,:]-1)
            energy[:,:,1:,:] += rx[:,:,:-1,:]*(2*u[:,:,:-1,:]-1)
            energy[:,:,:,:-1] += ry[:,:,:,:-1]*(2*u[:,:,:,1:]-1)
            energy[:,:,:,1:] += ry[:,:,:,:-1]*(2*u[:,:,:,:-1]-1)
            u_new = torch.exp(energy) / torch.sum(torch.exp(energy),1)
            change = torch.max(torch.abs(u_new-u))
            u_new = u
            
            iter+=1
            if change < epsilon:
                break
            
        ctx.save_for_backward(energy,rx)
        return energy
            
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
            grad_rx[:,:,:-1,:] += (2*u[:,:,1:,:]-1)* dE[:,:,:-1,:]
            grad_rx[:,:,:-1,:] += (2*u[:,:,:-1,:]-1)* dE[:,:,1:,:]
            grad_ry[:,:,:,:-1] += (2*u[:,:,:,1:]-1)* dE[:,:,:,:-1]
            grad_ry[:,:,:,:-1] += (2*u[:,:,:,:-1]-1)* dE[:,:,:,1:]
            
            new_du = torch.zeros_like(u)
            new_du[:,:,1:,:] += 2*dE[:,:,:-1,:]*rx[:,:,:-1,:]
            new_du[:,:,:-1,:] += 2*dE[:,:,1:,:]*rx[:,:,:-1,:]
            new_du[:,:,:,1:] += 2*dE[:,:,:,:-1]*ry[:,:,:,:-1]
            new_du[:,:,:,:-1] += 2*dE[:,:,:,1:]*ry[:,:,:,:-1]
            
            new_dE = new_du * (u - torch.sum(torch.unsqueeze(u,2),torch.unsqueeze(u,1),dim=1))
            
            change=torch.max(torch.abs(new_dE-dE))
            dE = new_dE
            
            iter+=1
            if change < epsilon:
                break
        
        return grad_d, grad_rx

class Potts_LBP2d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-4
    
    @staticmethod
    def forward(ctx, d, rx, ry):
        DeepFlowFunction.check_var_dims([d,rx,ry],2)
        b,c,sx,sy = d.shape[0],d.shape[1],d.shape[2],d.shape[3]
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
        
        min_er = min(torch.min(torch.prod(torch.sort(erx,dim=1)[0][:,0:2,:,:],dim=1)),
                     torch.min(torch.prod(torch.sort(ery,dim=1)[0][:,0:2,:,:],dim=1)))
        decay_const = 2 * (4-1) * (1-math.sqrt(min_er)) / (1+math.sqrt(min_er))
        if decay_const >= 1-Potts_LBP2d_PytorchNative.epsilon:
            #not guaranteed to converge, should throw error
            max_num_iters = 10*(sx+sy)
        elif decay_const <= Potts_LBP2d_PytorchNative.epsilon:
            max_num_iters = 1
        else:
            max_num_iters = int(math.log(Potts_LBP1d_PytorchNative.epsilon) / math.log(decay_const))
        
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
            mexd = mvxd.clone()
            a = torch.sum(erx[:,:,:-1,:] * mvxd, dim=1)
            for i in range(c):
                mexd[:,i,:,:] += erx[:,i,:-1,:] * (a - erx[:,i,:-1,:]*mvxd[:,i,:,:])
            mexu = mvxu.clone()
            a = torch.sum(erx[:,:,:-1,:] * mvxu, dim=1)
            for i in range(c):
                mexu[:,i,:,:] += erx[:,i,:-1,:] * (a - erx[:,i,:-1,:]*mvxu[:,i,:,:])
            meyd = mvyd.clone()
            a = torch.sum(ery[:,:,:,:-1] * mvyd, dim=1)
            for i in range(c):
                meyd[:,i,:,:] += ery[:,i,:,:-1] * (a - ery[:,i,:,:-1]*mvyd[:,i,:,:])
            meyu = mvyu.clone()
            a = torch.sum(ery[:,:,:,:-1] * mvyu, dim=1)
            for i in range(c):
                meyu[:,i,:,:] += ery[:,i,:,:-1] * (a - ery[:,i,:,:-1]*mvyu[:,i,:,:])
            
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
            grad_mvxd = grad_mexd.clone()
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    grad_mvxd[:,j,:,:] += erx[:,i,:-1,:]*erx[:,j,:-1,:]*grad_mexd[:,i,:,:]
                    grad_rx[:,j,:-1,:] += erx[:,i,:-1,:]*mvxd[:,j,:,:]*grad_mexd[:,i,:,:]
            grad_mvxu = grad_mexu.clone()
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    grad_mvxu[:,j,:,:] += erx[:,i,:-1,:]*erx[:,j,:-1,:]*grad_mexu[:,i,:,:]
                    grad_rx[:,j,:-1,:] += erx[:,i,:-1,:]*mvxu[:,j,:]*grad_mexu[:,i,:,:]
            grad_mvyd = grad_meyd.clone()
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    grad_mvyd[:,j,:,:] += ery[:,i,:,:-1]*ery[:,j,:,:-1]*grad_meyd[:,i,:,:]
                    grad_ry[:,j,:,:-1] += ery[:,i,:,:-1]*mvyd[:,j,:,:]*grad_meyd[:,i,:,:]
            grad_mvyu = grad_meyu.clone()
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    grad_mvyu[:,j,:,:] += ery[:,i,:,:-1]*ery[:,j,:,:-1]*grad_meyu[:,i,:,:]
                    grad_ry[:,j,:,:-1] += ery[:,i,:,:-1]*mvyu[:,j,:,:]*grad_meyu[:,i,:,:]
            
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
            grad_mexd[:,:,1:,:] += to_add[:,:,:-1,:]
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
            
            #backprop back to edge messages in y direction
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
    
class Potts_Mean3d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3)
        output = torch.zeros_like(d)
        if d.is_cuda:
            deepflow.potts_meanpass_3d_gpu_forward(d,rx, ry, rz, output)
        else:
            deepflow.potts_meanpass_3d_cpu_forward(d,rx, ry, rz, output)
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
            deepflow.potts_meanpass_3d_gpu_backward(grad_output, u, rx, ry, rz, grad_d, grad_rx, grad_ry, grad_rz)
        else:
            deepflow.potts_meanpass_3d_cpu_backward(grad_output, u, rx, ry, rz, grad_d, grad_rx, grad_ry, grad_rz)
        return grad_d, grad_rx, grad_ry, grad_rz
        
class Potts_Mean3d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-6
    
    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3)
        u = torch.exp(d) / torch.sum(torch.exp(d),1)
        iter = 0
        while True:
            
            energy = torch.zeros_like(u)
            energy[:,:,:-1,:,:] += rx[:,:,:-1,:,:]*(2*u[:,:,1:,:,:]-1)
            energy[:,:,1:,:,:] += rx[:,:,:-1,:,:]*(2*u[:,:,:-1,:,:]-1)
            energy[:,:,:,:-1,:] += ry[:,:,:,:-1,:]*(2*u[:,:,:,1:,:]-1)
            energy[:,:,:,1:,:] += ry[:,:,:,:-1,:]*(2*u[:,:,:,:-1,:]-1)
            energy[:,:,:,:,:-1] += rz[:,:,:,:,:-1]*(2*u[:,:,:,:,1:]-1)
            energy[:,:,:,:,1:] += rz[:,:,:,:,:-1]*(2*u[:,:,:,:,:-1]-1)
            u_new = torch.exp(energy) / torch.sum(torch.exp(energy),1)
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
            
            new_dE = new_du * (u - torch.sum(torch.unsqueeze(u,2),torch.unsqueeze(u,1),dim=1))
            
            change=torch.max(torch.abs(new_dE-dE))
            dE = new_dE
            
            iter+=1
            if change < epsilon:
                break
        
        return grad_d, grad_rx, grad_ry, grad_rz
    
    
class Potts_LBP3d_PytorchNative(torch.autograd.Function):

    epsilon = 10**-4
    
    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3)
        b,c,sx,sy,sz = d.shape[0],d.shape[1],d.shape[2],d.shape[3],d.shape[4]
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
        
        min_er = min([torch.min(torch.prod(torch.sort(erx,dim=1)[0][:,0:2,:,:,:],dim=1)),
                      torch.min(torch.prod(torch.sort(ery,dim=1)[0][:,0:2,:,:,:],dim=1)),
                      torch.min(torch.prod(torch.sort(erz,dim=1)[0][:,0:2,:,:,:],dim=1))])
        decay_const = 2 * (6-1) * (1-math.sqrt(min_er)) / (1+math.sqrt(min_er))
        if decay_const >= 1-Potts_LBP3d_PytorchNative.epsilon:
            #not guaranteed to converge, should throw error
            max_num_iters = 10*(sx+sy+sz)
        elif decay_const <= Potts_LBP3d_PytorchNative.epsilon:
            max_num_iters = 1
        else:
            max_num_iters = int(math.log(Potts_LBP1d_PytorchNative.epsilon) / math.log(decay_const))
        
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
            mexd = mvxd.clone()
            a = torch.sum(erx[:,:,:-1,:,:] * mvxd, dim=1)
            for i in range(c):
                mexd[:,i,:,:,:] += erx[:,i,:-1,:,:] * (a - erx[:,i,:-1,:,:]*mvxd[:,i,:,:,:])
            mexu = mvxu.clone()
            a = torch.sum(erx[:,:,:-1,:,:] * mvxu, dim=1)
            for i in range(c):
                mexu[:,i,:,:,:] += erx[:,i,:-1,:,:] * (a - erx[:,i,:-1,:,:]*mvxu[:,i,:,:,:])
            meyd = mvyd.clone()
            a = torch.sum(ery[:,:,:,:-1,:] * mvyd, dim=1)
            for i in range(c):
                meyd[:,i,:,:,:] += ery[:,i,:,:-1,:] * (a - ery[:,i,:,:-1,:]*mvyd[:,i,:,:,:])
            meyu = mvyu.clone()
            a = torch.sum(ery[:,:,:,:-1,:] * mvyu, dim=1)
            for i in range(c):
                meyu[:,i,:,:,:] += ery[:,i,:,:-1,:] * (a - ery[:,i,:,:-1,:]*mvyu[:,i,:,:,:])
            mezd = mvzd.clone()
            a = torch.sum(erz[:,:,:,:,:-1] * mvzd, dim=1)
            for i in range(c):
                mezd[:,i,:,:,:] += erz[:,i,:,:,:-1] * (a - erz[:,i,:,:,:-1]*mvzd[:,i,:,:,:])
            mezu = mvzu.clone()
            a = torch.sum(erz[:,:,:,:,:-1] * mvzu, dim=1)
            for i in range(c):
                mezu[:,i,:,:,:] += erz[:,i,:,:,:-1] * (a - erz[:,i,:,:,:-1]*mvzu[:,i,:,:,:])
            
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
            grad_mvxd = grad_mexd.clone()
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    grad_mvxd[:,j,:,:,:] += erx[:,i,:-1,:,:]*erx[:,j,:-1,:,:]*grad_mexd[:,i,:,:,:]
                    grad_rx[:,j,:-1,:,:] += erx[:,i,:-1,:,:]*mvxd[:,j,:,:,:]*grad_mexd[:,i,:,:,:]
            grad_mvxu = grad_mexu.clone()
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    grad_mvxu[:,j,:,:,:] += erx[:,i,:-1,:,:]*erx[:,j,:-1,:,:]*grad_mexu[:,i,:,:,:]
                    grad_rx[:,j,:-1,:,:] += erx[:,i,:-1,:,:]*mvxu[:,j,:,:,:]*grad_mexu[:,i,:,:,:]
            grad_mvyd = grad_meyd.clone()
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    grad_mvyd[:,j,:,:,:] += ery[:,i,:,:-1,:]*ery[:,j,:,:-1,:]*grad_meyd[:,i,:,:,:]
                    grad_ry[:,j,:,:-1,:] += ery[:,i,:,:-1,:]*mvyd[:,j,:,:,:]*grad_meyd[:,i,:,:,:]
            grad_mvyu = grad_meyu.clone()
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    grad_mvyu[:,j,:,:,:] += ery[:,i,:,:-1,:]*ery[:,j,:,:-1,:]*grad_meyu[:,i,:,:,:]
                    grad_ry[:,j,:,:-1,:] += ery[:,i,:,:-1,:]*mvyu[:,j,:,:,:]*grad_meyu[:,i,:,:,:]
            grad_mvzd = grad_mezd.clone()
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    grad_mvzd[:,j,:,:,:] += erz[:,i,:,:,:-1]*erz[:,j,:,:,:-1]*grad_mezd[:,i,:,:,:]
                    grad_rz[:,j,:,:,:-1] += erz[:,i,:,:,:-1]*mvzd[:,j,:,:,:]*grad_mezd[:,i,:,:,:]
            grad_mvzu = grad_mezu.clone()
            for i in range(c):
                for j in range(c):
                    if i == j:
                        continue
                    grad_mvzu[:,j,:,:,:] += erz[:,i,:,:,:-1]*erz[:,j,:,:,:-1]*grad_mezu[:,i,:,:,:]
                    grad_rz[:,j,:,:,:-1] += erz[:,i,:,:,:-1]*mvzu[:,j,:,:,:]*grad_mezu[:,i,:,:,:]
            
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