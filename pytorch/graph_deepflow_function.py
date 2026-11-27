import torch
import math

def categorical_input_check(d,r):
    b,c,N = d.shape[0],d.shape[1],d.shape[2]
    if len(d.shape) != 3:
        raise Exception("Data term tensor has wrong number of dimensions")
    if len(r.shape) != 5:
        raise Exception("Regularisation tensor has wrong number of dimensions")
    if r.shape[0] != b:
        raise Exception("Regularisation tensor shape doesn't reflect number of batches")
    if r.shape[1] != c or r.shape[2] != c:
        raise Exception("Regularisation tensor shape doesn't reflect number of labels")
    if r.shape[3] != N or r.shape[4] != N:
        raise Exception("Regularisation tensor shape doesn't reflect number of nodes")
    return b,c,N

def binary_input_check(d,r):
    b,c,N = d.shape[0],d.shape[1],d.shape[2]
    if len(d.shape) != 3:
        raise Exception("Data term tensor has wrong number of dimensions")
    if len(r.shape) != 4:
        raise Exception("Regularisation tensor has wrong number of dimensions")
    if r.shape[0] != b:
        raise Exception("Regularisation tensor shape doesn't reflect number of batches")
    if r.shape[1] != c:
        raise Exception("Regularisation tensor shape doesn't reflect number of labels")
    if r.shape[2] != N or r.shape[3] != N:
        raise Exception("Regularisation tensor shape doesn't reflect number of nodes")
    return b,c,N

class DenseGraph_LBP_Categorical_PytorchNative(torch.autograd.Function):

    epsilon = 10**-8
    epsilon_low = 10**-15
    
    @staticmethod
    def forward(ctx, d, r):
        b,c,N = categorical_input_check(d,r)
            
        ed = torch.exp(d)
        er = torch.exp(-r)
        me = torch.ones_like(er[:,:,0,:,:])
                
        #get the L1 decay constant to determine number of iterations
        n_neigh = torch.count_nonzero(r[:,:,:,0,1], dim=1)
        pm = torch.min(torch.prod(torch.sort(er,dim=1)[0][:,0:2,:],dim=1))
        if abs(pm-1) < DenseGraph_LBP_Categorical_PytorchNative.epsilon:
            decay_const = -0.25*(pm-1) #linear approximation near 0/0 point at pm=1
        else:
            decay_const = torch.max(2*(n_neigh-1)*(1-torch.sqrt(pm))**2/(1-pm)) #full equation otherwise
            
        #use the decay constant to find the number of iterations to perform
        if decay_const >= 1-DenseGraph_LBP_Categorical_PytorchNative.epsilon:
            #not guaranteed to converge, should throw error
            max_num_iters = 10*N
        elif decay_const <= DenseGraph_LBP_Categorical_PytorchNative.epsilon:
            max_num_iters = 1
        else:
            max_num_iters = int(math.log(DenseGraph_LBP_Categorical_PytorchNative.epsilon) / math.log(decay_const))+1
        
        iter = 0
        while True:
            
            #compute vertex to edge messages and renormalize
            mv = torch.unsqueeze(ed,dim=-1) * (torch.unsqueeze(torch.prod(me,dim=2),dim=-1) / torch.permute(me,(0,1,3,2)))+DenseGraph_LBP_Categorical_PytorchNative.epsilon_low
            mv /= torch.sum(mv,1,keepdim=True)
            
            #compute edge to vertex messages
            me = torch.sum(er * torch.unsqueeze(mv,dim=2), dim=1)
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #save context for backwards pass and return the marginal
        ctx.save_for_backward(ed,er,me,torch.tensor(max_num_iters))
        return me
    
    @staticmethod 
    def backward(ctx, grad_me):
        ed, er, me, max_num_iters = ctx.saved_tensors
        b,c,N = ed.shape[0],ed.shape[1],ed.shape[2]
        max_num_iters = int(max_num_iters.cpu().item())

        #compute the missing parts of the forward pass
        prod_me = torch.prod(me,dim=2)
        mv = torch.unsqueeze(ed,dim=-1) * (torch.unsqueeze(torch.prod(me,dim=2),dim=-1) / torch.permute(me,(0,1,3,2)))+ DenseGraph_LBP_Binary_PytorchNative.epsilon_low
        av = torch.sum(mv,1,keepdim=True)
        mv /= av

        grad_d =  torch.zeros_like(ed) #(will multiply out the ed at the end, keeps everything in exponential domain)
        grad_r = torch.zeros_like(er)
        
        iter = 0
        while True:
            
            #backprop back to vertex messages
            #me = torch.sum(er * torch.unsqueeze(mv,dim=2), dim=1)
            grad_mv = torch.sum(torch.unsqueeze(grad_me,dim=1)*er,2)
            grad_r += torch.unsqueeze(grad_me,1)*torch.unsqueeze(mv,dim=2)
                        
            #backprop through renormalisation
            grad_mv = (grad_mv - torch.sum(grad_mv*mv,1,keepdim=True)) / av
            
            #backprop back to edge messages
            #mv = torch.unsqueeze(ed,dim=-1) * (torch.unsqueeze(torch.prod(me,dim=2),dim=-1) / torch.permute(me,(0,1,3,2)))
            grad_d += torch.sum(grad_mv * torch.unsqueeze(prod_me,dim=-1) / torch.permute(me,(0,1,3,2)),dim=-1)
            grad_me = torch.unsqueeze(ed * prod_me,dim=2) * (torch.permute(torch.sum( torch.unsqueeze(grad_mv,dim=-1) / torch.permute((torch.unsqueeze(me,dim=3)*torch.unsqueeze(me,dim=2)),(0,1,4,2,3)),dim=3),(0,1,3,2)) - torch.permute( grad_mv / torch.permute(me**2,(0,1,3,2)),(0,1,3,2)))
                        
            iter+=1
            if iter >= max_num_iters:
                break
        
        #correct to no longer be in the exponential domain
        grad_r *= er
        grad_d *= ed
        
        return grad_d, grad_r
    
    
    
class DenseGraph_LBP_Binary_PytorchNative(torch.autograd.Function):

    epsilon = 10**-8
    epsilon_low = 10**-15
    
    @staticmethod
    def forward(ctx, d, r):
        b,c,N = binary_input_check(d,r)
            
        ed = torch.exp(d)
        er = torch.exp(-r)
        me1 = torch.ones_like(er)
        me0 = torch.ones_like(er)
        
        #get the L_infy decay constant to determine number of iterations
        decay_const = torch.max(torch.unsqueeze(torch.sum((1-er)/ (1+er),dim=2),dim=3) - torch.permute((1-er)/ (1+er),(0,1,3,2)))
            
        #use the decay constant to find the number of iterations to perform
        if decay_const >= 1-DenseGraph_LBP_Binary_PytorchNative.epsilon:
            #not guaranteed to converge, should throw error
            max_num_iters = 2*N
        elif decay_const <= DenseGraph_LBP_Binary_PytorchNative.epsilon:
            max_num_iters = 1
        else:
            max_num_iters = int(math.log(DenseGraph_LBP_Binary_PytorchNative.epsilon) / math.log(decay_const))+1
        
        iter = 0
        while True:
            
            #compute vertex to edge messages and renormalize
            mv0 =                              (torch.unsqueeze(torch.prod(me0,dim=2),dim=-1) / torch.permute(me0,(0,1,3,2))) + DenseGraph_LBP_Binary_PytorchNative.epsilon_low
            mv1 = torch.unsqueeze(ed,dim=-1) * (torch.unsqueeze(torch.prod(me1,dim=2),dim=-1) / torch.permute(me1,(0,1,3,2))) + DenseGraph_LBP_Binary_PytorchNative.epsilon_low
            av = (mv0+mv1)
            mv0, mv1 = mv0/av, mv1/av
            
            #compute edge to vertex messages
            me0 = er * mv1 + mv0
            me1 = er * mv0 + mv1
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #save context for backwards pass and return the marginal
        ctx.save_for_backward(ed,er,me0,me1,torch.tensor(max_num_iters))
        return me0, me1
    
    @staticmethod 
    def backward(ctx, grad_me0, grad_me1):
        ed, er, me0, me1, max_num_iters = ctx.saved_tensors
        b,c,N = ed.shape[0],ed.shape[1],ed.shape[2]
        max_num_iters = int(max_num_iters.cpu().item())

        #compute the missing parts of the forward pass
        prod_me0 = torch.prod(me0,dim=2)
        prod_me1 = torch.prod(me1,dim=2)
        grad_prod_me0 = torch.sum(grad_me0/torch.permute(me0, (0, 1, 3, 2)),dim=2)
        prod_me1 = torch.prod(me1,dim=2)
        mv0 =                              (torch.unsqueeze(prod_me0,dim=-1) / torch.permute(me0,(0,1,3,2))) + DenseGraph_LBP_Binary_PytorchNative.epsilon_low
        mv1 = torch.unsqueeze(ed,dim=-1) * (torch.unsqueeze(prod_me1,dim=-1) / torch.permute(me1,(0,1,3,2))) + DenseGraph_LBP_Binary_PytorchNative.epsilon_low
        av = mv0+mv1
        mv0, mv1 = mv0/av, mv1/av

        grad_d =  torch.zeros_like(ed) #(will multiply out the ed at the end, keeps everything in exponential domain)
        grad_r = torch.zeros_like(er)

        iter = 0
        while True:
            
            #backprop back to vertex messages
            grad_mv0 = grad_me0 + er * grad_me1
            grad_mv1 = grad_me1 + er * grad_me0
            grad_r += grad_me1*mv0 + grad_me0*mv1
                        
            #backprop through renormalisation
            grad_mv0, grad_mv1 = (grad_mv0 - grad_mv1) * mv1 / av, (grad_mv1 - grad_mv0) * mv0 / av
            
            #backprop back to edge messages
            grad_d += torch.sum(grad_mv1 * torch.unsqueeze(prod_me1,dim=-1) / torch.permute(me1,(0,1,3,2)),dim=-1)
            grad_me0 = (prod_me0 / (me0 ** 2)) * torch.sum(grad_mv0 * mv0, dim=-1,keepdim=True) - grad_mv0 * mv0 / me0
            grad_me1 = ed.unsqueeze(-1) * ((prod_me1 / (me1 ** 2)) * torch.sum(grad_mv1 * mv1, dim=-1, keepdim=True) - (grad_mv1 * mv1 / me1))
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #correct to no longer be in the exponential domain
        grad_r *= er
        grad_d *= ed
        return grad_d, grad_r
    
    
class DenseGraph_MFE_Categorical_PytorchNative(torch.autograd.Function):

    epsilon = 10**-8
    
    @staticmethod
    def forward(ctx, d, r):
        b,c,N = categorical_input_check(d,r)
            
        #initial marginal estimate
        marginal = torch.softmax(d,dim=1)
            
        #get the L_infinity decay constant to determine number of iterations
        edge_weight = torch.max(torch.sum(torch.abs(torch.unsqueeze(r,dim=1)-torch.unsqueeze(r,dim=2)),dim=3),dim=2)[0]
        node_weight = 0.5 * torch.sum(edge_weight,dim=-1)
        decay_const = torch.max(node_weight)
        
        #use the decay constant to find the number of iterations to perform
        if decay_const >= 1-DenseGraph_MFE_Categorical_PytorchNative.epsilon:
            #not guaranteed to converge, should throw error
            max_num_iters = 10*N
        elif decay_const <= DenseGraph_MFE_Categorical_PytorchNative.epsilon:
            max_num_iters = 1
        else:
            max_num_iters = int(math.log(DenseGraph_MFE_Categorical_PytorchNative.epsilon) / math.log(decay_const))
        
        
        iter = 0
        while True:
            
            #estimate the local energy from current estimates
            energy = d + torch.sum(torch.sum(torch.permute(r,(0,1,2,4,3)) * torch.unsqueeze(torch.unsqueeze(marginal-0.5,dim=2),dim=-1),dim=1),dim=-2)
            
            #compute the new marginal estimate
            marginal = torch.softmax(energy,dim=1)
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #save context for backwards pass and return the updated energy estimate (softmax outside to get marginal)
        ctx.save_for_backward(d,r,marginal,torch.tensor(max_num_iters))
        energy = d + torch.sum(torch.sum(torch.permute(r,(0,1,2,4,3)) * torch.unsqueeze(torch.unsqueeze(marginal-0.5,dim=2),dim=-1),dim=1),dim=-2)
        return energy
    
    @staticmethod 
    def backward(ctx, grad_output):
        d,r,marginal, max_num_iters = ctx.saved_tensors
        b,c,N = d.shape[0],d.shape[1],d.shape[2]
        max_num_iters = int(max_num_iters.cpu().item())

        #backprop through final layer
        #output = d + torch.sum(torch.sum(torch.permute(r,(0,1,2,4,3)) * torch.unsqueeze(torch.unsqueeze(marginal-0.5,dim=2),dim=-1),dim=1),dim=-2)
        grad_d =  grad_output.clone()
        grad_r = torch.permute(torch.unsqueeze(torch.unsqueeze(grad_output,dim=-2),dim=1) * torch.unsqueeze(torch.unsqueeze(marginal-0.5,dim=2),dim=-1),(0,1,2,4,3))
        grad_m = torch.sum(torch.sum(torch.unsqueeze(torch.unsqueeze(grad_output,dim=-2),dim=1) * torch.permute(r,(0,1,2,4,3)),dim=2),dim=-1)
        
        iter = 0
        while True:
            
            #backprop back through softmax to energy
            #marginal = torch.softmax(energy,dim=1)
            grad_e = marginal * (grad_m - torch.sum(grad_m * marginal, dim=1, keepdim=True))
                        
            #backprop back to edge messages
            #energy = d + torch.sum(torch.sum(torch.permute(r,(0,1,2,4,3)) * torch.unsqueeze(torch.unsqueeze(marginal-0.5,dim=2),dim=-1),dim=1),dim=-2)
            grad_d +=  grad_e
            grad_r += torch.permute(torch.unsqueeze(torch.unsqueeze(grad_e,dim=-2),dim=1) * torch.unsqueeze(torch.unsqueeze(marginal-0.5,dim=2),dim=-1),(0,1,2,4,3))
            grad_m = torch.sum(torch.sum(torch.unsqueeze(torch.unsqueeze(grad_e,dim=-2),dim=1) * torch.permute(r,(0,1,2,4,3)),dim=2),dim=-1)
            
            iter+=1
            if iter >= max_num_iters:
                break
                
        return grad_d, grad_r
    
    
class DenseGraph_MFE_Binary_PytorchNative(torch.autograd.Function):

    epsilon = 10**-8
    
    @staticmethod
    def forward(ctx, d, r):
        b,c,N = binary_input_check(d,r)
            
        #initial marginal estimate
        marginal = torch.sigmoid(d)
        
        #get the L_infinity decay constant to determine number of iterations
        decay_const = torch.max(torch.sum(r,dim=-1))
        
        #use the decay constant to find the number of iterations to perform
        if decay_const >= 1-DenseGraph_MFE_Binary_PytorchNative.epsilon:
            #not guaranteed to converge, should throw error
            max_num_iters = 10*N
        elif decay_const <= DenseGraph_MFE_Binary_PytorchNative.epsilon:
            max_num_iters = 1
        else:
            max_num_iters = int(math.log(DenseGraph_MFE_Binary_PytorchNative.epsilon) / math.log(decay_const))
        
        iter = 0
        while True:
            
            #estimate the local energy from current estimates
            energy = d + torch.sum(r * torch.unsqueeze(marginal-0.5,dim=-1),dim=-2)
            
            #compute the new marginal estimate
            marginal = torch.sigmoid(energy)
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #save context for backwards pass and return the updated energy estimate (softmax outside to get marginal)
        ctx.save_for_backward(d,r,marginal,torch.tensor(max_num_iters))
        energy = d + torch.sum(torch.permute(r,(0,1,3,2)) * torch.unsqueeze(marginal-0.5,dim=-1),dim=-2)
        return energy
    
    @staticmethod 
    def backward(ctx, grad_output):
        d,r,marginal, max_num_iters = ctx.saved_tensors
        b,c,N = d.shape[0],d.shape[1],d.shape[2]
        max_num_iters = int(max_num_iters.cpu().item())

        #backprop through final layer
        #output = d + torch.sum(torch.permute(r,(0,1,3,2)) * torch.unsqueeze(marginal-0.5,dim=-1),dim=-2)
        grad_d =  grad_output.clone()
        grad_r = torch.permute(torch.unsqueeze(grad_output,dim=-2) * torch.unsqueeze(marginal-0.5,dim=-1),(0,1,3,2))
        grad_m = torch.sum(torch.unsqueeze(grad_output,dim=-2) * torch.permute(r,(0,1,3,2)),dim=-1)
        
        iter = 0
        while True:
            
            #backprop back through softmax to energy
            #marginal = torch.simoid(energy,dim=1)
            grad_e = grad_m * marginal*(1 - marginal)
                        
            #backprop back to edge messages
            #energy = d + torch.sum(torch.permute(r,(0,1,3,2)) * torch.unsqueeze(marginal-0.5,dim=-1),dim=-2)
            grad_d +=  grad_e
            grad_r += torch.permute(torch.unsqueeze(grad_e,dim=-2) * torch.unsqueeze(marginal-0.5,dim=-1),(0,1,3,2))
            grad_m = torch.sum(torch.unsqueeze(grad_e,dim=-2) * torch.permute(r,(0,1,3,2)),dim=-1)
            
            iter+=1
            if iter >= max_num_iters:
                break
                
        return grad_d, grad_r
    
    
    
class DenseGraph_MAP_Binary_PytorchNative(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, d, r):
        b,c,N = binary_input_check(d,r)
        
        epsilon = 10**-4
        cc = 0.25
        tau = 0.1/(N-1)

        #flow and constraint variables
        p_s = d * (d > 0)
        p_t = -d * (d < 0)
        p = torch.zeros_like(r)
        div = torch.zeros_like(d)
        u = 1.0*(d>0)
        
        #check max number of iterations allowed (which is probably way too high)
        iter = 10*N^2
        
        #iterate
        for i in range(iter):
            
            #update spatial flows
            capacity = div + p_t - p_s - u/cc
            p += tau*(torch.unsqueeze(capacity,dim=-2) - torch.unsqueeze(capacity,dim=-1))
            p = (torch.abs(p) <= r)*p + (torch.abs(p) > r)*torch.sign(p)*r
            div = torch.sum(p,dim=-1)
            
            #update sink/source flows
            capacity = div + p_t - u/cc
            p_s = 1/cc + capacity
            p_s = (d>0)*(d*(p_s > d) + p_s*(p_s < d))
            capacity = p_s - div + u/cc
            p_t = capacity
            p_t = (d<0)*(-d*(p_t > -d) + p_t*(p_t < -d))
            
            #update constraint multipliers and constrain to [0,1] interval
            erru = (p_s - p_t - div)
            u += cc*erru
            u = torch.clip(u,min=0,max=1)
            
            #early termination
            tot_change = torch.sum(torch.abs(erru))
            if tot_change < N*epsilon:
                break
            
        #return constraint multipliers which should reflect MAP solution
        return u
    
    #method is not differentiable
    @staticmethod 
    def backward(ctx, grad_output):
        return None
    
        
class DenseGraph_MAP_Categorical_PytorchNative(torch.autograd.Function):

    
    @staticmethod
    def forward(ctx, d, r):
        b,c,N = categorical_input_check(d,r)
        
        #invert data terms to save on lots of bit flips later
        d = -d
        
        epsilon = 10**-4
        cc = 0.25
        tau = 0.1/(N-1)

        #flow and constraint variables
        p_s = torch.max(d,dim=1,keepdim=True)[0]
        p_t = torch.zeros_like(d)
        p = torch.zeros_like(r)
        div = torch.zeros_like(d)
        u = 1.0*(d==torch.min(d,dim=1,keepdim=True)[0])
        u /= torch.sum(u,dim=1,keepdim=True)
        
        #check max number of iterations allowed (which is probably way too high)
        max_iter = 10*N**2
        
        #iterate
        for i in range(max_iter):
            
            #update spatial flows
            capacity = div + p_t - p_s - u/cc
            p += tau*(torch.unsqueeze(torch.unsqueeze(capacity,dim=1),dim=-2) - torch.unsqueeze(torch.unsqueeze(capacity,dim=2),dim=-1))
            p = (torch.abs(p) <= r)*p + (torch.abs(p) > r)*torch.sign(p)*r
            div = torch.sum(p,dim=(2,-1))
            
            #update sink/source flows
            capacity = div + p_t - u/cc
            p_s = torch.unsqueeze((1/cc + torch.sum(capacity,dim=1))/c,dim=1)
            capacity = p_s - div + u/cc
            p_t = capacity
            p_t = torch.minimum(p_t, d)
            
            #update constraint multipliers and constraint to simplex
            erru = p_s - p_t - div
            u += cc*erru
            u = torch.clip(u,min=0)
            u /= torch.sum(u,dim=1,keepdim=True)
            
            #early termination
            tot_change = torch.sum(torch.abs(erru))
            if tot_change < N*epsilon:
                break
                
            
        #return constraint multipliers which should reflect MAP solution
        return u
    
    #method is not differentiable
    @staticmethod 
    def backward(ctx, grad_output):
        return None