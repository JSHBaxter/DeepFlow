import torch

class DenseGraph_LBP_PytorchNative(torch.autograd.Function):

    epsilon = 10**-4
    
    @staticmethod
    def forward(ctx, d, r):
        b,c,N = d.shape[0],d.shape[1],d.shape[2]
        if len(d.shape) != 3:
            raise Exception("Data term tensor has wrong number of dimensions")
        if len(r.shape) != 5:
            raise Exception("Regularisation tensor has wrong number of dimensions")
        if r.shape[0] != b:
            raise Exception("Regularisation tensor shape doesn't reflect number of batches")
        if r.shape[1] != c or r.shape[2] != c:
            raise Exception("Regularisation tensor shape doesn't reflect number of nodes")
        if r.shape[3] != N or r.shape[4] != N:
            raise Exception("Regularisation tensor shape doesn't reflect number of labels")
            
        ed = torch.exp(d)
        er = torch.exp(-r)
        mv = torch.ones_like(er[:,:,0,:,:])
        me = torch.ones_like(er[:,:,0,:,:])
        
        #get the L1 decay constant to determine number of iterations //TODO
        n_neigh = torch.count_nonzero(r[:,:,:,0,1], dim=1)
        torch.min(torch.prod(torch.sort(er,dim=1)[0][:,0:2,:],dim=1))
            
            
        #use the decay constant to find the number of iterations to perform
        if decay_const >= 1-DenseGraph_LBP_PytorchNative.epsilon:
            #not guaranteed to converge, should throw error
            max_num_iters = 10*sx
        elif decay_const <= DenseGraph_LBP_PytorchNative.epsilon:
            max_num_iters = 1
        else:
            max_num_iters = int(math.log(DenseGraph_LBP_PytorchNative.epsilon) / math.log(decay_const))
        
        iter = 0
        while True:
            
            #compute vertex to edge messages
            mv = ed * (torch.product(me,dim=2) / torch.permute(me,(0,1,3,2)))
            #for i in range(N): for j in range(N): mv[:,:,i,j] = ed[:,:,i] * torch.product(me[:,:,:,i],dim=-1) / me[:,:,j,i]
            
            
            #renormalise messages
            av = torch.sum(mvxu,1,keepdim=True)
            mv /= av
            
            #compute edge to vertex messages
            me = torch.sum(er * torch.unsqueeze(torch.permute(mv,(0,1,3,2)),dim=2), dim=2)
                
            iter+=1
            if iter >= max_num_iters:
                break
        
        #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
        marginal = d + torch.sum(torch.log(me,dim=2))
        
        #save context for backwards pass and return the marginal
        ctx.save_for_backward(ed,er,mv,me,av,torch.tensor(max_num_iters))
        return marginal
    
    @staticmethod 
    def backward(ctx, grad_output):
        ed,erx,mvxu,mvxd,mexu,mexd,avxu,avxd,max_num_iters = ctx.saved_tensors
        b,c,sx = ed.shape[0],ed.shape[1],ed.shape[2]
        max_num_iters = int(max_num_iters.cpu().item())
        grad_d =  torch.zeros_like(ed) #(will multiply out the ed at the end, keeps everything in exponential domain)
        grad_rx = torch.zeros_like(erx)
        prod_me = torch.product(me,dim=2)
        
        #backprop through final layer
        #marginal = d.clone() + torch.sum(torch.log(me,dim=2))
        grad_me = torch.unsqueeze(grad_output,2) / me
        
        iter = 0
        while True:
            
            #backprop back to vertex messages
            #me = torch.sum(er * torch.unsqueeze(torch.permute(mv,(0,1,3,2)),dim=2), dim=2)
            grad_mv = torch.permute(torch.sum(torch.unsqueeze(grad_me,2)*er,2),(0,1,3,2))
            grad_rx += torch.unsqueeze(grad_me,2)*torch.unsqueeze(torch.permute(mv,(0,1,3,2)),dim=2)
                        
            #backprop through renormalisation
            grad_mv = (grad_mv - torch.sum(grad_mv*mv,1,keepdim=True)) / av
            
            #backprop back to edge messages //TODO
            #mv = ed * torch.product(me,dim=2) / torch.permute(me,(0,1,3,2))
            grad_ed = torch.sum(grad_mv * (torch.product(me,dim=2) / torch.permute(me,(0,1,3,2))), -1)
            grad_me = -torch.permute(grad_mv * ed * torch.product(me,dim=2),(0,1,3,2)) / me**2
            for j in range(N):
                grad_me += torch.unsqueeze((grad_mv[:,:,:,j] * ed * prod_me),2) / (me*me[:,:,j,:])
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #correct to no longer be in the exponential domain
        grad_rx *= erx
        grad_d = grad_d*ed + grad_output
        
        return grad_d, grad_rx