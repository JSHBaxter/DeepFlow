import torch
from itertools import product


#Repurpose and improve on the efficiency of the 1D LTL case manager
# Guaranteed to converge to correct marginals in a linear number of steps regardless of constraints or regularisation
# Constraints are only applied locally, so before/after are simplified to notprecede and notsuceed. For example, if A before B, then BCA is valid here.
class LBP_LTL_LOCAL_PytorchNative:

    epsilon = 10**-4
    
    @staticmethod
    def constraint_application(constraint_list,hierarchy,d,r_eff=None):
        r_eff_multiplier = torch.ones((1,len(hierarchy.leaves),len(hierarchy.leaves),1),dtype=d.dtype,device=d.device)
        
        for constraint_A, constraint_B, constraint_type in constraint_list:
            
            #if A comes before B, then it cannot be that a child of B occurs immediately before a child of A
            if constraint_type == "BEFORE":
                for child_A, child_B in product(constraint_A.leaves,constraint_B.leaves):
                    r_eff_multiplier[:,child_B.index,child_A.index,:] = 0
            
            #if A comes after B, then it cannot be that a child of A occurs immediately before a child of B
            elif constraint_type == "AFTER":
                for child_A, child_B in product(constraint_A.leaves,constraint_B.leaves):
                    r_eff_multiplier[:,child_A.index,child_B.index,:] = 0
            
            #if A must preced B, then no label other than A or B can come before any label of B
            elif constraint_type == "PRECEDE":
                for child_C, child_B in product(hierarchy.leaves - constraint_A.leaves - constraint_B.leaves, constraint_B.leaves):
                    r_eff_multiplier[:,child_C.index,child_B.index,:] = 0
            
            #if A must not precede B, then no label in A can come before any label of B
            elif constraint_type == "NOTPRECEDE":
                for child_A, child_B in product(constraint_A.leaves, constraint_B.leaves):
                    r_eff_multiplier[:,child_A.index,child_B.index,:] = 0
            
            #if A must suceed B, then no label other than A or B can come after any label of B
            elif constraint_type == "SUCEED":
                for child_C, child_B in product(hierarchy.leaves - constraint_A.leaves - constraint_B.leaves, constraint_B.leaves):
                    r_eff_multiplier[:,child_B.index,child_C.index,:] = 0
            
            #if A must suceed B, then no label in A can come after any label of B
            elif constraint_type == "NOTSUCEED":
                for child_A, child_B in product(constraint_A.leaves, constraint_B.leaves):
                    r_eff_multiplier[:,child_B.index,child_A.index,:] = 0
            
            #if A is final, everything after it must remain in A
            elif constraint_type == "FINAL":
                for child_A, child_notA in product(constraint_A.leaves, hierarchy.leaves - constraint_A.leaves):
                    r_eff_multiplier[:,child_A.index,child_notA.index,:] = 0
            
            #if A is final, everything before it must remain in A
            elif constraint_type == "INITIAL":
                for child_A, child_notA in product(constraint_A.leaves, hierarchy.leaves - constraint_A.leaves):
                    r_eff_multiplier[:,child_notA.index,child_A.index,:] = 0
            
            else:
                print("Constraint",constraint_A,constraint_B,constraint_type,"is not recognised")
                #raise Exception("Constraint not recognised.")
                
        if r_eff is not None:
            return d, r_eff*r_eff_multiplier.detach()
        else:
            return d, r_eff_multiplier.detach()
        
    @staticmethod
    def cleanup(hierarchy):
        return
    
    class MARG(torch.autograd.Function):
        @staticmethod
        def forward(ctx, d, erx):
            b,c,sx = d.shape[0],d.shape[1],d.shape[2]
            ed = torch.exp(d)
            mvxu = torch.ones((b,c,sx-1),device=d.device)
            mvxd = torch.ones((b,c,sx-1),device=d.device)
            mexu = torch.ones((b,c,sx-1),device=d.device)
            mexd = torch.ones((b,c,sx-1),device=d.device)

            max_num_iters = sx
            epsilon = 10**-6

            iter = 0
            while True:

                #compute vertex to edge messages
                #messages in x
                mvxd = ed[:,:,1:].clone()
                mvxd[:,:,:-1] *= mexd[:,:,1:]
                mvxu = ed[:,:,:-1].clone()
                mvxu[:,:,1:] *= mexu[:,:,:-1]

                #renormalise messages
                avxu = torch.sum(mvxu+epsilon,1,keepdim=True)
                mvxu = (mvxu+epsilon) / avxu
                avxd = torch.sum(mvxd+epsilon,1,keepdim=True)
                mvxd = (mvxd+epsilon) / avxd

                #compute edge to vertex messages
                mexd = torch.sum(erx * torch.unsqueeze(mvxd,1), dim=2)
                mexu = torch.sum(erx * torch.unsqueeze(mvxu,2), dim=1)

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

            #exit immediately if no gradient
            if torch.amax(torch.abs(grad_output)) == 0:
                return grad_d, grad_rx

            #backprop through final layer
            grad_mexu = grad_output[:,:,1:] / mexu
            grad_mexd = grad_output[:,:,:-1] / mexd

            iter = 0
            while True:

                #backprop back to vertex messages
                grad_mvxd = torch.sum(erx*torch.unsqueeze(grad_mexd,2),1)
                grad_mvxu = torch.sum(erx*torch.unsqueeze(grad_mexu,1),2)
                if grad_rx.shape[-1] == 1:
                    if grad_rx.shape[0] == 1:
                        update = torch.unsqueeze(mvxu,2)*torch.unsqueeze(grad_mexu,1) + torch.unsqueeze(mvxd,1)*torch.unsqueeze(grad_mexd,2)
                        grad_rx += torch.sum(torch.sum(update,-1,keepdim=True),0,keepdim=True)
                    else:
                        grad_rx += torch.sum(torch.unsqueeze(mvxu,2)*torch.unsqueeze(grad_mexu,1) + torch.unsqueeze(mvxd,1)*torch.unsqueeze(grad_mexd,2),-1,keepdim=True)
                else:
                    grad_rx += torch.unsqueeze(mvxu,2)*torch.unsqueeze(grad_mexu,1) + torch.unsqueeze(mvxd,1)*torch.unsqueeze(grad_mexd,2)

                #backprop through renormalisation
                grad_mvxd = (grad_mvxd - torch.sum(grad_mvxd*mvxd,1,keepdim=True)) / avxd
                grad_mvxu = (grad_mvxu - torch.sum(grad_mvxu*mvxu,1,keepdim=True)) / avxu

                grad_mexd *= 0
                grad_mexu *= 0

                #backprop back to edge messages in x direction
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

            #correct to no longer be in the exponential domain for the data term
            grad_d *= ed
            grad_d += grad_output

            return grad_d, grad_rx

    class MAP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, d, erx):
            b,c,sx = d.shape[0],d.shape[1],d.shape[2]
            mvxu = torch.zeros((b,c,sx-1),device=d.device)
            mvxd = torch.zeros((b,c,sx-1),device=d.device)
            mexu = torch.zeros((b,c,sx-1),device=d.device)
            mexd = torch.zeros((b,c,sx-1),device=d.device)
            rx = torch.log(erx)

            max_num_iters = sx-1

            iter = 0
            while True:

                #compute vertex to edge messages
                #messages in x
                mvxd = d[:,:,1:].clone()
                mvxd[:,:,:-1] += mexd[:,:,1:]
                mvxu = d[:,:,:-1].clone()
                mvxu[:,:,1:] += mexu[:,:,:-1]

                #compute edge to vertex messages
                mexd = torch.amax(rx + torch.unsqueeze(mvxd,1), dim=2)
                mexu = torch.amax(rx + torch.unsqueeze(mvxu,2), dim=1)
                
                #normalise to prevent this from getting too big or too small
                mexu -= torch.amax(mexu,1,keepdim=True)
                mexd -= torch.amax(mexd,1,keepdim=True)

                iter+=1
                if iter >= max_num_iters:
                    break

            #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
            marginal = d.clone()
            marginal[:,:,:-1] += mexd
            marginal[:,:,1:] += mexu

            ctx.save_for_backward(d,erx,mvxu,mvxd,mexu,mexd,torch.tensor(max_num_iters))
            
            return marginal
        
        @staticmethod 
        def backward(ctx, grad_output):

            d,erx,mvxu,mvxd,mexu,mexd,max_num_iters = ctx.saved_tensors
            b,c,sx = d.shape[0],d.shape[1],d.shape[2]
            max_num_iters = int(max_num_iters.cpu().item())
            rx = torch.log(erx)
            grad_rx = torch.zeros_like(erx)

            #exit immediately if no gradient
            if torch.amax(torch.abs(grad_output)) == 0:
                return grad_d, grad_rx

            #backprop through final layer
            grad_mexu = grad_output[:,:,1:]
            grad_mexd = grad_output[:,:,:-1]
            grad_d =  grad_output.clone()

            #find max mask
            mask_mvxd = ((rx + torch.unsqueeze(mvxd,1)) == torch.amax(rx + torch.unsqueeze(mvxd,1), dim=2, keepdims=True)).float()
            num_mask_mvxd = torch.sum(mask_mvxd,dim=2,keepdim=True)
            mask_mvxd /= num_mask_mvxd
            mask_mvxu = ((rx + torch.unsqueeze(mvxu,2)) == torch.amax(rx + torch.unsqueeze(mvxu,2), dim=1, keepdims=True)).float()
            num_mask_mvxu = torch.sum(mask_mvxu,dim=1,keepdim=True)
            mask_mvxu /= num_mask_mvxu
            
            mask_aexu = (mexu == 0).float()
            num_mask_aexu = torch.sum(mask_aexu,dim=1,keepdim=True)
            mask_aexu /= num_mask_aexu
            mask_aexd = (mexd == 0).float()
            num_mask_aexd = torch.sum(mask_aexd,dim=1,keepdim=True)
            mask_aexd /= num_mask_aexd
            
            iter = 0
            while True:

                #backprob through normalisation
                grad_mexu = grad_mexu - mask_aexu * torch.sum(grad_mexu,1,keepdims=True)
                grad_mexd = grad_mexd - mask_aexd * torch.sum(grad_mexd,1,keepdims=True)
                
                #backprop back to vertex messages
                grad_mvxd = torch.sum(mask_mvxd*torch.unsqueeze(grad_mexd,2),1)
                grad_mvxu = torch.sum(mask_mvxu*torch.unsqueeze(grad_mexu,1),2)
                update = mask_mvxu*torch.unsqueeze(grad_mexu,1) + mask_mvxd*torch.unsqueeze(grad_mexd,2)
                if grad_rx.shape[-1] == 1:
                    if grad_rx.shape[0] == 1:
                        grad_rx += torch.sum(torch.sum(update,-1,keepdim=True),0,keepdim=True)
                    else:
                        grad_rx += torch.sum(update,-1,keepdim=True)
                else:
                    grad_rx += update

                grad_mexd *= 0
                grad_mexu *= 0

                #backprop back to edge messages in x direction
                grad_d[:,:,1:] += grad_mvxd
                grad_mexd[:,:,1:] += grad_mvxd[:,:,:-1]
                grad_d[:,:,:-1] += grad_mvxu 
                grad_mexu[:,:,:-1] += grad_mvxu[:,:,1:]

                iter+=1
                if iter >= max_num_iters:
                    break

            #correct regularisation term gradient to be pack in the expo domain
            grad_erx = grad_rx * erx

            return grad_d, grad_erx
        
    