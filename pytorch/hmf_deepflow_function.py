import deepflow
import numpy as np
import torch
from deepflow_function import DeepFlowFunction


def check_ordering(p,b,l):
    p = p.detach().cpu().numpy().astype(int)
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

def make_leaf_branch_list(p,l,c):
    leaves = [[] for i in range(l)]
    branches = [[i] for i in range(l)]
    for curr, par in enumerate(p):
        if par != -1:
            branches[curr] += branches[par]
    for curr, par in reversed(list(enumerate(p))):
        if curr >= l-c:
            leaves[curr] = [curr]
        if par != -1:
            leaves[par] += leaves[curr]
    antileaves = [[i for i in range(l-c,l) if i not in leaf] for leaf in leaves]
    return leaves, antileaves, branches
        
def mix_regularisation_terms(p,b,l,device,r):
    p = p.detach().cpu().numpy().astype(int)
    lb = len(p)
    sz = list(r[0].shape[2:])
    r_eff = [torch.zeros([b,l,l]+sz,device=device) for rd in r]
        
    #find the leaves
    _, _, branches = make_leaf_branch_list(p,lb,l)
    
    #go through every pair and add reg terms for non-shared parents
    for i in range(l):
        for j in range(i+1,l):
            branch_idx = [branch for branch in branches[lb-l+i] if branch not in branches[lb-l+j]]
            for rd,rd_eff in zip(r,r_eff):
                rd_eff[:,i,j,...] = torch.sum(rd[:,branch_idx,...],dim=1)
                rd_eff[:,j,i,...] = rd_eff[:,i,j,...]

    return r_eff
           
def mix_marginal_fields(p,y):
    p = p.detach().cpu().numpy().astype(int)
    b = y.shape[0]
    bc = len(p)
    c = y.shape[1]
    
    #make a container for the returned marginal thing
    ret_y = torch.zeros(size=tuple([b,bc]+list(y.shape[2:])),device=y.device)
    ret_y[:,(bc-c):,...] = y.clone()
    
    #move up the tree
    for curr_index, par_index in reversed(list(enumerate(p))):
        if par_index != -1:
            ret_y[:,par_index,...] += ret_y[:,curr_index,...]
    
    return ret_y


def sum_up_joint_marginals(p,marginals_j):
    # dims = len(marginals_j)
    p = p.detach().cpu().numpy().astype(int)
    # b = marginals_j[0].shape[0]
    c = marginals_j[0].shape[1]
    bc = len(p)
    leaves, antileaves, _ = make_leaf_branch_list(p,bc,c)
    leaves = [[l-(bc-c) for l in ll] for ll in leaves]
    antileaves = [[l-(bc-c) for l in ll] for ll in antileaves]
    
    #do some nested list comprehension to save on for loops
    #ie for each joint marginal (i.e. dimension) transform it from Leaf-x-Leaf to Branch by (for each branch) finding the probability of moving into/out-of said branch and concatenating
    new_marginals_j = [torch.concatenate([ torch.sum(torch.sum(mj[:,leaves[curr],...],dim=1)[:,antileaves[curr],...],dim=1,keepdim=True) 
                                          +torch.sum(torch.sum(mj[:,antileaves[curr],...],dim=1)[:,leaves[curr],...],dim=1,keepdim=True) 
                                          for curr in range(bc)],dim=1)
                       for mj in marginals_j]
    
    return new_marginals_j
        

#acculate buffer downwards down the tree 
def sum_down_tree(p,buffer):
    p = p.detach().cpu().numpy().astype(int)
    for curr_index, par_index in enumerate(p):
        if par_index != -1:
            buffer[:,curr_index,...] += buffer[:,par_index,...]
            
def acc_up_tree(p,buffer):
    p = p.detach().cpu().numpy().astype(int)
    for curr_index, par_index in reversed(list(enumerate(p))):
        if par_index != -1:
            buffer[:,par_index,...] += buffer[:,curr_index,...]
    
    
class HMFDeepFlow1dLoss_functional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, rx, marginals_d, marginals_rx, y, p):
        epsilon = 10**-12
        ctx.save_for_backward(marginals_d, marginals_rx, y, p)
        loss = torch.mean(-torch.sum(y*torch.log(marginals_d+epsilon),1))
        return loss
            
    @staticmethod
    def backward(ctx, grad_l):
        marginals_d, marginals_rx, y = ctx.saved_tensors

        #unpack y to be hierarchical
        Nu = y.shape[0]*y.shape[1]*y.shape[2]
        y_mix = mix_marginal_fields(p,y)
        yrx = torch.abs(y_mix[:,:,1:]-y_mix[:,:,:-1])

        #get gradient for negative log-likelihood loss        
        return grad_l*(marginals_d-y)/Nu, \
               grad_l*(torch.nn.functional.pad(yrx-marginals_rx,(0,1),value=0))/Nu, \
               torch.zeros_like(marginals_d), \
               torch.zeros_like(marginals_rx), \
               torch.zeros_like(y), torch.zeros_like(p)
    
class HMFDeepFlow2dLoss_functional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, rx, ry, marginals_d, marginals_rx, marginals_ry, y, p):
        epsilon = 10**-12
        ctx.save_for_backward(marginals_d, marginals_rx, marginals_ry, y, p)
        loss = torch.mean(-torch.sum(y*torch.log(marginals_d+epsilon),1))
        return loss
            
    @staticmethod
    def backward(ctx, grad_l):
        marginals_d, marginals_rx, marginals_ry, y, p = ctx.saved_tensors
        
        #unpack y to be hierarchical
        Nu = y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]
        y_mix = mix_marginal_fields(p,y)
        yrx = torch.abs(y_mix[:,:,1:,:]-y_mix[:,:,:-1,:])
        yry = torch.abs(y_mix[:,:,:,1:]-y_mix[:,:,:,:-1])
        
        return grad_l*(marginals_d-y)/Nu, \
               grad_l*(torch.nn.functional.pad(yrx-marginals_rx,(0,0,0,1),value=0))/Nu, \
               grad_l*(torch.nn.functional.pad(yry-marginals_ry,(0,1,0,0),value=0))/Nu, \
               torch.zeros_like(marginals_d), \
               torch.zeros_like(marginals_rx), \
               torch.zeros_like(marginals_ry), \
               torch.zeros_like(y), torch.zeros_like(p)
    
            
class HMFDeepFlow3dLoss_functional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, rx, ry, rz, marginals_d, marginals_rx, marginals_ry, marginals_rz, y, p):
        epsilon = 10**-12
        ctx.save_for_backward(marginals_d, marginals_rx, marginals_ry, marginals_rz, y, p)
        loss = torch.mean(-torch.sum(y*torch.log(marginals_d+epsilon),1))
        return loss
            
    @staticmethod
    def backward(ctx, grad_l):
        marginals_d, marginals_rx, marginals_ry, marginals_rz, y = ctx.saved_tensors
        
        #unpack y to be hierarchical
        Nu = y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]*y.shape[4]
        y_mix = mix_marginal_fields(p,y)
        yrx = torch.abs(y_mix[:,:,1:,:,:]-y_mix[:,:,:-1,:,:])
        yry = torch.abs(y_mix[:,:,:,1:,:]-y_mix[:,:,:,:-1,:])
        yrz = torch.abs(y_mix[:,:,:,:,1:]-y_mix[:,:,:,:,:-1])
        
        return grad_l*(marginals_d-y)/Nu, \
               grad_l*(torch.nn.functional.pad(yrx-marginals_rx,(0,0,0,0,0,1),value=0))/Nu, \
               grad_l*(torch.nn.functional.pad(yry-marginals_ry,(0,0,0,1,0,0),value=0))/Nu, \
               grad_l*(torch.nn.functional.pad(yrz-marginals_rz,(0,1,0,0,0,0),value=0))/Nu, \
               torch.zeros_like(marginals_d), \
               torch.zeros_like(marginals_rx), \
               torch.zeros_like(marginals_ry), \
               torch.zeros_like(marginals_rz), \
               torch.zeros_like(y), torch.zeros_like(p)

#function for mixing together marginal estimates up the tree to get the regularisation terms
def reg_terms_up(p,u,r):
    u_mix = mix_marginal_fields(p,u)
    r_mix = torch.zeros_like(u_mix)
    for i,rd in enumerate(r):
        if i == 0:
            r_mix[:,:,:-1,...] += rd[:,:,:-1,...]*(2*u_mix[:,:,1:,...]-1)
            r_mix[:,:,1:,...] += rd[:,:,:-1,...]*(2*u_mix[:,:,:-1,...]-1)
        elif i == 1:
            r_mix[:,:,:,:-1,...] += rd[:,:,:,:-1,...]*(2*u_mix[:,:,:,1:,...]-1)
            r_mix[:,:,:,1:,...] += rd[:,:,:,:-1,...]*(2*u_mix[:,:,:,:-1,...]-1)
        elif i == 2:
            r_mix[:,:,:,:,:-1,...] += rd[:,:,:,:,:-1,...]*(2*u_mix[:,:,:,:,1:,...]-1)
            r_mix[:,:,:,:,1:,...] += rd[:,:,:,:,:-1,...]*(2*u_mix[:,:,:,:,:-1,...]-1)
    return r_mix


def HMF_MeanNd_calculate_joint(d,r,u,p):
    bc,b,c = p.shape[0],d.shape[0],d.shape[1]
    
    #calculate the joint marginals
    r_mix = reg_terms_up(p,u,r)
    sum_down_tree(p,r_mix)
    energy = d + r_mix[:,(bc-c):,...]
    u = torch.softmax(energy,dim=1)
    u_mix = mix_marginal_fields(p,u)
    _, _, branches = make_leaf_branch_list(p,bc,c)
    marginals_j = []
    for i,rd in enumerate(r):
        
        #get energy to subtract (i.e. the energy associated with the join)
        if i == 0:
            energy_to_sub_lo = rd[:,:,:-1,...] * (2*u_mix[:,:,1:]-1)
            energy_to_sub_hi = rd[:,:,:-1,...] * (2*u_mix[:,:,:-1]-1)
        elif i == 1:
            energy_to_sub_lo = rd[:,:,:,:-1,...] * (2*u_mix[:,:,:,1:]-1)
            energy_to_sub_hi = rd[:,:,:,:-1,...] * (2*u_mix[:,:,:,:-1]-1)
        elif i == 2:
            energy_to_sub_lo = rd[:,:,:,:,:-1,...] * (2*u_mix[:,:,:,:,1:]-1)
            energy_to_sub_hi = rd[:,:,:,:,:-1,...] * (2*u_mix[:,:,:,:,:-1]-1)
        sum_down_tree(p,energy_to_sub_lo)
        energy_to_sub_lo = energy_to_sub_lo[:,(bc-c):,...]
        sum_down_tree(p,energy_to_sub_hi)
        energy_to_sub_hi = energy_to_sub_hi[:,(bc-c):,...]
                
        #calculate the joint marginals with respect to pairs of leaves
        size_tuple = list([b,c,c]+list(rd.shape[2:]))
        size_tuple[3+i] -= 1
        
        branch_mask = torch.zeros(c, bc, dtype=torch.bool)
        for ci in range(c):
            branch_mask[ci, branches[bc-c+ci]] = True
        unshared_mask = branch_mask[:, None, :] ^ branch_mask[None, :, :]
        if i == 0:
            rd_slice = rd[:,:,:-1,...]
        elif i == 1:
            rd_slice = rd[:,:,:,:-1,...]
        elif i == 2:
            rd_slice = rd[:,:,:,:,:-1,...]
        penalty = (rd_slice[:, None, None] *
                   unshared_mask[None, :, :, :, *((None,)*(rd_slice.ndim-2))]
                   ).sum(dim=3)
        if i == 0:
            marginals_pair = torch.unsqueeze(energy[:,:,:-1,...] - energy_to_sub_lo,dim=2) + torch.unsqueeze(energy[:,:,1:,...] - energy_to_sub_hi,dim=1) - penalty
        elif i == 1:
            marginals_pair = torch.unsqueeze(energy[:,:,:,:-1,...] - energy_to_sub_lo,dim=2) + torch.unsqueeze(energy[:,:,:,1:,...] - energy_to_sub_hi,dim=1) - penalty
        elif i == 2:
            marginals_pair = torch.unsqueeze(energy[:,:,:,:,:-1,...] - energy_to_sub_lo,dim=2) + torch.unsqueeze(energy[:,:,:,:,1:,...] - energy_to_sub_hi,dim=1) - penalty
        
        #for c1,c2 in product(range(c),range(c)):
        #    unshared_branches =  [b for b in branches[bc-c+c1] if b not in branches[bc-c+c2]]
        #    unshared_branches += [b for b in branches[bc-c+c2] if b not in branches[bc-c+c1]]
        #    #get energies of the joint distribution
        #    if i == 0:
        #        marginals_pair[:,c1,c2,...] -= torch.sum(rd[:,unshared_branches,:-1,...],dim=1)
        #    elif i == 1:
        #        marginals_pair[:,c1,c2,...] -= torch.sum(rd[:,unshared_branches,:,:-1,...],dim=1)
        #    elif i == 2:
        #        marginals_pair[:,c1,c2,...] -= torch.sum(rd[:,unshared_branches,:,:,:-1,...],dim=1)
        
        #do softmax yourself, subtracting our maximum for stability purposes (ensures denom is between 1 and N^2)
        marginals_pair = torch.exp(marginals_pair - torch.max(torch.max(marginals_pair,dim=1,keepdim=True)[0],dim=2,keepdim=True)[0])
        marginals_pair = marginals_pair / torch.sum(marginals_pair,dim=(1,2),keepdim=True)
    
        #reformulate to be for branches in the tree
        marginals_j += [m.clone() for m in sum_up_joint_marginals(p,[marginals_pair,])]
    
    return marginals_j

def HMF_MeanNd_PytorchNative_forward(d,r,p):
    # dims = len(r)
    bc,_,c = p.shape[0],d.shape[0],d.shape[1]

    max_iter = 2*sum(d.shape[2:]) #todo put in a real convergence criterion
    u = torch.softmax(d,dim=1)

    #perform regular mean field iterations
    iter = 0
    while True:

        #sum the marginal estimates up the tree to get the regularisation values
        r_mix = reg_terms_up(p,u,r)
        
        #sum the regularisation energies down the tree, combine with data term and normalise for new estimate
        sum_down_tree(p,r_mix)
        energy = d + r_mix[:,(bc-c):,...]
        u = torch.softmax(energy,dim=1)

        #check if converged to exit
        iter += 1
        if iter > max_iter:
            break
        
    return u, max_iter

def HMF_MeanNd_PytorchNative_backwards(max_iter,d,r,u,grad_u_org,p):
    # dims = len(r)
    bc,_,c = p.shape[0],d.shape[0],d.shape[1]
    # eye_matrix = torch.eye(c,device=u.device).view(tuple([1,c,c]+[1,]*(len(d.shape)-2)))
    grad_d = torch.zeros_like(d)
    grad_r = [torch.zeros_like(rd) for rd in r]
    u_mix = mix_marginal_fields(p,u)
    grad_u_mix = torch.zeros_like(u_mix)
    grad_e_mix = torch.zeros_like(u_mix)
    # leaves, antileaves, branches = make_leaf_branch_list(p,bc,c)
    
    grad_u_mix[:,(bc-c):,...] += grad_u_org
    
    #undo the regular marginal calculation (add one for the last iteration)
    for iter in range(max_iter+1):
        
        #undo softmax to each component
        grad_e_mix[:,:(bc-c),...] *= 0
        grad_e_mix[:,(bc-c):,...] = u * (grad_u_mix[:,(bc-c):,...] - torch.sum(u*grad_u_mix[:,(bc-c):,...], 1, keepdims=True)) #torch.sum(torch.unsqueeze(grad_u_mix[:,(bc-c):,...] * u,2) * (eye_matrix - torch.unsqueeze(u,1)),2)
        grad_d += grad_e_mix[:,(bc-c):,...]
        
        #propagate gradients up tree for regualirisation then back down for the marginal estimate
        grad_u_mix *= 0
        acc_up_tree(p,grad_e_mix)
        for i,(rd,g_rd) in enumerate(zip(r,grad_r)):
            if i == 0:
                grad_u_mix[:,:,1:,...] += grad_e_mix[:,:,:-1,...] * rd[:,:,:-1,...] #(put in the multiplication by 2 at the end)
                grad_u_mix[:,:,:-1,...] += grad_e_mix[:,:,1:,...] * rd[:,:,:-1,...] #(put in the multiplication by 2 at the end)
                g_rd[:,:,:-1,...] += grad_e_mix[:,:,:-1,...] * (2*u_mix[:,:,1:,...]-1)
                g_rd[:,:,:-1,...] += grad_e_mix[:,:,1:,...] * (2*u_mix[:,:,:-1,...]-1)
            elif i == 1:
                grad_u_mix[:,:,:,1:,...] += grad_e_mix[:,:,:,:-1,...] * rd[:,:,:,:-1,...] #(put in the multiplication by 2 at the end)
                grad_u_mix[:,:,:,:-1,...] += grad_e_mix[:,:,:,1:,...] * rd[:,:,:,:-1,...] #(put in the multiplication by 2 at the end)
                g_rd[:,:,:,:-1,...] += grad_e_mix[:,:,:,:-1,...] * (2*u_mix[:,:,:,1:,...]-1)
                g_rd[:,:,:,:-1,...] += grad_e_mix[:,:,:,1:,...] * (2*u_mix[:,:,:,:-1,...]-1)
            elif i == 3:
                grad_u_mix[:,:,:,:,1:,...] += grad_e_mix[:,:,:,:,:-1,...] * rd[:,:,:,:,:-1,...] #(put in the multiplication by 2 at the end)
                grad_u_mix[:,:,:,:,:-1,...] += grad_e_mix[:,:,:,:,1:,...] * rd[:,:,:,:,:-1,...] #(put in the multiplication by 2 at the end)
                g_rd[:,:,:,:,:-1,...] += grad_e_mix[:,:,:,:,:-1,...] * (2*u_mix[:,:,:,:,1:,...]-1)
                g_rd[:,:,:,:,:-1,...] += grad_e_mix[:,:,:,:,1:,...] * (2*u_mix[:,:,:,:,:-1,...]-1)
        sum_down_tree(p,grad_u_mix)
        grad_u_mix[:,(bc-c):,...] *= 2 #and here it is
    
    return grad_d, grad_r

class HMF_Mean1d_PytorchNative(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, p):
        marginals_u, max_iter = HMF_MeanNd_PytorchNative_forward(d,[rx,],p)
        ctx.save_for_backward(d,rx,marginals_u,p,torch.tensor(max_iter))
        return marginals_u
    
    @staticmethod
    def backward(ctx, grad_marginal_u):
        d,rx,u,p,max_iter = ctx.saved_tensors
        grad_d, grad_r = HMF_MeanNd_PytorchNative_backwards(int(max_iter.item()),d,[rx,],u,grad_marginal_u,p)
        return grad_d, grad_r[0], torch.zeros_like(p)
    
class HMF_Mean2d_PytorchNative(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry, p):
        marginals_u, max_iter = HMF_MeanNd_PytorchNative_forward(d,[rx,ry,],p)
        ctx.save_for_backward(d,rx,ry,marginals_u,p,torch.tensor(max_iter))
        return marginals_u
    
    @staticmethod
    def backward(ctx, grad_marginal_u):
        d,rx,ry,u,p,max_iter = ctx.saved_tensors
        grad_d, grad_r = HMF_MeanNd_PytorchNative_backwards(int(max_iter.item()),d,[rx,ry,],
                                                            u,grad_marginal_u,p)
        return grad_d, grad_r[0], grad_r[1], torch.zeros_like(p)
    
class HMF_Mean3d_PytorchNative(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry, rz, p):
        marginals_u, max_iter = HMF_MeanNd_PytorchNative_forward(d,[rx,ry,rz,],p)
        ctx.save_for_backward(d,rx,ry,rz,marginals_u,p,torch.tensor(max_iter))
        return marginals_u
    
    @staticmethod
    def backward(ctx, grad_marginal_u):
        d,rx,ry,rz,u,p,max_iter = ctx.saved_tensors
        grad_d, grad_r = HMF_MeanNd_PytorchNative_backwards(int(max_iter.item()),d,[rx,ry,rz,],
                                                            u,grad_marginal_u,p)
        return grad_d, grad_r[0], grad_r[1], grad_r[2], torch.zeros_like(p)
    
    
def HMF_MAPNd_PytorchNative_ALMM_forward(d,r,order):
    d_eff_mean = -torch.mean(torch.sort(d-torch.amax(d,1,keepdims=True),dim=1)[0][:,-2,...])+10**-16
    DeepFlowFunction.check_var_dims([d,]+r,len(r),True)
    check_ordering(order,r[0].shape[1],d.shape[1])
    order_forward = list(enumerate(order.detach().cpu().numpy().astype(int)))
    order_backward = list(reversed(order_forward))
    b,bc,c,sz = d.shape[0], r[0].shape[1], d.shape[1],list(d.shape[2:])
    D = -(d - torch.max(d,dim=1,keepdim=True)[0]) / d_eff_mean
    # D_eff_mean = torch.mean(torch.sort(D,dim=1)[0][:,1,...]).item()
    div = torch.zeros_like(r[0])
    r = [rd/d_eff_mean for rd in r]
    q = [torch.zeros_like(rd) for rd in r]
    cc = 0.25
    icc = 1/cc
    tau = 0.1
    conv_crit = 0.001
    
    max_iter = 20*sum(sz)
    min_iter = 10
    
    #get the number of children for each node
    num_children = [0]*bc
    for curr,par in order_backward:
        num_children[par] += 1
        
    #initalise systems
    ps = torch.amin(D,dim=1)#torch.zeros(size=[b,]+sz,device=D.device)#
    old_sum_ps = torch.sum(ps)
    p = torch.repeat_interleave(torch.unsqueeze(ps,1),bc,dim=1)
    u = torch.zeros_like(p)
    u[:,(bc-c):,...] = 0.1*torch.softmax(d,1)
    u[:,(bc-c):,...] += 0.9*(p[:,(bc-c):,...] == D).float()
    u[:,(bc-c):,...] /= torch.sum(u[:,(bc-c):,...],1,keepdim=True)
    for curr,par in order_backward:
        if par != -1:
            u[:,par,...] += u[:,curr,...]
    #print(u)
    
    iter = 0
    while True:
        
        #update capacities and spatial flows
        for inner_iter in range(100):
            cap = div + p - u*icc
            for curr,par in order_backward:
                if par == -1:
                    cap[:,curr,...] -= ps
                else:
                    cap[:,curr,...] -= p[:,par,...]
            for i,(rd,qd) in enumerate(zip(r,q)):
                if i == 0:
                    qd[:,:,:-1,...] -= tau * (cap[:,:,1:,...]-cap[:,:,:-1,...])
                elif i == 1:
                    qd[:,:,:,:-1,...] -= tau * (cap[:,:,:,1:,...]-cap[:,:,:,:-1,...])
                elif i == 2:
                    qd[:,:,:,:,:-1,...] -= tau * (cap[:,:,:,:,1:,...]-cap[:,:,:,:,:-1,...])
                qd[qd > rd] = rd[qd > rd]
                qd[qd < -rd] = -rd[qd < -rd]

            #calculate divergences
            div *= 0
            for i,qd in enumerate(q):
                if i == 0:
                    div[:,:,:-1,...] -= qd[:,:,:-1,...]
                    div[:,:,1:,...] += qd[:,:,:-1,...]
                elif i == 1:
                    div[:,:,:,:-1,...] -= qd[:,:,:,:-1,...]
                    div[:,:,:,1:,...] += qd[:,:,:,:-1,...]
                elif i == 2:
                    div[:,:,:,:,:-1,...] -= qd[:,:,:,:,:-1,...]
                    div[:,:,:,:,1:,...] += qd[:,:,:,:,:-1,...]
        
        #update source/sink flows (leaves first)
        caps = torch.ones_like(ps) * icc
        cap = u*icc - div
        for curr,par in order_backward[:c]:
            if par != -1:
                cap[:,curr,...] += p[:,par,...]
            else:
                cap[:,curr,...] += ps
        p[:,(bc-c):,...] = torch.clip(torch.minimum(D,cap[:,(bc-c):,...]),min=0)
        cap[:,(bc-c):,...] = p[:,(bc-c):,...] + div[:,(bc-c):,...] - u[:,(bc-c):,...]*icc
        for curr,par in order_backward[:c]:
            if par != -1:
                cap[:,par,...] += cap[:,curr,...]
            else:
                caps += cap[:,curr,...]
            
        #now branches
        for curr,par in order_backward[c:]:
            if par != -1:
                cap[:,curr,...] += p[:,par,...]
                p[:,curr,...] = torch.clip(cap[:,curr,...]/(num_children[curr]+1),min=0)
                cap[:,par,...] += p[:,curr,...] + div[:,curr,...] - u[:,curr,...]*icc
            else:
                cap[:,curr,...] += ps
                p[:,curr,...] = torch.clip(cap[:,curr,...]/(num_children[curr]+1),min=0)
                caps += p[:,curr,...] + div[:,curr,...] - u[:,curr,...]*icc
        
        #update source flow last
        ps = torch.clip(caps / num_children[-1],min=0)
        
        #update labels
        u_diff = div + p
        for curr,par in order_backward:
            if par == -1:
                u_diff[:,curr,...] -= ps
            else:
                u_diff[:,curr,...] -= p[:,par,...]
        u_diff *= cc
        # most_changed = torch.unravel_index(torch.argmax(torch.sum(torch.abs(u_diff),1)),[b,]+sz)
        #print(most_changed)
        #print(u[most_changed[0],:,most_changed[1],most_changed[2]])
        u -= u_diff
        #print(u[most_changed[0],:,most_changed[1],most_changed[2]])
        conv_test = torch.max(torch.abs(u_diff)).item()
        
        new_sum_ps = torch.sum(ps)
        perc_impr = 2*(new_sum_ps-old_sum_ps) / (new_sum_ps+old_sum_ps)
        if perc_impr < 0:
            cc *= 1.001
        elif perc_impr > 0.0:
            cc /= 1.001
        icc = 1/cc
        #print(iter,"tot ps",new_sum_ps,old_sum_ps,perc_impr,cc,"conv test",conv_test)
        old_sum_ps = new_sum_ps
        iter += 1
        if iter > min_iter and (iter > max_iter or conv_test < conv_crit):
            break
            
    #and binarise
    u = (u[:,(bc-c):,...]==torch.amax(u[:,(bc-c):,...],1,keepdims=True)).float()
    u /= torch.sum(u,1,keepdims=True)
    #print("Took "+str(iter)+" iterations.")
    #exit()
    return u
    
    
def HMF_MAPNd_PytorchNative_Bregman_forward(d,r,order):
    d_eff_mean = -torch.mean(torch.sort(d-torch.amax(d,1,keepdims=True),dim=1)[0][:,-2,...])+10**-16
    DeepFlowFunction.check_var_dims([d,]+r,len(r),True)
    check_ordering(order,r[0].shape[1],d.shape[1])
    order_forward = list(enumerate(order.detach().cpu().numpy().astype(int)))
    order_backward = list(reversed(order_forward))
    b,bc,c,sz = d.shape[0], r[0].shape[1], d.shape[1],list(d.shape[2:])
    ds = torch.zeros(size=[b,]+sz,device=d.device)
    D = -(d - torch.max(d,dim=1,keepdim=True)[0]) / d_eff_mean
    r = [rd / d_eff_mean for rd in r]
    u = (d == torch.max(d,dim=1,keepdim=True)[0]).float()
    u = 0.1*u/torch.sum(u,dim=1,keepdim=True) + 0.9 / c
    d = torch.zeros_like(r[0])
    q = [torch.zeros_like(rd) for rd in r]
    # D_eff_mean = torch.mean(torch.sort(D,dim=1)[0][:,1,...]).item()
    cc = 0.25
    icc = 1/cc
    tau = 0.1*cc
    
    
    max_iter = bc*sum(sz)
    min_iter = min(sz)/2
    
    iter = 0
    while True:
        #push down costs
        div = torch.zeros_like(r[0])
        for i,qd in enumerate(q):
            if i == 0:
                div[:,:,:-1,...] -= qd[:,:,:-1,...]
                div[:,:,1:,...] += qd[:,:,:-1,...]
            elif i == 1:
                div[:,:,:,:-1,...] -= qd[:,:,:,:-1,...]
                div[:,:,:,1:,...] += qd[:,:,:,:-1,...]
            elif i == 2:
                div[:,:,:,:,:-1,...] -= qd[:,:,:,:,:-1,...]
                div[:,:,:,:,1:,...] += qd[:,:,:,:,:-1,...]
        for curr, par in order_forward:
            if par == -1:
                d[:,curr,...] = ds + div[:,curr,...]
            else:
                d[:,curr,...] = d[:,par,...] + div[:,curr,...]
        d[:,(bc-c):,...] += D
                
        #normalise across leaves
        d[:,(bc-c):,...] -= torch.max(d[:,(bc-c):,...],1,keepdim=True)[0]
        d[:,(bc-c):,...] = u * torch.exp(-icc * d[:,(bc-c):,...])
        u_old = u.clone()
        u = d[:,(bc-c):,...] / torch.sum(d[:,(bc-c):,...],dim=1,keepdim=True)
        conv_test = torch.max(torch.abs(u_old - u)).item()
        del u_old
        
        #push up capacities
        ds *= 0
        d[:,:(bc-c),...] *= 0
        for curr,par in order_backward:
            if par == -1:
                ds += d[:,curr,...]
            else:
                d[:,par,...] += d[:,curr,...]
                
        #redo spatial flows
        for i,(rd,qd) in enumerate(zip(r,q)):
            if i == 0:
                qd[:,:,:-1,...] += tau * (d[:,:,:-1,...]-d[:,:,1:,...])
            elif i == 1:
                qd[:,:,:,:-1,...] += tau * (d[:,:,:,:-1,...]-d[:,:,:,1:,...])
            elif i == 2:
                qd[:,:,:,:,:-1,...] += tau * (d[:,:,:,:,:-1,...]-d[:,:,:,:,1:,...])
            qd[qd > rd] = rd[qd > rd]
            qd[qd < -rd] = -rd[qd < -rd]
        
        iter += 1
        if iter > min_iter and (iter > max_iter or  conv_test < 0.01):
            break
    return u
    
    
def HMF_MAP1d_PytorchNative_forward(d, rx, p):
    """Solve the hierarchical chain exactly, including branch boundary costs."""
    DeepFlowFunction.check_var_dims([d, rx], 1, True)
    b, c, n = d.shape
    bc = rx.shape[1]
    check_ordering(p, bc, c)
    parents = p.detach().cpu().tolist()

    # A transition pays for every hierarchy node whose membership changes.
    membership = d.new_zeros((bc, c))
    for leaf in range(c):
        node = bc - c + leaf
        while node != -1:
            membership[node, leaf] = 1
            node = parents[node]
    differences = torch.abs(membership[:, :, None] - membership[:, None, :])

    scores = d[:, :, 0]
    predecessors = torch.empty((b, c, n), dtype=torch.long, device=d.device)
    for site in range(1, n):
        transition_cost = torch.einsum("bk,kij->bij", rx[:, :, site - 1], differences)
        best_scores, previous = torch.max(scores[:, :, None] - transition_cost, dim=1)
        scores = best_scores + d[:, :, site]
        # Remove a common offset to limit roundoff on long chains.
        scores = scores - scores.max(dim=1, keepdim=True).values
        predecessors[:, :, site] = previous

    labels = torch.empty((b, n), dtype=torch.long, device=d.device)
    labels[:, -1] = scores.argmax(dim=1)
    batch = torch.arange(b, device=d.device)
    for site in range(n - 1, 0, -1):
        labels[:, site - 1] = predecessors[batch, labels[:, site], site]
    return torch.zeros_like(d).scatter_(1, labels[:, None, :], 1)


class HMF_MAP1d_PytorchNative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, rx, p):
        return HMF_MAP1d_PytorchNative_forward(d, rx, p)
    @staticmethod
    def backward(ctx, g_u):
        raise Exception("No backwards pass for MAP")
    
class HMF_MAP2d_PytorchNative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, rx, ry, p):
        return HMF_MAPNd_PytorchNative_ALMM_forward(d,[rx,ry],p)
    @staticmethod
    def backward(ctx, g_u):
        raise Exception("No backwards pass for MAP")
    
class HMF_MAP3d_PytorchNative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, rx, ry, rz, p):
        return HMF_MAPNd_PytorchNative_ALMM_forward(d,[rx,ry,rz],p)
    @staticmethod
    def backward(ctx, g_u):
        raise Exception("No backwards pass for MAP")
    
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
        
    #For the optimisers, there is no well defined backwards
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
