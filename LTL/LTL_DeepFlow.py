import torch
from LTL_Node import *
from LBP_LTL_LOCAL_PytorchNative import LBP_LTL_LOCAL_PytorchNative
from LBP_LTL_GLOBAL_PytorchNative import LBP_LTL_GLOBAL_PytorchNative
from ALMM_LTL_GLOBAL_PytorchNative import ALMM_LTL_GLOBAL_PytorchNative
from itertools import product
import copy


class LTL_DeepFlow(torch.nn.Module):
    
    # Constructor for the activation layer
    # - creates the layer given the root node 
    #
    # Inputs
    # structure - grouping structure
    # constraint_list - list of three-tuples representing BEFORE, AFTER, PRECEDE, NOTPRECEDE, SUCEED, NOTSUCEED, INITIAL, FINAL, and UNIQUE constraints
    # constraint_type - whether to use a local belief propagation (LOCAL), global loopy belief propagation (GLOBAL), or augmented Lagrangian method of multipliers (ALMM)
    # solve_type - whether to use sum-product (MARG) or max-product (MAP) in training time
    #
    def __init__(self, structure : LTL_GroupingStructure,
                       constraint_list : List[Tuple[LTL_Node, LTL_Node, str]],
                       constraint_type="GLOBAL",solve_type="MARG") -> None:
        super().__init__()
        self.solve_type = solve_type#copy.deepcopy(solve_type)
        self.structure = structure.copy()
        self.constraint_list = [c for c in constraint_list]
        self.constraint_type = copy.deepcopy(constraint_type)
        
    def forward_general(self,d,r=None,solve_type="MARG"):
        
        #check input to make sure it is the right size
        self.structure.check_vars_dims(d,r)
        b,ct,N = d.shape[0],d.shape[1],d.shape[2]
        c = len(self.structure.leaves)
        function_base = (LBP_LTL_LOCAL_PytorchNative if self.constraint_type == "LOCAL" else
                         LBP_LTL_GLOBAL_PytorchNative if self.constraint_type == "GLOBAL" else
                         ALMM_LTL_GLOBAL_PytorchNative)
        function = (function_base.MARG if solve_type == "MARG" else function_base.MAP)
        if solve_type == "MARG" and function_base == ALMM_LTL_GLOBAL_PytorchNative:
            raise ValueError("Augmented Lagrangian method of multipliers only has MAP functionality")
        
        #create container for regularisation terms
        d_eff, r_eff = function_base.make_dr_eff(d,r,self.structure.groups,self.structure.leaves)
        
        #apply constraints
        if r_eff is not None:
            args = function_base.constraint_application(self.constraint_list,self.structure,d_eff,r_eff)
        else:
            args = function_base.constraint_application(self.constraint_list,self.structure,d_eff)
        
        #run solver
        leaf_probs = torch.softmax(function.apply(*args),axis=1)

        #construct back out the terms for the groups
        p_d = torch.zeros((1,c,ct), dtype=d.dtype, device=d.device)
        for l in self.structure.leaves:
            p_d[:,l.index,l.index] = 1
        for g in self.structure.groups:
            for l in g.leaves:
                p_d[:,l.index,g.index] = 1
        full_probs = torch.matmul(torch.permute(p_d,(0,2,1)).detach(),leaf_probs)
        
        #clean up and return
        function_base.cleanup(self.structure)
        return full_probs

    def forward(self, d, r=None):
        if self.training:
            return self.forward_general(d,r,self.solve_type)
        else:
            return self.MAP(d,r)

    def MAP(self, d, r=None):
        with torch.no_grad():
            map_marginal = self.forward_general(d,r,"MAP")
            map_marginal[:,:len(self.structure.leaves),:] = (map_marginal[:,:len(self.structure.leaves),:] == torch.amax(map_marginal[:,:len(self.structure.leaves),:],1,keepdims=True)).to(torch.float32)
            for b in self.structure.groups:
                map_marginal[:,b.index,:] = torch.sum(map_marginal[:,[l.index for l in b.leaves],:],1)
        return map_marginal
    
    #internal autograd function
    @staticmethod
    class LBP_LTL_LOSS_PytorchNative_with_R(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gt, p, d, r,leaves):
            ctx.save_for_backward(gt,p)
            return -torch.mean(torch.sum(gt[:,:int(leaves.item()),:]*torch.log(p[:,:int(leaves.item()),:]),1))
        
        #TODO need to update with r
        @staticmethod
        def backward(ctx, grad_l):
            gt,p = saved_tensors
            grad_d = grad_l*(p-gt)/(d.shape[0]*d.shape[-1])
            return torch.zeros_like(gt), torch.zeros_like(p), grad_d, grad_r, torch.zeros(1)
        
    @staticmethod
    class LBP_LTL_LOSS_PytorchNative_without_R(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gt, p, d, leaves):
            ctx.save_for_backward(gt,p)
            return -torch.mean(torch.sum(gt[:,:int(leaves.item()),:]*torch.log(p[:,:int(leaves.item()),:]),1))
        
        @staticmethod
        def backward(ctx, grad_l):
            gt,p = ctx.saved_tensors
            grad_d = grad_l*(p-gt)/(gt.shape[0]*gt.shape[-1])
            return torch.zeros_like(gt), torch.zeros_like(p), grad_d, torch.zeros(1)
            
    
    def loss(self,gt,d,r=None,p=None):
        if p is None:
            p = self.forward(d,r).detach()
        if r is None:
            return LTL_DeepFlow.LBP_LTL_LOSS_PytorchNative_without_R.apply(gt,p,d,torch.tensor(len(self.structure.leaves)).detach())
        else:
            return LTL_DeepFlow.LBP_LTL_LOSS_PytorchNative_with_R.apply(gt,p,d,r,torch.tensor(len(self.structure.leaves)).detach())
        
    
    
    
            