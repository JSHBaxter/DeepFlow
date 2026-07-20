import torch
from LBP_LTL_LOCAL_PytorchNative import LBP_LTL_LOCAL_PytorchNative
from LBP_LTL_GLOBAL_PytorchNative import LBP_LTL_GLOBAL_PytorchNative
from typing import List, Tuple
from sortedcontainers import SortedSet
from itertools import product
import copy

class LTL_Node:
    def __init__(self,name,index):
        self.name = name
        self.index = index
        self.leaves = SortedSet(self.get_leaves())
        self.level = 0
    
    def __str__(self):
        return self.name
    
    def __lt__(self, other):
        return self.index < other.index
    
    # These methods need to be implemented by the Leaf and Group subclasses
    def get_leaves(self):
        raise NotImplement
    def verify_correctness(self,leaves,groups):
        raise NotImplement

class LTL_LeafNode(LTL_Node):
    def __init__(self,name,index):
        super().__init__(name,index)
        
    def get_leaves(self):
        return [self]
    
    def verify_correctness(self,leaves,groups):
        if self not in leaves:
            raise ValueError("Group verification failed - leaf "+self.name+" missing from list")
    
class LTL_GroupNode(LTL_Node):
    def __init__(self,name,index,children):
        self.children = [c for c in children]
        super().__init__(name,index)
        self.level = max([c.level for c in self.children])+1
    
    def __str__(self):
        stringname = self.name + " {"
        for c in self.children:
            stringname += c.name+", "
        return stringname[:-2] + "}"
    
    def __lt__(self, other):
        return self.level < other.level
            
    def __len__(self):
        return len(self.children)
    
    def get_leaves(self):
        ret_list = []
        for c in self.children:
            ret_list += c.get_leaves()
        return ret_list
    
    #verify groups are in list and form a DAG
    def verify_correctness(self,leaves,groups):
        if self.flag:
            raise ValueError("Group verification failed - not a DAG")
        if self not in groups:
            raise ValueError("Group verification failed - group "+self.name+" missing from list")
        self.flag = True
        for c in self.children:
            if not issubclass(type(c),LTL_Node):
                raise ValueError("Group verification failed - group "+self.name+" contains invalid child")
    
class LTL_GroupingStructure:
    def __init__(self, leaves : List[LTL_LeafNode],
                       group_list : List[LTL_GroupNode]):
        self.leaves = SortedSet(leaves)
        self.groups = SortedSet(group_list)
        
        #verify index correctness
        indices = SortedSet()
        for l in self.leaves:
            if l.index not in indices:
                indices.add(l.index)
            else:
                raise ValueError("Leaf index collision : "+str(l.index))
        for i in range(len(self.leaves)):
            if i not in indices:
                raise ValueError("Leaf index missing: "+str(i))
        for g in self.groups:
            if g.index not in indices:
                indices.add(g.index)
            else:
                raise ValueError("Group index collision : "+str(g.index))
        for i in range(len(self.leaves),len(self.leaves)+len(self.groups)):
            if i not in indices:
                raise ValueError("Group index missing: "+str(i))
        
        #verify groups are all included and only reference other groups and leaves - should raise error if incorrect
        for g in self.groups:
            for n in self.groups:
                g.flag=False
            g.verify_correctness(self.leaves,self.groups)
    
    def __str__(self):
        stringname = "Leaves : {"
        for l in self.leaves:
            stringname += l.name+", "
        stringname = stringname[:-2]+"}\nGroups: {"
        for g in self.groups:
            stringname += g.name+",\n         "
        stringname = stringname[:-11]+"}"
        return stringname
            
    def __len__(self):
        return len(self.leaves)+len(self.groups)
        
    def check_vars_dims(self, d, r):
        if r is not None and d.shape[0] != r.shape[0]:
            raise ValueError("Batch size needs to be the same for both the data and regularisation terms")
        if r is not None and d.shape[1] != r.shape[1]:
            raise ValueError("Number of channels needs to be the same for both the data and regularisation terms")
        if r is not None and d.shape[2] != r.shape[2]+1:
            raise ValueError("The length of the regularisation signal must be one less than that of the data signal")
        if d.shape[1] != len(self):
            raise ValueError("Number of channels disagrees with number of leaves and groups in the hierarchy")

    def copy(self):
        return LTL_GroupingStructure(list(self.leaves),list(self.groups))


class LTL_DeepFlow(torch.nn.Module):
    
    # Constructor for the activation layer
    # - creates the layer given the root node 
    #
    # Inputs
    # structure - grouping structure
    # constraint_list - list of three-tuples representing BEFORE, AFTER, PRECEDE, NOTPRECEDE, SUCEED, NOTSUCEED, INITIAL, FINAL, and UNIQUE constraints
    # constraint_type - whether to use a local or a global estimate (only LOCAL currently implemented)
    # solve_type - whether to use sum-product (MARG) or max-product (MAP) in training time
    #
    def __init__(self, structure : LTL_GroupingStructure,
                       constraint_list : List[Tuple[LTL_Node, LTL_Node, str]],
                       constraint_type="LOCAL",solve_type="MARG") -> None:
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
        function_base = (LBP_LTL_LOCAL_PytorchNative if self.constraint_type == "LOCAL" else LBP_LTL_GLOBAL_PytorchNative)
        function = (function_base.MARG if solve_type == "MARG" else function_base.MAP)
        
        #create container for data terms
        p_d = torch.zeros((1,c,ct), dtype=d.dtype, device=d.device)
        for l in self.structure.leaves:
            p_d[:,l.index,l.index] = 1
        for g in self.structure.groups:
            for l in g.leaves:
                p_d[:,l.index,g.index] = 1
        d_eff = torch.matmul(p_d.detach(),d)
        
        #create container for regularisation terms
        p_r = torch.zeros((1,c*c,ct), dtype=d.dtype, device=d.device)
        unshared = {}
        for l1,l2 in product(self.structure.leaves,self.structure.leaves):
            unshared[(l1,l2)] = [] if l1 == l2 else [l1,l2]
        for g in self.structure.groups:
            l_in = g.leaves
            l_out = self.structure.leaves - l_in
            for l1,l2 in product(l_in,l_out):
                unshared[(l1,l2)].append(g)
                unshared[(l2,l1)].append(g)
        for l1,l2 in product(self.structure.leaves,self.structure.leaves):
            for g in unshared[(l1,l2)]:
                p_r[:,l1.index*c+l2.index,g.index] = 1
        if r is not None:
            r_eff = torch.exp(-torch.matmul(p_r.detach(),r.unsqueeze(0)).view(b,c,c,N-1))
        else:
            r_eff = None
        
        #apply constraints
        if r_eff is not None:
            args = function_base.constraint_application(self.constraint_list,self.structure,d_eff,r_eff)
        else:
            args = function_base.constraint_application(self.constraint_list,self.structure,d_eff)
        
        #run solver
        leaf_probs = torch.softmax(function.apply(*args),axis=1)

        #construct back out the terms for the groups
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
        
    
    
    
            