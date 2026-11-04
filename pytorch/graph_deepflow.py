import torch
from graph_deepflow_function import DenseGraph_LBP_Categorical_PytorchNative, DenseGraph_MFE_Categorical_PytorchNative, DenseGraph_MAP_Categorical_PytorchNative
from graph_deepflow_function import DenseGraph_LBP_Binary_PytorchNative, DenseGraph_MFE_Binary_PytorchNative, DenseGraph_MAP_Binary_PytorchNative
import math


class BinaryDeepFlow(torch.nn.Module):
    def __init__(self, solve_type="LBP",test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
    
    def forward(self,d,r):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            N = d.shape[2]
            eye = torch.unsqueeze(torch.unsqueeze(torch.eye(N).to(d.device),dim=0),dim=1)
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP" or self.solve_type == "FULL_MARGINALS":
                me0, me1 = DenseGraph_LBP_Binary_PytorchNative.apply(d,r)
                
                #estimate the marginals for each node
                ed = torch.exp(d)
                marginal_node_0 = torch.prod(me0,dim=2)
                marginal_node_1 = ed*torch.prod(me1,dim=2)
                am = marginal_node_0+marginal_node_1
                
                #estimate the joint marginals for each edge, usng the unnormalised marginals to speed things up #TODO check accuracy
                er = torch.exp(-r)
                marginal_joint_00 =    torch.unsqueeze(marginal_node_0,dim=-1)*torch.unsqueeze(marginal_node_0,dim=-2) / (me0*torch.permute(me0,(0,1,3,2)))
                marginal_joint_10 = er*torch.unsqueeze(marginal_node_1,dim=-1)*torch.unsqueeze(marginal_node_0,dim=-2) / (me0*torch.permute(me1,(0,1,3,2)))
                marginal_joint_01 = er*torch.unsqueeze(marginal_node_0,dim=-1)*torch.unsqueeze(marginal_node_1,dim=-2) / (me1*torch.permute(me0,(0,1,3,2)))
                marginal_joint_11 =    torch.unsqueeze(marginal_node_1,dim=-1)*torch.unsqueeze(marginal_node_1,dim=-2) / (me1*torch.permute(me1,(0,1,3,2)))
                marginal_joint_norm = marginal_joint_00 + marginal_joint_01 + marginal_joint_10 + marginal_joint_11
                
                #assert(self.solve_type == "FULL_MARGINALS")
                return marginal_node_1 / am, (marginal_joint_01+marginal_joint_10)*(1-eye) / marginal_joint_norm
            
                
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                energy = DenseGraph_MFE_Binary_PytorchNative.apply(d,r)
                marginal = torch.sigmoid(energy)
                
                e01 = torch.unsqueeze(energy,dim=-2) - 2*r*torch.unsqueeze(marginal,dim=-1)
                e10 = torch.unsqueeze(energy,dim=-1) - 2*r*torch.unsqueeze(marginal,dim=-2)
                e11 = torch.unsqueeze(energy,dim=-1) + torch.unsqueeze(energy,dim=-2) - 2*r*(torch.unsqueeze(marginal,dim=-1)+torch.unsqueeze(marginal,dim=-2)-1)
                e00 = -torch.clip(torch.maximum(torch.maximum(e01,e10),e11),min=0)
                u00 = torch.exp(e00)
                u01 = torch.exp(e01+e00)
                u10 = torch.exp(e10+e00)
                u11 = torch.exp(e11+e00)
                marginal_nonmatch = ((u10+u01)/(u00+u01+u10+u11))*(1-eye)
                
                return marginal, marginal_nonmatch
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
            
        elif self.test_type == "MAP":
            return DenseGraph_MAP_Binary_PytorchNative.apply(d,r)
            
        else:
                raise Exception("Uknown test-time solve type: " + str(self.solve_type))
            
            

class PottsDeepFlow(torch.nn.Module):
    def __init__(self, solve_type="LBP",activation=True,test_type=None) -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        self.activation = activation
    
    def forward(self,d,r):
                
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #transform Potts regularisation to more general form (for non-MAP solvers)
            N = d.shape[2]
            r = (torch.unsqueeze(r,dim=1)+torch.unsqueeze(r,dim=2)) * (1-torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.eye(N),dim=0),dim=1),dim=2))
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP" or self.solve_type == "FULL_MARGINALS":
                me = DenseGraph_LBP_Categorical_PytorchNative.apply(d,r)
                
                #calculate the marginal in the logistic domain
                if self.solve_type == "LBP" and self.activation == False:
                    marginal_logit = d + torch.sum(torch.log(me),dim=2)
                    return marginal_logit
                
                epsilon = 10**-10
                me = DenseGraph_LBP_Categorical_PytorchNative.apply(d,r)
                
                #estimate the marginals for each node
                marginal_node = torch.exp(d)*torch.prod(me0,dim=2)
                if self.solve_type == "LBP" and self.activation == True:
                    return marginal_node / torch.sum(marginal_node,dim=1)
                
                #estimate the joint marginals for each edge
                marginal_joint = torch.exp(-r) * torch.unsqueeze(torch.unsqueeze(marginal_node_0,dim=2),dim=4)*torch.unsqueeze(torch.unsqueeze(marginal_node_0,dim=1),dim=3) / (torch.unsqueeze(me,dim=1)*torch.unsqueeze(torch.permute(me,(0,1,3,2)),dim=2))
                                
                #assert(self.solve_type == "FULL_MARGINALS")
                return marginal_node / torch.sum(marginal_node,dim=1), marginal_joint / torch.sum(torch.sum(marginal_joint_norm,dim=1,keepdim=True),dim=2,keepdim=True) * (1-torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.eye(N),dim=0),dim=1),dim=2))
            
                
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                marginal_logit = DenseGraph_MFE_Categorical_PytorchNative.apply(d,r)
                return torch.softmax(marginal_logit,dim=1) if self.activation else marginal_logit
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
            
        elif self.test_type == "MAP":
            #transform Potts regularisation to more general form (for MAP solvers)
            c, N = d.shape[1], d.shape[2]
            r = torch.unsqueeze(r,dim=2)*torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.eye(c),dim=0),dim=-1),dim=-1)*(1-torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.eye(N),dim=0),dim=1),dim=2))
            
            return DenseGraph_MAP_Categorical_PytorchNative.apply(d,r)
            
        else:
            raise Exception("Uknown test-time solve type: " + str(self.solve_type))
            