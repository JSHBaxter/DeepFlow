import torch
from graph_deepflow_function import DenseGraph_LBP_Categorical_PytorchNative, DenseGraph_MFE_Categorical_PytorchNative, DenseGraph_MAP_Categorical_PytorchNative
from graph_deepflow_function import DenseGraph_LBP_Binary_PytorchNative, DenseGraph_MFE_Binary_PytorchNative, DenseGraph_MAP_Binary_PytorchNative
import math


class BinaryDeepFlow(torch.nn.Module):
    def __init__(self, solve_type="LBP",activation=True,test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        self.activation = activation
    
    def forward(self,d,r):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP":
                me0, me1 = DenseGraph_LBP_Binary_PytorchNative.apply(d,r)
                marginal_logit = d + torch.sum(torch.log(me1),dim=2) - torch.sum(torch.log(me0),dim=2)
                
            #if the testing time type is set to the logarithm of the partition function, use a Bethe approximation (side result of LBP)
            elif self.solve_type == "LPF":
                epsilon = 10**-10
                N = d.shape[2]
                me0, me1 = DenseGraph_LBP_Binary_PytorchNative.apply(d,r)
                
                #estimate the marginals for each node
                marginal_node_0 = torch.prod(me0,dim=2)
                marginal_node_1 = torch.exp(d)*torch.prod(me1,dim=2)
                marginal_node_0, marginal_node_1 = marginal_node_0 / (marginal_node_0+marginal_node_1), marginal_node_1 / (marginal_node_0+marginal_node_1)
                
                #estimate the joint marginals for each edge
                marginal_joint_00 = torch.unsqueeze(marginal_node_0,dim=-2)*torch.unsqueeze(marginal_node_0,dim=-1) / me0**2
                marginal_joint_10 = torch.unsqueeze(marginal_node_1,dim=-2)*torch.unsqueeze(marginal_node_0,dim=-1) / (me0*me1)
                marginal_joint_01 = torch.unsqueeze(marginal_node_0,dim=-2)*torch.unsqueeze(marginal_node_1,dim=-1) / (me0*me1)
                marginal_joint_11 = torch.unsqueeze(marginal_node_1,dim=-2)*torch.unsqueeze(marginal_node_1,dim=-1) / me1**2
                marginal_joint_norm = marginal_joint_00 + marginal_joint_01 + marginal_joint_10 + marginal_joint_11
                marginal_joint_00 = marginal_joint_00 / marginal_joint_norm * (1-torch.unsqueeze(torch.unsqueeze(torch.eye(N),dim=0),dim=1))
                marginal_joint_01 = marginal_joint_01 / marginal_joint_norm * (1-torch.unsqueeze(torch.unsqueeze(torch.eye(N),dim=0),dim=1))
                marginal_joint_10 = marginal_joint_10 / marginal_joint_norm * (1-torch.unsqueeze(torch.unsqueeze(torch.eye(N),dim=0),dim=1))
                marginal_joint_11 = marginal_joint_11 / marginal_joint_norm * (1-torch.unsqueeze(torch.unsqueeze(torch.eye(N),dim=0),dim=1))
                
                #estimate the partition function using Bethe approximation
                log_ZB = torch.sum( -marginal_joint_00*torch.log(marginal_joint_00+epsilon)
                                   + marginal_joint_10*(r-torch.log(marginal_joint_10+epsilon)) 
                                   + marginal_joint_01*(r-torch.log(marginal_joint_01+epsilon))
                                   - marginal_joint_11*torch.log(marginal_joint_11+epsilon), dim=(2,3))
                log_ZB += (N-1)*torch.sum(  marginal_node_0*torch.log(marginal_node_0+epsilon)
                                          + marginal_node_1*torch.log(marginal_node_1+epsilon))
                
                #return partition function
                return log_ZB
            
                
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                marginal_logit = DenseGraph_MFE_Binary_PytorchNative.apply(d,r)
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
                
            #return the marginal estimate, pushed through the final activation function if need be
            return torch.sigmoid(marginal_logit) if self.activation else marginal_logit
            
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
            r = torch.unsqueeze(r,dim=1)+torch.unsqueeze(r,dim=2) * (1-torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.eye(N),dim=0),dim=1),dim=2))
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP":
                me = DenseGraph_LBP_Categorical_PytorchNative.apply(d,r)
                marginal_logit = d + torch.sum(torch.log(me),dim=2)
                
            #if the testing time type is set to the logarithm of the partition function, use a Bethe approximation (side result of LBP)
            elif self.solve_type == "LPF":
                epsilon = 10**-10
                me = DenseGraph_LBP_Categorical_PytorchNative.apply(d,r)
                
                #estimate the marginals for each node
                marginal_node = torch.exp(d)*torch.prod(me0,dim=2)
                marginal_node /= torch.sum(marginal_node,dim=1)
                
                #estimate the joint marginals for each edge
                marginal_joint = torch.unsqueeze(torch.unsqueeze(marginal_node_0,dim=-2),dim=1)*torch.unsqueeze(torch.unsqueeze(marginal_node_0,dim=-1),dim=2) / (me*torch.permute(me,(0,2,1,3,4)))
                marginal_joint = marginal_joint / torch.sum(torch.sum(marginal_joint_norm,dim=1,keepdim=True),dim=2,keepdim=True) * (1-torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.eye(N),dim=0),dim=1),dim=2))
                
                #estimate the partition function using Bethe approximation
                log_ZB = torch.sum( -marginal_joint_00*torch.log(marginal_joint_00+epsilon)
                                   + marginal_joint_10*(r-torch.log(marginal_joint_10+epsilon)) 
                                   + marginal_joint_01*(r-torch.log(marginal_joint_01+epsilon))
                                   - marginal_joint_11*torch.log(marginal_joint_11+epsilon), dim=(2,3))
                log_ZB += (N-1)*torch.sum(  marginal_node_0*torch.log(marginal_node_0+epsilon)
                                          + marginal_node_1*torch.log(marginal_node_1+epsilon))
                
                #return partition function
                return log_ZB
            
                
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                marginal_logit = DenseGraph_MFE_Categorical_PytorchNative.apply(d,r)
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
                
            #return the marginal estimate, pushed through the final activation function if need be
            return torch.softmax(marginal_logit,dim=1) if self.activation else marginal_logit
            
        elif self.test_type == "MAP":
            #transform Potts regularisation to more general form (for MAP solvers)
            c, N = d.shape[1], d.shape[2]
            r = torch.unsqueeze(r,dim=2)*torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.eye(c),dim=0),dim=-1),dim=-1)*(1-torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.eye(N),dim=0),dim=1),dim=2))
            
            return DenseGraph_MAP_Categorical_PytorchNative.apply(d,r)
            
        else:
            raise Exception("Uknown test-time solve type: " + str(self.solve_type))
            