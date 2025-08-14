import torch
from binary_deepflow_function import Binary_MAP1d, Binary_Mean1d, Binary_Mean1d_PytorchNative, Binary_LBP1d_PytorchNative
from binary_deepflow_function import Binary_MAP2d, Binary_Mean2d, Binary_Mean2d_PytorchNative, Binary_LBP2d_PytorchNative
from binary_deepflow_function import Binary_MAP3d, Binary_Mean3d, Binary_Mean3d_PytorchNative, Binary_LBP3d_PytorchNative
import math

class BinaryDeepFlow1d(torch.nn.Module):
    def __init__(self, solve_type="LBP",activation=True,test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        self.activation = activation
        
    def forward(self,d,rx):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP" or self.solve_type == "FULL_MARGINALS":
                mexu0,mexu1,mexd0,mexd1 = Binary_LBP1d_PytorchNative.apply(d,rx)
                
                #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
                if self.solve_type == "LBP" and self.activation == False:
                    marginal_logit = d.clone()
                    marginal_logit[:,:,:-1] -= torch.log(mexd0)
                    marginal_logit[:,:,1:] -= torch.log(mexu0)
                    marginal_logit[:,:,:-1] += torch.log(mexd1)
                    marginal_logit[:,:,1:] += torch.log(mexu1)
                    return marginal_logit
                
                #calculate the marginal in a less expensive way (still need to normalise)
                ed = torch.exp(d)
                marginal_node_0 = torch.ones_like(d)
                marginal_node_0[:,:,:-1] *= mexd0
                marginal_node_0[:,:,1:] *= mexu0
                marginal_node_1 = ed.clone()
                marginal_node_1[:,:,:-1] *= mexd1
                marginal_node_1[:,:,1:] *= mexu1
                am = marginal_node_0+marginal_node_1
                if self.solve_type == "LBP" and self.activation == True:
                    return marginal_node_1 / am
                
                #estimate the joint marginals for each edge (can use the unnormalised marginals to speed things up, assuming all things are non-zero)
                erx = torch.exp(-rx)
                marginal_joint_x_00 =                marginal_node_0[:,:,:-1] * marginal_node_0[:,:,1:] / (mexu0*mexd0)
                marginal_joint_x_01 = erx[:,:,:-1] * marginal_node_0[:,:,:-1] * marginal_node_1[:,:,1:] / (mexu1*mexd0)
                marginal_joint_x_10 = erx[:,:,:-1] * marginal_node_1[:,:,:-1] * marginal_node_0[:,:,1:] / (mexu0*mexd1)
                marginal_joint_x_11 =                marginal_node_1[:,:,:-1] * marginal_node_1[:,:,1:] / (mexu1*mexd1)
                marginal_joint_x_norm = marginal_joint_x_00 + marginal_joint_x_01 + marginal_joint_x_10 + marginal_joint_x_11
                
                #assert(self.solve_type == "FULL_MARGINALS")
                return marginal_node_1 / am, (marginal_joint_x_01+marginal_joint_x_10) / marginal_joint_x_norm
                
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                marginal_logit = Binary_Mean1d_PytorchNative.apply(d,rx)
                return torch.sigmoid(marginal_logit) if self.activation else marginal_logit
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
            
        elif self.test_type == "MAP":
            return Binary_MAP1d.apply(d,rx)
            
        else:
            raise Exception("Uknown test-time solve type: " + str(self.solve_type))
                
                
                
class BinaryDeepFlow2d(torch.nn.Module):
    def __init__(self, solve_type="LBP",activation=True,test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        self.activation = activation
        
    def forward(self,d,rx,ry):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP" or self.solve_type == "FULL_MARGINALS":
                mexu0,mexu1,mexd0,mexd1,meyu0,meyu1,meyd0,meyd1 = Binary_LBP2d_PytorchNative.apply(d,rx,ry)
                
                #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
                if self.solve_type == "LBP" and self.activation == False:
                    marginal_logit = d.clone()
                    marginal_logit[:,:,:-1,:] -= torch.log(mexd0)
                    marginal_logit[:,:,1:,:] -= torch.log(mexu0)
                    marginal_logit[:,:,:-1,:] += torch.log(mexd1)
                    marginal_logit[:,:,1:,:] += torch.log(mexu1)
                    marginal_logit[:,:,:,:-1] -= torch.log(meyd0)
                    marginal_logit[:,:,:,1:] -= torch.log(meyu0)
                    marginal_logit[:,:,:,:-1] += torch.log(meyd1)
                    marginal_logit[:,:,:,1:] += torch.log(meyu1)
                    return marginal_logit
                
                #estimate the marginals for each node
                ed = torch.exp(d)
                marginal_node_0 = torch.ones_like(d)
                marginal_node_0[:,:,:-1,:] *= mexd0
                marginal_node_0[:,:,1:,:] *= mexu0
                marginal_node_0[:,:,:,:-1] *= meyd0
                marginal_node_0[:,:,:,1:] *= meyu0
                marginal_node_1 = ed.clone()
                marginal_node_1[:,:,:-1,:] *= mexd1
                marginal_node_1[:,:,1:,:] *= mexu1
                marginal_node_1[:,:,:,:-1] *= meyd1
                marginal_node_1[:,:,:,1:] *= meyu1
                am = marginal_node_0+marginal_node_1
                if self.solve_type == "LBP" and self.activation == True:
                    return marginal_node_1 / am
                
                #estimate the joint marginals for each edge (can use the unnormalised marginals to speed things up, assuming all things are non-zero)
                erx = torch.exp(-rx)
                marginal_joint_x_00 =                  marginal_node_0[:,:,:-1,:] * marginal_node_0[:,:,1:,:] / (mexu0*mexd0)
                marginal_joint_x_01 = erx[:,:,:-1,:] * marginal_node_0[:,:,:-1,:] * marginal_node_1[:,:,1:,:] / (mexu1*mexd0)
                marginal_joint_x_10 = erx[:,:,:-1,:] * marginal_node_1[:,:,:-1,:] * marginal_node_0[:,:,1:,:] / (mexu0*mexd1)
                marginal_joint_x_11 =                  marginal_node_1[:,:,:-1,:] * marginal_node_1[:,:,1:,:] / (mexu1*mexd1)
                ery = torch.exp(-ry)
                marginal_joint_y_00 =                  marginal_node_0[:,:,:,:-1] * marginal_node_0[:,:,:,1:] / (meyu0*meyd0)
                marginal_joint_y_01 = ery[:,:,:,:-1] * marginal_node_0[:,:,:,:-1] * marginal_node_1[:,:,:,1:] / (meyu1*meyd0)
                marginal_joint_y_10 = ery[:,:,:,:-1] * marginal_node_1[:,:,:,:-1] * marginal_node_0[:,:,:,1:] / (meyu0*meyd1)
                marginal_joint_y_11 =                  marginal_node_1[:,:,:,:-1] * marginal_node_1[:,:,:,1:] / (meyu1*meyd1)
                marginal_joint_x_norm = marginal_joint_x_00 + marginal_joint_x_01 + marginal_joint_x_10 + marginal_joint_x_11
                marginal_joint_y_norm = marginal_joint_y_00 + marginal_joint_y_01 + marginal_joint_y_10 + marginal_joint_y_11
                
                #assert(self.solve_type == "FULL_MARGINALS")
                return marginal_node_1 / am, (marginal_joint_x_01+marginal_joint_x_10) / marginal_joint_x_norm, (marginal_joint_y_01+marginal_joint_y_10) / marginal_joint_y_norm
            
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                marginal_logit = Binary_Mean2d_PytorchNative.apply(d,rx,ry)
                return torch.sigmoid(marginal_logit) if self.activation else marginal_logit
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
                
            
        elif self.test_type == "MAP":
            return Binary_MAP2d.apply(d,rx,ry)
            
        else:
                raise Exception("Uknown test-time solve type: " + str(self.solve_type))
                
                
                
class BinaryDeepFlow3d(torch.nn.Module):
    def __init__(self, solve_type="LBP",activation=True,test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        self.activation = activation
        
    def forward(self,d,rx,ry):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP" or self.solve_type == "FULL_MARGINALS":
                mexu0,mexu1,mexd0,mexd1,meyu0,meyu1,meyd0,meyd1,mezu0,mezu1,mezd0,mezd1 = Binary_LBP3d_PytorchNative.apply(d,rx,ry,rz)
                
                #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
                if self.solve_type == "LBP" and self.activation == False:
                    marginal_logit = d.clone()
                    marginal_logit[:,:,:-1,:,:] -= torch.log(mexd0)
                    marginal_logit[:,:,1:,:,:] -= torch.log(mexu0)
                    marginal_logit[:,:,:-1,:,:] += torch.log(mexd1)
                    marginal_logit[:,:,1:,:,:] += torch.log(mexu1)
                    marginal_logit[:,:,:,:-1,:] -= torch.log(meyd0)
                    marginal_logit[:,:,:,1:,:] -= torch.log(meyu0)
                    marginal_logit[:,:,:,:-1,:] += torch.log(meyd1)
                    marginal_logit[:,:,:,1:,:] += torch.log(meyu1)
                    marginal_logit[:,:,:,:,:-1] -= torch.log(mezd0)
                    marginal_logit[:,:,:,:,1:] -= torch.log(mezu0)
                    marginal_logit[:,:,:,:,:-1] += torch.log(mezd1)
                    marginal_logit[:,:,:,:,1:] += torch.log(mezu1)
                    return marginal_logit
                
                #estimate the marginals for each node
                ed = torch.exp(d)
                marginal_node_0 = torch.ones_like(d)
                marginal_node_0[:,:,:-1,:,:] *= mexd0
                marginal_node_0[:,:,1:,:,:] *= mexu0
                marginal_node_0[:,:,:,:-1,:] *= meyd0
                marginal_node_0[:,:,:,1:,:] *= meyu0
                marginal_node_0[:,:,:,:,:-1] *= mezd0
                marginal_node_0[:,:,:,:,1:] *= mezu0
                marginal_node_1 = ed.clone()
                marginal_node_1[:,:,:-1,:,:] *= mexd1
                marginal_node_1[:,:,1:,:,:] *= mexu1
                marginal_node_1[:,:,:,:-1,:] *= meyd1
                marginal_node_1[:,:,:,1:,:] *= meyu1
                marginal_node_1[:,:,:,:,:-1] *= mezd1
                marginal_node_1[:,:,:,:,1:] *= mezu1
                am = marginal_node_0+marginal_node_1
                if self.solve_type == "LBP" and self.activation == True:
                    return marginal_node_1 / am
                
                #estimate the joint marginals for each edge (can use the unnormalised marginals to speed things up, assuming all things are non-zero)
                erx = torch.exp(-rx)
                marginal_joint_x_00 =                    marginal_node_0[:,:,:-1,:,:] * marginal_node_0[:,:,1:,:,:] / (mexu0*mexd0)
                marginal_joint_x_01 = erx[:,:,:-1,:,:] * marginal_node_0[:,:,:-1,:,:] * marginal_node_1[:,:,1:,:,:] / (mexu1*mexd0)
                marginal_joint_x_10 = erx[:,:,:-1,:,:] * marginal_node_1[:,:,:-1,:,:] * marginal_node_0[:,:,1:,:,:] / (mexu0*mexd1)
                marginal_joint_x_11 =                    marginal_node_1[:,:,:-1,:,:] * marginal_node_1[:,:,1:,:,:] / (mexu1*mexd1)
                ery = torch.exp(-ry)
                marginal_joint_y_00 =                    marginal_node_0[:,:,:,:-1,:] * marginal_node_0[:,:,:,1:,:] / (meyu0*meyd0)
                marginal_joint_y_01 = ery[:,:,:,:-1,:] * marginal_node_0[:,:,:,:-1,:] * marginal_node_1[:,:,:,1:,:] / (meyu1*meyd0)
                marginal_joint_y_10 = ery[:,:,:,:-1,:] * marginal_node_1[:,:,:,:-1,:] * marginal_node_0[:,:,:,1:,:] / (meyu0*meyd1)
                marginal_joint_y_11 =                    marginal_node_1[:,:,:,:-1,:] * marginal_node_1[:,:,:,1:,:] / (meyu1*meyd1)
                erz = torch.exp(-rz)
                marginal_joint_z_00 =                    marginal_node_0[:,:,:,:,:-1] * marginal_node_0[:,:,:,:,1:] / (mezu0*mezd0)
                marginal_joint_z_01 = ery[:,:,:,:,:-1] * marginal_node_0[:,:,:,:,:-1] * marginal_node_1[:,:,:,:,1:] / (mezu1*mezd0)
                marginal_joint_z_10 = ery[:,:,:,:,:-1] * marginal_node_1[:,:,:,:,:-1] * marginal_node_0[:,:,:,:,1:] / (mezu0*mezd1)
                marginal_joint_z_11 =                    marginal_node_1[:,:,:,:,:-1] * marginal_node_1[:,:,:,:,1:] / (mezu1*mezd1)
                marginal_joint_x_norm = marginal_joint_x_00 + marginal_joint_x_01 + marginal_joint_x_10 + marginal_joint_x_11
                marginal_joint_y_norm = marginal_joint_y_00 + marginal_joint_y_01 + marginal_joint_y_10 + marginal_joint_y_11
                marginal_joint_z_norm = marginal_joint_z_00 + marginal_joint_z_01 + marginal_joint_z_10 + marginal_joint_z_11
                
                #assert(self.solve_type == "FULL_MARGINALS")
                return marginal_node_1 / am, (marginal_joint_x_01+marginal_joint_x_10) / marginal_joint_x_norm, (marginal_joint_y_01+marginal_joint_y_10) / marginal_joint_y_norm, (marginal_joint_z_01+marginal_joint_z_10) / marginal_joint_z_norm
            
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                marginal_logit = Binary_Mean3d_PytorchNative.apply(d,rx,ry,rz)
                return torch.sigmoid(marginal_logit) if self.activation else marginal_logit
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
            
        elif self.test_type == "MAP":
            return Binary_MAP3d.apply(d,rx,ry,rz)
            
        else:
                raise Exception("Uknown test-time solve type: " + str(self.solve_type))