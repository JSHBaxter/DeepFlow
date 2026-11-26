import math
from binary_lbp_utils import BinaryLBPOptions

import torch
from binary_deepflow_function import (
    Binary_LBP1d_PytorchNative,
    Binary_LBP2d_PytorchNative,
    Binary_LBP3d_PytorchNative,
    Binary_MAP1d,
    Binary_MAP1dStar,
    Binary_MAP2d,
    Binary_MAP2dStar,
    Binary_MAP3d,
    Binary_MAP3dStar,
    Binary_Mean1d_PytorchNative,
    Binary_Mean2d_PytorchNative,
    Binary_Mean3d_PytorchNative,
    BinaryDeepFlow1dLoss_functional,
    BinaryDeepFlow2dLoss_functional,
    BinaryDeepFlow3dLoss_functional,
)
from torch.nn.functional import pad


class BinaryDeepFlow1dLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = BinaryDeepFlow1dLoss_functional()
    def forward(self, d, rx, marginals_u, marginals_jx, y):
        return self.f.apply(d, rx, marginals_u, marginals_jx, y)

class BinaryDeepFlow2dLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = BinaryDeepFlow2dLoss_functional()
    def forward(self, d, rx, ry, marginals_u, marginals_jx, marginals_jy, y):
        return self.f.apply(d, rx, ry, marginals_u, marginals_jx, marginals_jy, y)
                
class BinaryDeepFlow3dLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = BinaryDeepFlow3dLoss_functional()
    def forward(self, d, rx, ry, rz, marginals_u, marginals_jx, marginals_jy, marginals_jz):
        return self.f.apply(d, rx, ry, rz, marginals_u, marginals_jx, marginals_jy, marginals_jz, y)
    
class BinaryDeepFlow1d(torch.nn.Module):
    def __init__(self, solve_type="LBP",test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        
    def forward(self,d,rx):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP":
                mexu0,mexu1,mexd0,mexd1 = Binary_LBP1d_PytorchNative.apply(d,rx,rx)
                
                marginal_node_0 = torch.exp(-d * (d > 0) ) * pad(mexd0,(0,1),value=1) * pad(mexu0,(1,0),value=1)
                marginal_node_1 = torch.exp( d * (d < 0) ) * pad(mexd1,(0,1),value=1) * pad(mexu1,(1,0),value=1)
                am = marginal_node_0 + marginal_node_1
                marginals_u = marginal_node_1 / am
                
                epsilon = 10**-16
                er = torch.exp(-rx)
                marginal_joint_00 =               marginal_node_0[:,:,:-1] * marginal_node_0[:,:,1:] / (mexu0*mexd0+epsilon)
                marginal_joint_01 = er[:,:,:-1] * marginal_node_0[:,:,:-1] * marginal_node_1[:,:,1:] / (mexu1*mexd0+epsilon)
                marginal_joint_10 = er[:,:,:-1] * marginal_node_1[:,:,:-1] * marginal_node_0[:,:,1:] / (mexu0*mexd1+epsilon)
                marginal_joint_11 =               marginal_node_1[:,:,:-1] * marginal_node_1[:,:,1:] / (mexu1*mexd1+epsilon)
                
                marginal_joint_num = marginal_joint_01 + marginal_joint_10
                marginal_joint_den = marginal_joint_00 + marginal_joint_01 + marginal_joint_10 + marginal_joint_11
                marginals_j = marginal_joint_num / marginal_joint_den
                
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                marginals_u, marginals_j = Binary_Mean1d_PytorchNative.apply(d,rx)
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
                
            return marginals_u, marginals_j
            
        elif self.test_type == "MAP":
            return Binary_MAP1d.apply(d,rx)
            
        else:
            raise Exception("Uknown test-time solve type: " + str(self.solve_type))

class BinaryDeepFlow2d(torch.nn.Module):
    
    def __init__(self, solve_type="LBP",test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        
    def forward(self,d,rx,ry):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP":
                mexu0,mexu1,mexd0,mexd1,meyu0,meyu1,meyd0,meyd1 = Binary_LBP2d_PytorchNative.apply(d,rx,rx,ry,ry)
                
                marginal_node_0 = torch.exp(-d * (d > 0) ) * pad(mexd0,(0,0,0,1),value=1) * pad(mexu0,(0,0,1,0),value=1) * pad(meyd0,(0,1,0,0),value=1) * pad(meyu0,(1,0,0,0),value=1)
                marginal_node_1 = torch.exp( d * (d < 0) ) * pad(mexd1,(0,0,0,1),value=1) * pad(mexu1,(0,0,1,0),value=1) * pad(meyd1,(0,1,0,0),value=1) * pad(meyu1,(1,0,0,0),value=1)
                am = marginal_node_0 + marginal_node_1
                marginals_u = marginal_node_1 / am
                
                epsilon = 10**-16
                erx = torch.exp(-rx)
                ery = torch.exp(-ry)
                marginal_joint_00_x =                  marginal_node_0[:,:,:-1,:] * marginal_node_0[:,:,1:,:] / (mexu0*mexd0+epsilon)
                marginal_joint_01_x = erx[:,:,:-1,:] * marginal_node_0[:,:,:-1,:] * marginal_node_1[:,:,1:,:] / (mexu1*mexd0+epsilon)
                marginal_joint_10_x = erx[:,:,:-1,:] * marginal_node_1[:,:,:-1,:] * marginal_node_0[:,:,1:,:] / (mexu0*mexd1+epsilon)
                marginal_joint_11_x =                  marginal_node_1[:,:,:-1,:] * marginal_node_1[:,:,1:,:] / (mexu1*mexd1+epsilon)
                marginal_joint_00_y =                  marginal_node_0[:,:,:,:-1] * marginal_node_0[:,:,:,1:] / (meyu0*meyd0+epsilon)
                marginal_joint_01_y = ery[:,:,:,:-1] * marginal_node_0[:,:,:,:-1] * marginal_node_1[:,:,:,1:] / (meyu1*meyd0+epsilon)
                marginal_joint_10_y = ery[:,:,:,:-1] * marginal_node_1[:,:,:,:-1] * marginal_node_0[:,:,:,1:] / (meyu0*meyd1+epsilon)
                marginal_joint_11_y =                  marginal_node_1[:,:,:,:-1] * marginal_node_1[:,:,:,1:] / (meyu1*meyd1+epsilon)
                
                marginal_joint_num_x = marginal_joint_01_x + marginal_joint_10_x
                marginal_joint_den_x = marginal_joint_00_x + marginal_joint_01_x + marginal_joint_10_x + marginal_joint_11_x
                marginals_jx = marginal_joint_num_x / marginal_joint_den_x
                marginal_joint_num_y = marginal_joint_01_y + marginal_joint_10_y
                marginal_joint_den_y = marginal_joint_00_y + marginal_joint_01_y + marginal_joint_10_y + marginal_joint_11_y
                marginals_jy = marginal_joint_num_y / marginal_joint_den_y
            
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                marginals_u, marginals_jx, marginals_jy = Binary_Mean2d_PytorchNative.apply(d,rx,ry)
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
            
            return marginals_u, marginals_jx, marginals_jy
            
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
        
    def forward(self,d,rx,ry,rz):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP":
                mexu0,mexu1,mexd0,mexd1,meyu0,meyu1,meyd0,meyd1,mezu0,mezu1,mezd0,mezd1 = Binary_LBP3d_PytorchNative.apply(d,rx,rx,ry,ry,rz,rz)
                
                marginal_node_0 = torch.exp(-d * (d > 0) ) * pad(mexd0,(0,0,0,0,0,1),value=1) * pad(mexu0,(0,0,0,0,1,0),value=1) * pad(meyd0,(0,0,0,1,0,0),value=1) * pad(meyu0,(0,0,1,0,0,0),value=1) * pad(mezd0,(0,1,0,0,0,0),value=1) * pad(mezu0,(1,0,0,0,0,0),value=1)
                marginal_node_1 = torch.exp( d * (d < 0) ) * pad(mexd1,(0,0,0,0,0,1),value=1) * pad(mexu1,(0,0,0,0,1,0),value=1) * pad(meyd1,(0,0,0,1,0,0),value=1) * pad(meyu1,(0,0,1,0,0,0),value=1) * pad(mezd1,(0,1,0,0,0,0),value=1) * pad(mezu1,(1,0,0,0,0,0),value=1)
                am = marginal_node_0 + marginal_node_1
                marginals_u = marginal_node_1 / am
                
                epsilon = 10**-16
                erx = torch.exp(-rx)
                ery = torch.exp(-ry)
                erz = torch.exp(-rz)
                marginal_joint_00_x =                    marginal_node_0[:,:,:-1,:,:] * marginal_node_0[:,:,1:,:,:] / (mexu0*mexd0+epsilon)
                marginal_joint_01_x = erx[:,:,:-1,:,:] * marginal_node_0[:,:,:-1,:,:] * marginal_node_1[:,:,1:,:,:] / (mexu1*mexd0+epsilon)
                marginal_joint_10_x = erx[:,:,:-1,:,:] * marginal_node_1[:,:,:-1,:,:] * marginal_node_0[:,:,1:,:,:] / (mexu0*mexd1+epsilon)
                marginal_joint_11_x =                    marginal_node_1[:,:,:-1,:,:] * marginal_node_1[:,:,1:,:,:] / (mexu1*mexd1+epsilon)
                marginal_joint_00_y =                    marginal_node_0[:,:,:,:-1,:] * marginal_node_0[:,:,:,1:,:] / (meyu0*meyd0+epsilon)
                marginal_joint_01_y = ery[:,:,:,:-1,:] * marginal_node_0[:,:,:,:-1,:] * marginal_node_1[:,:,:,1:,:] / (meyu1*meyd0+epsilon)
                marginal_joint_10_y = ery[:,:,:,:-1,:] * marginal_node_1[:,:,:,:-1,:] * marginal_node_0[:,:,:,1:,:] / (meyu0*meyd1+epsilon)
                marginal_joint_11_y =                    marginal_node_1[:,:,:,:-1,:] * marginal_node_1[:,:,:,1:,:] / (meyu1*meyd1+epsilon)
                marginal_joint_00_z =                    marginal_node_0[:,:,:,:,:-1] * marginal_node_0[:,:,:,:,1:] / (mezu0*mezd0+epsilon)
                marginal_joint_01_z = erz[:,:,:,:,:-1] * marginal_node_0[:,:,:,:,:-1] * marginal_node_1[:,:,:,:,1:] / (mezu1*mezd0+epsilon)
                marginal_joint_10_z = erz[:,:,:,:,:-1] * marginal_node_1[:,:,:,:,:-1] * marginal_node_0[:,:,:,:,1:] / (mezu0*mezd1+epsilon)
                marginal_joint_11_z =                    marginal_node_1[:,:,:,:,:-1] * marginal_node_1[:,:,:,:,1:] / (mezu1*mezd1+epsilon)
                
                marginal_joint_num_x = marginal_joint_01_x + marginal_joint_10_x
                marginal_joint_den_x = marginal_joint_00_x + marginal_joint_01_x + marginal_joint_10_x + marginal_joint_11_x
                marginals_jx = marginal_joint_num_x / marginal_joint_den_x
                marginal_joint_num_y = marginal_joint_01_y + marginal_joint_10_y
                marginal_joint_den_y = marginal_joint_00_y + marginal_joint_01_y + marginal_joint_10_y + marginal_joint_11_y
                marginals_jy = marginal_joint_num_y / marginal_joint_den_y
                marginal_joint_num_z = marginal_joint_01_z + marginal_joint_10_z
                marginal_joint_den_z = marginal_joint_00_z + marginal_joint_01_z + marginal_joint_10_z + marginal_joint_11_z
                marginals_jz = marginal_joint_num_z / marginal_joint_den_z
            
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                marginals_u, marginals_jx, marginals_jy, marginals_jz = Binary_Mean3d_PytorchNative.apply(d,rx,ry,rz)
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
            
            return marginals_u, marginals_jx, marginals_jy, marginals_jz
            
        elif self.test_type == "MAP":
            return Binary_MAP3d.apply(d,rx,ry,rz)
            
        else:
            raise Exception("Uknown test-time solve type: " + str(self.solve_type))
            
    
class BinaryDeepFlow1dStar(torch.nn.Module):
    def __init__(self, test_type="MAP") -> None:
        super().__init__()
        self.test_type = test_type
        
    def forward(self,d,rx,lx):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            mexu0,mexu1,mexd0,mexd1 = Binary_LBP1d_PytorchNative.apply(d, rx + torch.nan_to_num(math.inf*(lx==1),0,math.inf),
                                                                          rx + torch.nan_to_num(math.inf*(lx==-1),0,math.inf))

            marginal_node_0 = torch.exp(-d * (d > 0) ) * pad(mexd0,(0,1),value=1) * pad(mexu0,(1,0),value=1)
            marginal_node_1 = torch.exp( d * (d < 0) ) * pad(mexd1,(0,1),value=1) * pad(mexu1,(1,0),value=1)
            am = marginal_node_0 + marginal_node_1
            marginals_u = marginal_node_1 / am

            epsilon = 10**-16
            eru = torch.exp(-rx) * (lx!=1)
            erd = torch.exp(-rx) * (lx!=-1)
            marginal_joint_00 =                marginal_node_0[:,:,:-1] * marginal_node_0[:,:,1:] / (mexu0*mexd0+epsilon)
            marginal_joint_01 = eru[:,:,:-1] * marginal_node_0[:,:,:-1] * marginal_node_1[:,:,1:] / (mexu1*mexd0+epsilon)
            marginal_joint_10 = erd[:,:,:-1] * marginal_node_1[:,:,:-1] * marginal_node_0[:,:,1:] / (mexu0*mexd1+epsilon)
            marginal_joint_11 =                marginal_node_1[:,:,:-1] * marginal_node_1[:,:,1:] / (mexu1*mexd1+epsilon)

            marginal_joint_num = marginal_joint_01 + marginal_joint_10
            marginal_joint_den = marginal_joint_00 + marginal_joint_01 + marginal_joint_10 + marginal_joint_11
            marginals_j = marginal_joint_num / marginal_joint_den
                
            return marginals_u, marginals_j
            
        elif self.test_type == "MAP":
            return Binary_MAP1dStar.apply(d,rx,lx)
            
        else:
            raise Exception("Uknown test-time solve type: " + str(self.test_type))

class BinaryDeepFlow2dStar(torch.nn.Module):
    
    """Binary star LBP with configurable convergence checks and optional fusion.

    `last_convergence` contains scalar-only forward/backward diagnostics from the
    latest invocation. Compilation is opt-in and may change floating-point rounding.
    `check_every=1, compile_updates=False` reproduces the original check schedule.
    """
    def __init__(self, test_type="MAP", check_every=100, compile_updates=False,
                 record_history=True, report_every=0) -> None:
        super().__init__()
        self.test_type = test_type
        self.lbp_options = BinaryLBPOptions(check_every, compile_updates, record_history, report_every)
        self.last_convergence = {}
        
    def forward(self,d,rx,ry,lx,ly):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            # New dictionaries per call: graphs retain their own diagnostic target.
            self.last_convergence = {
                "forward": {"history": [], "status": "not_run"},
                "backward": {"history": [], "status": "not_run"},
            }
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            mexu0,mexu1,mexd0,mexd1,meyu0,meyu1,meyd0,meyd1 = Binary_LBP2d_PytorchNative.apply(d,rx + torch.nan_to_num(math.inf*(lx==1),0,math.inf),
                                                                                                 rx + torch.nan_to_num(math.inf*(lx==-1),0,math.inf),
                                                                                                 ry + torch.nan_to_num(math.inf*(ly==1),0,math.inf),
                                                                                                 ry + torch.nan_to_num(math.inf*(ly==-1),0,math.inf),
                                                                                                 self.lbp_options, self.last_convergence)

            marginal_node_0 = torch.exp(-d * (d > 0) ) * pad(mexd0,(0,0,0,1),value=1) * pad(mexu0,(0,0,1,0),value=1) * pad(meyd0,(0,1,0,0),value=1) * pad(meyu0,(1,0,0,0),value=1)
            marginal_node_1 = torch.exp( d * (d < 0) ) * pad(mexd1,(0,0,0,1),value=1) * pad(mexu1,(0,0,1,0),value=1) * pad(meyd1,(0,1,0,0),value=1) * pad(meyu1,(1,0,0,0),value=1)
            am = marginal_node_0 + marginal_node_1
            marginals_u = marginal_node_1 / am

            epsilon = 10**-16
            erxu = torch.exp(-rx) * (lx!=-1)
            erxd = torch.exp(-rx) * (lx!=1)
            eryu = torch.exp(-ry) * (ly!=-1)
            eryd = torch.exp(-ry) * (ly!=1)
            marginal_joint_00_x =                   marginal_node_0[:,:,:-1,:] * marginal_node_0[:,:,1:,:] / (mexu0*mexd0+epsilon)
            marginal_joint_01_x = erxu[:,:,:-1,:] * marginal_node_0[:,:,:-1,:] * marginal_node_1[:,:,1:,:] / (mexu1*mexd0+epsilon)
            marginal_joint_10_x = erxd[:,:,:-1,:] * marginal_node_1[:,:,:-1,:] * marginal_node_0[:,:,1:,:] / (mexu0*mexd1+epsilon)
            marginal_joint_11_x =                   marginal_node_1[:,:,:-1,:] * marginal_node_1[:,:,1:,:] / (mexu1*mexd1+epsilon)
            marginal_joint_00_y =                   marginal_node_0[:,:,:,:-1] * marginal_node_0[:,:,:,1:] / (meyu0*meyd0+epsilon)
            marginal_joint_01_y = eryu[:,:,:,:-1] * marginal_node_0[:,:,:,:-1] * marginal_node_1[:,:,:,1:] / (meyu1*meyd0+epsilon)
            marginal_joint_10_y = eryd[:,:,:,:-1] * marginal_node_1[:,:,:,:-1] * marginal_node_0[:,:,:,1:] / (meyu0*meyd1+epsilon)
            marginal_joint_11_y =                   marginal_node_1[:,:,:,:-1] * marginal_node_1[:,:,:,1:] / (meyu1*meyd1+epsilon)

            marginal_joint_num_x = marginal_joint_01_x + marginal_joint_10_x
            marginal_joint_den_x = marginal_joint_00_x + marginal_joint_01_x + marginal_joint_10_x + marginal_joint_11_x
            marginals_jx = marginal_joint_num_x / marginal_joint_den_x
            marginal_joint_num_y = marginal_joint_01_y + marginal_joint_10_y
            marginal_joint_den_y = marginal_joint_00_y + marginal_joint_01_y + marginal_joint_10_y + marginal_joint_11_y
            marginals_jy = marginal_joint_num_y / marginal_joint_den_y
            
            return marginals_u, marginals_jx, marginals_jy
            
        elif self.test_type == "MAP":
            return Binary_MAP2dStar.apply(d,rx,ry,lx,ly)
            
        else:
            raise Exception("Uknown test-time solve type: " + str(self.test_type))


class BinaryDeepFlow3dStar(torch.nn.Module):
    def __init__(self, activation=True,test_type="MAP") -> None:
        super().__init__()
        self.test_type = test_type
        self.activation = activation
        
    def forward(self,d,rx,ry,rz,lx,ly,lz):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            mexu0,mexu1,mexd0,mexd1,meyu0,meyu1,meyd0,meyd1,mezu0,mezu1,mezd0,mezd1 = Binary_LBP3d_PytorchNative.apply(d,rx + torch.nan_to_num(math.inf*(lx==1),0,math.inf),
                                                                                                                         rx + torch.nan_to_num(math.inf*(lx==-1),0,math.inf),
                                                                                                                         ry + torch.nan_to_num(math.inf*(ly==1),0,math.inf),
                                                                                                                         ry + torch.nan_to_num(math.inf*(ly==-1),0,math.inf),
                                                                                                                         rz + torch.nan_to_num(math.inf*(lz==1),0,math.inf),
                                                                                                                         rz + torch.nan_to_num(math.inf*(lz==-1),0,math.inf))

            marginal_node_0 = torch.exp(-d * (d > 0) ) * pad(mexd0,(0,0,0,0,0,1),value=1) * pad(mexu0,(0,0,0,0,1,0),value=1) * pad(meyd0,(0,0,0,1,0,0),value=1) * pad(meyu0,(0,0,1,0,0,0),value=1) * pad(mezd0,(0,1,0,0,0,0),value=1) * pad(mezu0,(1,0,0,0,0,0),value=1)
            marginal_node_1 = torch.exp( d * (d < 0) ) * pad(mexd1,(0,0,0,0,0,1),value=1) * pad(mexu1,(0,0,0,0,1,0),value=1) * pad(meyd1,(0,0,0,1,0,0),value=1) * pad(meyu1,(0,0,1,0,0,0),value=1) * pad(mezd1,(0,1,0,0,0,0),value=1) * pad(mezu1,(1,0,0,0,0,0),value=1)
            am = marginal_node_0 + marginal_node_1
            marginals_u = marginal_node_1 / am

            epsilon = 10**-16
            erxu = torch.exp(-rx) * (lx!=-1)
            erxd = torch.exp(-rx) * (lx!=1)
            eryu = torch.exp(-ry) * (ly!=-1)
            eryd = torch.exp(-ry) * (ly!=1)
            erzu = torch.exp(-rz) * (lz!=-1)
            erzd = torch.exp(-rz) * (lz!=1)
            marginal_joint_00_x =                     marginal_node_0[:,:,:-1,:,:] * marginal_node_0[:,:,1:,:,:] / (mexu0*mexd0+epsilon)
            marginal_joint_01_x = erxu[:,:,:-1,:,:] * marginal_node_0[:,:,:-1,:,:] * marginal_node_1[:,:,1:,:,:] / (mexu1*mexd0+epsilon)
            marginal_joint_10_x = erxd[:,:,:-1,:,:] * marginal_node_1[:,:,:-1,:,:] * marginal_node_0[:,:,1:,:,:] / (mexu0*mexd1+epsilon)
            marginal_joint_11_x =                     marginal_node_1[:,:,:-1,:,:] * marginal_node_1[:,:,1:,:,:] / (mexu1*mexd1+epsilon)
            marginal_joint_00_y =                     marginal_node_0[:,:,:,:-1,:] * marginal_node_0[:,:,:,1:,:] / (meyu0*meyd0+epsilon)
            marginal_joint_01_y = eryu[:,:,:,:-1,:] * marginal_node_0[:,:,:,:-1,:] * marginal_node_1[:,:,:,1:,:] / (meyu1*meyd0+epsilon)
            marginal_joint_10_y = eryd[:,:,:,:-1,:] * marginal_node_1[:,:,:,:-1,:] * marginal_node_0[:,:,:,1:,:] / (meyu0*meyd1+epsilon)
            marginal_joint_11_y =                     marginal_node_1[:,:,:,:-1,:] * marginal_node_1[:,:,:,1:,:] / (meyu1*meyd1+epsilon)
            marginal_joint_00_z =                     marginal_node_0[:,:,:,:,:-1] * marginal_node_0[:,:,:,:,1:] / (mezu0*mezd0+epsilon)
            marginal_joint_01_z = erzu[:,:,:,:,:-1] * marginal_node_0[:,:,:,:,:-1] * marginal_node_1[:,:,:,:,1:] / (mezu1*mezd0+epsilon)
            marginal_joint_10_z = erzd[:,:,:,:,:-1] * marginal_node_1[:,:,:,:,:-1] * marginal_node_0[:,:,:,:,1:] / (mezu0*mezd1+epsilon)
            marginal_joint_11_z =                     marginal_node_1[:,:,:,:,:-1] * marginal_node_1[:,:,:,:,1:] / (mezu1*mezd1+epsilon)

            marginal_joint_num_x = marginal_joint_01_x + marginal_joint_10_x
            marginal_joint_den_x = marginal_joint_00_x + marginal_joint_01_x + marginal_joint_10_x + marginal_joint_11_x
            marginals_jx = marginal_joint_num_x / marginal_joint_den_x
            marginal_joint_num_y = marginal_joint_01_y + marginal_joint_10_y
            marginal_joint_den_y = marginal_joint_00_y + marginal_joint_01_y + marginal_joint_10_y + marginal_joint_11_y
            marginals_jy = marginal_joint_num_y / marginal_joint_den_y
            marginal_joint_num_z = marginal_joint_01_z + marginal_joint_10_z
            marginal_joint_den_z = marginal_joint_00_z + marginal_joint_01_z + marginal_joint_10_z + marginal_joint_11_z
            marginals_jz = marginal_joint_num_z / marginal_joint_den_z
            
            return marginals_u, marginals_jx, marginals_jy, marginals_jz
            
        elif self.test_type == "MAP":
            return Binary_MAP3dStar.apply(d,rx,ry,rz,lx,ly,lz)
            
        else:
            raise Exception("Uknown test-time solve type: " + str(self.test_type))
            
