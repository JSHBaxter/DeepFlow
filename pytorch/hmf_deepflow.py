import torch
from deepflow_function import DeepFlowFunction
from hmf_deepflow_function import (
    HMF_MAP1d_PytorchNative,
    HMF_MAP2d_PytorchNative,
    HMF_MAP3d_PytorchNative,
    HMF_Mean1d_PytorchNative,
    HMF_Mean2d_PytorchNative,
    HMF_Mean3d_PytorchNative,
    HMF_MeanNd_calculate_joint,
    HMFDeepFlow1dLoss_functional,
    HMFDeepFlow2dLoss_functional,
    HMFDeepFlow3dLoss_functional,
    check_ordering,
    mix_regularisation_terms,
    sum_up_joint_marginals,
)
from unhmf_deepflow_function import (
    UNHMF_LBP1d_PytorchNative,
    UNHMF_LBP2d_PytorchNative,
    UNHMF_LBP3d_PytorchNative,
    UNHMFMarginalFromMessageGradientSafe1D,
    UNHMFMarginalFromMessageGradientSafe2D,
    UNHMFMarginalFromMessageGradientSafe3D,
)


class HMFDeepFlow1dLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = HMFDeepFlow1dLoss_functional()
    def forward(self, d, rx, marginals_u, marginals_jx, y, p):
        return self.f.apply(d, rx, marginals_u, marginals_jx, y, p)

class HMFDeepFlow2dLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = HMFDeepFlow2dLoss_functional()
    def forward(self, d, rx, ry, marginals_u, marginals_jx, marginals_jy, y, p):
        return self.f.apply(d, rx, ry, marginals_u, marginals_jx, marginals_jy, y, p)
                
class HMFDeepFlow3dLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = HMFDeepFlow3dLoss_functional()
    def forward(self, d, rx, ry, rz, marginals_u, marginals_jx, marginals_jy, marginals_jz, y, p):
        return self.f.apply(d, rx, ry, rz, marginals_u, marginals_jx, marginals_jy, marginals_jz, y, p)
    
class HMFDeepFlow1d(torch.nn.Module):
    def __init__(self, solve_type="LBP",test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        
    def forward(self,d,rx,p):
        DeepFlowFunction.check_var_dims([d,rx],1,True)
        b,c,_ = d.shape[0],d.shape[1],d.shape[2]
        check_ordering(p,rx.shape[1],d.shape[1])
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP":
                r_eff = mix_regularisation_terms(p,b,c,d.device,[rx,])
                mexu,mexd = UNHMF_LBP1d_PytorchNative.apply(d,r_eff[0])
                marginals_u, marginals_j = UNHMFMarginalFromMessageGradientSafe1D.apply(d,r_eff[0],
                                                                                        mexd,mexu)
                marginals_j = sum_up_joint_marginals(p,[marginals_j])[0]
                
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                marginals_u = HMF_Mean1d_PytorchNative.apply(d,rx,p)
                marginals_j = HMF_MeanNd_calculate_joint(d,[rx,],marginals_u,p)[0]
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
                
            return marginals_u, marginals_j
            
        elif self.test_type == "MAP":
            return HMF_MAP1d_PytorchNative.apply(d,rx,p)
            
        else:
            raise Exception("Unknown test-time solve type: " + str(self.solve_type))

class HMFDeepFlow2d(torch.nn.Module):
    
    def __init__(self, solve_type="LBP",test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        
    def forward(self,d,rx,ry,p):
        DeepFlowFunction.check_var_dims([d,rx,ry],2,True)
        b,c,_,_ = d.shape[0],d.shape[1],d.shape[2],d.shape[3]
        check_ordering(p,rx.shape[1],d.shape[1])
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP":
                r_eff = mix_regularisation_terms(p,b,c,d.device,[rx,ry,])
                mexu,mexd,meyu,meyd = UNHMF_LBP2d_PytorchNative.apply(d,r_eff[0],r_eff[1])
                marginals_u, marginals_jx, marginals_jy = UNHMFMarginalFromMessageGradientSafe2D.apply(d,r_eff[0],r_eff[1],
                                                                                                       mexd,mexu,meyd,meyu)
                marginals_j = sum_up_joint_marginals(p,[marginals_jx, marginals_jy])
                marginals_jx, marginals_jy = marginals_j[0], marginals_j[1]
            
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                marginals_u = HMF_Mean2d_PytorchNative.apply(d,rx,ry,p)
                marginals_j = HMF_MeanNd_calculate_joint(d,[rx,ry,],marginals_u,p)
                marginals_jx, marginals_jy = marginals_j[0], marginals_j[1]
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
            
            return marginals_u, marginals_jx, marginals_jy
            
        elif self.test_type == "MAP":
            return HMF_MAP2d_PytorchNative.apply(d,rx,ry,p)
            
        else:
            raise Exception("Unknown test-time solve type: " + str(self.solve_type))


class HMFDeepFlow3d(torch.nn.Module):
    def __init__(self, solve_type="LBP",activation=True,test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        self.activation = activation
        
    def forward(self,d,rx,ry,rz,p):
        DeepFlowFunction.check_var_dims([d,rx,ry,rz],3,True)
        b,c,_,_,_ = d.shape[0],d.shape[1],d.shape[2],d.shape[3],d.shape[4]
        check_ordering(p,rx.shape[1],d.shape[1])
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP":
                r_eff = mix_regularisation_terms(p,b,c,d.device,[rx,ry,rz,])
                mexu,mexd,meyu,meyd,mezu,mezd = UNHMF_LBP3d_PytorchNative.apply(d,r_eff[0],r_eff[1],r_eff[2])
                marginals_u, marginals_jx, marginals_jy, marginals_jz = UNHMFMarginalFromMessageGradientSafe3D.apply(d,r_eff[0],r_eff[0],r_eff[0],
                                                                                                                     mexd,mexu,meyd,meyu,mezd,mezu)
                marginals_j = sum_up_joint_marginals(p,[marginals_jx, marginals_jy, marginals_jz])
                marginals_jx, marginals_jy, marginals_jz = marginals_j[0], marginals_j[1], marginals_j[2]
            
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                marginals_u = HMF_Mean3d_PytorchNative.apply(d,rx,ry,rz,p)
                marginals_j = HMF_MeanNd_calculate_joint(d,[rx,ry,rz,],marginals_u,p)
                marginals_jx, marginals_jy, marginals_jz = marginals_j[0], marginals_j[1], marginals_j[2]
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
            
            return marginals_u, marginals_jx, marginals_jy, marginals_jz
            
        elif self.test_type == "MAP":
            return HMF_MAP3d_PytorchNative.apply(d,rx,ry,rz,p)
            
        else:
            raise Exception("Unknown test-time solve type: " + str(self.solve_type))