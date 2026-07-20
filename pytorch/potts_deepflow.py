import torch
from potts_deepflow_function import Potts_MAP1d, Potts_Mean1d, Potts_Mean1d_PytorchNative, Potts_LBP1d_PytorchNative
from potts_deepflow_function import Potts_MAP2d, Potts_Mean2d, Potts_Mean2d_PytorchNative, Potts_LBP2d_PytorchNative
from potts_deepflow_function import Potts_MAP3d, Potts_Mean3d, Potts_Mean3d_PytorchNative, Potts_LBP3d_PytorchNative
import math

from torch.nn.functional import pad

class PottsDeepFlow1d(torch.nn.Module):
    def __init__(self, solve_type="LBP",test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        
    def forward(self,d,rx):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            b = d.shape[0]
            c = d.shape[1]
            eye = torch.eye(c).view(1,c,c,1).to(d.device)
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP":
                mexu, mexd = Potts_LBP1d_PytorchNative.apply(d,rx)
                
                #calculate the marginal in a less expensive way (still need to normalise)
                marginal_node = torch.exp(d) * pad(mexd, (0,1), value=1) * pad(mexu, (1,0), value=1)
                #marginal_node[:,:,:-1] *= mexd
                #marginal_node[:,:,1:] *= mexu
                am = torch.sum(marginal_node,dim=1,keepdim=True)
                
                #estimate the joint marginals for each edge (can use the unnormalised marginals to speed things up, assuming all things are non-zero)
                erx = torch.exp(-(torch.unsqueeze(rx,dim=1)+torch.unsqueeze(rx,dim=2))*(1-eye))
                marginal_joint_x_pre = erx[:,:,:,:-1] * torch.unsqueeze(marginal_node[:,:,:-1],dim=2) * torch.unsqueeze(marginal_node[:,:,1:],dim=1) / (torch.unsqueeze(mexu,dim=1)*torch.unsqueeze(mexd,dim=2))
                marginal_joint_x = marginal_joint_x_pre / torch.sum(marginal_joint_x_pre,dim=(1,2),keepdim=True)
                marginal_nonmatch_all_x = marginal_joint_x * (1-eye)
                marginal_nonmatch_x = torch.sum(marginal_nonmatch_all_x,dim=1) + torch.sum(marginal_nonmatch_all_x,dim=2)
                
                #assert(self.solve_type == "FULL_MARGINALS")
                return marginal_node / am, marginal_nonmatch_x
                
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                energy = Potts_Mean1d_PytorchNative.apply(d,rx)
                marginal = torch.softmax(energy,1)
                energy_joint_all = (torch.unsqueeze(energy[:,:,:-1],2) + torch.unsqueeze(energy[:,:,1:],1)
                                      - 2*torch.unsqueeze(rx[:,:,:-1],1)*(torch.unsqueeze(marginal[:,:,1:],2) - 1) 
                                      - 2*torch.unsqueeze(rx[:,:,:-1],2)*(torch.unsqueeze(marginal[:,:,:-1],1) - 1) 
                                      + 2*torch.unsqueeze(rx[:,:,:-1],1)*eye)
                marginal_joint_all = torch.softmax(energy_joint_all.view(b,c*c,-1),1).view(b,c,c,-1)
                marginal_match_x = torch.sum(marginal_joint_all * eye,dim=1)
                marginal_nonmatch = torch.sum(marginal_joint_all,1)+torch.sum(marginal_joint_all,2)-2*marginal_match_x
                return marginal, marginal_nonmatch
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
            
        elif self.test_type == "MAP":
            return Potts_MAP1d.apply(d,rx)
            
        else:
            raise Exception("Uknown test-time solve type: " + str(self.solve_type))
            
            
class PottsDeepFlow2d(torch.nn.Module):
    def __init__(self, solve_type="LBP",test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        
    def forward(self,d,rx,ry):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            b = d.shape[0]
            c = d.shape[1]
            sjx = d.shape[2]
            sjy = d.shape[3]
            eye = torch.eye(c).view(1,c,c,1,1).to(d.device)
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP":
                mexu, mexd, meyu, meyd = Potts_LBP2d_PytorchNative.apply(d,rx,ry)
                
                #calculate the marginal in a less expensive way (still need to normalise)
                marginal_node = torch.exp(d) * pad(mexd,(0,0,0,1),value=1) * pad(mexu,(0,0,1,0),value=1) * pad(meyd,(0,1,0,0),value=1) * pad(meyu,(1,0,0,0),value=1)
                #marginal_node[:,:,:-1,:] *= mexd
                #marginal_node[:,:,1:,:] *= mexu
                #marginal_node[:,:,:,:-1] *= meyd
                #marginal_node[:,:,:,1:] *= meyu
                am = torch.sum(marginal_node,dim=1,keepdim=True)
                
                #estimate the joint marginals for each edge (can use the unnormalised marginals to speed things up, assuming all things are non-zero)
                erx = torch.exp(-(torch.unsqueeze(rx,dim=1)+torch.unsqueeze(rx,dim=2))*(1-eye))
                marginal_joint_x_pre = erx[:,:,:,:-1,:] * torch.unsqueeze(marginal_node[:,:,:-1,:],dim=2) * torch.unsqueeze(marginal_node[:,:,1:,:],dim=1) / (torch.unsqueeze(mexu,dim=1)*torch.unsqueeze(mexd,dim=2))
                marginal_joint_x = marginal_joint_x_pre / torch.sum(marginal_joint_x_pre,dim=(1,2),keepdim=True)
                marginal_match_x = torch.sum(marginal_joint_x * eye,dim=1)
                marginal_nonmatch_x = torch.sum(marginal_joint_x,dim=1) + torch.sum(marginal_joint_x,dim=2) - 2*marginal_match_x
                ery = torch.exp(-(torch.unsqueeze(ry,dim=1)+torch.unsqueeze(ry,dim=2))*(1-eye))
                marginal_joint_y_pre = ery[:,:,:,:,:-1] * torch.unsqueeze(marginal_node[:,:,:,:-1],dim=2) * torch.unsqueeze(marginal_node[:,:,:,1:],dim=1) / (torch.unsqueeze(meyu,dim=1)*torch.unsqueeze(meyd,dim=2))
                marginal_joint_y = marginal_joint_y_pre / torch.sum(marginal_joint_y_pre,dim=(1,2),keepdim=True)
                marginal_match_y = torch.sum(marginal_joint_y * eye,dim=1)
                marginal_nonmatch_y = torch.sum(marginal_joint_y,dim=1) + torch.sum(marginal_joint_y,dim=2) - 2*marginal_match_y
                
                #assert(self.solve_type == "FULL_MARGINALS")
                return marginal_node / am, marginal_nonmatch_x, marginal_nonmatch_y
                
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                energy = Potts_Mean2d_PytorchNative.apply(d,rx,ry)
                marginal = torch.softmax(energy,1)
                energy_joint_all = (torch.unsqueeze(energy[:,:,:-1,:],2) + torch.unsqueeze(energy[:,:,1:,:],1)
                                      - 2*torch.unsqueeze(rx[:,:,:-1,:],1)*(torch.unsqueeze(marginal[:,:,1:,:],2) - 1) 
                                      - 2*torch.unsqueeze(rx[:,:,:-1,:],2)*(torch.unsqueeze(marginal[:,:,:-1,:],1) - 1) 
                                      + 2*torch.unsqueeze(rx[:,:,:-1,:],1)*eye)
                marginal_joint_all = torch.softmax(energy_joint_all.view(b,c*c,sjx-1,sjy),1).view(b,c,c,sjx-1,sjy)
                marginal_nonmatch_all = marginal_joint_all - marginal_joint_all*eye
                marginal_nonmatch_x = torch.sum(marginal_nonmatch_all,1)+torch.sum(marginal_nonmatch_all,2)
                energy_joint_all = (torch.unsqueeze(energy[:,:,:,:-1],2) + torch.unsqueeze(energy[:,:,:,1:],1)
                                      - 2*torch.unsqueeze(ry[:,:,:,:-1],1)*(torch.unsqueeze(marginal[:,:,:,1:],2) - 1) 
                                      - 2*torch.unsqueeze(ry[:,:,:,:-1],2)*(torch.unsqueeze(marginal[:,:,:,:-1],1) - 1) 
                                      + 2*torch.unsqueeze(ry[:,:,:,:-1],1)*eye)
                marginal_joint_all = torch.softmax(energy_joint_all.view(b,c*c,sjx,sjy-1),1).view(b,c,c,sjx,sjy-1)
                marginal_nonmatch_all = marginal_joint_all - marginal_joint_all*eye
                marginal_nonmatch_y = torch.sum(marginal_nonmatch_all,1)+torch.sum(marginal_nonmatch_all,2)
                return marginal, marginal_nonmatch_x, marginal_nonmatch_y
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
            
        elif self.test_type == "MAP":
            return Potts_MAP2d.apply(d,rx,ry)
            
        else:
            raise Exception("Uknown test-time solve type: " + str(self.solve_type))
            
            
class PottsDeepFlow3d(torch.nn.Module):
    def __init__(self, solve_type="LBP",test_type="MAP") -> None:
        super().__init__()
        self.solve_type = solve_type
        self.test_type = test_type
        
    def forward(self,d,rx,ry,rz):
        
        #if we are in training or only "soft" mode, always use training solvers
        if self.training or self.test_type is None:
            b = d.shape[0]
            c = d.shape[1]
            sjx = d.shape[2]
            sjy = d.shape[3]
            sjz = d.shape[3]
            eye = torch.eye(c).view(1,c,c,1,1,1).to(d.device)
            
            #if we are in parallel loopy belief propagation, get the final converged messages and calculate the marginal
            if self.solve_type == "LBP":
                mexu, mexd, meyu, meyd, mezu, mezd = Potts_LBP3d_PytorchNative.apply(d,rx,ry,rz)
                
                #calculate the marginal in a less expensive way (still need to normalise)
                marginal_node = torch.exp(d) * pad(mexd,(0,0,0,0,0,1),value=1) * pad(mexu,(0,0,0,0,1,0),value=1) * pad(meyd,(0,0,0,1,0,0),value=1) * pad(meyu,(0,0,1,0,0,0),value=1) * pad(mezd,(0,1,0,0,0,0),value=1) * pad(mezu,(1,0,0,0,0,0),value=1)
                #marginal_node = torch.exp(d)
                #marginal_node[:,:,:-1,:,:] *= mexd
                #marginal_node[:,:,1:,:,:] *= mexu
                #marginal_node[:,:,:,:-1,:] *= meyd
                #marginal_node[:,:,:,1:,:] *= meyu
                #marginal_node[:,:,:,:,:-1] *= mezd
                #marginal_node[:,:,:,:,1:] *= mezu
                am = torch.sum(marginal_node,dim=1,keepdim=True)
                
                #estimate the joint marginals for each edge (can use the unnormalised marginals to speed things up, assuming all things are non-zero)
                erx = torch.exp(-(torch.unsqueeze(rx,dim=1)+torch.unsqueeze(rx,dim=2))*(1-eye))
                marginal_joint_x_pre = erx[:,:,:,:-1,:,:] * torch.unsqueeze(marginal_node[:,:,:-1,:,:],dim=2) * torch.unsqueeze(marginal_node[:,:,1:,:,:],dim=1) / (torch.unsqueeze(mexu,dim=1)*torch.unsqueeze(mexd,dim=2))
                marginal_joint_x = marginal_joint_x_pre / torch.sum(marginal_joint_x_pre,dim=(1,2),keepdim=True)
                marginal_match_x = torch.sum(marginal_joint_x * eye,dim=1)
                marginal_nonmatch_x = torch.sum(marginal_joint_x,dim=1) + torch.sum(marginal_joint_x,dim=2) - 2*marginal_match_x
                ery = torch.exp(-(torch.unsqueeze(ry,dim=1)+torch.unsqueeze(ry,dim=2))*(1-eye))
                marginal_joint_y_pre = ery[:,:,:,:,:-1,:] * torch.unsqueeze(marginal_node[:,:,:,:-1,:],dim=2) * torch.unsqueeze(marginal_node[:,:,:,1:,:],dim=1) / (torch.unsqueeze(meyu,dim=1)*torch.unsqueeze(meyd,dim=2))
                marginal_joint_y = marginal_joint_y_pre / torch.sum(marginal_joint_y_pre,dim=(1,2),keepdim=True)
                marginal_match_y = torch.sum(marginal_joint_y * eye,dim=1)
                marginal_nonmatch_y = torch.sum(marginal_joint_y,dim=1) + torch.sum(marginal_joint_y,dim=2) - 2*marginal_match_y
                erz = torch.exp(-(torch.unsqueeze(rz,dim=1)+torch.unsqueeze(rz,dim=2))*(1-eye))
                marginal_joint_z_pre = erz[:,:,:,:,:,:-1] * torch.unsqueeze(marginal_node[:,:,:,:,:-1],dim=2) * torch.unsqueeze(marginal_node[:,:,:,:,1:],dim=1) / (torch.unsqueeze(mezu,dim=1)*torch.unsqueeze(mezd,dim=2))
                marginal_joint_z = marginal_joint_z_pre / torch.sum(marginal_joint_z_pre,dim=(1,2),keepdim=True)
                marginal_match_z = torch.sum(marginal_joint_z * eye,dim=1)
                marginal_nonmatch_z = torch.sum(marginal_joint_z,dim=1) + torch.sum(marginal_joint_z,dim=2) - 2*marginal_match_z
                
                #assert(self.solve_type == "FULL_MARGINALS")
                return marginal_node / am, marginal_nonmatch_x, marginal_nonmatch_y, marginal_nonmatch_z
                
            #if we are in mean field estimation, get the marginal estimate directly
            elif self.solve_type == "MFE":
                energy = Potts_Mean3d_PytorchNative.apply(d,rx,ry,rz)
                marginal = torch.softmax(energy,1)
                energy_joint_all = (torch.unsqueeze(energy[:,:,:-1,:,:],2) + torch.unsqueeze(energy[:,:,1:,:,:],1)
                                      - 2*torch.unsqueeze(rx[:,:,:-1,:,:],1)*(torch.unsqueeze(marginal[:,:,1:,:,:],2) - 1) 
                                      - 2*torch.unsqueeze(rx[:,:,:-1,:,:],2)*(torch.unsqueeze(marginal[:,:,:-1,:,:],1) - 1) 
                                      + 2*torch.unsqueeze(rx[:,:,:-1,:,:],1)*eye)
                marginal_joint_all = torch.softmax(energy_joint_all.view(b,c*c,sjx-1,sjy,sjz),1).view(b,c,c,sjx-1,sjy,sjz)
                marginal_nonmatch_all = marginal_joint_all - marginal_joint_all*eye
                marginal_nonmatch_x = torch.sum(marginal_nonmatch_all,1)+torch.sum(marginal_nonmatch_all,2)
                energy_joint_all = (torch.unsqueeze(energy[:,:,:,:-1,:],2) + torch.unsqueeze(energy[:,:,:,1:,:],1)
                                      - 2*torch.unsqueeze(ry[:,:,:,:-1,:],1)*(torch.unsqueeze(marginal[:,:,:,1:,:],2) - 1) 
                                      - 2*torch.unsqueeze(ry[:,:,:,:-1,:],2)*(torch.unsqueeze(marginal[:,:,:,:-1,:],1) - 1) 
                                      + 2*torch.unsqueeze(ry[:,:,:,:-1,:],1)*eye)
                marginal_joint_all = torch.softmax(energy_joint_all.view(b,c*c,sjx,sjy-1,sjz),1).view(b,c,c,sjx,sjy-1,sjz)
                marginal_nonmatch_all = marginal_joint_all - marginal_joint_all*eye
                marginal_nonmatch_y = torch.sum(marginal_nonmatch_all,1)+torch.sum(marginal_nonmatch_all,2)
                energy_joint_all = (torch.unsqueeze(energy[:,:,:,:,:-1],2) + torch.unsqueeze(energy[:,:,:,:,1:],1)
                                      - 2*torch.unsqueeze(rz[:,:,:,:,:-1],1)*(torch.unsqueeze(marginal[:,:,:,:,1:],2) - 1) 
                                      - 2*torch.unsqueeze(rz[:,:,:,:,:-1],2)*(torch.unsqueeze(marginal[:,:,:,:,:-1],1) - 1) 
                                      + 2*torch.unsqueeze(rz[:,:,:,:,:-1],1)*eye)
                marginal_joint_all = torch.softmax(energy_joint_all.view(b,c*c,sjx,sjy,sjz-1),1).view(b,c,c,sjx,sjy,sjz-1)
                marginal_nonmatch_all = marginal_joint_all - marginal_joint_all*eye
                marginal_nonmatch_z = torch.sum(marginal_nonmatch_all,1)+torch.sum(marginal_nonmatch_all,2)
                return marginal, marginal_nonmatch_x, marginal_nonmatch_y, marginal_nonmatch_z
                
            #throw error if solution type is unknown
            else:
                raise Exception("Uknown solve type: " + str(self.solve_type))
            
        elif self.test_type == "MAP":
            return Potts_MAP3d.apply(d,rx,ry,rz)
            
        else:
            raise Exception("Uknown test-time solve type: " + str(self.solve_type))