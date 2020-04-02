import torch
from torch import *
from torch.nn import *
from torch.autograd import Function

class BinaryAuglag3dFun(Function):
    @staticmethod
    def forward(ctx, data, rx, ry, rz):
        epsilon = 1e-3
        tau = 0.1
        icc = 0.25
        cc = 1.0 / icc
        inner_its = data.shape[2]+data.shape[3]+data.shape[4]
        
        cs = torch.clamp(data, min=0)
        ct = -torch.clamp(data, max=0)
        ps = torch.zeros_like(rx)
        pt = torch.zeros_like(rx)
        px = torch.zeros_like(rx)
        py = torch.zeros_like(rx)
        pz = torch.zeros_like(rx)
        div = torch.zeros_like(rx)
        g = torch.zeros_like(rx)
        u = torch.sigmoid(data)
        
        for iter in range(100):
            
            #calculate capacities and flow updates
            g = tau * (div + pt - ps - u * icc)
            px[:, :, :-1, :, :] += g[:, :, 1:, :, :] - g[:, :, :-1, :, :]
            px = torch.max(torch.min(px, rx), -rx)
            py[:, :, :, :-1, :] += g[:, :, :, 1:, :] - g[:, :, :, :-1, :]
            py = torch.max(torch.min(py, ry), -ry)
            pz[:, :, :, :, :-1] += g[:, :, :, :, 1:] - g[:, :, :, :, :-1]
            pz = torch.max(torch.min(pz, rz), -rz)
            
            # update divergence
            div = px + py + pz
            div[:,:,1:,:,:] -= px[:, :, :-1, :, :]
            div[:,:,:,1:,:] -= py[:, :, :, :-1, :]
            div[:,:,:,:,1:] -= pz[:, :, :, :, :-1]
         
            # update the source and sink flows and Lagrangian
            ps = torch.min(icc + pt + div - u * icc, cs)
            pt = torch.min(ps - div + u * icc, ct)
            erru = cc * (div - ps + pt)
            u = u - erru

            # convergence criteria and error check
            if torch.max(erru) < epsilon and torch.min(erru) > -epsilon:
                break

            if (torch.max(torch.isnan(u)) > 0):
                raise Exception("Invalid labeling detected")
            
        return torch.clamp(u, 0.0, 1.0)

class BinaryMeanpass3dFun(Function):
    @staticmethod
    def forward(ctx, data, rx, ry, rz):
        damper = 0.5
        epsilon = 1e-2
        inner_its = data.shape[2]+data.shape[3]+data.shape[4]
        
        m = BinaryAuglag3dFun.apply(data, rx, ry, rz)
        
        #simple forward iteration of the algorithm
        def calculate_marginal_energy():
            marginal_energy = data.clone()
            marginal_energy[:,:, :-1, :,:] += rx[:,:, :-1, :,:] * (2.0*m[:,:, 1:, :,:]-1.0)
            marginal_energy[:,:, 1:, :,:] += rx[:,:, :-1, :,:] * (2.0*m[:,:, :-1, :,:]-1.0)
            marginal_energy[:,:,:, :-1, :] += ry[:,:,:, :-1, :] * (2.0*m[:,:,:, 1:, :]-1.0)
            marginal_energy[:,:,:, 1:, :] += ry[:,:,:, :-1, :] * (2.0*m[:,:,:, :-1, :]-1.0)
            marginal_energy[:,:,:,:, :-1] += rz[:,:,:,:, :-1] * (2.0*m[:,:,:,:, 1:]-1.0)
            marginal_energy[:,:,:,:, 1:] += rz[:,:,:,:, :-1] * (2.0*m[:,:,:,:, :-1]-1.0)
            return marginal_energy
            
        def iteration():
            new_m = torch.sigmoid(calculate_marginal_energy())
            return new_m
        
        #run until converged (checking ever so and so iterations)
        #then run extra set of so and so iterations to ensure it has stabilized fully
        for j in range(100):
            for i in range(inner_its):
                new_m = iteration()
                m = damper*m + (1-damper)*new_m
            diff = torch.max(torch.abs(new_m-m))
            if diff < epsilon:
                break
        for i in range(inner_its):
            m = damper*m + (1-damper)*iteration()
        m = calculate_marginal_energy()
            
        #save vars for backwards pass
        ctx.save_for_backward(rx.clone(),ry.clone(),rz.clone(),m.clone())
        
        return m
    
    @staticmethod
    def backward(ctx, grad_output):
        damper = 0.5
        epsilon = 1e-2 * torch.max(torch.max(grad_output),-torch.min(grad_output))
        rx, ry, rz, mi = ctx.saved_tensors
        inner_its = 1#mi.shape[2]+mi.shape[3]+mi.shape[4]
        m = torch.sigmoid(mi)
        grad_rx = torch.zeros_like(rx)
        grad_ry = torch.zeros_like(rx)
        grad_rz = torch.zeros_like(rx)
        grad_m = torch.zeros_like(grad_output)
        
        #backprop through first iteration (note no damper or sigmoid)
        grad_e = grad_output.clone()
        grad_data = grad_output.clone()
        grad_rx[:,:, :-1, :,:] += grad_e[:,:, :-1, :,:]*(2.0*m[:,:, 1:, :,:]-1.0)
        grad_rx[:,:, :-1, :,:] +=  grad_e[:,:, 1:, :,:] * (2.0*m[:,:, :-1, :,:]-1.0)
        grad_ry[:,:,:, :-1, :] += grad_e[:,:,:, :-1, :] * (2.0*m[:,:,:, 1:, :]-1.0)
        grad_ry[:,:,:, :-1, :] += grad_e[:,:,:, 1:, :] * (2.0*m[:,:,:, :-1, :]-1.0)
        grad_rz[:,:,:,:, :-1] += grad_e[:,:,:,:, :-1] * (2.0*m[:,:,:,:, 1:]-1.0)
        grad_rz[:,:,:,:, :-1] += grad_e[:,:,:,:, 1:] * (2.0*m[:,:,:,:, :-1]-1.0)
        grad_m[:,:, 1:, :,:] += 2.0*rx[:,:, :-1, :,:] * grad_e[:,:, :-1, :,:]
        grad_m[:,:, :-1, :,:] += 2.0*rx[:,:, :-1, :,:] * grad_e[:,:, 1:, :,:]
        grad_m[:,:,:, 1:, :] += 2.0*ry[:,:,:, :-1, :] * grad_e[:,:,:, :-1, :]
        grad_m[:,:,:, :-1, :] += 2.0*ry[:,:,:, :-1, :] * grad_e[:,:,:, 1:, :]
        grad_m[:,:,:,:, 1:] += 2.0*rz[:,:,:,:, :-1] * grad_e[:,:,:,:, :-1]
        grad_m[:,:,:,:, :-1] += 2.0*rz[:,:,:,:, :-1] * grad_e[:,:,:,:, 1:]
        
        def iteration(grad_data, grad_rx, grad_ry, grad_rz, grad_m):
            #prop back through damper and sigmoid to get gradient energy
            grad_e = (1-damper) * m * (1-m) * grad_m
            
            #prop back through energy equation (and rest of damper)
            grad_data += grad_e
            grad_rx[:,:, :-1, :,:] += grad_e[:,:, :-1, :,:]*(2.0*m[:,:, 1:, :,:]-1.0)
            grad_rx[:,:, :-1, :,:] +=  grad_e[:,:, 1:, :,:] * (2.0*m[:,:, :-1, :,:]-1.0)
            grad_ry[:,:,:, :-1, :] += grad_e[:,:,:, :-1, :] * (2.0*m[:,:,:, 1:, :]-1.0)
            grad_ry[:,:,:, :-1, :] += grad_e[:,:,:, 1:, :] * (2.0*m[:,:,:, :-1, :]-1.0)
            grad_rz[:,:,:,:, :-1] += grad_e[:,:,:,:, :-1] * (2.0*m[:,:,:,:, 1:]-1.0)
            grad_rz[:,:,:,:, :-1] += grad_e[:,:,:,:, 1:] * (2.0*m[:,:,:,:, :-1]-1.0)
            grad_m *= damper
            grad_m[:,:, 1:, :,:] += 2.0*rx[:,:, :-1, :,:] * grad_e[:,:, :-1, :,:]
            grad_m[:,:, :-1, :,:] += 2.0*rx[:,:, :-1, :,:] * grad_e[:,:, 1:, :,:]
            grad_m[:,:,:, 1:, :] += 2.0*ry[:,:,:, :-1, :] * grad_e[:,:,:, :-1, :]
            grad_m[:,:,:, :-1, :] += 2.0*ry[:,:,:, :-1, :] * grad_e[:,:,:, 1:, :]
            grad_m[:,:,:,:, 1:] += 2.0*rz[:,:,:,:, :-1] * grad_e[:,:,:,:, :-1]
            grad_m[:,:,:,:, :-1] += 2.0*rz[:,:,:,:, :-1] * grad_e[:,:,:,:, 1:]
            grad_m *= 0.99
            return grad_data.clone(), grad_rx.clone(), grad_ry.clone(), grad_rz.clone(), grad_m.clone()
        
        #run until converged (checking ever so and so iterations)
        for j in range(100):
            for i in range(inner_its):
                grad_data, grad_rx, grad_ry, grad_rz, grad_m = iteration(grad_data, grad_rx, grad_ry, grad_rz, grad_m)
            diff = torch.max(torch.max(grad_m),-torch.min(grad_m))
            if diff < epsilon:
                break
        
        #return gradient approximations
        return grad_data, grad_rx, grad_ry, grad_rz
            

class BinaryMeanpass3d(Module):
    def __init__(self):
        super(BinaryMeanpass3d, self).__init__()
        
    def forward(self, d, rx, ry, rz):
        return BinaryMeanpass3dFun.apply(d, rx, ry, rz)
        
    