import torch
import sys
import deepflow

class Potts_MAP1d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx):
        if len(d.shape) == 3 and len(rx.shape) == 3:
            output = torch.zeros_like(d)
            if d.is_cuda:
                deepflow.potts_auglag_1d_gpu_forward(d,rx, output)
            else:
                deepflow.potts_auglag_1d_cpu_forward(d,rx, output)
            return output
        else:
            sys.stderr.write("Gave enough smoothness terms for 1D deepflow, but wrong dimensionality. \n")
            return
            
    #For the optimisers, there is no well defined backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input
        
class Potts_MAP2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry):
        if len(d.shape) == 4 and len(rx.shape) == 4 and len(ry.shape) == 4:
            output = torch.zeros_like(d)
            if d.is_cuda:
                deepflow.potts_auglag_2d_gpu_forward(d,rx, ry, output)
            else:
                deepflow.potts_auglag_2d_cpu_forward(d,rx, ry, output)
            return output
        else:
            sys.stderr.write("Gave enough smoothness terms for 2D deepflow, but wrong dimensionality. \n")
            return
            
    #For the optimisers, there is no well defined backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input
            
class Potts_MAP3d(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        if len(d.shape) == 5 and len(rx.shape) == 5 and len(ry.shape) == 5 and len(rz.shape) == 5:
            output = torch.zeros_like(d)
            if d.is_cuda:
                deepflow.potts_auglag_3d_gpu_forward(d, rx, ry, rz, output)
            else:
                deepflow.potts_auglag_3d_cpu_forward(d, rx, ry, rz, output)
            return output
        else:
            sys.stderr.write("Gave enough smoothness terms for 3D deepflow, but wrong dimensionality. \n")
            return
        
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input *= 0
        return grad_input
        
        
class Potts_Mean1d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx):
        if len(d.shape) == 3 and len(rx.shape) == 3:
            output = torch.zeros_like(d)
            if d.is_cuda:
                deepflow.potts_meanpass_1d_gpu_forward(d,rx, output)
            else:
                deepflow.potts_meanpass_1d_cpu_forward(d,rx, output)
            ctx.save_for_backward(output,rx)
            return output
        else:
            sys.stderr.write("Gave enough smoothness terms for 1D deepflow, but wrong dimensionality. \n")
            return
            
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        u,rx, = ctx.saved_tensors
        grad_output = grad_output.clone()
        u = u.clone()
        rx = rx.clone()
        grad_d =  torch.zeros_like(u)
        grad_rx = torch.zeros_like(rx)
        if u.is_cuda:
            deepflow.potts_meanpass_1d_gpu_backward(grad_output, u, rx, grad_d, grad_rx)
        else:
            deepflow.potts_meanpass_1d_cpu_backward(grad_output, u, rx, grad_d, grad_rx)
        return grad_d, grad_rx

class Potts_Mean2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry):
        if len(d.shape) == 4 and len(rx.shape) == 4 and len(ry.shape) == 4:
            output = torch.zeros_like(d)
            if d.is_cuda:
                deepflow.potts_meanpass_2d_gpu_forward(d,rx, ry, output)
            else:
                deepflow.potts_meanpass_2d_cpu_forward(d,rx, ry, output)
            ctx.save_for_backward(output,rx,ry)
            return output
        else:
            sys.stderr.write("Gave enough smoothness terms for 2D deepflow, but wrong dimensionality. \n")
            return
            
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        u,rx,ry, = ctx.saved_tensors
        grad_output = grad_output.clone()
        u = u.clone()
        rx = rx.clone()
        ry = ry.clone()
        grad_d =  torch.zeros_like(u)
        grad_rx = torch.zeros_like(rx)
        grad_ry = torch.zeros_like(ry)
        if u.is_cuda:
            deepflow.potts_meanpass_2d_gpu_backward(grad_output, u, rx, ry, grad_d, grad_rx, grad_ry)
        else:
            deepflow.potts_meanpass_2d_cpu_backward(grad_output, u, rx, ry, grad_d, grad_rx, grad_ry)
        return grad_d, grad_rx, grad_ry
        
class Potts_Mean3d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, d, rx, ry, rz):
        if len(d.shape) == 5 and len(rx.shape) == 5 and len(ry.shape) == 5 and len(rz.shape) == 5:
            output = torch.zeros_like(d)
            if d.is_cuda:
                deepflow.potts_meanpass_3d_gpu_forward(d,rx, ry, rz, output)
            else:
                deepflow.potts_meanpass_3d_cpu_forward(d,rx, ry, rz, output)
            ctx.save_for_backward(output,rx,ry,rz)
            return output
        else:
            sys.stderr.write("Gave enough smoothness terms for 3D deepflow, but wrong dimensionality. \n")
            return
            
    #For the optimisers, there is no well defind backwards
    @staticmethod
    def backward(ctx, grad_output):
        u,rx,ry,rz, = ctx.saved_tensors
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
            deepflow.potts_meanpass_3d_gpu_backward(grad_output, u, rx, ry, rz, grad_d, grad_rx, grad_ry, grad_rz)
        else:
            deepflow.potts_meanpass_3d_cpu_backward(grad_output, u, rx, ry, rz, grad_d, grad_rx, grad_ry, grad_rz)
        return grad_d, grad_rx, grad_ry, grad_rz
        