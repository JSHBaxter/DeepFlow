import numpy as np

import time

import torch
from binary_deepflow import Binary_MAP1d,Binary_MAP2d,Binary_MAP3d
from binary_deepflow import Binary_Mean1d,Binary_Mean2d,Binary_Mean3d

b=1
c=3
x=32
epsilon = 0.000000001


def test_dimensionality(d):
    
    size_info = tuple( [b,c]+[x for i in range(d)] )
    size_red_info = tuple( [1,c]+[x for i in range(d)] )
    
    data_t = (0.5*np.random.normal(0,1,size=size_info)).astype(np.float32)
    data_r = (1*np.clip(0.25*np.random.normal(0,1,size=size_info)+np.array([i % 2 for i in range(c*x**d)]).reshape(size_red_info),0,100)).astype(np.float32)

    t = torch.tensor(data_t, device=torch.device("cuda"))
    r = torch.tensor(data_r, device=torch.device("cuda"))
    t.requires_grad = True
    r.requires_grad = True

    if d == 1:
        oa_1 = torch.exp(Binary_MAP1d.apply(t,r))
        om_1 = Binary_Mean1d.apply(t,r)
    elif d == 2:
        oa_1 = torch.exp(Binary_MAP2d.apply(t,r,r))
        om_1 = Binary_Mean2d.apply(t,r,r)
    elif d == 3:
        oa_1 = torch.exp(Binary_MAP3d.apply(t,r,r,r))
        om_1 = Binary_Mean3d.apply(t,r,r,r)


        
    print("\nGPU " + str(d) + "D results:")
    print("data sum")
    print(torch.sum(t,dim=2))
    print("avg MAP")
    print(torch.sum(oa_1,dim=2)/x)
    print("data")
    print(t)
    print("reg")
    print(r)
    print("mean pass")
    print(om_1)
    print("data mean pass diff")
    print((om_1-t))
    print("MAP res")
    print(oa_1)
    
    loss = -torch.sum(om_1)
    loss.backward()

    t_g1 = t.grad
    r_g1 = r.grad
    
    print("data grad")
    print(t_g1)
    print("reg grad")
    print(r_g1)
    print(torch.sum(r_g1))



    t = torch.tensor(data_t, device=torch.device("cpu")).contiguous()
    r = torch.tensor(data_r, device=torch.device("cpu")).contiguous()
    t.requires_grad = True
    r.requires_grad = True

    if d == 1:
        oa_2 = torch.exp(Binary_MAP1d.apply(t,r))
        om_2 = Binary_Mean1d.apply(t,r)
    elif d == 2:
        oa_2 = torch.exp(Binary_MAP2d.apply(t,r,r))
        om_2 = Binary_Mean2d.apply(t,r,r)
    elif d == 3:
        oa_2 = torch.exp(Binary_MAP3d.apply(t,r,r,r))
        om_2 = Binary_Mean3d.apply(t,r,r,r)

        
    print("\nCPU " + str(d) + "D results:")
    print("data sum")
    print(torch.sum(t,dim=2))
    print("avg MAP")
    print(torch.sum(oa_2,dim=2)/x)
    print("data")
    print(t)
    print("reg")
    print(r)
    print("mean pass")
    print(om_2)
    print("data mean pass diff")
    print(om_2-t)
    print("MAP res")
    print(oa_2)

    loss = -torch.sum(om_2)
    loss.backward()

    t_g2 = t.grad
    r_g2 = r.grad
    
    print("data grad")
    print(t_g2)
    print("reg grad")
    print(r_g2)
    print(torch.sum(r_g2))


    print("\nDifferences " + str(d) + "D:")
    print("auglag res")
    #print(2*(oa_1.cpu()-oa_2)/(abs(oa_1.cpu())+abs(oa_2)+epsilon))
    print(torch.max(torch.abs((oa_1.cpu()-oa_2))))
    print("meanpass res")
    #print(2*(om_1.cpu()-om_2)/(abs(om_1.cpu())+abs(om_2)+epsilon))
    print(torch.max(torch.abs((om_1.cpu()-om_2))))
    print("t_g")
    #print(2*(t_g1.cpu()-t_g2)/(abs(t_g1.cpu())+abs(t_g2)+epsilon))
    print(torch.max(torch.abs((t_g1.cpu()-t_g2))))
    print("r_g")
    #print(2*(t_g1.cpu()-t_g2)/(abs(t_g1.cpu())+abs(t_g2)+epsilon))
    print(torch.max(torch.abs((r_g1.cpu()-r_g2))))


for d in range(1,2):
    test_dimensionality(1)
    