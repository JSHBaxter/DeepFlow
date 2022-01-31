import numpy as np

import torch
from hmf_deepflow import HMF_MAP1d,HMF_MAP2d,HMF_MAP3d
from hmf_deepflow import HMF_Mean1d,HMF_Mean2d,HMF_Mean3d

b=1
l=3
c=2**l
br=2**(l+1)-2
x=32
epsilon = 0.000000001

def test_dimensionality(d):

    size_info = tuple( [b,c]+[x for i in range(d)] )
    size_redc_info = tuple( [1,c]+[1 for i in range(d)] )
    size_red_info = tuple( [1,1]+[x for i in range(d)] )
    br_size_info = tuple( [b,br]+[x for i in range(d)] )
    
    data_t = (1*(10*np.random.normal(0,1,size=size_info)+np.array(range(c)).reshape(size_redc_info))).astype(np.float32)
    data_r = (0.25*np.clip(0.25*np.random.normal(0,1,size=br_size_info)+0.1*np.array([(i+1) % 2 for i in range(x**d)]).reshape(size_red_info),0,100)).astype(np.float32)

    parentage = np.zeros(shape=(br,)).astype(np.int32)
    for i in range(br):
        parentage[i] = i//2-1




    t = torch.tensor(data_t, device=torch.device("cuda"))
    r = torch.tensor(data_r, device=torch.device("cuda"))
    p = torch.tensor(parentage, device=torch.device("cuda"))
    t.requires_grad = True
    r.requires_grad = True

    if d == 1:
        oa_1 = torch.exp(HMF_MAP1d.apply(t,r,p))
        om_1 = HMF_Mean1d.apply(t,r,p)
    elif d == 2:
        oa_1 = torch.exp(HMF_MAP2d.apply(t,r,r,p))
        om_1 = HMF_Mean2d.apply(t,r,r,p)
    elif d == 3:
        oa_1 = torch.exp(HMF_MAP3d.apply(t,r,r,r,p))
        om_1 = HMF_Mean3d.apply(t,r,r,r,p)

    loss = -torch.sum(om_1.flatten()[0])
    loss.backward()

    t_g1 = t.grad.clone()
    r_g1 = r.grad.clone()

    print("\nGPU results:")
    print("data sum")
    print(torch.sum(t,dim=1))
    print("data")
    print(t)
    print("mean pass")
    print(om_1)
    print("data mean pass diff")
    print(om_1-t)
    print("MAP res")
    print(oa_1)
    #print("data grad")
    #print(t_g1)
    #print(torch.mean(torch.abs(t_g1)))
    #print("reg grad")
    #print(r_g1)
    #print(torch.mean(torch.abs(r_g1)))




    t = torch.tensor(data_t, device=torch.device("cpu"))
    r = torch.tensor(data_r, device=torch.device("cpu"))
    p = torch.tensor(parentage, device=torch.device("cpu"))
    t.requires_grad = True
    r.requires_grad = True

    if d == 1:
        oa_2 = torch.exp(HMF_MAP1d.apply(t,r,p))
        om_2 = HMF_Mean1d.apply(t,r,p)
    elif d == 2:
        oa_2 = torch.exp(HMF_MAP2d.apply(t,r,r,p))
        om_2 = HMF_Mean2d.apply(t,r,r,p)
    elif d == 3:
        oa_2 = torch.exp(HMF_MAP3d.apply(t,r,r,r,p))
        om_2 = HMF_Mean3d.apply(t,r,r,r,p)
        
    loss = -torch.sum(om_2.flatten()[0])
    loss.backward()

    t_g2 = t.grad.clone()
    r_g2 = r.grad.clone()


    print("\nCPU results:")
    print("data sum")
    print(torch.sum(t,dim=1))
    print("data")
    print(t)
    print("mean pass")
    print(om_2)
    print("data mean pass diff")
    print(om_2-t)
    print("MAP res")
    print(oa_2)
    #print("data grad")
    #print(t_g2)
    #print(torch.mean(torch.abs(t_g2)))
    #print("reg grad")
    #print(r_g2)
    #print(torch.mean(torch.abs(r_g2)))



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
    test_dimensionality(3)
