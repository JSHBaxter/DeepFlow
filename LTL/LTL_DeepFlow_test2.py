import numpy as np
import torch
from LTL_DeepFlow import *
from LTL_DeepFlow_test import get_exact_answer

print("Make the label hierarchy")
c1 = LTL_LeafNode("c1",0)
c2 = LTL_LeafNode("c2",1)
c3 = LTL_LeafNode("c3",2)
c4 = LTL_LeafNode("c4",3)

b1 = LTL_GroupNode("b1",4,[c2,c3])

structure = LTL_GroupingStructure([c1,c2,c3,c4],[b1])

print(str(structure))

print("Design some simple bias/reg terms to use")
torch.manual_seed(1)
N = 7
d = 0.1*torch.normal(0,1,(1,len(structure),N))
r = 0.0*torch.abs(torch.normal(0,1,(1,len(structure),N-1)))
d[:,c1.index,0] += np.log(7)
d[:,c4.index,-1] += np.log(7)
print(d)
print(r)
print()

print("Design some constraints")
constraint_list = [#(c1,b1,"BEFORE"),(c4,b1,"AFTER"),
                   (c1,c1,"UNIQUE"),(c4,c4,"UNIQUE"),
                   (c2,c2,"UNIQUE"),(c3,c3,"UNIQUE"),
                   (c1,c1,"INITIAL"),(c4,c4,"FINAL"),
                   (c1,b1,"PRECEDE"),(b1,c1,"SUCEED"),
                   (c4,b1,"SUCEED"),(b1,c4,"PRECEDE")
                  ]
print()

print("Get EXACT answer")
marginal_exact, map_exact = get_exact_answer(structure,constraint_list,N,d,r)
print(marginal_exact)
print(map_exact)
epsilon = 10**-16
print()

print("Use data terms only")
d_d = d.clone().detach()
d_prime = d[:,:4,:]
d_prime[:,1:3,:] += d[:,4,:]
marginal_est_d = torch.softmax(d_prime,1)
print(marginal_est_d)
print((marginal_est_d == torch.amax(marginal_est_d,1,keepdims=True)).float())
print()

print("Use LOCAL estimator")
module_l_marg = LTL_DeepFlow(structure,constraint_list,constraint_type="LOCAL",solve_type="MARG")
module_l_map = LTL_DeepFlow(structure,constraint_list,constraint_type="LOCAL",solve_type="MAP")
d_l = d.clone().detach()
d_l2 = d.clone().detach()
r_l = r.clone().detach()
r_l2 = r.clone().detach()
d_l.requires_grad = True
d_l2.requires_grad = True
r_l.requires_grad = True
r_l2.requires_grad = True
module_l_marg.train()
module_l_map.train()
marginal_est_l = module_l_marg(d_l,r_l)
mapginal_est_l = module_l_map(d_l2,r_l)
module_l_marg.eval()
map_est_l = module_l_marg(d_l,r_l)
print(marginal_est_l)
print(mapginal_est_l)
print(map_est_l)
loss = -torch.sum(marginal_exact*torch.log(marginal_est_l)+(1-marginal_exact)*torch.log(1-marginal_est_l))
loss.backward()
print(d_l.grad)
loss = -torch.sum(marginal_exact*torch.log(mapginal_est_l)+(1-marginal_exact)*torch.log(1-mapginal_est_l))
loss.backward()
print(d_l2.grad)
print()

print("Use GLOBAL estimator")
module_g_marg = LTL_DeepFlow(structure,constraint_list,constraint_type="GLOBAL",solve_type="MARG")
module_g_map = LTL_DeepFlow(structure,constraint_list,constraint_type="GLOBAL",solve_type="MAP")
d_g = d.clone().detach()
d_g2 = d.clone().detach()
r_g = r.clone().detach()
r_g2 = r.clone().detach()
d_g.requires_grad = True
d_g2.requires_grad = True
r_g.requires_grad = True
r_g2.requires_grad = True
module_g_marg.train()
module_g_map.train()
marginal_est_g = module_g_marg(d_g,r_g)
mapginal_est_g = module_g_map(d_g2,r_g)
module_g_marg.eval()
map_est_g = module_g_marg(d_g,r_g)
print(marginal_est_g)
print(mapginal_est_g)
print(map_est_g)
loss = -torch.sum(marginal_exact*torch.log(marginal_est_g)+(1-marginal_exact)*torch.log(1-marginal_est_g))
loss.backward()
print(d_g.grad)
loss = -torch.sum(marginal_exact*torch.log(mapginal_est_g)+(1-marginal_exact)*torch.log(1-mapginal_est_g))
loss.backward()
print(d_g2.grad)
print()