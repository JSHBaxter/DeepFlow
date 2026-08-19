import numpy as np
import torch
from LTL_DeepFlow import *
from LTL_Node import *
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
batch = 8
N = 8
d = 0.1*torch.normal(0,1,(batch,len(structure),N))
r = 0.0*torch.abs(torch.normal(0,1,(batch,len(structure),N-1)))
d[:,c1.index,0] += np.log(1)
d[:,c4.index,-1] += np.log(1)
print(d)
print(r)
print()

print("Design some constraints")
constraint_list = [(c1,c1,"UNIQUE"),(c2,c2,"UNIQUE"),
                   (c3,c3,"UNIQUE"),(c4,c4,"UNIQUE"),
                   (c1,c1,"INITIAL"),(c4,c4,"FINAL"),
                   (c1,b1,"BEFORE"),(b1,c1,"AFTER"),
                   (c4,b1,"AFTER"),(b1,c4,"BEFORE"),
                   (c1,c4,"BEFORE"),(c4,c1,"AFTER"),
                   #(c1,b1,"BEFORE"),
                   #(c1,c4,"BEFORE"),
                   #(c2,c4,"BEFORE"),
                   #(c3,c4,"BEFORE")
                  ]
print()

print("Get EXACT answer")
marginal_exact = []
map_exact = []
for b in range(batch):
    marginal_exact_b, map_exact_b = get_exact_answer(structure,constraint_list,N,d[b:b+1,...],r[b:b+1,...])
    marginal_exact.append(marginal_exact_b)
    map_exact.append(map_exact_b)
marginal_exact = torch.cat(marginal_exact,0)
map_exact = torch.cat(map_exact,0)
print(marginal_exact)
print(map_exact)
print()

print("Use data terms only")
d_d = d.clone().detach()
d_prime = d[:,:len(structure.leaves),:]
for g in structure.groups:
    d_prime[:,[l.index for l in g.leaves],:] += d[:,g.index:(g.index+1),:]
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
mapginal_est_l = module_l_map(d_l2,r_l2)
module_l_marg.eval()
map_est_l = module_l_marg(d_l,r_l)
print(marginal_est_l)
print(mapginal_est_l)
print(map_est_l)
#loss = -torch.sum(marginal_exact*torch.log(marginal_est_l)+(1-marginal_exact)*torch.log(1-marginal_est_l))
#loss.backward()
#print(d_l.grad)
#loss = -torch.sum(marginal_exact*torch.log(mapginal_est_l)+(1-marginal_exact)*torch.log(1-mapginal_est_l))
#loss.backward()
#print(d_l2.grad)
print(torch.sum(torch.sum(d*map_est_l,-1),-1))
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
mapginal_est_g = module_g_map(d_g2,r_g2)
module_g_marg.eval()
map_est_g = module_g_marg(d_g,r_g)
print(marginal_est_g)
print(mapginal_est_g)
print(map_est_g)
#loss = -torch.sum(marginal_exact*torch.log(marginal_est_g)+(1-marginal_exact)*torch.log(1-marginal_est_g))
#loss.backward()
#print(d_g.grad)
#loss = -torch.sum(marginal_exact*torch.log(mapginal_est_g)+(1-marginal_exact)*torch.log(1-mapginal_est_g))
#loss.backward()
#print(d_g2.grad)
print(torch.sum(torch.sum(d*map_est_g,-1),-1))
print()

print("Use ALMM estimator")
module_a_map = LTL_DeepFlow(structure,constraint_list,constraint_type="ALMM",solve_type="MAP")
d_g = d.clone().detach()
r_g = r.clone().detach()
module_a_map.eval()
map_est_a = module_a_map(d_g,r_g)
print(map_est_a)
print(torch.sum(torch.sum(d*map_est_a,-1),-1))
print()