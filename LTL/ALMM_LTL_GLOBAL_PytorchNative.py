import torch
from torch.nn.functional import pad

import numpy as np
from itertools import product

from matplotlib import pyplot as plt
import time

torch.set_printoptions(linewidth=1000, threshold=1000000)

#Repurpose and improve on the efficiency of the 1D LTL case manager
# Guaranteed to converge to correct marginals in a linear number of steps regardless of constraints or regularisation
# Constraints are only applied locally, for example, if A before B, then BCA is valid
class ALMM_LTL_GLOBAL_PytorchNative:
    
    epsilon = 10**-4
    
    @staticmethod
    def constraint_application(constraint_list,hierarchy,d,r_eff=None,logic_constraint_weight=None, verbose=False):
        
        if len(hierarchy.groups) == 0:
            raise ValueException("We need at least one group for this method at the moment.")
        
        #need to update this to remove these constraints and replace with something equivalent that works in the current framework (i.e. form a NOT A group)
        new_constraint_list = []
        def should_add_to_constraint_list(new_constraint, curr_list):
            if len(curr_list) == 0:
                return True
            equals = [(c[0]==new_constraint[0]) and (c[1]==new_constraint[1]) and c[2]==new_constraint[2] for c in curr_list]
            return not max(equals)
        for constraint in constraint_list:
            if constraint[2] in ["NOTPRECEDE","NOTSUCEED"]:
                print("Do not currently handle this kind of constraint:" + str(constraint) )
                continue
            if should_add_to_constraint_list(constraint, new_constraint_list):
                new_constraint_list.append(constraint)
            if constraint[2] == "BEFORE" or constraint[2] == "AFTER":
                add_equi_constraint = (constraint[1], constraint[0], "AFTER" if constraint[2] == "BEFORE" else "BEFORE")
                if should_add_to_constraint_list(add_equi_constraint, new_constraint_list):
                    new_constraint_list.append(add_equi_constraint)
        
        b = d.shape[0]
        if r_eff == None:
            r_eff = torch.zeros((1,len(hierarchy.leaves)+len(hierarchy.groups),1),device=d.device)
        
        #arguments for the apply method
        return d, r_eff, hierarchy, new_constraint_list, True#verbose

    @staticmethod
    def make_dr_eff(d,r,groups,leaves):
        b,ct,N = d.shape[0],d.shape[1],d.shape[2]
        c = len(leaves)
        g = len(groups)
        return d, r
    
    @staticmethod
    def cleanup(hierarchy):
        pass
    
    class MAP:
        @staticmethod
        @torch.no_grad()
        def apply(d, r_eff, hierarchy, constraint_list, verbose=False):
            b,ct,N = d.shape[0],d.shape[1],d.shape[2]
            tau = 0.1
            cc = 25
            icc = 1/cc
            conv_criteria_u = 0.01
            conv_criteria_g = 10**-4 * N * b
            patience = 10
            max_iters = 100*N
            
            l_num = len(hierarchy.leaves)
            g_num = len(hierarchy.groups)
            l_idx_map = {l:l.index for idx_l, l in enumerate(hierarchy.leaves)}
            g_idx_map = {g:g.index-l_num for idx_g, g in enumerate(hierarchy.groups)}
            
            #sort the constraints into groups since we don't need all of them
            foldable_constraint_types = ["INITIAL","FINAL"]
            foldable_constraints = [c for c in constraint_list if (c[2] in foldable_constraint_types)]
            simple_constraint_types = ["PRECEDE","SUCEED"]
            simple_constraints = [c for c in constraint_list if (c[2] in simple_constraint_types)]
            inter_label_constraint_types = ["BEFORE","AFTER"]
            inter_label_constraints = [c for c in constraint_list if (c[2] in inter_label_constraint_types)]
            unique_constraints = [c for c in constraint_list if (c[2] == "UNIQUE")]
            
            #make d into a sink constraint
            D = torch.zeros((b,l_num,N), device=d.device)
            for g in hierarchy.groups:
                for l in g.leaves:
                    D[:,l_idx_map[l],:] += d[:,g.index,:]
            for l in hierarchy.leaves:
                D[:,l_idx_map[l],:] += d[:,l.index,:]
            D = -(D-torch.amax(D,1,keepdim=True))
            D_range = torch.amin(torch.sum(D,2,keepdim=True),1,keepdim=True)
            D /= D_range
            
            #implement INITIAL and FINAL constraints as modifications of the bounds for the spatial flows
            r_bounds_u = r_eff/D_range
            r_bounds_l = -r_bounds_u
            for constraint_A, constraint_B, constraint_type in foldable_constraints:
                if constraint_type == "INITIAL":
                    r_bounds_l[:,constraint_A.index,:] = -np.inf
                if constraint_type == "FINAL":
                    r_bounds_u[:,constraint_A.index,:] = np.inf
            r_g_l = r_bounds_l[:,l_num:,:]
            r_g_u = r_bounds_u[:,l_num:,:]
            r_l_l = r_bounds_l[:,:l_num,:]
            r_l_u = r_bounds_u[:,:l_num,:]
            
            #create flow fields for all the simple constraints
            simple_flows = [torch.zeros((b,N-1),device=d.device) for c in simple_constraints]
            
            #create BDA layers (and buffers for flows to/from said layers)
            bda_idx_map = {} #maps and sets for storing the BDA involves labels and layers
            bda_q_idx_map = {}
            bda_c_idx_map = {}
            bda_d_map = {}
            bda_unique = set([c[0] for c in unique_constraints])
            bda_non_unique = set([c[1] for c in inter_label_constraints if c[1] not in bda_unique])
            for constraint in unique_constraints:
                constraint_A = constraint[0]
                if constraint_A not in bda_idx_map.keys():
                    bda_idx_map[constraint_A] = len(bda_idx_map)
                if constraint_A not in bda_d_map.keys():
                    bda_d_map[constraint_A] = bda_idx_map[constraint_A]
            for constraint in inter_label_constraints:
                constraint_A, constraint_B = constraint[0],constraint[1]
                if constraint_B not in bda_idx_map.keys():
                    bda_idx_map[constraint_B] = len(bda_idx_map)
                if constraint_B not in bda_d_map.keys():
                    bda_d_map[constraint_B] = bda_idx_map[constraint_B]
                bda_c_idx_map[constraint] = len(bda_c_idx_map)
            bda_num = len(bda_idx_map)
            bda_c_num = len(bda_c_idx_map)
            
            #create nice matrices to capture the parent-child flow relationships
            match_lg = torch.zeros((1,l_num,g_num),device=d.device)
            match_lng = torch.zeros((1,l_num,g_num),device=d.device)
            for g in hierarchy.groups:
                for l in g.leaves:
                    match_lg[:,l_idx_map[l],g_idx_map[g]] = 1
            match_lng = torch.ones_like(match_lg) - match_lg
            match_gl = torch.transpose(match_lg,1,2)
            match_ngl = torch.transpose(match_lng,1,2)
            
            #set up the max-flow problem buffers
            p_l = torch.repeat_interleave(torch.amin(D,1,keepdim=True),l_num,1)
            q_l = torch.zeros((b,l_num,N-1), device=d.device)
            ps_g = torch.repeat_interleave(torch.amin(D,1,keepdim=True),g_num,1) / g_num
            p_g = ps_g.clone()
            q_g = torch.zeros((b,g_num,N-1), device=d.device)
            p_ng = ps_g.clone()
            ps_bda = torch.zeros((b,bda_num,N), device=d.device)
            pt_bda = torch.zeros((b,3*bda_num,N), device=d.device)
            q_bda = torch.zeros((b,2*bda_num,N-1), device=d.device)
            c_bda = torch.zeros((b,bda_c_num,N), device=d.device)
            d_bda = torch.zeros((b,bda_num,N), device=d.device)
                
            #get initial labels
            u_l = (D==p_l).float()#torch.zeros_like(D)
            u_l /= torch.sum(u_l,1,keepdim=True)
            u_g = torch.matmul(match_gl,u_l)
            
            u_ng = 1-u_g#torch.zeros_like(u_g)
            u_bda = torch.zeros((b,3*bda_num,N), device=d.device)
            for k,v in bda_idx_map.items():
                u_bda[:,3*v+1,:] = u_l[:,l_idx_map[k],:] if k in hierarchy.leaves else u_g[:,g_idx_map[k],:]
                for idx_b in range(b):
                    u_d = u_bda[idx_b,3*v+1,:].cpu().detach().numpy()
                    u_b = np.zeros_like(u_d)
                    u_a = np.zeros_like(u_d)
                    acc_a = 0
                    for t in range(0,N):
                        if u_d[t] >= acc_a:
                            acc_a = u_d[t]
                        else:
                            u_a[t] = acc_a - u_d[t]
                    acc_b = 0
                    for t in range(N-1,-1,-1):
                        if u_d[t] >= acc_b:
                            acc_b = u_d[t]
                        else:
                            u_b[t] = acc_b - u_d[t]
                    s_u_ab = u_a+u_b
                    u_a[s_u_ab > 0] /= s_u_ab[s_u_ab > 0]
                    u_b[s_u_ab > 0] /= s_u_ab[s_u_ab > 0]
                    u_bda[idx_b,3*v+0,:] = torch.Tensor(u_b,device=d.device)
                    u_bda[idx_b,3*v+1,:] = torch.Tensor(u_d,device=d.device)
                    u_bda[idx_b,3*v+2,:] = torch.Tensor(u_a,device=d.device)
            
            #get initial capacities
            cap_l = -icc*u_l
            cap_g = -icc*u_g
            cap_ng = -icc*u_ng
            cap_bda = -icc*u_bda
            old_tot_flow =  torch.sum(ps_g) + torch.sum(ps_bda)
            
            
            #separate out several steps into their own "inline" functions for reusability
            def update_capacities():
                #capacity is "capacity to receive flow" - i.e. outgoing flows minus incoming flows
                cap_g[...] = -ps_g + p_g - u_g*icc
                cap_g[:,:,:-1] += q_g
                cap_g[:,:,1:] -= q_g
                cap_ng[...] = -ps_g + p_ng - u_ng*icc
                cap_l[...] = -torch.matmul(match_lg,p_g)-torch.matmul(match_lng,p_ng) + p_l - u_l*icc
                cap_l[:,:,:-1] += q_l
                cap_l[:,:,1:] -= q_l
                for c_idx, (c, constraint) in enumerate(zip(simple_flows,simple_constraints)):
                    if constraint[2] == "SUCEED":
                        if constraint[0] in l_idx_map.keys():
                            cap_l[:,l_idx_map[constraint[0]],1:] -= c
                        elif constraint[0] in g_idx_map.keys():
                            cap_g[:,g_idx_map[constraint[0]],1:] -= c
                        if constraint[1] in l_idx_map.keys():
                            cap_l[:,l_idx_map[constraint[1]],1:] -= c
                            cap_l[:,l_idx_map[constraint[1]],:-1] += c
                        elif constraint[1] in g_idx_map.keys():
                            cap_g[:,g_idx_map[constraint[1]],1:] -= c
                            cap_g[:,g_idx_map[constraint[1]],:-1] += c
                    elif constraint[2] == "PRECEDE":
                        if constraint[0] in l_idx_map.keys():
                            cap_l[:,l_idx_map[constraint[0]],:-1] -= c
                        elif constraint[0] in g_idx_map.keys():
                            cap_g[:,g_idx_map[constraint[0]],:-1] -= c
                        if constraint[1] in l_idx_map.keys():
                            cap_l[:,l_idx_map[constraint[1]],:-1] -= c
                            cap_l[:,l_idx_map[constraint[1]],1:] += c
                        elif constraint[1] in g_idx_map.keys():
                            cap_g[:,g_idx_map[constraint[1]],:-1] -= c
                            cap_g[:,g_idx_map[constraint[1]],1:] += c
                cap_bda[...] = -torch.repeat_interleave(ps_bda,3,dim=1) + pt_bda - u_bda*icc
                cap_bda[:,0::3,:-1] += q_bda[:,0::2,:]
                cap_bda[:,0::3,1:] -= q_bda[:,0::2,:]
                cap_bda[:,2::3,:-1] += q_bda[:,1::2,:]
                cap_bda[:,2::3,1:] -= q_bda[:,1::2,:]
                for k,v in bda_d_map.items():
                    if k in hierarchy.leaves:
                        cap_l[:,l_idx_map[k],:] += d_bda[:,v,:]
                    else:
                        cap_g[:,g_idx_map[k],:] += d_bda[:,v,:]
                    cap_bda[:,3*v+1,:] -= d_bda[:,v,:]
                for constraint, c_bda_idx in bda_c_idx_map.items():
                    cap_bda[:,3*bda_idx_map[constraint[1]]+(2 if constraint[2] == "AFTER" else 0),:] -= c_bda[:,c_bda_idx,:]
                    if constraint[0] in hierarchy.leaves:
                        cap_l[:,l_idx_map[constraint[0]],:] += c_bda[:,c_bda_idx,:]
                    else:
                        cap_g[:,g_idx_map[constraint[0]],:] += c_bda[:,c_bda_idx,:]
            
            
            #Pock-Chambolle iteration
            def update_spatial_flows(cap,q,r_l,r_u):
                update_quant = tau*(cap[:,:,1:]-cap[:,:,:-1])
                cap[:,:,1:] += q
                cap[:,:,:-1] -= q
                q += update_quant
                if r_l.shape[2] == 1:
                    q[q < r_l] = r_l
                    q[q > r_u] = r_u
                else:
                    q[q < r_l] = r_l[q < r_l]
                    q[q > r_u] = r_u[q > r_u]
                cap[:,:,1:] -= q
                cap[:,:,:-1] += q
                
                
            
            def update_simple_constraints():
                for c_idx, (c, constraint) in enumerate(zip(simple_flows,simple_constraints)):
                    #find the new optimal value (may need to be iterated)
                    update_quant = c.clone()
                    if constraint[2] == "SUCEED":
                        update_quant += 0.5*(cap_l[:,l_idx_map[constraint[0]],1:] if constraint[0] in hierarchy.leaves else cap_g[:,g_idx_map[constraint[0]],1:])
                        if constraint[1] in hierarchy.leaves:
                            update_quant += 0.5*(cap_l[:,l_idx_map[constraint[1]],1:] - cap_l[:,l_idx_map[constraint[1]],:-1])
                        else:
                            update_quant +=0.5*(cap_g[:,g_idx_map[constraint[1]],1:] - cap_g[:,g_idx_map[constraint[1]],:-1])
                    else: #constraint[2] == "PRECEDE"
                        update_quant += 0.5*(cap_l[:,l_idx_map[constraint[0]],:-1] if constraint[0] in hierarchy.leaves else cap_g[:,g_idx_map[constraint[0]],:-1])
                        if constraint[1] in hierarchy.leaves:
                            update_quant += 0.5*(cap_l[:,l_idx_map[constraint[1]],:-1] - cap_l[:,l_idx_map[constraint[1]],1:])
                        else:
                            update_quant += 0.5*(cap_g[:,g_idx_map[constraint[1]],:-1] - cap_g[:,g_idx_map[constraint[1]],1:])
                    update_quant[update_quant<0] *= 0
                    
                    #correct the capacities calculations
                    if constraint[2] == "SUCEED":
                        if constraint[0] in hierarchy.leaves:
                            cap_l[:,l_idx_map[constraint[0]],1:] += c - update_quant
                        else:
                            cap_g[:,g_idx_map[constraint[0]],1:] += c - update_quant
                        if constraint[1] in hierarchy.leaves:
                            cap_l[:,l_idx_map[constraint[1]],1:] += c - update_quant
                            cap_l[:,l_idx_map[constraint[1]],:-1] += update_quant - c
                        else:
                            cap_g[:,g_idx_map[constraint[1]],1:] += c - update_quant
                            cap_g[:,g_idx_map[constraint[1]],:-1] += update_quant - c
                            
                    else: #constraint[2] == "PRECEDE"
                        if constraint[0] in hierarchy.leaves:
                            cap_l[:,l_idx_map[constraint[0]],:-1] += c - update_quant
                        else:
                            cap_g[:,g_idx_map[constraint[0]],:-1] += c - update_quant
                        if constraint[1] in hierarchy.leaves:
                            cap_l[:,l_idx_map[constraint[1]],:-1] += c - update_quant
                            cap_l[:,l_idx_map[constraint[1]],1:] += update_quant - c
                        else:
                            cap_g[:,g_idx_map[constraint[1]],:-1] += c - update_quant
                            cap_g[:,g_idx_map[constraint[1]],1:] += update_quant - c
                            
                    c[...] = update_quant
                    del update_quant
                    
            def update_bda_regular_constraints():
                
                #for each BDA layer, update the flows to/from the regular layers
                for l in bda_unique - hierarchy.groups:
                    update_quant = (cap_bda[:,3*bda_idx_map[l]+1,:] - cap_l[:,l_idx_map[l],:])/2 + d_bda[:,bda_d_map[l],:]
                    cap_l[:,l_idx_map[l],:] += update_quant - d_bda[:,bda_d_map[l],:] 
                    cap_bda[:,3*bda_idx_map[l]+1,:] += d_bda[:,bda_d_map[l],:] - update_quant
                    d_bda[:,bda_d_map[l],:] = update_quant
                    del update_quant
                for g in bda_unique - hierarchy.leaves:
                    update_quant = (cap_bda[:,3*bda_idx_map[g]+1,:] - cap_g[:,g_idx_map[g],:])/2 + d_bda[:,bda_d_map[g],:]
                    cap_g[:,g_idx_map[g],:] += update_quant - d_bda[:,bda_d_map[g],:] 
                    cap_bda[:,3*bda_idx_map[g]+1,:] += d_bda[:,bda_d_map[g],:] - update_quant
                    d_bda[:,bda_d_map[g],:] = update_quant
                    del update_quant
                for l in bda_non_unique - hierarchy.groups:
                    update_quant = (cap_bda[:,3*bda_idx_map[l]+1,:] - cap_l[:,l_idx_map[l],:])/2 + d_bda[:,bda_d_map[l],:]
                    update_quant[update_quant < 0] = 0
                    cap_l[:,l_idx_map[l],:] += update_quant - d_bda[:,bda_d_map[l],:] 
                    cap_bda[:,3*bda_idx_map[l]+1,:] += d_bda[:,bda_d_map[l],:] - update_quant
                    d_bda[:,bda_d_map[l],:] = update_quant
                    del update_quant
                for g in bda_non_unique - hierarchy.leaves:
                    update_quant = (cap_bda[:,3*bda_idx_map[g]+1,:] - cap_g[:,g_idx_map[g],:])/2 + d_bda[:,bda_d_map[g],:]
                    update_quant[update_quant < 0] *= 0
                    cap_g[:,g_idx_map[g],:] += update_quant - d_bda[:,bda_d_map[g],:] 
                    cap_bda[:,3*bda_idx_map[g]+1,:] += d_bda[:,bda_d_map[g],:] - update_quant
                    d_bda[:,bda_d_map[g],:] = update_quant
                    del update_quant

                #for BDA layers, update spatial flows
                quant = (cap_bda[:,0::3,1:]-cap_bda[:,0::3,:-1])/2 + q_bda[:,0::2,:]
                quant[quant > 0] *= 0
                cap_bda[:,0::3,1:] += q_bda[:,0::2,:] - quant
                cap_bda[:,0::3,:-1] += quant - q_bda[:,0::2,:]
                q_bda[:,0::2,:] = quant

                quant = (cap_bda[:,2::3,1:]-cap_bda[:,2::3,:-1])/2 + q_bda[:,1::2,:]
                quant[quant < 0] *= 0
                cap_bda[:,2::3,1:] += q_bda[:,1::2,:] - quant
                cap_bda[:,2::3,:-1] += quant - q_bda[:,1::2,:]
                q_bda[:,1::2,:] = quant
                
                
                

            def update_bda_remaining_constraints():
                #for each non-unique BDA constraint, update the flows to/from the regular layers
                for constraint, c_bda_idx in bda_c_idx_map.items():
                    update_quant = (cap_bda[:,3*bda_idx_map[constraint[1]]+(2 if constraint[2] == "AFTER" else 0),:] - (
                                         cap_l[:,l_idx_map[constraint[0]],:] if constraint[0] in hierarchy.leaves 
                                    else cap_g[:,g_idx_map[constraint[0]],:])
                                   )/2 + c_bda[:,c_bda_idx,:]
                    update_quant[update_quant < 0] = 0
                    
                    cap_bda[:,3*bda_idx_map[constraint[1]]+(2 if constraint[2] == "AFTER" else 0),:] += c_bda[:,c_bda_idx,:] - update_quant
                    if constraint[0] in hierarchy.leaves:
                        cap_l[:,l_idx_map[constraint[0]],:] += update_quant - c_bda[:,c_bda_idx,:]
                    else:
                        cap_g[:,g_idx_map[constraint[0]],:] += update_quant - c_bda[:,c_bda_idx,:]

                    c_bda[:,c_bda_idx,:] = update_quant
                    del update_quant
                    
                    
            #run the solution algorithm
            iter = 0
            patience_count = 0
            
            #initialise some flows that take a bit to stabilise
            for iter in range(int(1/tau+0.5)):
                for sub_iter in range(int(N)):
                    update_bda_regular_constraints()
                update_bda_remaining_constraints()
            for sub_iter in range(int(N/tau+0.5)):
                update_simple_constraints()
                update_spatial_flows(cap_g,q_g,r_g_l,r_g_u)
                update_spatial_flows(cap_l,q_l,r_l_l,r_l_u)
            
            #find optimal starting values for u based on the newly updated capacities - closer to being a valid labelling
            u_l = -cc*cap_l
            u_g = -cc*cap_g
            u_ng = -cc*cap_ng
            u_bda = -cc*cap_bda
            
            
            #run the iterative gradient-descent-style solver
            while True:
                    
                #update all spatial flows and to/from BDA layer flows
                for sub_iter in range(5):
                    update_bda_regular_constraints()
                    update_bda_remaining_constraints()
                    update_simple_constraints()
                    update_spatial_flows(cap_g,q_g,r_g_l,r_g_u)
                    update_spatial_flows(cap_l,q_l,r_l_l,r_l_u)
                
                #update flows at the sinks and update the capacities for the leaf labels
                cap_l -= p_l
                p_l[...] = -cap_l
                p_l[p_l > D] = D[p_l > D]
                cap_l += p_l

                #update flows at the groups
                quant_g = 1/(1+torch.sum(match_gl,2,keepdim=True))*(torch.matmul(match_gl,cap_l) - cap_g) + p_g
                quant_ng = 1/(1+torch.sum(match_ngl,2,keepdim=True))*(torch.matmul(match_ngl,cap_l) - cap_ng) + p_ng
                cap_l += torch.matmul(match_lg,p_g) + torch.matmul(match_lng,p_ng) - torch.matmul(match_lg,quant_g) - torch.matmul(match_lng,quant_ng)
                cap_g += quant_g - p_g
                cap_ng += quant_ng - p_ng
                p_g[...] = quant_g
                p_ng[...] = quant_ng
                del quant_g, quant_ng

                #update flows at the source
                quant = 0.5*(icc + cap_g + cap_ng) + ps_g
                cap_g += ps_g - quant
                cap_ng += ps_g - quant
                ps_g[...] = quant
                del quant
                
                #update sink flows at the BDA layers
                cap_bda -= pt_bda
                pt_bda[...] = -cap_bda
                pt_bda[pt_bda > 0] = 0
                cap_bda += pt_bda

                #update source flows at the BDA layers
                quant = (1/3) * (icc + cap_bda[:,0::3,:] + cap_bda[:,1::3,:] + cap_bda[:,2::3,:]) + ps_bda
                cap_bda += torch.repeat_interleave(ps_bda - quant,3,1)
                ps_bda[...] = quant
                del quant
                
                #update multipliers at the sinks
                err_u_l = cc * cap_l + u_l
                u_l -= tau * err_u_l
                u_l[u_l < 0] = 0
                u_l /= torch.sum(u_l,1,keepdim=True)
                
                #update multipliers at the groups
                err_u_g = cc * cap_g + u_g
                u_g -= tau * err_u_g
                err_u_ng = cc * cap_ng + u_ng
                u_ng -= tau * err_u_ng
                
                #update multipliers at the BDA layers
                err_u_bda = cc * cap_bda + u_bda
                u_bda -= tau * err_u_bda
                u_bda[u_bda < 0] = 0
                u_bda /= torch.repeat_interleave(u_bda[:,0::3,:]+u_bda[:,1::3,:]+u_bda[:,2::3,:],3,1)
            
                update_capacities()
                
                #compute dual value
                d_val = torch.sum(D * u_l).item()
                r_val = torch.sum(r_eff[:,:l_num,:] * torch.abs(u_l[:,:,1:]-u_l[:,:,:-1])).item() + torch.sum(r_eff[:,l_num:,:] * torch.abs(u_g[:,:,1:]-u_g[:,:,:-1])).item()
                dual_val = d_val+r_val
            
                #check convergence and update cc
                conv_crit_check = max(torch.amax(err_u_l).cpu().item(),-torch.amin(err_u_l).cpu().item())
                
                tot_flow =  torch.sum(ps_g).cpu().item() + torch.sum(ps_bda).cpu().item()
                gap = dual_val-tot_flow
                #if we have a primal-dual gap and the primal maximum is still increasing, loosen the penalties
                if tot_flow > old_tot_flow and gap > 0:
                    cc /= 1.001
                else: #increase the penalties gradually
                    cc *= 1.001
                icc = 1/cc
                old_tot_flow = tot_flow
                
                #for safety, update the capacities so they don't drift (also covers updated from changed u)
                update_capacities()
                
                if verbose:
                    print("\033[2K"+str(iter), torch.sum(ps_g).cpu().item(), "\t", torch.sum(ps_bda).cpu().item(),"\t",dual_val-tot_flow, "\t", cc, "\t", conv_crit_check,end="\r")
                
                iter += 1
                if iter > max_iters or (conv_crit_check < conv_criteria_u and abs(gap) < conv_criteria_g):
                    patience_count += 1
                else:
                    patience_count = 0
                if patience_count > patience:
                    break
                    
            if verbose:
                print("")
                
            #final normalisation before returning
            u_l[u_l < 0] = 0
            u_l /= torch.sum(u_l,1,keepdim=True)
            return torch.log(u_l)
