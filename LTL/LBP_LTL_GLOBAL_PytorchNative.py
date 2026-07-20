import torch
from itertools import product
from LBP_LTL_LOCAL_PytorchNative import LBP_LTL_LOCAL_PytorchNative

from matplotlib import pyplot as plt


#Repurpose and improve on the efficiency of the 1D LTL case manager
# Guaranteed to converge to correct marginals in a linear number of steps regardless of constraints or regularisation
# Constraints are only applied locally, for example, if A before B, then BCA is valid
class LBP_LTL_GLOBAL_PytorchNative:
    
    epsilon = 10**-4
    
    @staticmethod
    def constraint_application(constraint_list,hierarchy,d,r_eff=None,logic_constraint_weight=None):
        
        simple_constraint_types = ["PRECEDE","SUCEED","NOTPRECEDE","NOTSUCEED","INITIAL","FINAL"]
        simple_constraints = [c for c in constraint_list if (c[2] in simple_constraint_types)]
        inter_label_constraint_types = ["BEFORE","AFTER"]
        inter_label_constraints = [c for c in constraint_list if (c[2] in inter_label_constraint_types)]
        unique_constraints = [c for c in constraint_list if (c[2] == "UNIQUE")]
        r_eff_multiplier = torch.ones((1,len(hierarchy.leaves),len(hierarchy.leaves),1),dtype=d.dtype,device=d.device)
        
        #make mapping to c_index
        c_index_top = 0
        for l in list(hierarchy.leaves)+list(hierarchy.groups):
            if hasattr(l,"c_index"):
                del l.c_index
        for constraint_A, constraint_B, _ in inter_label_constraints + unique_constraints:
            if not hasattr(constraint_A,"c_index"):
                constraint_A.c_index = c_index_top
                c_index_top += 3
            if constraint_B is not None and not hasattr(constraint_B,"c_index"):
                constraint_B.c_index = c_index_top
                c_index_top += 3
        num_extra_labels = c_index_top
        
        #make containers for new reg terms
        if logic_constraint_weight is None:
            logic_constraint_weight = (3 / num_extra_labels if num_extra_labels > 0 else -1)
        r_bda_inter = logic_constraint_weight*torch.ones((1,len(hierarchy.leaves),num_extra_labels,1),dtype=d.dtype,device=d.device)
        r_bda_tempo = torch.zeros((1,num_extra_labels,num_extra_labels,1),dtype=d.dtype,device=d.device)
        
        #apply the local constraints
        for constraint_A, constraint_B, constraint_type in simple_constraints:
            
            #if A must preced B, then no label other than A or B can come before any label of B
            if constraint_type == "PRECEDE":
                for child_C, child_B in product(hierarchy.leaves - constraint_A.leaves - constraint_B.leaves, constraint_B.leaves):
                    r_eff_multiplier[:,child_C.index,child_B.index,:] = 0
            
            #if A must not precede B, then no label in A can come before any label of B
            elif constraint_type == "NOTPRECEDE":
                for child_A, child_B in product(constraint_A.leaves, constraint_B.leaves):
                    r_eff_multiplier[:,child_A.index,child_B.index,:] = 0
            
            #if A must suceed B, then no label other than A or B can come after any label of B
            elif constraint_type == "SUCEED":
                for child_C, child_B in product(hierarchy.leaves - constraint_A.leaves - constraint_B.leaves, constraint_B.leaves):
                    r_eff_multiplier[:,child_B.index,child_C.index,:] = 0
            
            #if A must suceed B, then no label in A can come after any label of B
            elif constraint_type == "NOTSUCEED":
                for child_A, child_B in product(constraint_A.leaves, constraint_B.leaves):
                    r_eff_multiplier[:,child_B.index,child_A.index,:] = 0
            
            #if A is final, everything after it must remain in A
            elif constraint_type == "FINAL":
                for child_A, child_notA in product(constraint_A.leaves, hierarchy.leaves - constraint_A.leaves):
                    r_eff_multiplier[:,child_A.index,child_notA.index,:] = 0
            
            #if A is final, everything before it must remain in A
            elif constraint_type == "INITIAL":
                for child_A, child_notA in product(constraint_A.leaves, hierarchy.leaves - constraint_A.leaves):
                    r_eff_multiplier[:,child_notA.index,child_A.index,:] = 0
        
        #apply the temporal logic for global constraints
        #that is, - currently being a child of B means you cannot be BEFORE B nor AFTER B
        #         - BEFORE B cannot come after DURING B
        #         - BEFORE B cannot come after AFTER B
        #         - DURING B cannot come after AFTER B
        # Right now, we allow for it to go directly from BEFORE B to AFTER B (sub-resolution B) which we might want to exclude in future
        for constraint_A, constraint_B, _ in inter_label_constraints + unique_constraints:
            for child_A in constraint_A.leaves: 
                r_bda_inter[:,child_A.index,constraint_A.c_index+0,:] = 0.0
                r_bda_inter[:,child_A.index,constraint_A.c_index+2,:] = 0.0
            r_bda_tempo[:,constraint_A.c_index+0,constraint_A.c_index+0,:] = 1.0
            r_bda_tempo[:,constraint_A.c_index+0,constraint_A.c_index+1,:] = 1.0
            r_bda_tempo[:,constraint_A.c_index+1,constraint_A.c_index+1,:] = 1.0
            r_bda_tempo[:,constraint_A.c_index+1,constraint_A.c_index+2,:] = 1.0
            r_bda_tempo[:,constraint_A.c_index+2,constraint_A.c_index+2,:] = 1.0
            if constraint_B is None:
                continue
            for child_B in constraint_B.leaves: 
                r_bda_inter[:,child_B.index,constraint_B.c_index+0,:] = 0.0
                r_bda_inter[:,child_B.index,constraint_B.c_index+2,:] = 0.0
            r_bda_tempo[:,constraint_B.c_index+0,constraint_B.c_index+0,:] = 1.0
            r_bda_tempo[:,constraint_B.c_index+0,constraint_B.c_index+1,:] = 1.0
            r_bda_tempo[:,constraint_B.c_index+1,constraint_B.c_index+1,:] = 1.0
            r_bda_tempo[:,constraint_B.c_index+1,constraint_B.c_index+2,:] = 1.0
            r_bda_tempo[:,constraint_B.c_index+2,constraint_B.c_index+2,:] = 1.0
            
        # apply specific temporal logic global constraints (BEFORE and AFTER)
        for constraint_A, constraint_B, constraint_type in inter_label_constraints:
            #if A must be BEFORE B, then any label in A must prevent DURING_B and AFTER_B
            #also implies A AFTER B,  so any label in B must prevent BEFORE_A and DURING_A
            if constraint_type == "BEFORE":
                for child_A in constraint_A.leaves:
                    r_bda_inter[:,child_A.index,constraint_B.c_index+1,:] = 0
                    r_bda_inter[:,child_A.index,constraint_B.c_index+2,:] = 0
                for child_B in constraint_B.leaves:
                    r_bda_inter[:,child_B.index,constraint_A.c_index+0,:] = 0
                    r_bda_inter[:,child_B.index,constraint_A.c_index+1,:] = 0
                #also add the local constraint for good measure and to help with convergence
                for child_A, child_B in product(constraint_A.leaves,constraint_B.leaves):
                    r_eff_multiplier[:,child_B.index,child_A.index,:] = 0
                            
            #same but in reverse
            elif constraint_type == "AFTER":
                for child_A in constraint_A.leaves:
                    r_bda_inter[:,child_A.index,constraint_B.c_index+0,:] = 0
                    r_bda_inter[:,child_A.index,constraint_B.c_index+1,:] = 0
                for child_B in constraint_B.leaves:
                    r_bda_inter[:,child_B.index,constraint_A.c_index+1,:] = 0
                    r_bda_inter[:,child_B.index,constraint_A.c_index+2,:] = 0
                #also add the local constraint for good measure and to help with convergence
                for child_A, child_B in product(constraint_A.leaves,constraint_B.leaves):
                    r_eff_multiplier[:,child_A.index,child_B.index,:] = 0
            
        #and apply uniqueness constraints (i.e. if A exists, then A implies DURING A (cannot be BEFORE or AFTER A)
        for constraint_A, _, _ in unique_constraints:
            for child_A in constraint_A.leaves:
                    r_bda_inter[:,child_A.index,constraint_A.c_index+0,:] = 0
                    r_bda_inter[:,child_A.index,constraint_A.c_index+2,:] = 0
        
        if r_eff is not None:
            r_eff_ret = r_eff*r_eff_multiplier.detach()
        else:
            r_eff_ret = r_eff_multiplier.detach()
            
        return d, r_eff_ret, r_bda_inter.detach(), r_bda_tempo.detach()

    @staticmethod
    def cleanup(hierarchy):
        for l in list(hierarchy.leaves)+list(hierarchy.groups):
            if hasattr(l,"c_index"):
                del l.c_index
    
    class MARG(torch.autograd.Function):
        @staticmethod
        def forward(ctx, d, er_eff, er_bda_inter, er_bda_tempo):
            epsilon = 10**-300
            b,c,sx = d.shape[0],d.shape[1],d.shape[2]
            ct = er_bda_tempo.shape[1]

            if ct == 0:
                return LBP_LTL_LOCAL_PytorchNative.MARG.forward(ctx,d,er_eff)

            mvxu = torch.ones((b,c,sx-1),device=d.device,dtype=torch.float64)
            mvxd = torch.ones((b,c,sx-1),device=d.device,dtype=torch.float64)
            mexu = torch.ones((b,c,sx-1),device=d.device,dtype=torch.float64)
            mexd = torch.ones((b,c,sx-1),device=d.device,dtype=torch.float64)

            mvtu = torch.ones((b,ct,sx-1),device=d.device,dtype=torch.float64)
            mvtd = torch.ones((b,ct,sx-1),device=d.device,dtype=torch.float64)
            metu = torch.ones((b,ct,sx-1),device=d.device,dtype=torch.float64)
            metd = torch.ones((b,ct,sx-1),device=d.device,dtype=torch.float64)

            mvxt = torch.ones((b,c,sx),device=d.device,dtype=torch.float64)
            mvtx = torch.ones((b,ct,sx),device=d.device,dtype=torch.float64)
            mext = torch.ones((b,ct,sx),device=d.device,dtype=torch.float64)
            metx = torch.ones((b,c,sx),device=d.device,dtype=torch.float64)

            max_num_iters = 3*sx
            first_type_iters = 2*sx
            max_num_iters = 2*sx

            iter = 0
            ed = torch.exp((d-torch.amax(d,1,keepdims=True)))
            while True:
                #ed = torch.exp((d-torch.amax(d,1,keepdims=True))*min([1,((iter+1)/(first_type_iters+1))])).double()
                #print("\nMARG Iteration:",iter)

                #compute vertex to edge messages in label layer (through time)
                mvxd = ed[:,:,1:] * metx[:,:,1:]
                mvxd[:,:,:-1] *= mexd[:,:,1:]
                mvxd += epsilon
                mvxu = ed[:,:,:-1]* metx[:,:,:-1]
                mvxu[:,:,1:] *= mexu[:,:,:-1]
                mvxu += epsilon
                #compute vertex to edge messages in BDA layer (through time)
                mvtd = mext[:,:,1:].clone()
                mvtd[:,:,:-1] *= metd[:,:,1:]
                mvtd += epsilon
                mvtu = mext[:,:,:-1].clone()
                mvtu[:,:,1:] *= metu[:,:,:-1]
                mvtu += epsilon
                #compute vertex to edge messages from the labels to the BDA (per each time)
                mvxt = torch.ones((b,c,sx),device=d.device,dtype=torch.float64)
                mvxt[:,:,:-1] *= mexd
                mvxt[:,:,1:] *= mexu
                mvxt += epsilon
                #compute vertex to edge messages from the BDA to the labels (per each time)
                mvtx = torch.ones((b,ct,sx),device=d.device,dtype=torch.float64)
                mvtx[:,:,:-1] *= metd
                mvtx[:,:,1:] *= metu
                mvtx += epsilon

                #renormalise messages in label layer
                avxu = torch.sum(mvxu,1,keepdim=True)
                mvxu /= avxu
                avxd = torch.sum(mvxd,1,keepdim=True)
                mvxd /= avxd
                #renormalise messages in BDA layer
                avtu = mvtu[:,0::3,:]+mvtu[:,1::3,:]+mvtu[:,2::3,:]
                mvtu[:,0::3,:] /= avtu
                mvtu[:,1::3,:] /= avtu
                mvtu[:,2::3,:] /= avtu
                avtd = mvtd[:,0::3,:]+mvtd[:,1::3,:]+mvtd[:,2::3,:]
                mvtd[:,0::3,:] /= avtd
                mvtd[:,1::3,:] /= avtd
                mvtd[:,2::3,:] /= avtd
                #renormalise messages between label and BDA layers
                avtx = mvtx[:,0::3,:]+mvtx[:,1::3,:]+mvtx[:,2::3,:]
                mvtx[:,0::3,:] /= avtx
                mvtx[:,1::3,:] /= avtx
                mvtx[:,2::3,:] /= avtx
                avxt = torch.sum(mvxt,1,keepdim=True)
                mvxt /= avxt
                
                #compute edge to vertex messages
                mexd = torch.sum(er_eff * torch.unsqueeze(mvxd,1), dim=2)
                mexu = torch.sum(er_eff * torch.unsqueeze(mvxu,2), dim=1)
                metd = torch.sum(er_bda_tempo * torch.unsqueeze(mvtd,1), dim=2)
                metu = torch.sum(er_bda_tempo * torch.unsqueeze(mvtu,2), dim=1)
                mext = torch.sum(er_bda_inter * torch.unsqueeze(mvxt,2), dim=1)
                metx = torch.sum(er_bda_inter * torch.unsqueeze(mvtx,1), dim=2)
                
                iter+=1
                if iter >= max_num_iters:
                    break

            #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
            marginal = d.clone()
            marginal += torch.log(metx)
            marginal[:,:,:-1] += torch.log(mexd/torch.amax(mexd,1,keepdims=True))
            marginal[:,:,1:] += torch.log(mexu/torch.amax(mexu,1,keepdims=True))
            #marginal_plt = torch.softmax(marginal,1)
            #lines = [plt.plot(marginal_plt[0,i,:],label=str(iter)+"-"+str(i)) for i in range(marginal.shape[1])]
            #plt.legend()
            #plt.show()
            
            #print(iter, [torch.amin(torch.amax(t,axis=1)).item() for t in [mexu,mexd,metu,metd,metx,mext]])

            ctx.save_for_backward(d, er_eff, er_bda_inter, er_bda_tempo,
                                  mvxu,mvxd,mexu,mexd,avxu,avxd,
                                  mvtu,mvtd,metu,metd,avtu,avtd,
                                  mvxt,mvtx,mext,metx,avxt,avtx,
                                  torch.tensor(max_num_iters))
            return marginal

        
        @staticmethod
        def backward(ctx, grad_output):
            if len(ctx.saved_tensors) == 9:
                grad_ed, grad_rx = LBP_LTL_LOCAL_PytorchNative.MARG.backward(ctx,grad_output)
                return grad_ed, grad_rx, None, None
            (d, er_eff, er_bda_inter, er_bda_tempo, 
             mvxu,mvxd,mexu,mexd,avxu,avxd,
             mvtu,mvtd,metu,metd,avtu,avtd,
             mvxt,mvtx,mext,metx,avxt,avtx,
             max_num_iters) = ctx.saved_tensors
            b,c,sx = d.shape[0],d.shape[1],d.shape[2]
            ct = er_bda_tempo.shape[1]
            max_num_iters = int(max_num_iters.cpu().item())
            ed = torch.exp(d)
            grad_ed = torch.zeros_like(ed)
            grad_rx = torch.zeros_like(er_eff)

            #exit immediately if no gradient
            if torch.amax(torch.abs(grad_output)) == 0:
                return grad_ed, grad_rx, torch.zeros_like(er_bda_inter), torch.zeros_like(er_bda_tempo)

            #backprop through final marginal prob calculation
            grad_mexu = grad_output[:,:,1:] / mexu
            grad_mexd = grad_output[:,:,:-1] / mexd
            grad_metx = grad_output / metx
            grad_mext = torch.zeros_like(mext)
            grad_metd = torch.zeros_like(metd)
            grad_metu = torch.zeros_like(metu)

            iter = 0
            while True:

                #backprop through edge-to-vertex messages in the regular temporal space
                grad_mvxd = torch.sum(er_eff*torch.unsqueeze(grad_mexd,2),1)
                grad_mvxu = torch.sum(er_eff*torch.unsqueeze(grad_mexu,1),2)
                if grad_rx.shape[-1] == 1:
                    if grad_rx.shape[0] == 1:
                        update = torch.unsqueeze(mvxu,2)*torch.unsqueeze(grad_mexu,1) + torch.unsqueeze(mvxd,1)*torch.unsqueeze(grad_mexd,2)
                        grad_rx += torch.sum(torch.sum(update,-1,keepdim=True),0,keepdim=True)
                    else:
                        grad_rx += torch.sum(torch.unsqueeze(mvxu,2)*torch.unsqueeze(grad_mexu,1) + torch.unsqueeze(mvxd,1)*torch.unsqueeze(grad_mexd,2),-1,keepdim=True)
                else:
                    grad_rx += torch.unsqueeze(mvxu,2)*torch.unsqueeze(grad_mexu,1) + torch.unsqueeze(mvxd,1)*torch.unsqueeze(grad_mexd,2)

                #... and in the BDA space and between spaces
                grad_mvtd = torch.sum(er_bda_tempo*torch.unsqueeze(grad_metd,2),1)
                grad_mvtu = torch.sum(er_bda_tempo*torch.unsqueeze(grad_metu,1),2)
                grad_mvxt = torch.sum(er_bda_inter*torch.unsqueeze(grad_mext,1),2)
                grad_mvtx = torch.sum(er_bda_inter*torch.unsqueeze(grad_metx,2),1)

                #backprob through normalisation (to finish)
                grad_mvxd = (grad_mvxd - torch.sum(grad_mvxd*mvxd,1,keepdim=True)) / avxd
                grad_mvxu = (grad_mvxu - torch.sum(grad_mvxu*mvxu,1,keepdim=True)) / avxu
                grad_mvxt = (grad_mvxt - torch.sum(grad_mvxt*mvxt,1,keepdim=True)) / avxt
                for i in range(ct//3):
                    grad_mvtu[:,(3*i):(3*i+3),:] = (grad_mvtu[:,(3*i):(3*i+3),:] - torch.sum(grad_mvtu[:,(3*i):(3*i+3),:]*mvtu[:,(3*i):(3*i+3),:],1,keepdim=True)) / avtu[:,i:(i+1),:]
                    grad_mvtd[:,(3*i):(3*i+3),:] = (grad_mvtd[:,(3*i):(3*i+3),:] - torch.sum(grad_mvtd[:,(3*i):(3*i+3),:]*mvtd[:,(3*i):(3*i+3),:],1,keepdim=True)) / avtd[:,i:(i+1),:]
                    grad_mvtx[:,(3*i):(3*i+3),:] = (grad_mvtx[:,(3*i):(3*i+3),:] - torch.sum(grad_mvtx[:,(3*i):(3*i+3),:]*mvtx[:,(3*i):(3*i+3),:],1,keepdim=True)) / avtx[:,i:(i+1),:]


                #backprob through vertex-to-edge messages
                #clear buffers to start
                grad_mexu *= 0
                grad_mexd *= 0
                grad_metu *= 0
                grad_metd *= 0
                grad_metx *= 0
                grad_mext *= 0

                ##compute vertex to edge messages in label layer (through time)
                #mvxd = ed[:,:,1:] * metx[:,:,1:]
                #mvxd[:,:,:-1] *= mexd[:,:,1:]
                grad_metx[:,:,1:-1] += ed[:,:,1:-1] * mexd[:,:,:-1] * grad_mvxd[:,:,:-1]
                grad_metx[:,:,-1] += ed[:,:,-1] * grad_mvxd[:,:,-1]
                grad_ed[:,:,1:-1] += metx[:,:,1:-1] * mexd[:,:,:-1] * grad_mvxd[:,:,:-1]
                grad_ed[:,:,-1] += metx[:,:,-1] * grad_mvxd[:,:,-1]
                grad_mexd[:,:,1:] += ed[:,:,1:-1] * metx[:,:,1:-1] * grad_mvxd[:,:,:-1]

                #mvxu = ed[:,:,:-1]* metx[:,:,:-1]
                #mvxu[:,:,1:] *= mexu[:,:,:-1]
                grad_metx[:,:,1:-1] += ed[:,:,1:-1] * mexu[:,:,:-1] * grad_mvxu[:,:,1:]
                grad_metx[:,:,0] += ed[:,:,0] * grad_mvxu[:,:,0]
                grad_ed[:,:,1:-1] += metx[:,:,1:-1] * mexu[:,:,:-1] * grad_mvxu[:,:,1:]
                grad_ed[:,:,0] += metx[:,:,0] * grad_mvxu[:,:,0]
                grad_mexu[:,:,:-1] += ed[:,:,1:-1] * metx[:,:,1:-1]* grad_mvxu[:,:,1:]

                ##compute vertex to edge messages in BDA layer (through time)
                #mvtd = mext[:,:,1:].clone()
                #mvtd[:,:,:-1] *= metd[:,:,1:]
                grad_mext[:,:,1:-1] += metd[:,:,:-1] * grad_mvtd[:,:,:-1]
                grad_mext[:,:,-1] += grad_mvtd[:,:,-1]
                grad_metd[:,:,1:] += mext[:,:,1:-1] * grad_mvtd[:,:,:-1]

                #mvtu = mext[:,:,:-1].clone()
                #mvtu[:,:,1:] *= metu[:,:,:-1]
                grad_mext[:,:,1:-1] += metu[:,:,:-1] * grad_mvtu[:,:,1:]
                grad_mext[:,:,0] += grad_mvtu[:,:,0]
                grad_metu[:,:,:-1] += mext[:,:,1:-1]* grad_mvtu[:,:,1:]

                ##compute vertex to edge messages from the labels to the BDA (per each time)
                #mvxt = torch.ones((b,c,sx),device=d.device)
                #mvxt[:,:,:-1] *= mexd
                #mvxt[:,:,1:] *= mexu
                grad_mexd[:,:,0] += grad_mvxt[:,:,0]
                grad_mexd[:,:,1:] += mexu[:,:,:-1]*grad_mvxt[:,:,1:-1]
                grad_mexu[:,:,-1] += grad_mvxt[:,:,-1]
                grad_mexu[:,:,:-1] += mexd[:,:,1:]*grad_mvxt[:,:,1:-1]

                ##compute vertex to edge messages from the BDA to the labels (per each time)
                #mvtx = torch.ones((b,ct,sx),device=d.device)
                #mvtx[:,:,:-1] *= metd
                #mvtx[:,:,1:] *= metu
                grad_metd[:,:,0] += grad_mvtx[:,:,0]
                grad_metd[:,:,1:] += metu[:,:,:-1]*grad_mvtx[:,:,1:-1]
                grad_metu[:,:,-1] += grad_mvtx[:,:,-1]
                grad_metu[:,:,:-1] += metd[:,:,1:]*grad_mvtx[:,:,1:-1]
                
                iter+=1
                if iter >= max_num_iters:
                    break

            #correct to no longer be in the exponential domain
            grad_ed *= ed
            grad_ed += grad_output

            return grad_ed, grad_rx, torch.zeros_like(er_bda_inter), torch.zeros_like(er_bda_tempo)

    
    class MAP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, d, er_eff, er_bda_inter, er_bda_tempo):
            b,c,sx = d.shape[0],d.shape[1],d.shape[2]
            ct = er_bda_tempo.shape[1]

            if ct == 0:
                return LBP_LTL_LOCAL_PytorchNative.MAP.forward(ctx,d,er_eff)
            r_eff = torch.log(er_eff)
            r_bda_inter = torch.log(er_bda_inter)
            r_bda_tempo = torch.log(er_bda_tempo)

            mvxu = torch.zeros((b,c,sx-1),device=d.device)
            mvxd = torch.zeros((b,c,sx-1),device=d.device)
            mexu = torch.zeros((b,c,sx-1),device=d.device)
            mexd = torch.zeros((b,c,sx-1),device=d.device)

            mvtu = torch.zeros((b,ct,sx-1),device=d.device)
            mvtd = torch.zeros((b,ct,sx-1),device=d.device)
            metu = torch.zeros((b,ct,sx-1),device=d.device)
            metd = torch.zeros((b,ct,sx-1),device=d.device)

            mvxt = torch.zeros((b,c,sx),device=d.device)
            mvtx = torch.zeros((b,ct,sx),device=d.device)
            mext = torch.zeros((b,ct,sx),device=d.device)
            metx = torch.zeros((b,c,sx),device=d.device)

            max_num_iters = 2*sx-1
            
            iter = 0
            while True:
                #print("\nMAP Iteration:",iter)

                #compute vertex to edge messages in label layer (through time)
                mvxd = d[:,:,1:] + metx[:,:,1:]
                mvxd[:,:,:-1] += mexd[:,:,1:]
                mvxu = d[:,:,:-1] + metx[:,:,:-1]
                mvxu[:,:,1:] += mexu[:,:,:-1]
                #compute vertex to edge messages in BDA layer (through time)
                mvtd = mext[:,:,1:].clone()
                mvtd[:,:,:-1] += metd[:,:,1:]
                mvtu = mext[:,:,:-1].clone()
                mvtu[:,:,1:] += metu[:,:,:-1]
                #compute vertex to edge messages from the labels to the BDA (per each time)
                mvxt = torch.zeros((b,c,sx),device=d.device)
                mvxt[:,:,:-1] += mexd
                mvxt[:,:,1:] += mexu
                #compute vertex to edge messages from the BDA to the labels (per each time)
                mvtx = torch.zeros((b,ct,sx),device=d.device)
                mvtx[:,:,:-1] += metd
                mvtx[:,:,1:] += metu

                #compute edge to vertex messages
                mexd = torch.amax(r_eff + torch.unsqueeze(mvxd,1), dim=2)
                mexu = torch.amax(r_eff + torch.unsqueeze(mvxu,2), dim=1)
                metd = torch.amax(r_bda_tempo + torch.unsqueeze(mvtd,1), dim=2)
                metu = torch.amax(r_bda_tempo + torch.unsqueeze(mvtu,2), dim=1)
                mext = torch.amax(r_bda_inter + torch.unsqueeze(mvxt,2), dim=1)
                metx = torch.amax(r_bda_inter + torch.unsqueeze(mvtx,1), dim=2)
                
                #renormalise messages in label layer (should be enough to ensure stability)
                mexu -= torch.amax(mexu,1,keepdim=True)
                mexd -= torch.amax(mexd,1,keepdim=True)

                iter+=1
                if iter >= max_num_iters:
                    break

            #calculate the marginal (logistic domain so normalise by subtracting out the contributions to label 0)
            marginal = d + metx
            marginal[:,:,:-1] += mexd
            marginal[:,:,1:] += mexu

            ctx.save_for_backward(d, er_eff, er_bda_inter, er_bda_tempo,
                                  mvxu,mvxd,mexu,mexd,
                                  mvtu,mvtd,metu,metd,
                                  mvxt,mvtx,mext,metx,
                                  torch.tensor(max_num_iters))
            return marginal

        
        @staticmethod
        def backward(ctx, grad_output):
            if len(ctx.saved_tensors) == 9:
                grad_ed, grad_rx = LBP_LTL_LOCAL_PytorchNative.MAP.backward(ctx,grad_output)
                return grad_ed, grad_rx, None, None
            (d, er_eff, er_bda_inter, er_bda_tempo, 
             mvxu,mvxd,mexu,mexd,
             mvtu,mvtd,metu,metd,
             mvxt,mvtx,mext,metx,
             max_num_iters) = ctx.saved_tensors
            b,c,sx = d.shape[0],d.shape[1],d.shape[2]
            ct = er_bda_tempo.shape[1]
            max_num_iters = int(max_num_iters.cpu().item())
            ed = torch.exp(d)
            grad_d = torch.zeros_like(d)
            grad_rx = torch.zeros_like(er_eff)

            #exit immediately if no gradient
            if torch.amax(torch.abs(grad_output)) == 0:
                return grad_d, grad_rx
            r_eff = torch.log(er_eff)
            r_bda_inter = torch.log(er_bda_inter)
            r_bda_tempo = torch.log(er_bda_tempo)

            #backprop through final marginal prob calculation
            grad_d += grad_output.clone()
            grad_mexu = grad_output[:,:,1:]
            grad_mexd = grad_output[:,:,:-1]
            grad_metx = grad_output
            grad_mext = torch.zeros_like(mext)
            grad_metd = torch.zeros_like(metd)
            grad_metu = torch.zeros_like(metu)

            #compute the max masks
            mask_mvxd = ((r_eff + torch.unsqueeze(mvxd,1)) == torch.amax(r_eff + torch.unsqueeze(mvxd,1), dim=2, keepdims=True)).float()
            num_mask_mvxd = torch.sum(mask_mvxd,dim=2,keepdim=True)
            mask_mvxd /= num_mask_mvxd
            mask_mvxu = ((r_eff + torch.unsqueeze(mvxu,2)) == torch.amax(r_eff + torch.unsqueeze(mvxu,2), dim=1, keepdims=True)).float()
            num_mask_mvxu = torch.sum(mask_mvxu,dim=1,keepdim=True)
            mask_mvxu /= num_mask_mvxu
            
            mask_mvtd = ((r_bda_tempo + torch.unsqueeze(mvtd,1)) == torch.amax(r_bda_tempo + torch.unsqueeze(mvtd,1), dim=2, keepdims=True)).float()
            num_mask_mvtd = torch.sum(mask_mvtd,dim=2,keepdim=True)
            mask_mvtd /= num_mask_mvtd
            mask_mvtu = ((r_bda_tempo + torch.unsqueeze(mvtu,2)) == torch.amax(r_bda_tempo + torch.unsqueeze(mvtu,2), dim=1, keepdims=True)).float()
            num_mask_mvtu = torch.sum(mask_mvtu,dim=1,keepdim=True)
            mask_mvtu /= num_mask_mvtu
            
            mask_mvxt = ((r_bda_inter + torch.unsqueeze(mvxt,2)) == torch.amax(r_bda_inter + torch.unsqueeze(mvxt,2), dim=1, keepdims=True)).float()
            num_mask_mvxt = torch.sum(mask_mvxt,dim=1,keepdim=True)
            mask_mvxt /= num_mask_mvxt
            mask_mvtx = ((r_bda_inter + torch.unsqueeze(mvtx,1)) == torch.amax(r_bda_inter + torch.unsqueeze(mvtx,1), dim=2, keepdims=True)).float()
            num_mask_mvtx = torch.sum(mask_mvtx,dim=2,keepdim=True)
            mask_mvtx /= num_mask_mvtx
            
            mask_aexu = (mexu == 0).float()
            num_mask_aexu = torch.sum(mask_aexu,dim=1,keepdim=True)
            mask_aexu /= num_mask_aexu
            mask_aexd = (mexd == 0).float()
            num_mask_aexd = torch.sum(mask_aexd,dim=1,keepdim=True)
            mask_aexd /= num_mask_aexd
            
            iter = 0
            while True:

                #backprob through normalisation
                grad_mexu = grad_mexu - mask_aexu * torch.sum(grad_mexu,1,keepdims=True)
                grad_mexd = grad_mexd - mask_aexd * torch.sum(grad_mexd,1,keepdims=True)
                
                #backprop through edge-to-vertex messages in the regular temporal space
                grad_mvxd = torch.sum(mask_mvxd*torch.unsqueeze(grad_mexd,2),1)
                grad_mvxu = torch.sum(mask_mvxu*torch.unsqueeze(grad_mexu,1),2)
                update = mask_mvxu*torch.unsqueeze(grad_mexu,1) + mask_mvxd*torch.unsqueeze(grad_mexd,2)
                if grad_rx.shape[-1] == 1:
                    if grad_rx.shape[0] == 1:
                        grad_rx += torch.sum(torch.sum(update,-1,keepdim=True),0,keepdim=True)
                    else:
                        grad_rx += torch.sum(update,-1,keepdim=True)
                else:
                    grad_rx += update

                #... and in the BDA space and between spaces
                grad_mvtd = torch.sum(mask_mvtd*torch.unsqueeze(grad_metd,2),1)
                grad_mvtu = torch.sum(mask_mvtu*torch.unsqueeze(grad_metu,1),2)
                grad_mvxt = torch.sum(mask_mvxt*torch.unsqueeze(grad_mext,1),2)
                grad_mvtx = torch.sum(mask_mvtx*torch.unsqueeze(grad_metx,2),1)


                #backprob through vertex-to-edge messages
                #clear buffers to start
                grad_mexu *= 0
                grad_mexd *= 0
                grad_metu *= 0
                grad_metd *= 0
                grad_metx *= 0
                grad_mext *= 0

                ##compute vertex to edge messages in label layer (through time)
                #mvxd = ed[:,:,1:] * metx[:,:,1:]
                #mvxd[:,:,:-1] *= mexd[:,:,1:]
                grad_metx[:,:,1:] += grad_mvxd
                grad_d[:,:,1:] += grad_mvxd
                grad_mexd[:,:,1:] += grad_mvxd[:,:,:-1]

                #mvxu = ed[:,:,:-1]* metx[:,:,:-1]
                #mvxu[:,:,1:] *= mexu[:,:,:-1]
                grad_metx[:,:,:-1] += grad_mvxu
                grad_d[:,:,:-1] += grad_mvxu
                grad_mexu[:,:,:-1] += grad_mvxu[:,:,1:]

                ##compute vertex to edge messages in BDA layer (through time)
                #mvtd = mext[:,:,1:].clone()
                #mvtd[:,:,:-1] *= metd[:,:,1:]
                grad_mext[:,:,1:] += grad_mvtd
                grad_metd[:,:,1:] += grad_mvtd[:,:,:-1]

                #mvtu = mext[:,:,:-1].clone()
                #mvtu[:,:,1:] *= metu[:,:,:-1]
                grad_mext[:,:,:-1] += grad_mvtu
                grad_metu[:,:,:-1] += grad_mvtu[:,:,1:]

                ##compute vertex to edge messages from the labels to the BDA (per each time)
                #mvxt = torch.ones((b,c,sx),device=d.device)
                #mvxt[:,:,:-1] *= mexd
                #mvxt[:,:,1:] *= mexu
                grad_mexd += grad_mvxt[:,:,:-1]
                grad_mexu += grad_mvxt[:,:,1:]

                ##compute vertex to edge messages from the BDA to the labels (per each time)
                #mvtx = torch.ones((b,ct,sx),device=d.device)
                #mvtx[:,:,:-1] *= metd
                #mvtx[:,:,1:] *= metu
                grad_metd += grad_mvtx[:,:,:-1]
                grad_metu += grad_mvtx[:,:,1:]

                iter+=1
                if iter >= max_num_iters:
                    break
                    
            #move regularisation terms back into logarithmic domain
            grad_erx = grad_rx * r_eff

            return grad_d, grad_erx, torch.zeros_like(er_bda_inter), torch.zeros_like(er_bda_tempo)
