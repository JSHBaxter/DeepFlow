import torch
from torch_geometric.data import Data, Batch
from itertools import product

class Geometric_LBP_PytorchNative(torch.autograd.Function):

    epsilon = 10**-4
    
    @staticmethod
    def internal_forward_binary(graph: Data):
        N,c = graph.x.shape[0], graph.x.shape[1]
        e = graph.edge_index
        
        #checks on input shape
        if len(graph.x.shape) > 2:
            raise Exception("The vertices can only have a single dimensional channel")
        if not hasattr(graph,edge_attr):
            raise Exception("Missing regularisation terms")
        if len(graph.edge_attr.shape) != 2:
            raise Exception("The edges can only have a single regularisation term per channel")
        if graph.edge_attr.shape[1] != N:
            raise Exception("The edges have incorrect number of terms. Have " + str(graph.edge_attr.shape[1]) "; expecting "+ str(N))
        
        #change weights to exponential domain
        ed = torch.exp(graph.x)
        er = torch.exp(-graph.edge_attr)
        meu0, meu1, med0, med1 = torch.ones_like(er), torch.ones_like(er), torch.ones_like(er), torch.ones_like(er)
        
        #get the L_infinity decay constant %TODO
        max_num_iters = 10
        
        iter = 0
        while True:
            
            #compute vertex to edge messages
            mvu0, mvu1, mvd0, mvd1 = torch.ones_like(er), torch.ones_like(er), torch.ones_like(er), torch.ones_like(er)
            for (oe_idx, out_edge), (ie_idx, in_edge) in product(enumerate(zip(graph.edge_index[0],graph.edge_index[1])),
                                                                 enumerate(zip(graph.edge_index[0],graph.edge_index[1]))):
                if oe_idx == ie_idx: #edges are the same so skip
                    continue
                if out_edge[0] == in_edge[0]:
                    mvu0[oe_idx] *= med0[ie_idx]
                    mvu1[oe_idx] *= med1[ie_idx]
                elif out_edge[0] == in_edge[1]:
                    mvu0[oe_idx] *= meu0[ie_idx]
                    mvu1[oe_idx] *= meu1[ie_idx]
                elif out_edge[1] == in_edge[0]:
                    mvd0[oe_idx] *= med0[ie_idx]
                    mvd1[oe_idx] *= med1[ie_idx]
                elif out_edge[1] == in_edge[1]:
                    mvd0[oe_idx] *= meu0[ie_idx]
                    mvd1[oe_idx] *= meu1[ie_idx]
                #else edges don't connect at a shared vertex so skip
                
            #renormalise messages
            avu, avd = mvu0+mvu1, mvd0+mvd1
            mvu0, mvu1, mvd0, mvd1 = mvu0/avu, mvu1/avu, mvd0/avd, mvd1/avd
            
            #compute vertex to edge messages
            meu0 = mvd0 + er*mvd1
            meu1 = mvd1 + er*mvd0
            med0 = mvu0 + er*mvu1
            med1 = mvu1 + er*mvu0
            
            iter+=1
            if iter >= max_num_iters:
                break
                
        #calculate the marginals
        marginal = d.clone()
        for idx_edge, edge in enumerate(zip(graph.edge_index[0],graph.edge_index[1])):
            marginal[edge[0]] += log(med1[idx_edge])-log(med0[idx_edge])
            marginal[edge[1]] += log(meu1[idx_edge])-log(meu0[idx_edge])
        
        #return tensors for saving
        context = (ed, er, graph.edge_index, mvu0*avu, mvu1*avu, mvd0*avd, mvd1*avd)
        return marginal, len(context), context
    
    @staticmethod
    def internal_forward_categorical(graph: Data):
        N,c = graph.x.shape[0], graph.x.shape[1]
        e = graph.edge_index
        
        #checks on input shape
        if len(graph.x.shape) > 2:
            raise Exception("The vertices can only have a single dimensional channel")
        if not hasattr(graph,edge_attr):
            raise Exception("Missing regularisation terms")
        if len(graph.edge_attr.shape) < 2 or len(graph.edge_attr.shape) > 3:
            raise Exception("The edges can only have either a single regularisation term per channel, or one for each pair of channels")
        if graph.edge_attr.shape[1] != N:
            raise Exception("The edges have incorrect number of terms. Have " + str(graph.edge_attr.shape[1]) "; expecting "+ str(N))
        if len(graph.edge_attr.shape) == 3 and graph.edge_attr.shape[2] != N:
            raise Exception("The edges have incorrect number of terms. Have " + str(graph.edge_attr.shape[2]) "; expecting "+ str(N))
            
        pairwise = (len(graph.edge_attr.shape) == 3)
        
        #change weights to exponential domain
        ed = torch.exp(graph.x)
        er = torch.exp(-graph.edge_attr)
        if pairwise:
            meu, med = torch.ones_like(er[:,:,0]), torch.ones_like(er[:,:,0])
            #set the values along the main diagonal to be unit (saves time later)
            for i in range(c):
                er[:,i,i] = 1
        else:
            meu, med = torch.ones_like(er), torch.ones_like(er)
        
        #get the L_infinity decay constant %TODO
        max_num_iters = 10
        
        iter = 0
        while True:
            
            #compute vertex to edge messages
            mvu, mvd = torch.ones_like(meu), torch.ones_like(med)
            for (oe_idx, out_edge), (ie_idx, in_edge) in product(enumerate(zip(graph.edge_index[0],graph.edge_index[1])),
                                                                 enumerate(zip(graph.edge_index[0],graph.edge_index[1]))):
                if oe_idx == ie_idx: #edges are the same so skip
                    continue
                if out_edge[0] == in_edge[0]:
                    mvu[oe_idx] *= med[ie_idx]
                elif out_edge[0] == in_edge[1]:
                    mvu[oe_idx] *= meu[ie_idx]
                elif out_edge[1] == in_edge[0]:
                    mvd[oe_idx] *= med[ie_idx]
                elif out_edge[1] == in_edge[1]:
                    mvd[oe_idx] *= meu[ie_idx]
                #else edges don't connect at a shared vertex so skip
                
            #renormalise messages
            avu, avd = torch.sum(mvu,dim=-1), avd = torch.sum(mvd,dim=-1)
            mvu, mvd = mvu/avu, mvd/avd
            
            #compute vertex to edge messages
            if pairwise:
                med = torch.sum(er * torch.unsqueeze(mvd,1), dim=2)
            else:
                med = mvu.clone()
                a = torch.sum(er * mvd, 1, keepdim=True)
                med = += erx[:,:,:-1] * (a - erx[:,:,:-1]*mvxd)
            
            iter+=1
            if iter >= max_num_iters:
                break
                
        #calculate the marginals
        marginal = d.clone()
        for idx_edge, edge in enumerate(zip(graph.edge_index[0],graph.edge_index[1])):
            marginal[edge[0]] += log(med)
            marginal[edge[1]] += log(meu)
        
        #return tensors for saving
        context = (ed, er, graph.edge_index, mvu*avu, mvd*avd)
        return marginal, len(context), context
    
    @staticmethod
    def forward(ctx, graph, output_type="binary"):
        
        if output_type != "binary" and output_type != categorical:
            raise Exception("Invalid model type: " + str(output_type))
        
        results = []
        context = []
        if type(graph) is Batch:
            for i in range(b):
                if output_type == "binary":
                    r,n,c = internal_forward_binary(graph[i])
                else:
                    r,n,c = internal_forward_categorical(graph[i])
                results.append(Data(x=r,edge_index=graph[i].edge_index.clone()))
                context += list(c)
        elif type(graph) is Data:
            if output_type == "binary":
                r,n,c = internal_forward_binary(graph)
            else:
                r,n,c = internal_forward_categorical(graph)
            context = list(c)
            results = Data(x=r,edge_index=graph.edge_index.clone())
            
        #save context for backwards pass and return the marginal
        ctx.save_for_backward(tuple(context))
        ctx.n = n
        
        #return the estimates
        if type(graph) is Batch:
            return Batch.from_data_list(results)
        elif type(graph) is Data:
            return results
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        
        
        
        ed,erx,mvxu,mvxd,mexu,mexd,avxu,avxd,max_num_iters = ctx.saved_tensors
        b,c,sx = ed.shape[0],ed.shape[1],ed.shape[2]
        max_num_iters = int(max_num_iters.cpu().item())
        grad_d =  torch.zeros_like(ed) #(will multiply out the ed at the end, keeps everything in exponential domain)
        grad_rx = torch.zeros_like(erx)
        prod_me = torch.product(me,dim=2)
        
        #backprop through final layer
        #marginal = d.clone() + torch.sum(torch.log(me,dim=2))
        grad_me = torch.unsqueeze(grad_output,2) / me
        
        iter = 0
        while True:
            
            #backprop back to vertex messages
            #me = torch.sum(er * torch.unsqueeze(torch.permute(mv,(0,1,3,2)),dim=2), dim=2)
            grad_mv = torch.permute(torch.sum(torch.unsqueeze(grad_me,2)*er,2),(0,1,3,2))
            grad_rx += torch.unsqueeze(grad_me,2)*torch.unsqueeze(torch.permute(mv,(0,1,3,2)),dim=2)
                        
            #backprop through renormalisation
            grad_mv = (grad_mv - torch.sum(grad_mv*mv,1,keepdim=True)) / av
            
            #backprop back to edge messages //TODO
            #mv = ed * torch.product(me,dim=2) / torch.permute(me,(0,1,3,2))
            grad_ed = torch.sum(grad_mv * (torch.product(me,dim=2) / torch.permute(me,(0,1,3,2))), -1)
            grad_me = -torch.permute(grad_mv * ed * torch.product(me,dim=2),(0,1,3,2)) / me**2
            for j in range(N):
                grad_me += torch.unsqueeze((grad_mv[:,:,:,j] * ed * prod_me),2) / (me*me[:,:,j,:])
            
            iter+=1
            if iter >= max_num_iters:
                break
        
        #correct to no longer be in the exponential domain
        grad_rx *= erx
        grad_d = grad_d*ed + grad_output
        
        return grad_d, grad_rx