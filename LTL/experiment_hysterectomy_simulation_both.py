import numpy as np
from scipy.stats import ortho_group
from scipy.ndimage import gaussian_filter1d
import torch
from LTL_DeepFlow import *
from LTL_DeepFlow_test import find_rule_break
import copy
import random
import pickle
from itertools import product
import os

def make_label_hierarchy():
    
    leaves = [LTL_LeafNode("Surgical scene preparation",0),
              LTL_LeafNode("Surgical approach left",1),
              LTL_LeafNode("Adnexal resection left",2),
              LTL_LeafNode("Surgical approach right",3),
              LTL_LeafNode("Adnexal resection right",4),
              LTL_LeafNode("Anterior dissection",5),
              LTL_LeafNode("Colpostomy and uterus removal",6),
              LTL_LeafNode("Closing and end of procedure",7)]
    b_left = LTL_GroupNode("Left-side phases",8,[leaves[1],leaves[2]])
    b_right = LTL_GroupNode("Right-side phases",9,[leaves[3],leaves[4]])
    b_lat = LTL_GroupNode("Lateral phases",10,[b_left,b_right])
    structure = LTL_GroupingStructure(leaves,[b_left,b_right,b_lat])
    plain_structure = LTL_GroupingStructure(leaves,[])
    
    constraints = [(leaves[0],leaves[0],"INITIAL"),
                   (leaves[1],leaves[2],"PRECEDE"),
                   (leaves[2],leaves[1],"SUCEED"),
                   (leaves[3],leaves[4],"PRECEDE"),
                   (leaves[4],leaves[3],"SUCEED"),
                   (leaves[0],b_lat,"PRECEDE"),
                   (b_lat,leaves[0],"SUCEED"),
                   (b_lat,leaves[5],"PRECEDE"),
                   (leaves[5],b_lat,"SUCEED"),
                   (leaves[5],leaves[6],"PRECEDE"),
                   (leaves[6],leaves[5],"SUCEED"),
                   (leaves[6],leaves[7],"PRECEDE"),
                   (leaves[7],leaves[6],"SUCEED"),
                   (leaves[7],leaves[7],"FINAL")]
    
    constraints_iSPM1 = [(leaves[0],leaves[0],"INITIAL"),
                   (leaves[0],leaves[1],"PRECEDE"),
                   (leaves[1],leaves[0],"SUCEED"),
                   (leaves[1],leaves[2],"PRECEDE"),
                   (leaves[2],leaves[1],"SUCEED"),
                   (leaves[2],leaves[3],"PRECEDE"),
                   (leaves[3],leaves[2],"SUCEED"),
                   (leaves[3],leaves[4],"PRECEDE"),
                   (leaves[4],leaves[3],"SUCEED"),
                   (leaves[5],leaves[4],"SUCEED"),
                   (leaves[5],leaves[6],"PRECEDE"),
                   (leaves[6],leaves[5],"SUCEED"),
                   (leaves[6],leaves[7],"PRECEDE"),
                   (leaves[7],leaves[6],"SUCEED"),
                   (leaves[7],leaves[7],"FINAL")]
    
    constraints_iSPM2 = [(leaves[0],leaves[0],"INITIAL"),
                   (leaves[0],leaves[3],"PRECEDE"),
                   (leaves[3],leaves[0],"SUCEED"),
                   (leaves[3],leaves[4],"PRECEDE"),
                   (leaves[4],leaves[3],"SUCEED"),
                   (leaves[4],leaves[1],"PRECEDE"),
                   (leaves[1],leaves[4],"SUCEED"),
                   (leaves[1],leaves[2],"PRECEDE"),
                   (leaves[2],leaves[1],"SUCEED"),
                   (leaves[5],leaves[2],"SUCEED"),
                   (leaves[5],leaves[6],"PRECEDE"),
                   (leaves[6],leaves[5],"SUCEED"),
                   (leaves[6],leaves[7],"PRECEDE"),
                   (leaves[7],leaves[6],"SUCEED"),
                   (leaves[7],leaves[7],"FINAL")]
    
    uniqueness = [(leaves[0],leaves[0],"UNIQUE"),
                  (leaves[1],leaves[1],"UNIQUE"),
                  (leaves[2],leaves[2],"UNIQUE"),
                  (leaves[3],leaves[3],"UNIQUE"),
                  (leaves[4],leaves[4],"UNIQUE"),
                  (leaves[5],leaves[5],"UNIQUE"),
                  (leaves[6],leaves[6],"UNIQUE"),
                  (leaves[7],leaves[7],"UNIQUE"),
                  (b_left,b_left,"UNIQUE"),
                  (b_right,b_right,"UNIQUE"),
                  (b_lat,b_lat,"UNIQUE"),]
    
    return structure, constraints, plain_structure, {"LeftFirst":constraints_iSPM1, "RightFirst":constraints_iSPM2}, uniqueness






#create very simple CNN model for signal segmentation
class ExperimentModel(torch.nn.Module):
    def __init__(self,in_channels,out_channels,conv_width,kernel_width,num_convs=3) -> None:
        super().__init__()
        self.convF = torch.nn.Conv1d(in_channels,conv_width,kernel_width,padding="same")
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(conv_width,conv_width,kernel_width,padding="same") for i in range(num_convs-2)])
        self.convL = torch.nn.Conv1d(conv_width,out_channels,kernel_width,padding="same")
        self.dropout = torch.nn.ModuleList([torch.nn.Dropout1d(0.25) for i in range(num_convs)])
    def forward(self,x):
        relu = torch.relu
        x = relu(self.convF(self.dropout[0](x)))
        for d,c in zip(self.dropout[1:-1],self.convs):
            x = relu(c(d(x)))
        x = self.convL(self.dropout[-1](x))
        return x


class RegularSoftmax(torch.nn.Module):
    
    def forward(self,d):
        return torch.softmax(d,1)
    
    def MAP(self,d,gt_type=None):
        return (d == torch.amax(d,dim=1)).to(float).detach()
    
    def loss(self,gt,d,p=None):
        if p is None:
            p = torch.softmax(d,1)
        return -torch.mean(gt*torch.log(p))
    
class ExperimentStructure(torch.nn.Module):
    
    def __init__(self,structure,constraints):
        super().__init__()
        self.module_ltl = LTL_DeepFlow(structure,constraints,constraint_type="GLOBAL")
    
    def forward(self,d):
        dprime = torch.zeros((d.shape[0],len(self.module_ltl.structure),d.shape[2]))
        dprime[:,:len(self.module_ltl.structure.leaves),:] += d
        marginal = self.module_ltl(dprime)
        return marginal[:,:len(self.module_ltl.structure.leaves),:]
    
    def MAP(self,d,gt_type=None):
        dprime = torch.zeros((d.shape[0],len(self.module_ltl.structure),d.shape[2]))
        dprime[:,:len(self.module_ltl.structure.leaves),:] += d
        orgMAP = self.module_ltl.MAP(dprime).detach()
        return orgMAP[:,:len(self.module_ltl.structure.leaves),:]
    
    def loss(self,gt,d,p):
        dprime = torch.zeros((d.shape[0],len(self.module_ltl.structure),d.shape[2]))
        dprime[:,:len(self.module_ltl.structure.leaves),:] += d
        
        if p is not None:
            pprime = torch.zeros((d.shape[0],len(self.module_ltl.structure),d.shape[2]))
            pprime[:,:len(self.module_ltl.structure.leaves),:] += p
            for c in self.module_ltl.structure.groups:
                for l in c.leaves:
                    pprime[:,c.index,:] += p[:,l.index,:] 
        else:
            pprime = self.module_ltl(dprime)
            p = pprime[:,:len(self.module_ltl.structure.leaves),:]
        
        gtnew = torch.zeros((d.shape[0],len(self.module_ltl.structure),d.shape[2]))
        gtnew[:,:len(self.module_ltl.structure.leaves),:] += gt
        for c in self.module_ltl.structure.groups:
            for l in c.leaves:
                gtnew[:,c.index,:] += gt[:,l.index,:] 
        
        return self.module_ltl.loss(gtnew,dprime,r=None,p=pprime)
    
#create a MAP-only structure for evaluation on iSPM
    
class iSPMStructure(torch.nn.Module):
    
    def __init__(self,structure,constraints_dict):
        super().__init__()
        self.module_ltl = {key:LTL_DeepFlow(structure,constraints_dict[key],constraint_type="LOCAL") for key in constraints_dict.keys()}
    
    def MAP(self,d,iSPM):
        orgMAP = self.module_ltl[iSPM].MAP(d).detach()
        return orgMAP
        
def train(networks, optimisers, train_epochs, training_set, validation_set=None, training_type="training"):
    for epoch in range(train_epochs):
        print("Running " + training_type + " epoch",epoch,end=" ")
        random.shuffle(training_set)
        batch_loss_number = 10
        losses = [0 for n in networks]
        batch_count = 0
        soft_acc = None
        hard_acc = None
        for n in networks:
            n[0].train()
            n[1].train()
        for s_idx,(s,gt) in enumerate(training_set):
            s,gt = torch.Tensor(s).unsqueeze(0), torch.Tensor(gt).unsqueeze(0)
            s = s + 0.01*torch.randn(s.shape)
            pre = [n[0](s) for n in networks]
            post = [n[1](p) for n,p in zip(networks,pre)]
            map_conf = [n[1].MAP(p) for n,p in zip(networks,pre)]
            
            #compute CE losses
            for i,(n,d,pred) in enumerate(zip(networks,pre,post)):
                losses[i] = losses[i]+n[1].loss(gt,d,pred)
            if soft_acc is None:
                soft_acc = np.array([torch.mean(torch.sum(gt*pred,1)).detach().item() for pred in post])
                hard_acc = np.array([torch.mean(torch.sum(gt*pred,1)).detach().item() for pred in map_conf])
            else:
                soft_acc += np.array([torch.mean(torch.sum(gt*pred,1)).detach().item() for pred in post])
                hard_acc += np.array([torch.mean(torch.sum(gt*pred,1)).detach().item() for pred in map_conf])
            #print(epoch,s_idx,soft_acc)
            
            #test if pretrained has failed (i.e. nans) and flag if that is the case
            if np.any(np.isnan(soft_acc)):
                return False
        
            #optimise weights if we are on schedule
            batch_count += 1
            if batch_count == batch_loss_number or s_idx == len(training_set)-1:
                for loss,opt in zip(losses,optimisers):
                    (loss/batch_count).backward()
                    opt.step()
                    opt.zero_grad()
                batch_count = 0
                losses = [0 for n in networks]
        #print some accuracy info so we know where we are, approx
        print("Soft accurary:",soft_acc/len(training_set), "Hard accurary:",hard_acc/len(training_set),end=" " if validation_set is not None else "\n")
        

        if validation_set is None:
            continue
        hard_acc = None
        for n in networks:
            n[0].eval()
            n[1].eval()
        for s_idx,(s,gt) in enumerate(validation_set):
            s,gt = torch.Tensor(s).unsqueeze(0), torch.Tensor(gt).unsqueeze(0)
            s = s + 0.01*torch.randn(s.shape)
            pre = [n[0](s) for n in networks]
            map_conf = [n[1].MAP(p) for n,p in zip(networks,pre)]
            
            #compute CE losses
            if hard_acc is None:
                hard_acc = np.array([torch.mean(torch.sum(gt*pred,1)).detach().item() for pred in map_conf])
            else:
                hard_acc += np.array([torch.mean(torch.sum(gt*pred,1)).detach().item() for pred in map_conf])
        #print some accuracy info so we know where we are, approx
        print("Validation accurary:",hard_acc/len(validation_set))
    return True

#run a single experiment and save the results
def replication(number, num_signals=200, testing_perc=0.5, input_channels=16, kernel_width=3, hidden_channels=64, train_epochs=100, pretrain_epochs=100, sampling_rate=1, learning_rate=10**-4):

    print("Starting repetition",number)
    
    #skip if already done
    filename = "experiment_hysterectomy_simulation_BOTH_rep_"+str(number)+".pkl"
    if os.path.exists(filename):
        return
    
    #get the topological structure
    structure, constraints, plain_structure, constraints_dict, uniqueness = make_label_hierarchy()
    
    #generate the distribution parameters for each phase
    num_phases = len(structure.leaves)
    phase_means = np.random.normal(size=(num_phases,input_channels)) / np.sqrt(input_channels)
    alpha = 0.95
    phase_means[[1,3],:] = phase_means[[1,3],:]*(1-alpha) + alpha*np.mean(phase_means[[1,3],:],0)
    phase_means[[2,4,5],:] = phase_means[[2,4,5],:]*(1-alpha) + alpha*np.mean(phase_means[[2,4,5],:],0)
    phase_cov_P = np.zeros((num_phases,input_channels,input_channels))
    phase_cov_D = np.zeros((num_phases,input_channels,))
    for phase in range(num_phases):
        sigma = np.random.lognormal(size=(input_channels,))
        rot = ortho_group.rvs(dim=input_channels)
        phase_cov_P[phase,:,:] = 1*np.eye(input_channels)+np.matmul(rot*sigma,np.transpose(rot))
    phase_cov_P[[1,3],:,:] = phase_cov_P[[1,3],:,:]*(1-alpha) + alpha*np.sum(phase_cov_P[[1,3],:,:],0)/2
    phase_cov_P[[2,4,5],:,:] = phase_cov_P[[2,4,5],:,:]*(1-alpha) + alpha*np.sum(phase_cov_P[[2,4,5],:,:],0)/2
    for phase in range(num_phases):
        phase_cov_eigs = np.linalg.eigh(phase_cov_P[phase,:,:])
        phase_cov_P[phase,:,:] = phase_cov_eigs[1]
        phase_cov_D[phase,:] = phase_cov_eigs[0]
        
    #get length distribution information
    margin_lengths = int(100/sampling_rate)
    lengths_means = [630.264,82.032,395.5104,82.032,395.5104,682.4592,504.9648,236.32992]
    lengths_stds  = [111.2064,73.8336,228.5328,73.8336,228.5328,130.224,274.9968,122.00832]

    #generate the signals and ground truths
    topology = []
    labels = []
    labels_one_hot = []
    signals = []
    for signal_idx in range(num_signals):

        #get a random topology
        if np.random.uniform()<0.5:
            topology.append([0,1,2,3,4,5,6,7])
        else:
            topology.append([0,3,4,1,2,5,6,7])

        #generate signal ground truth labels
        labels.append([])
        for it, t in enumerate(topology[-1]):
            length = -1
            while length < 1:
                length = int(np.random.poisson(lam=lengths_means[t])/sampling_rate + 0.5)
            length += margin_lengths if (it == 0 or it == len(topology[-1])-1) else 0
            labels[-1] += length*[t,]

        #generate non-noised features
        devs = np.random.normal(size=(input_channels,))
        signal = np.transpose(np.array([phase_means[l,:]+np.matmul(phase_cov_P[l,:,:],phase_cov_D[l,:]*devs) for l in labels[-1]]))

        #smooth the signals
        signal = gaussian_filter1d(signal, int(20/sampling_rate), axis=1)
        
        #truncate, add noise and append to list
        labels[-1] = np.array(labels[-1][margin_lengths:-margin_lengths])
        label_one_hot = np.zeros(shape=(num_phases,len(labels[-1])))
        for t,l in enumerate(labels[-1]):
            label_one_hot[l,t] = 1
        labels_one_hot.append(label_one_hot)
        signal = signal[:,margin_lengths:-margin_lengths].copy()
        signal += np.random.normal(size=signal.shape)
        signals.append(signal)

    #split the data
    training_set = list(zip(signals[:int((1-testing_perc)*num_signals)],labels_one_hot[:int((1-testing_perc)*num_signals)]))
    testing_set = list(zip(signals[int((1-testing_perc)*num_signals):],labels_one_hot[int((1-testing_perc)*num_signals):]))
    
    #create the underlying network and pretrain
    while True:
        baseNetwork = (ExperimentModel(input_channels,len(structure.leaves),hidden_channels,kernel_width,3),RegularSoftmax(),"REG")
        networks = [baseNetwork,]
        baseoptimiser = torch.optim.Adam(baseNetwork[0].parameters(), lr=learning_rate)
        optimisers = [baseoptimiser,]
        pretrain_sucessful = train(networks, optimisers, pretrain_epochs, training_set, testing_set, "pre-training")
        if pretrain_sucessful:
            break
    
    #copy the network and optimiser over
    networks.append((copy.deepcopy(networks[0][0]), 
                     ExperimentStructure(structure,constraints),
                     "gSPM-L"))
    networks.append((copy.deepcopy(networks[0][0]), 
                     ExperimentStructure(structure,constraints+uniqueness),
                     "gSPM-G"))
    optimisers = [torch.optim.Adam(n[0].parameters(), lr=learning_rate) for n in networks]
    for o in optimisers:
        o.load_state_dict(baseoptimiser.state_dict())
    
        
    #finish the training
    training_passed = train(networks, optimisers, train_epochs, training_set, testing_set, "training")
    if not training_passed:
        print("Training failed")
        exit()

    #run the evaluation
    networks.append((None,iSPMStructure(plain_structure, constraints_dict),"iSPM"))
    acc_results = {(n_map,n_train) : [] for n_map,n_train in product(networks,networks)}
    ord_results = {(n_map,n_train) : [] for n_map,n_train in product(networks,networks)}
    uni_results = {(n_map,n_train) : [] for n_map,n_train in product(networks,networks)}
    for n in networks:
        if n[0] is not None:
            n[0].eval()
        n[1].eval()
    for s,gt in testing_set:
        s,gt = torch.Tensor(s).unsqueeze(0), torch.Tensor(gt).unsqueeze(0)
        
        #get MAP estimates
        for n1,n0 in acc_results.keys():
            if n0[0] is None:
                continue
            
            first_per_row = np.apply_along_axis(lambda x: np.flatnonzero(x)[0] if np.flatnonzero(x).size > 0 else -1, axis=1, arr=gt[0,...])
            
            gt_type = "LeftFirst" if first_per_row[1] < first_per_row[3] else "RightFirst"
                
            pred = n1[1].MAP(n0[0](s),gt_type)
            ord_broken = find_rule_break(structure,constraints,signal_onehot=pred[0,...])
            uni_broken = find_rule_break(structure,uniqueness,signal_onehot=pred[0,...])
            acc_results[(n1,n0)].append(torch.mean(torch.sum(gt*pred,1)).detach().cpu().item())
            ord_results[(n1,n0)].append(len(ord_broken))
            uni_results[(n1,n0)].append(len(uni_broken))
            
    for n1,n0 in acc_results.keys():
        acc_results[(n1,n0)] = np.array(acc_results[(n1,n0)])
        ord_results[(n1,n0)] = np.array(ord_results[(n1,n0)])
        uni_results[(n1,n0)] = np.array(uni_results[(n1,n0)])
    
    print("Results of repetition",number)
    for MAPN, TrainN in acc_results.keys():
        if len(acc_results[(MAPN,TrainN)]) == 0:
            continue
        print("Train type:",TrainN[2],"\tMAP type:",MAPN[2])
        print("mean:",np.mean(acc_results[(MAPN,TrainN)]),"\tstd:",np.std(acc_results[(MAPN,TrainN)]))
        print("mean:",np.mean(ord_results[(MAPN,TrainN)]),"\tstd:",np.std(ord_results[(MAPN,TrainN)]))
        print("mean:",np.mean(uni_results[(MAPN,TrainN)]),"\tstd:",np.std(uni_results[(MAPN,TrainN)]))
        
    print("Difference due to training")
    for MAPN in networks:
        paired_diffs = acc_results[(MAPN,networks[1])]-acc_results[(MAPN,networks[0])]
        print(np.mean(paired_diffs))
        print(np.std(paired_diffs))
    
    print("Difference due to MAP")
    for TrainN in networks:
        if TrainN[0] is None:
            continue
        paired_diffs = acc_results[(networks[1],TrainN)]-acc_results[(networks[0],TrainN)]
        print(np.mean(paired_diffs))
        print(np.std(paired_diffs))
    
    #save the results
    with open(filename, "wb") as file:
        pickle.dump({"acc_results":acc_results,"ord_results":ord_results,"uni_results":uni_results,},file)
        
        
#run the full repeated experiment
if __name__ == "__main__":
    for i in range(25):
        replication(i,train_epochs=100,pretrain_epochs=250,num_signals=200,testing_perc=0.5,sampling_rate=10)