import torch
import numpy as np
from itertools import product
from matplotlib import pyplot as plt

def find_rule_break(root, constraint_list, signal_labels=None, signal_onehot=None):
    if signal_labels is None and signal_onehot is None:
        raise ValueError("No signal given")
        
    #if one hot, change to label
    if signal_labels is None:
        signal_labels = np.argmax(signal_onehot, axis=0)
    N = len(signal_labels)
    conf = [int(s) for s in signal_labels]
    
    rules_broken = []
    for rule in constraint_list:
        prob = True
        l1 = rule[0]
        l2 = rule[1]
        if rule[2] == "PRECEDE":
            a1_invalid_set = set([l.index for l in root.leaves - l1.leaves - l2.leaves])
            a2_invalid_set = set([l.index for l in l2.leaves])
            for a1,a2 in zip(conf[:-1],conf[1:]):
                if a1 in a1_invalid_set and a2 in a2_invalid_set:
                    rules_broken.append(rule)
                    break
        elif rule[2] == "NOTPRECEDE":
            a1_invalid_set = set([l.index for l in l1.leaves])
            a2_invalid_set = set([l.index for l in l2.leaves])
            for a1,a2 in zip(conf[:-1],conf[1:]):
                if a1 in a1_invalid_set and a2 in a2_invalid_set:
                    rules_broken.append(rule)
                    break
        elif rule[2] == "SUCEED":
            a1_invalid_set = set([l.index for l in l2.leaves])
            a2_invalid_set = set([l.index for l in root.leaves - l1.leaves - l2.leaves])
            for a1,a2 in zip(conf[:-1],conf[1:]):
                if a1 in a1_invalid_set and a2 in a2_invalid_set:
                    rules_broken.append(rule)
                    break
        elif rule[2] == "NOTSUCEED":
            a1_invalid_set = set([l.index for l in l2.leaves])
            a2_invalid_set = set([l.index for l in l1.leaves])
            for a1,a2 in zip(conf[:-1],conf[1:]):
                if a1 in a1_invalid_set and a2 in a2_invalid_set:
                    rules_broken.append(rule)
                    break
        elif rule[2] == "BEFORE":
            a1_invalid_set = set([l.index for l in l2.leaves])
            a2_invalid_set = set([l.index for l in l1.leaves])
            for idx_1 in range(N-1):
                if conf[idx_1] not in a1_invalid_set:
                    continue
                for idx_2 in range(idx_1+1,N):
                    if conf[idx_2] in a2_invalid_set:
                        prob = None
                        break
                if prob is None:
                    rules_broken.append(rule)
        elif rule[2] == "AFTER":
            a1_invalid_set = set([l.index for l in l1.leaves])
            a2_invalid_set = set([l.index for l in l2.leaves])
            for idx_1 in range(N-1):
                if conf[idx_1] not in a1_invalid_set:
                    continue
                for idx_2 in range(idx_1+1,N):
                    if conf[idx_2] in a2_invalid_set:
                        prob = None
                        break
                if prob is None:
                    rules_broken.append(rule)
        elif rule[2] == "UNIQUE":
            leaves = set([l.index for l in l1.leaves])
            antileaves = set([l.index for l in root.leaves - l1.leaves])
            number_in = 0
            if conf[0] in leaves:
                number_in += 1
            for idx_1 in range(N-1):
                if conf[idx_1] in antileaves and conf[idx_1+1] in leaves:
                    number_in += 1
            if number_in > 1:
                rules_broken.append(rule)
        elif rule[2] == "INITIAL":
            leaves = set([l.index for l in l1.leaves])
            antileaves = set([l.index for l in root.leaves - l1.leaves])
            for idx_1 in range(N-1,1,-1):
                if conf[idx_1] in leaves and conf[idx_1-1] in antileaves:
                    rules_broken.append(rule)
                    break
        elif rule[2] == "FINAL":
            leaves = set([l.index for l in l1.leaves])
            antileaves = set([l.index for l in root.leaves - l1.leaves])
            for idx_1 in range(0,N-1):
                if conf[idx_1] in leaves and conf[idx_1+1] in antileaves:
                    rules_broken.append(rule)
                    break
        
            
    #print(len(rules_broken))
    #print(rules_broken)
    #plt.plot(conf)
    #plt.show()
    return rules_broken
    

def get_exact_answer(root,constraint_list,N,d,r):
    conf_space = len(root.leaves)**N
    marginal_probs = torch.zeros((1,len(root.leaves),N))
    map_log_prob = -np.inf
    map_conf = torch.zeros((1,len(root.leaves),N))
    all_prob = 0
    for i in range(conf_space):
        conf = [0]*N
        conf_t = torch.zeros((1,len(root),N))
        
        #construct the basic leaf sequence
        for t in range(N):
            conf_t[0,(i//len(root.leaves)**t) % len(root.leaves),t] = 1
            conf[t] = (i//len(root.leaves)**t) % len(root.leaves)
        
        #get the other group sequences as well
        for g in root.groups:
            for l in g.leaves:
                conf_t[:,g.index,:] += conf_t[:,l.index,:]
        
        prob = torch.sum(d*conf_t) - torch.sum(torch.abs(conf_t[:,:,:-1]-conf_t[:,:,1:])*r[:,:,:])
        prob = torch.exp(prob)
        
        broken = find_rule_break(root,constraint_list,conf)
        if len(broken) == 0:
            all_prob += prob
            if prob > map_log_prob:
                map_conf = conf_t
                map_log_prob = prob
            #print("\t",conf,prob)

            for idx,c in enumerate(conf):
                marginal_probs[0,c,idx] += prob

    marginal_probs /= all_prob
    all_probs = torch.zeros((1,len(root),N))
    all_probs[:,:len(root.leaves),:] += marginal_probs
    for g in root.groups:
        for l in g.leaves:
            all_probs[:,g.index,:] += marginal_probs[:,l.index,:]
    return all_probs.detach(), map_conf.detach()
