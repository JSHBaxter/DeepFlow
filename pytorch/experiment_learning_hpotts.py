import numpy as np
import time
import torch
import torchvision
from itertools import product
from math import sqrt, log, exp
import datetime
from random import shuffle
from multiprocessing import Pool
import time
import scipy
import pickle
import os
import gc
import copy
import matplotlib.pyplot as plt 
from simulate_mra import synthetic_mra

import matplotlib.image
from matplotlib import cm

from hmf_deepflow import HMFDeepFlow2d, HMFDeepFlow2dLoss
from hmf_deepflow_function import HMF_MAP2d_PytorchNative

torch.autograd.set_detect_anomaly(True)
debug_mode = False

def get_data_circle(size,number,channels=1,device="cuda"):
    base = np.zeros(shape=(number,channels,size,size),dtype=float)
    seg = np.zeros(shape=(number,1,size,size),dtype=float)
    x,y = np.meshgrid(range(size),range(size))
    p = torch.from_numpy(np.array([-1, -1, 1, 0, 0, 1, 2, 2]).astype(np.int32)).to(device)
    
    c2nr = 5
    radii = (0.375+0.125*np.random.uniform(size=(number,)))*size
    radius_giggle1 = np.sqrt((0.6+0.1*np.random.uniform(size=(number,))))*radii
    radius_giggle2 = np.sqrt((0.45+0.1*np.random.uniform(size=(number,))))*radius_giggle1
    centre = (0.4+0.2*np.random.uniform(size=(2,number,)))*size
    centre_giggle1 = centre + 0.1*np.random.uniform(size=(2,number,))*radii
    centre_giggle2 = centre_giggle1 + 0.1*np.random.uniform(size=(2,number,))*radius_giggle1
    offset1 = (0.4+0.2*np.random.uniform(size=(2,number,)))*size
    orientation1 = np.random.normal(size=(2,number,))
    orientation1 /= np.sqrt(np.sum(orientation1**2,axis=1,keepdims=True))
    orientation2 = np.random.normal(size=(2,number,))
    orientation2 /= np.sqrt(np.sum(orientation2**2,axis=1,keepdims=True))
    
    base = np.zeros(shape=(number,channels,size,size),dtype=float)
    seg = np.zeros(shape=(number,5,size,size),dtype=float)
    
    #create the underlying segmentation
    for n in range(number):
        seg[n,4,:,:] = ((x - centre[0,n])**2 + (y - centre[1,n])**2 < radii[n]**2)
        seg[n,0,:,:] = (1-seg[n,4,:,:])*( (x-offset1[0,n])*orientation1[0,n] 
                                         +(y-offset1[1,n])*orientation1[1,n] >= 0)
        seg[n,1,:,:] = (1-seg[n,4,:,:])*( (x-offset1[0,n])*orientation1[0,n] 
                                         +(y-offset1[1,n])*orientation1[1,n] < 0)
        seg[n,2,:,:] = seg[n,4,:,:] * ((x - centre_giggle1[0,n])**2 + (y - centre_giggle1[1,n])**2 > radius_giggle1[n]**2)
        seg[n,3,:,:] = seg[n,4,:,:] * (1-seg[n,2,:,:]) * ((x - centre_giggle2[0,n])**2 + (y - centre_giggle2[1,n])**2 > radius_giggle2[n]**2)
        seg[n,4,:,:] = seg[n,4,:,:] - seg[n,3,:,:] - seg[n,2,:,:]
    
    
        #now do the actual image
        for i in range(channels):
            base[n,i,:,:] = seg[n,2,:,:] - 0.9*seg[n,1,:,:] - seg[n,3,:,:] + 0.05*seg[n,4,:,:] - 0.05*seg[n,0,:,:]

            #make some brown noise to add to the image
            global_bias_field_freq = 10 * np.logical_or(
                                            np.logical_or( x**2+y**2<(size/32)**2,
                                                          (x-size+1)**2+y**2<(size/32)**2),
                                            np.logical_or( x**2+(y-size+1)**2<(size/32)**2,
                                                          (x-size+1)**2+(y-size+1)**2<(size/32)**2)
                                                  ) / (3.1415926*(size/32)**2)
            global_bias_field = np.real(np.fft.fft2((np.random.normal(size=(size,size))+
                                                    1.j*np.random.normal(size=(size,size)))*global_bias_field_freq))
            global_bias_field -= np.mean(global_bias_field)
            base[n,i,:,:] += global_bias_field/(2*c2nr)
            
            
    #and now add the white noise with specific contrast
    base += np.random.normal(size=base.shape)/(sqrt(channels)*2*c2nr)
    
    #visualise if needed for debugging
    if debug_mode:
        fig, ax = plt.subplots(nrows=number, ncols=1+channels)
        for ir, row in enumerate(ax):
            for ic, col in enumerate(row):
                if ic == 0:
                    seg_color = np.zeros(shape=(size,size,3))
                    seg_color[:,:,0] = seg[ir,0,:,:]+seg[ir,3,:,:]
                    seg_color[:,:,1] = seg[ir,1,:,:]+seg[ir,3,:,:]
                    seg_color[:,:,2] = seg[ir,2,:,:]
                    col.imshow(seg_color)
                    continue
                col.imshow(base[ir,ic-1,:,:], cmap = cm.gray, vmin=-1.5, vmax=1.5)
        plt.show()
    
    return torch.from_numpy(base).to(device).float(), torch.from_numpy(seg).to(device).float(), p

def dice(a,b):
    img_dice_num = 2*torch.sum(torch.sum(torch.abs(a*b),dim=-1),dim=-1) + 10**-2
    img_dice_den = torch.sum(torch.sum(torch.abs(a)+torch.abs(b),dim=-1),dim=-1) + 10**-2
    return torch.mean(img_dice_num/img_dice_den)

class ExperimentModel(torch.nn.Module):
    def __init__(self,in_channels,num_convs=2,act_type="reg",loss_weights=(1,0,0)) -> None:
        super().__init__()
        self.act_type = act_type
        width = 32
        
        self.conv1 = torch.nn.Conv2d(in_channels,width,5,padding="same")
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(width,width,5,padding="same") for i in range(num_convs)])
        
        self.conv3 = torch.nn.Conv2d(width,5,5,padding="same")
        baseline_data, baseline_seg, p = get_data_circle(256,64,device="cpu")
        baseline_prob = [torch.log(torch.mean(baseline_seg[:,i,...])).item() for i in range(5)]
        self.conv3.bias = torch.nn.Parameter(torch.Tensor(baseline_prob))
        
        self.conv4 = torch.nn.Conv2d(width,8,5,padding="same")
        self.conv4.weight = torch.nn.Parameter(self.conv4.weight.clone().detach()/32)
        self.conv4.bias = torch.nn.Parameter(-5*torch.ones((8,)))
        self.bin = HMFDeepFlow2d(solve_type=("LBP" if self.act_type == "reg" else "MFE"))
        self.max_rate_r = 0.95
        
        self.nll_loss_weight = loss_weights[0]
        self.dice_loss_weight = loss_weights[1]
        self.edge_dice_loss_rel_weight = loss_weights[2]
        self.dice_loss = (lambda x,y: 1-dice(x,y))
        self.mse_loss = (lambda x,y: torch.mean((x-y)**2))
    
    def forward(self,x,y_gt,ordering,iteration):
        
        #apply the network
        relu = torch.relu
        p = relu(self.conv1(x))
        for c in self.convs:
            p = relu(c(p))
        d = self.conv3(p)
        r = self.conv4(p)
        
        #use the final activation and calculate specific model losses
        if self.act_type == "reg" or self.act_type == "mfe":
            r = torch.sigmoid(r) * (2*log(7/5)/5 if self.act_type == "reg" else 1/(4*5)) * self.max_rate_r 
            rx = torch.nn.functional.pad(0.5*(r[:,:,:-1,:]+r[:,:,1:,:]),(0,0,0,1))
            ry = torch.nn.functional.pad(0.5*(r[:,:,:,:-1]+r[:,:,:,1:]),(0,1,0,0))
            q, jx, jy = self.bin(d,rx,ry,ordering)
            nll_loss = HMFDeepFlow2dLoss()(d,rx,ry,q,jx,jy,y_gt,ordering)
            dice_loss = self.dice_loss(q,y_gt)

            loss = self.nll_loss_weight*nll_loss + (self.dice_loss_weight*dice_loss if self.dice_loss_weight > 0 else 0)
            
            y_pred = HMF_MAP2d_PytorchNative.apply(d,rx,ry,ordering)
            y_pred = (y_pred == torch.max(y_pred,dim=1,keepdim=True)[0])
            return loss, d, rx, ry, q, jx, jy, y_pred
            
        else:
            d -= torch.max(d,dim=1,keepdim=True)[0]
            q = torch.softmax(d,dim=1)
            y_pred = (d == torch.max(d,dim=1,keepdim=True)[0])
            nll_loss = torch.mean(torch.log(torch.sum(torch.exp(d),dim=1)) - torch.sum(y_gt*d,dim=1))
            loss = self.nll_loss_weight*nll_loss + (self.dice_loss_weight*self.dice_loss(q,y_gt) if self.dice_loss_weight > 0 else 0)
            return loss, d, None, None, q, None, None, y_pred


def get_filename(experiment_iteration):
    return "../experiments/experiment_learning_"+str(experiment_iteration)+".pkl"    

def check_if_already_exists(experiment_iteration):    
    filename = get_filename(experiment_iteration)
    return (os.path.exists(filename) and os.path.isfile(filename))

def save_experiment(experiment_iteration, training_iters, models, all_data):
    print("Saving experiment " + str(experiment_iteration) + " results.")
    data_dict = {}
    for ix, x in enumerate(all_data):
        session_data = []
        for im, m in enumerate(models):
            loss,p,rx,ry,q,jx,jy,y_pred, = m(x[0],x[1],x[2],training_iters)
            #session_data.append(p.detach().cpu().numpy())
            #session_data.append(q.detach().cpu().numpy())
            #session_data.append(y_pred.detach().cpu().numpy())
            dice_eval = tuple([dice(x[1][b,0,:,:],y_pred[b,0,:,:]).detach().cpu().numpy().item() for b in range(p.shape[0])])
        
            data_dict[(ix,im)] = dice_eval
        
    pickle.dump(data_dict,open(get_filename(experiment_iteration),"wb"))
    del data_dict
    gc.collect()
    torch.cuda.empty_cache()
    
    
def experiment(loss_weights,offset_number,jump_number,experiment_number,depth):
    experiment_iteration = offset_number + jump_number*depth + experiment_number
    
    #if we have already run this experiment, move to the next
    if check_if_already_exists(experiment_iteration):
        return
    
    print("Running experiment " + str(experiment_iteration))
    batch_size = 32
    wait_iterations = 1000
    max_training_iters = wait_iterations+250
    viz_iters = None
    viz_number = 1
    #create the models and optimisers
    models = [ExperimentModel(1,depth,"noreg",loss_weights).to("cuda"),]
    #models.append(copy.deepcopy(models[-1]))
    #models[-1].loss_type = "both"
    #models[-1].dice_loss_weight = 4.9
    #models[-1].nll_loss_weight = 0.1
    learning_rate_1 = 10**-3
    learning_rate_2 = 10**-3
    optimisers_1 = [torch.optim.Adam(m.parameters(), lr=learning_rate_1) for m in models]
    optimisers_2 = None

    all_data = []

    for training_iters in range(max_training_iters):

        #see if we should get new data
        if len(all_data) == 0 or np.random.uniform() < 1/len(all_data):
            x = get_data_circle(256,batch_size,1)#
            all_data.append(x)
        else:
            x = all_data[int(np.random.uniform()*len(all_data))]
            if np.random.uniform() < 0.5:
                x = (torch.permute(x[0],(0,1,3,2)),
                     torch.permute(x[1],(0,1,3,2)),
                     x[2])
            if np.random.uniform() < 0.5:
                x = (torch.flip(x[0], (2,)),
                     torch.flip(x[1], (2,)),
                     x[2])
            if np.random.uniform() < 0.5:
                x = (torch.flip(x[0], (3,)),
                     torch.flip(x[1], (3,)),
                     x[2])
        
        #once the pre-training is done, copy over the models
        if training_iters == wait_iterations:
            for m in copy.deepcopy(models):
                models.append(copy.deepcopy(m))
                models[-1].act_type = "reg"
                models[-1].bin = HMFDeepFlow2d(solve_type="LBP")
                models.append(copy.deepcopy(m))
                models[-1].act_type = "mfe"
                models[-1].bin = HMFDeepFlow2d(solve_type="MFE")
            optimisers_2 = [torch.optim.Adam(m.parameters(), lr=learning_rate_2) for m in models]
            
        #apply the models and optimise
        res = []
        for m,opt in zip(models,(optimisers_1 if training_iters < wait_iterations else optimisers_2)): 
            if m is None:
                continue
            loss,p,rx,ry,q,jx,jy,y_pred, = m(x[0],x[1],x[2],training_iters)
            p.retain_grad()
            if rx is not None:
                rx.retain_grad()
                ry.retain_grad()
            dice_eval = dice(x[1],y_pred)
            loss.backward()
            res.append((loss.detach().cpu().item(),
                        dice_eval.detach().cpu().item(),
                        p.detach(),
                        p.grad.detach(),
                        None if rx is None else rx.detach(),
                        None if rx is None else rx.grad.detach(),
                        None if ry is None else ry.detach(),
                        None if ry is None else ry.grad.detach(),
                        q.detach(),
                        y_pred.detach(),
                        None if jx is None else jx.detach(),
                        None if jy is None else jy.detach()))
            opt.step()
            opt.zero_grad()

        #print the stats and visualise if on schedule
        print(training_iters,end="")
        for ir, r in enumerate(res):
            print(("\t" if ir > 0 else " ")+str(r[0])+"\t"+str(r[1]),end="")
        print("")

        #visualise if desired
        if (training_iters >= wait_iterations and viz_iters is not None and training_iters % viz_iters == viz_iters-1):
            for b in range(viz_number):
                fig, ax = plt.subplots(nrows=1+len(models), ncols=10)

                for ir, row in enumerate(ax):
                    for ic, col in enumerate(row):
                        colour = False
                        im = None
                        if ir < len(models):
                            if ic == 0:
                                col.imshow(res[ir][9].cpu().detach().numpy()[b,0,:,:],"gray",vmin=0,vmax=1)
                                im = res[ir][9].cpu().detach().numpy()[b,0,:,:]
                            elif ic == 1:
                                col.imshow(res[ir][8].cpu().detach().numpy()[b,0,:,:],"gray",vmin=0,vmax=1)
                                im = res[ir][8].cpu().detach().numpy()[b,0,:,:]
                            for idx, i in enumerate(list(range(2,8))+[10,11]):
                                if ic == idx+2 and res[ir][i] is not None:
                                    col.imshow(res[ir][i].cpu().detach().numpy()[b,0,:,:])
                                    im = res[ir][i].cpu().detach().numpy()[b,0,:,:]
                                    colour = True
                                    break
                        else:
                            if ic == 0:
                                col.imshow(x[ic].cpu().detach().numpy()[b,0,:,:])
                                colour = False
                                im = x[ic].cpu().detach().numpy()[b,0,:,:]
                                max_val = np.max(im) if (ic != 4 and ic != 6) else (np.log(2) if ir == 1 else 0.5)
                                min_val = -max_val if (ic != 4 and ic != 6) else 0
                                im = (im-np.min(im))/(max_val-np.min(im))
                            elif ic == 1:
                                col.imshow(x[ic].cpu().detach().numpy()[b,0,:,:],"gray",vmin=0,vmax=1)
                                colour = False
                                im = x[ic].cpu().detach().numpy()[b,0,:,:]
                        if im is not None:
                            if colour == True:
                                matplotlib.image.imsave("image_exp_"+str(experiment_iteration)+"_"+str(b)+"_"+str(ir)+"_"+str(ic)+".png", im)
                            else:
                                matplotlib.image.imsave("image_exp_"+str(experiment_iteration)+"_"+str(b)+"_"+str(ir)+"_"+str(ic)+".png", im, cmap = cm.gray, vmin=0, vmax=1)
                            
                                
                plt.show()
                
    del all_data
    del res
    del optimisers_1
    if optimisers_2:
        del optimisers_2
    gc.collect()
    torch.cuda.empty_cache()
    eval_data = [get_data_circle(256,64,1) for i in range(5)]
    save_experiment(experiment_iteration, max_training_iters, models, eval_data)
    print("\tDone!\n")

if __name__ == "__main__":
    jump_number = 10
    for depth,experiment_number in product(range(5),range(jump_number)):
        experiment((1,0,0),130000,jump_number,experiment_number,depth)
        gc.collect()
        torch.cuda.empty_cache()
    
""" 
#run a bunch of experiments
if __name__ == "__main__":
    experiment_configs = [(130000,10,(1,0,0)),
                          (140000,10,(0,1,0)),]
    for conf in experiment_configs:
        offset_number = conf[0]
        jump_number = conf[1]
        loss_weights = conf[2]
        for depth,experiment_number in product(range(5),range(jump_number)):
            experiment(loss_weights,offset_number,jump_number,experiment_number,depth)
"""