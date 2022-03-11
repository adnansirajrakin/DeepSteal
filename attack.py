from __future__ import print_function
import numpy as np
import pandas as pd
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from torch.nn import init
from collections import OrderedDict
import time
import shutil
import xlwt
from xlwt import Workbook 
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
# from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import random
random.seed(6)
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import torch as th
import operator



def uap_sgd(model, loader, nb_epoch, eps, beta = 12, step_decay = 0.8, y_target = None, loss_fn = None, layer_name = None, uap_init = None):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    beta        clamping value
    y_target    target class label for Targeted UAP variation
    loss_fn     custom loss function (default is CrossEntropyLoss)
    layer_name  target layer name for layer maximization attack
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})
    
    OUTPUT
    delta.data  adversarial perturbation
    losses      losses per iteration
    '''
    _, (x_val, y_val) = next(enumerate(loader))
    batch_size = len(x_val)
    if uap_init is None:
        batch_delta = torch.zeros_like(x_val) # initialize as zero vector
    else:
        batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1])
    delta = batch_delta[0]
    losses = []
    
    # loss function
    if layer_name is None:
        if loss_fn is None: loss_fn = nn.CrossEntropyLoss(reduction = 'none')
        beta = torch.cuda.FloatTensor([beta])
        def clamped_loss(output, target):
            loss = torch.mean(torch.min(loss_fn(output, target), beta))
            return loss
       
    # layer maximization attack
    else:
        def get_norm(self, forward_input, forward_output):
            global main_value
            main_value = torch.norm(forward_output, p = 'fro')
        for name, layer in model.named_modules():
            if name == layer_name:
                handle = layer.register_forward_hook(get_norm)
                
    batch_delta.requires_grad_()
    for epoch in range(nb_epoch):
        print('epoch %i/%i' % (epoch + 1, nb_epoch))
        
        # perturbation step size with decay
        eps_step = eps * step_decay
        
        for i, (x_val, y_val) in enumerate(loader):
            # for targeted UAP, switch output labels to y_target
            if y_target is not None: y_val = torch.ones(size = y_val.shape, dtype = y_val.dtype) * y_target
            if i ==38:
                break
            perturbed = torch.clamp((x_val + batch_delta).cuda(), 0, 1)
            outputs = model(perturbed)
            
            # loss function value
            if layer_name is None: loss = clamped_loss(outputs, y_val.cuda())
            else: loss = main_value
            
            if y_target is not None: loss = -loss # minimize loss for targeted UAP
            losses.append(torch.mean(loss))
            loss.backward()
            
            # batch update
            grad_sign = batch_delta.grad.data.mean(dim = 0).sign()
            delta = delta + grad_sign * eps_step
            delta = torch.clamp(delta, -eps, eps)
            batch_delta.data = delta.unsqueeze(0).repeat([batch_size, 1, 1, 1])
            batch_delta.grad.data.zero_()
    
    if layer_name is not None: handle.remove() # release hook
    
    return delta.data, losses


class DES_new(object):
    def __init__(self, criterion, layer =100, k_top=50, w_clk=1, s_clk=1, evolution=1000,probab =1 ):
        
        self.criterion = criterion.cuda() 
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        ## number of weights per clock or group
        self.N=w_clk
        ## number of wweight shift at one time
        self.S=s_clk
        ## tracking number of pool of weights
        self.k_top = k_top
        
        self.total=0
        self.layer = layer - 1
        ##evolution z
        self.epoch=evolution
        self.probab = probab ## probability of hard ware attack success rate $f_p$
        print(self.probab)
    def shift(self, m,f_index):
        ''' performs the s_clk number of shift starting at index f_index given a layers weights m'''

        # size of the layer
        self.total=m.weight.detach().view(-1).size()[0] ## size of the entire layer
        ranks=self.k_top ## initial pool 
        
        ranging=self.N ## number of weight in each clock
        
        params=m.weight.data.detach().clone() ##  weights
        param2= m.weight.data.detach().clone() ## keeping another copy for record
        
        
        
        
        param_flat=params.view(-1) ## 1D flattended
       
        param_flat = torch.flip(param_flat,[0]) ## flipping to make right shift real AWD genrates right shift thats why # for specific FPGA this may vary
        for y in range(self.S*ranging): 
            param_flat[f_index+y]=param_flat[f_index+ranging+y]  ## shifting the values 
        param_flat = torch.flip(param_flat,[0]) 
      
        param2 = param_flat.view(params.size())  # putting it back to the original matrix
        param_flipped=param2.detach().clone() ## copying the parameters

        return param_flipped

    def shift2(self, m,f_index):
        ''' performs the s_clk number of shift starting at index f_index given a layers weights m'''
        ## same as before not necessary 
        self.total=m.weight.detach().view(-1).size()[0] ## size of the entire layer
        ranks=self.k_top ## rank size can be different if needed to speed up
        
        ranging=self.N ## number of byte transferred
        
        params=m.weight.data.detach().clone() ##  weights
        param2= m.weight.data.detach().clone() 

        

        param_flat=params.view(-1)
        
        param_flat = torch.flip(param_flat,[0])
        for y in range(self.S*ranging):
            print("Old value, new value:")
            print(param_flat[f_index+y],param_flat[f_index+ranging+y])    
            param_flat[f_index+y]=param_flat[f_index+ranging+y]  ## shifting the values 
        param_flat = torch.flip(param_flat,[0])

      
        param2 = param_flat.view(params.size())
       
        param_flipped=param2.detach().clone() ## copying the parameters
        return param_flipped 


    def mutation(self,model,data,target,obj_func,x,y,layers,y_max,h,mutation=0):
        ''' this function performs the mutation step 
            model : network architecture
            data : test data
            target: label of the data
            obj_func: initial population mutation function values
            x: current x
            y: current y
            y_max: total weights at layer x
            mutation : which mutation strategy to use
            h: evolution number
        '''
        # random numbers to perform mutation
        train_indices = torch.from_numpy(np.random.choice(self.k_top, size=(5), replace=False)) 
        
        # normalization step
        x_norm=torch.clamp(x.float()/layers,0,1) ## normalize
        y_norm=torch.clamp(torch.div(y.float(),y_max.float()),0,1)  ## Normalize
        x,y=x.int(),y.int()
        
        # generating three alphas
        F_1 = torch.clamp(torch.rand(1),0.3,1) 
        F_2 = torch.clamp(torch.rand(1),0.3,1)  
        F_3 = torch.clamp(torch.rand(1),0.3,1)
         
        ## four mutantation strategy 
        if mutation == 3:
           _,indx = obj_func.topk(self.k_top)
           mut_x = x_norm[train_indices[0]] + F_1*(x_norm[[indx[0]]]-x_norm[[indx[-1]]])
           mut_y = y_norm[train_indices[0]] + F_1*(y_norm[[indx[0]]]-y_norm[[indx[-1]]])

        if mutation == 0:
           mut_x = x_norm[train_indices[0]] + F_1*(x_norm[train_indices[1]]-x_norm[train_indices[2]])
           mut_y = y_norm[train_indices[0]] + F_1*(y_norm[train_indices[1]]-y_norm[train_indices[2]])

        if mutation == 1:
           mut_x = x_norm[train_indices[0]] + F_1*(x_norm[train_indices[1]]-x_norm[train_indices[2]]) + F_2 * (x_norm[train_indices[3]]-x_norm[train_indices[4]])
           mut_y = y_norm[train_indices[0]] + F_1*(y_norm[train_indices[1]]-y_norm[train_indices[2]]) + F_2 * (y_norm[train_indices[3]]-y_norm[train_indices[4]])

        if mutation == 2:
           _,indx = obj_func.topk(self.k_top)
           mut_x = x_norm[train_indices[0]] + F_1*(x_norm[[indx[0]]]-x_norm[train_indices[0]]) + F_2 * (x_norm[train_indices[1]]-x_norm[train_indices[2]]) + F_3 * (x_norm[train_indices[3]]-x_norm[train_indices[4]])
           mut_y = y_norm[train_indices[0]] + F_1*(y_norm[[indx[0]]]-y_norm[train_indices[0]]) + F_2 * (y_norm[train_indices[1]]-y_norm[train_indices[2]]) +  F_3 * (y_norm[train_indices[3]]-y_norm[train_indices[4]])  
        
        # ignore the skip varibale for corssover case
        skip = 0

        ## mutatant should be within this range [0,1] or just replace with the parent features      
        if mut_x>1 or  mut_x<0:
           mut_x = x_norm[h]
        if mut_y>1 or mut_y <0:
           mut_y = y_norm[h]
        if skip == 0:
            

            ## denormalization of x and y
            mut_x=int(mut_x*(layers-1))
           
            n=0
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    
                    if n == mut_x:
                       
                       mut_y= (mut_y* m.weight.data.view(-1).detach().size()[0]-2*self.N*self.S) # denormalization of y
                       
                    n=n+1
            mut_y = int(mut_y)

            #compute the new objective for the new mutant vector
            obj_new = obj_func[h]*0.99
            n=0
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    
                    if n == mut_x:
                        prob =torch.Tensor([self.probab]) ## checking if the shift will be successful in hardware condsidering hardware AWD has a probability = probab to be successful
                        prob_out = torch.bernoulli(prob)
                        if prob_out == 1:
                            clean_weight = m.weight.data.detach().clone()
                            attack_weight=self.shift( m,mut_y)
                            m.weight.data = attack_weight
                            output=model(data)
                            obj_new=self.criterion(output,target).item() ## new mutation function evaluation at mut_x and mut_y
                            m.weight.data= clean_weight
                    n=n+1

            
            #compare with current population if causes more damage then replace
            if obj_func[h] < obj_new:
                #print(mutation)
                obj_func[h] = obj_new
                x[h]=mut_x
                y[h]=mut_y
        return obj_func, x, y


    def progressive_search(self, model, data, target,xs,ys):
      
        # set the model to evaluation mode
        model.eval()

        ## checking if the iteration end shift will be successful in hardware
        probab =self.probab
        prob =torch.Tensor([probab])
        prob_outs = torch.bernoulli(prob)
        

       
        # calculating total number of layers in the model we just attack convolution and linear layers
        n=0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
               n=n+1
        max_weight=torch.zeros([n+1])
        layers=self.layer
        print("Number of layers:" ,layers)
        
        # 3. setting up initial objective function,x anb y
        obj_func=torch.zeros([self.k_top])
        x=torch.randint(0, layers, ([self.k_top]))
        #print(x)
        y=torch.randint(0, layers, ([self.k_top]))
        y_max=torch.randint(0, layers, ([self.k_top])).float()

        if prob_outs == 1: 
        ## start the evolution
            for i in range(self.epoch):
            
            # only calculate the initial population objective for iteration 0
                if i == 0:
                    for k in range(self.k_top ):
                        n=0
                        for m in model.modules():
                            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                            
                                if n == x[k]:
                                    clean_weight = m.weight.data.detach().clone()
                                    y[k]=torch.from_numpy(np.random.choice(int(m.weight.data.view(-1).detach().size()[0]-2*self.N*self.S), size=(1), replace=False)) 
                                    y_max[k]= m.weight.data.view(-1).detach().size()[0]-2*self.N*self.S ## for each y value corresponding maximum y_max
                                    prob =torch.Tensor([probab])
                                    prob_out = torch.bernoulli(prob)
                                    if prob_out == 1: ## checking if the shift will be successful in hardware
                                        attack_weight=self.shift( m,y[k]) # attack 
                                        m.weight.data = attack_weight
                                        output=model(data)
                                        obj_func[k]=self.criterion(output,target).item() ## evaluate fitness function
                                        m.weight.data = clean_weight ## recover the weights
                                #print(obj_func[k])
                                n=n+1
            
            #perfomr four mutation strategy for each population candidate
                for z in range(4):
                    obj_func ,x ,y = self.mutation(model,data,target,obj_func,x,y,layers,y_max, i, mutation=z)

        ## This part checks if any previous shift were done at the best objective function index
            count = 0
            number = 0
            _,indx = obj_func.topk(self.k_top)
        
        ## This part checks if any previous shift were done at the best objective function index
            for k in range(indx.size()[0]):
                for i in range(len(xs)):
                    if (x[indx[number]],y[indx[number]]) == ( xs[i], ys[i]):
                        count = 1
                if count == 1:
                    number =  number +1
                    count= 0
                else:
                    break


                
        #This part checks if any previous shift were done at the best objective function index -1
            for k in range(indx.size()[0]):
                for i in range(len(xs)):
                    if (x[indx[number]],y[indx[number]]) == ( xs[i], ys[i]+1):
                        count = 1
                if count == 1:
                    number =  number +1
                    count=0
                else:
                    break
        #This part checks if any previous shift were done at the best objective function index +1 since the attack effects two weights
            for k in range(indx.size()[0]):
                for i in range(len(xs)):
                    if (x[indx[number]],y[indx[number]]) == ( xs[i], ys[i]-1):
                        count = 1
                if count == 1:
                    number =  number +1
                    count=0
                else:
                    break
        ## the reason we need to do that because lets assume attack at index 2 [1,2,3,4] after a shift [1,1,2,4] so basically we can not attack [. X X X] (2+1) and (2-1) anymore.
                
            print(number)
        ## after the check 'number' will indicate the index where we perform the shift (x[indx[number]],y[indx[number]])
            
            # Final shift at winner candidate hardwware check was done at the beginning.
            prob_out = 1
            if prob_out == 1:
                xs.append(x[indx[number]])
                ys.append(y[indx[number]])
                n=0
                for m in model.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        #print(n,name)
                        if n==x[indx[number]]:
                #print(name, self.loss.item(), loss_max)
                            attack_weight = self.shift2(m,y[indx[number]])
                            m.weight.data = attack_weight
                        n=n+1

                print("Layer numer, Index Number: ", x[indx[number]],y[indx[number]])
        
        return xs,ys


class BFA(object):
    def __init__(self, criterion, layer = 100, k_top=10):

        self.criterion = criterion
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0
        self.layer = layer

    def flip_bit(self, m):
        
        # 1. flatten the gradient tensor to perform topk
        self.k_top = m.weight.view(-1).size()[0]
        w_grad_topk, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(
            self.k_top)
        # update the b_grad to its signed representation
        w_grad_topk = m.weight.grad.detach().view(-1)[w_idx_topk]
        copy_weight = m.weight.data.detach().clone()

        # self.n_bits2flip loop 
        tracker=0
        i=0
        while tracker < self.n_bits2flip :
            # top1 check gradient  ++  dont
            # +- flip
            if(w_grad_topk[i].sign()+m.weight.data.view(-1)[w_idx_topk[i]].sign()==0):
                # logging.info(w_grad_topk[i].sign().item(),m.weight.data.view(-1)[w_idx_topk[i]].sign().item())
                # logging.info('before: ')
                # logging.info(m.weight.data.view(-1)[w_idx_topk[i]].item())        
                copy_weight.view(-1)[w_idx_topk[i]] = copy_weight.view(-1)[w_idx_topk[i]]*(-1.0)
                tracker+=1
                # logging.info('after: ')
                # logging.info(m.weight.data.view(-1)[w_idx_topk[i]].item())
            i+=1
            # logging.info(tracker)
        ww= copy_weight - m.weight.data
        
        return copy_weight

    def progressive_bit_search(self, model, data, target):
        
        model.eval()

        # 1. perform the inference w.r.t given data and target
        output = model(data)
        
        self.loss = self.criterion(output, target)
        # 2. zero out the grads first, then get the grads
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()

        self.loss.backward()
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()

        # 3. for each layer flip #bits = self.bits2flip
        while self.loss_max <= self.loss.item() and self.n_bits2flip<20 :
            
            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer
            n=0
                  
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    n=n+1
                    if n < self.layer:
                    #print(n,name)
                        clean_weight = module.weight.data.detach()
                        attack_weight = self.flip_bit(module)
                
                    # change the weight to attacked weight and get loss
                        module.weight.data = attack_weight
                        output = model(data)
                    # logging.info(name)
                        new_loss = self.criterion(output,target).item()
                    # if(new_loss!=self.loss_dict[name]):
                        # logging.info(new_loss)
                        self.loss_dict[name] = new_loss
                    # logging.info(self.loss_dict[name])                                      
                    # change the weight back to the clean weight
                        module.weight.data = clean_weight
            
            # after going through all the layer, now we find the layer with max loss
            max_loss_module = max(self.loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[max_loss_module]
        print(self.loss_dict.items())    
            # logging.info("loss dict: ", self.loss_dict) 
        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change the that layer's weight without putting back the clean weight
        
        
        n=0
        for name, module in model.named_modules():
            n=n+1
            #print(n,name)
            if name == max_loss_module:
                attack_weight = self.flip_bit(module)
                module.weight.data = attack_weight

        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        print(self.n_bits2flip)
        self.n_bits2flip = 0

        return
