#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:20:06 2020

@author: shanejackson
"""
import torch 
import math 
import matplotlib.pyplot as plt
import numpy as np
dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 100, 1, 22, 2
epochs = 10000

#x = torch.randn(N,D_in)#, device=device, dtype=dtype,requires_grad=False)
x = torch.linspace(0,math.pi,N)
x=x.view(N,D_in)
#y = torch.randn(N,D_out)#, device=device, dtype=dtype, requires_grad=False)

class MyModel(torch.nn.Module):
    def __init__(self, D_in,H,D_out):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        #g, dg, ddg activation function + 2 derivatvies with respect to input
        g    = torch.tanh        
        act1 = g(self.linear1(x))
        dg   = (1-act1.t()**2) * self.linear1.weight
        ddg  = -2*dg*self.linear1.weight *act1.t()
        
        Qt   = self.linear2(act1)
        dQt  = self.linear2.weight.mm(dg).t()
        ddQt = self.linear2.weight.mm(ddg).t()
        return torch.stack([Qt,dQt,ddQt],dim=0)

def diffLoss(y,ics,diffEq,setIcs):
    diff   = diffEq(y)#(y[0]+y[2]).pow(2).sum()
    initCond = setIcs(ics)
    loss     = diff+10*initCond
    return loss


## Define the differential Equation and Initial Conditions to pass
## into loss funtion
## Terms we will need each epoch 
## T - kinetic energy
## V - Potential energy 
## Raidii 
def sine(y):
    diffEq = (y[0]+y[2]).pow(2).sum()
    return diffEq
def setIC(ics):
    return (ics[0]).pow(2).sum()+(ics[1]-1.0).pow(2).sum()

def Action(Qt): ## OM Action
    dA = (Qt[2][:,:D_out] - fx) ** 2
    dA += (Qt[2][:,Np:2*Np] - fy) ** 2 
    dA += (Qt[2][:,2*Np:] - fz) ** 2   
    return dA

lr = 1e-4
model = MyModel(D_in,H,D_out)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for t in range(epochs):
    
    y =  model(x)
    ics = model(torch.tensor([[0.0]]))
    #loss = (y[0]+y[2]).pow(2).sum()+10*(ics[0]).pow(2).sum()+10*(ics[1]-1.0).pow(2).sum()
    loss=diffLoss(y,ics,sine,setIC)
    if t % 1000 == 999:
        print(loss)
    
    model.zero_grad()
    #Back prop calculate partial derivatives
    loss.backward()
    # Update weights
    optimizer.step()




plt.plot(x,y[0].detach(),'r')
plt.plot(x,np.sin(x),'b--')