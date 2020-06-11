#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:40:54 2020

Solving Differential Equations with a Neural Network Using Pytorch

@author: shanejackson
"""

import torch 
import math 
import matplotlib.pyplot as plt
import numpy as np
dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 100, 1, 22, 1
epochs = 50000

#x = torch.randn(N,D_in)#, device=device, dtype=dtype,requires_grad=False)
x = torch.linspace(0,math.pi/2,N)
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
lr = 1e-4
model = MyModel(D_in,H,D_out)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for t in range(epochs):
    
    y =  model(x)
    ics = model(torch.tensor([[0.0]]))
    loss = (y[0]+y[2]).pow(2).sum()+10*(ics[0]).pow(2).sum()+10*(ics[1]-1.0).pow(2).sum()
    if t % 1000 == 999:
        print(loss)
    
    model.zero_grad()
    #Back prop calculate partial derivatives
    loss.backward()
    # Update weights
    optimizer.step()




plt.plot(x,y[0].detach())
plt.plot(x,np.sin(x))