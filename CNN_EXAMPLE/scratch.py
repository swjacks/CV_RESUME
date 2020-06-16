#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:05:50 2020

Implementing neural network 
Scratch work 
@author: shanejackson
"""

import numpy as np
import torch as torch

dtype = torch.float
device = torch.device("cpu")

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, 1, 10, 3

#Create data
x = torch.randn(N,D_in,requires_grad=True)#, device=device, dtype=dtype,requires_grad=False)
y = torch.randn(N,D_out,requires_grad=True)#, device=device, dtype=dtype, requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

y1 = model(x)

'''
epochs        = 500
for t in range(epochs):
    #Forward Pass
    y_pred = model(x)
#    torch.autograd(y_pred,x)
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    model.zero_grad()
    #Back prop calculate partial derivatives
    loss.backward()
    # Update weights
    optimizer.step()


'''