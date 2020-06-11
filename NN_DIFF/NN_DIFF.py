#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:40:54 2020

Solving Differential Equations with a Neural Network Using Pytorch

@author: shanejackson
"""

import torch 
dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 10, 1, 8, 1
epochs = 5000

#x = torch.randn(N,D_in)#, device=device, dtype=dtype,requires_grad=False)
x = torch.linspace(0,1,N)
x=x.view(N,D_in)
y = torch.randn(N,D_out)#, device=device, dtype=dtype, requires_grad=False)

class MyModel(torch.nn.Module):
    def __init__(self, D_in,H,D_out):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        g    = torch.tanh
        
        act1 = g(self.linear1(x))
        dg   = (1-act1.t()**2) * self.linear1.weight
        Qt   = self.linear2(act1)
        dQt  = self.linear2.weight.mm(dg).t()
        return torch.stack([Qt,dQt],dim=0)

model = MyModel(D_in,H,D_out)
for t in range(epochs):
    
    y =  model(x)

    loss = (y[0]+y[1]).pow(2).sum()+(model(torch.tensor([[0.0]]))-1).pow(2).sum()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if t % 100 == 99:
        print(loss)
    
    model.zero_grad()
    #Back prop calculate partial derivatives
    loss.backward()
    # Update weights
    optimizer.step()
    

'''
loss_fn = torch.nn.MSELoss(reduction='sum')
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
y_pred = model(x)
print(x.grad)
#    torch.autograd(y_pred,x)
loss = loss_fn(y_pred, y)

model.zero_grad()
#Back prop calculate partial derivatives
loss.backward()
# Update weights
optimizer.step()
'''
'''
g    = torch.tanh
linear1 = torch.nn.Linear(D_in, H)
linear2 = torch.nn.Linear(H, D_out)        
act1 = g(linear1(x))
dg   = (1-act1.t()**2) * linear1.weight
Qt   = linear2(act1)
dQt  = linear2.weight.mm(dg).t()

def QT2(Params, t_seq):
  ## output of the network is qx, qy for each particle
  w0 = Params[:n]
  b0 = Params[n:2*n]
  w1 = Params[2*n:2*n+(n*d*Np)].reshape((d*Np,n))
  b1 = Params[2*n+(n*d*Np):2*n+(n*d*Np)+d*Np]
  tmp = np.outer(w0, t_seq) + b0[:,None]
  tmp1 = sigmoid(tmp)
  q = np.dot(w1,tmp1) + b1[:,None]
  tmp2 = (tmp1 - tmp1**2)*w0[:,None]
  dq = np.dot(w1, tmp2)
  tmp3 = tmp2*(1.0-2.0*tmp1)*w0[:,None]
  ddq = np.dot(w1, tmp3)
  ## output Q and dQ/dt matrix with dim (len(t_seq), d*Np)
  return q.T, dq.T, ddq.T
'''

