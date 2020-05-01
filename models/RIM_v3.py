import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

from models.GRU import GRU


class RIM(nn.Module):

    def __init__(self, input_size = 2, output_size = 1, st_size = 2, hidden_size = 2, bounded=-1, lr=.001):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.st_size = st_size
        self.lr = lr

        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=5, padding=2, groups=2)
        self.rnn_layer1 = GRU(hidden_size, st_size)
        self.conv2 = nn.Conv2d(st_size, hidden_size, kernel_size=3, padding=1)
        self.rnn_layer2 = GRU(hidden_size, st_size)
        self.conv3 = nn.Conv2d(hidden_size, output_size, kernel_size=3, padding=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.bounded = bounded

    def forward(self, xt, st1, st2, j):
        
        if j ==0:
            st1=torch.zeros(xt.size())
        
        out = f.relu(self.conv1.forward(xt))
        #out = nn.BatchNorm2d(out.shape[1])(out)
        st_out1 = self.rnn_layer1.forward(out, st1)
        out = f.relu(self.conv2.forward(st_out1))
        #out = nn.BatchNorm2d(out.shape[1])(out)

        
        if j ==0:
            st2=torch.zeros(out.size())
            
        st_out2 = self.rnn_layer2.forward(out, st2)

        if self.bounded > 0:
            out = torch.clamp(self.conv3.forward(st_out2), -self.bounded, self.bounded)
        else:
            out = self.conv3.forward(st_out2)
      

        return out, st_out1, st_out2

    def backprop(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def loss(self, theta, list_psi_t):
        loss_t = self.loss_func(theta, list_psi_t)
        return self.weight_func(loss_t)

    def init_hidden(self, batch_dim=1):
        return torch.zeros((batch_dim, self.st_size))