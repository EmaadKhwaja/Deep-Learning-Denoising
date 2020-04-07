
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

from models.GRU import GRU


class RIM(nn.Module):

    def __init__(self, input_size = 1, output_size = 1, st_size = 1, hidden_size = 1, bounded=-1, lr=.001):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.st_size = st_size
        self.lr = lr

        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2)
        self.rnn_layer = GRU(hidden_size, st_size)
        self.conv3 = nn.Conv2d(st_size, hidden_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_size, output_size, kernel_size=3, padding=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.bounded = bounded

    def forward(self, xt, st):
        out = f.relu(self.conv1.forward(xt))
        # out = nn.BatchNorm1d(out.shape[1])(out)
        out = f.relu(self.conv2.forward(out))
        # out = nn.BatchNorm1d(out.shape[1])(out)
        st_out = self.rnn_layer.forward(out, st)
        
        out = f.relu(self.conv3.forward(st_out))
        out = self.conv4.forward(out)


        # out = nn.BatchNorm1d(out.shape[1])(out)
        if self.bounded > 0:
            out = torch.clamp(self.conv4.forward(out), -self.bounded, self.bounded)
        else:
            out = self.conv4.forward(out)

        return out, st_out

    def backprop(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def loss(self, theta, list_psi_t):
        loss_t = self.loss_func(theta, list_psi_t)
        return self.weight_func(loss_t)

    def init_hidden(self, batch_dim=1):
        return torch.zeros((batch_dim, self.st_size))
