"""
Implements a plain CNN model for comparison with the equivariant architectures.\
Note that in order to be fair, we must give each CNN an equivalent number of parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainCNN(nn.Module):
    #applies ReLU after each conv layer except the last one
    def __init__(self, in_channels, out_channels, kernel_size, hidden_dim, num_gconvs, classes=10, group_order=4):
        super(PlainCNN, self).__init__()

        self.first = nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2,)
        self.hiddens = []
        for _ in range(num_gconvs):
            self.hiddens.append(nn.Conv2d(hidden_dim,
                                          hidden_dim,
                                          kernel_size=kernel_size,
                                          padding=kernel_size // 2))
            self.hiddens.append(nn.ReLU())
        self.hiddens = nn.Sequential(*self.hiddens)
        self.last = nn.Conv2d(hidden_dim, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.linear = nn.Linear(out_channels, classes) #we will do global average pooling before this layer

    def forward(self, x):
        x = F.relu(self.first(x))
        x = self.hiddens(x) #hiddens already has ReLU activations
        x = self.last(x)
        x = torch.mean(x, dim =(2, 3))  # Global average pooling
        logits = self.linear(x)
        return logits
