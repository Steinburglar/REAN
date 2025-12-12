"""
Creates a RelaxedP4 model for image classification, following the tutorial at:
https://github.com/Rui1521/Equivariant-CNNs-Tutorial/blob/main/Tutorial_Group_Convolution.ipynb
Credit the original author Rui Zhang, and the authors of "Approximately Equivariant Networks for Imperfectly Symmetric Dynamics"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .. utils import rot_img, count_parameters


class Relaxed_LiftingConvolution(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 group_order,
                 num_filter_banks,
                 activation = True
                 ):
        super(Relaxed_LiftingConvolution, self).__init__()

        self.num_filter_banks = num_filter_banks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation

        self.combination_weights = nn.Parameter(torch.ones(num_filter_banks, group_order).float()/num_filter_banks)

        # Initialize an unconstrained kernel.
        self.weight = torch.nn.Parameter(torch.zeros(self.num_filter_banks, # Additional dimension
                                                     self.out_channels,
                                                     self.in_channels,
                                                     self.kernel_size,
                                                     self.kernel_size))
        stdv = np.sqrt(1/(self.in_channels*self.kernel_size*self.kernel_size))
        self.weight.data.uniform_(-stdv, stdv)

        # If combination_weights are equal values, then the model is still equivariant
        self.combination_weights.data.uniform_(-stdv, stdv)

    def generate_filter_bank(self):
        """ Obtain a stack of rotated filters"""
        #collapse along the new appropiate filter bank dimension first, then do machinery of lifting
        weights = self.weight.reshape(self.num_filter_banks*self.out_channels,
                                      self.in_channels,
                                      self.kernel_size,
                                      self.kernel_size)
        filter_bank = torch.stack([rot_img(weights, -np.pi*2/self.group_order*i)
                                   for i in range(self.group_order)])
        filter_bank = filter_bank.transpose(0,1).reshape(self.num_filter_banks, # Additional dimension
                                                         self.out_channels,
                                                         self.group_order,
                                                         self.in_channels,
                                                         self.kernel_size,
                                                         self.kernel_size)
        return filter_bank


    def forward(self, x):
        # input shape: [bz, #in, h, w]
        # output shape: [bz, #out, group order, h, w]

        # generate filter bank given input group order
        filter_bank = self.generate_filter_bank()

        # for each rotation, we have a linear combination of multiple filters with different coefficients.
        relaxed_conv_weights = torch.einsum("na, noa... -> oa...", self.combination_weights, filter_bank) #apply weights to make final "kernel" tensor to hand to 2d conv

        # concatenate the first two dims before convolution.
        # ==============================
        x = F.conv2d(
            input=x,
            weight=relaxed_conv_weights.reshape(
                self.out_channels * self.group_order,
                self.in_channels,
                self.kernel_size,
                self.kernel_size
            ),
            padding = (self.kernel_size-1)//2
        )
        # ==============================

        # reshape output signal to shape [bz, #out, group order, h, w].
        # ==============================
        x = x.view(
            x.shape[0],
            self.out_channels,
            self.group_order,
            x.shape[-1],
            x.shape[-2]
        )
        # ==============================

        if self.activation:
            return F.relu(x)
        return x


class Relaxed_GroupConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 group_order,
                 num_filter_banks,
                 activation = True
                ):

        super(Relaxed_GroupConv, self).__init__()

        self.num_filter_banks = num_filter_banks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation


        ## Initialize weights
        self.combination_weights = nn.Parameter(torch.ones(group_order, num_filter_banks).float()/num_filter_banks)
        self.weight = nn.Parameter(torch.randn(self.num_filter_banks, ##additional dimension
                                               self.out_channels,
                                               self.in_channels,
                                               self.group_order,
                                               self.kernel_size,
                                               self.kernel_size))

        stdv = np.sqrt(1/(self.in_channels))
        self.weight.data.uniform_(-stdv, stdv)

        # If combination_weights are equal values, then the model is still equivariant
        # self.combination_weights.data.uniform_(-stdv, stdv)


    def generate_filter_bank(self):
        """ Obtain a stack of rotated and cyclic shifted filters"""
        filter_bank = []
        weights = self.weight.reshape(self.num_filter_banks*self.out_channels*self.in_channels, #similarly, collapse new dimension in with others, do machinery
                                      self.group_order,
                                      self.kernel_size,
                                      self.kernel_size)

        for i in range(self.group_order):
            # planar rotation
            rotated_filter = rot_img(weights, -np.pi*2/self.group_order*i)

            # cyclic shift
            shifted_indices = torch.roll(torch.arange(0, self.group_order, 1), shifts = i)
            shifted_rotated_filter = rotated_filter[:,shifted_indices]


            filter_bank.append(shifted_rotated_filter.reshape(self.num_filter_banks,
                                                              self.out_channels,
                                                              self.in_channels,
                                                              self.group_order,
                                                              self.kernel_size,
                                                              self.kernel_size))
        # stack
        filter_bank = torch.stack(filter_bank).permute(1,2,0,3,4,5,6)
        return filter_bank

    def forward(self, x):

        filter_bank = self.generate_filter_bank()

        relaxed_conv_weights = torch.einsum("na, aon... -> on...", self.combination_weights, filter_bank)
        #by now, since we have applied the weights, its the same as normall full group convolution

        x = torch.nn.functional.conv2d(
            input=x.reshape(
                x.shape[0],
                x.shape[1] * x.shape[2],
                x.shape[3],
                x.shape[4]
                ),
            weight=relaxed_conv_weights.reshape(
                self.out_channels * self.group_order,
                self.in_channels * self.group_order,
                self.kernel_size,
                self.kernel_size
            ),
            padding = (self.kernel_size-1)//2
        )

                # Reshape signal back [bz, #out * g_order, h, w] -> [bz, out, g_order, h, w]
        x = x.view(x.shape[0], self.out_channels, self.group_order, x.shape[-2], x.shape[-1])
        # ========================

        if self.activation:
            return F.relu(x)
        return x

class RelaxedP4CNN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, hidden_dim, group_order, num_gconvs, classes=10, factor=3.2):
        super().__init__()
        hidden_dim = hidden_dim // factor # adjusting hidden dim to account for extra parameters in relaxed convs
        hidden_dim = int(hidden_dim)
        out_channels = int(out_channels//factor)

        # First Layer
        self.lifting_conv = Relaxed_LiftingConvolution(in_channels = in_channels,
                                                       out_channels = hidden_dim,
                                                       kernel_size = kernel_size,
                                                       group_order = group_order,
                                                       num_filter_banks = 3,
                                                       activation = True)
        # Middle Layer
        self.gconvs = []
        for i in range(num_gconvs):
            self.gconvs.append(Relaxed_GroupConv(in_channels = hidden_dim,
                                                out_channels = hidden_dim,
                                                kernel_size = kernel_size,
                                                group_order = group_order,
                                                num_filter_banks = 3,
                                                activation = True))

        self.gconvs = nn.Sequential(*self.gconvs)


        # Final Layer
        self.final_layer = Relaxed_GroupConv(in_channels = hidden_dim,
                                            out_channels = out_channels,
                                            kernel_size = kernel_size,
                                            group_order = group_order,
                                            num_filter_banks = 3,
                                            activation = False)
        self.linear = nn.Linear(out_channels, classes)

        print(count_parameters(self), "trainable parameters in RelaxedP4CNN model")

    def forward(self, x):
        out = self.lifting_conv(x)

        out = self.gconvs(out)

        out = self.final_layer(out)
        out = torch.mean(out, dim = (2,3,4)) #adaptation to give full classification output
        logits = self.linear(out)
        return logits


def regularization_loss(model, alpha=1.0):
    """ computes the appropriate regularization loss for a relaxed equivariant model.
    Args: model: a relaxed equivariant model
    the model will have a first layer, some gconv layers, and a final layer (plus a linear classifier)
    each of these layers will have combination_weights attributes
    Returns:
        reg_loss: the regularization loss
    """
    combination_weights = [] #list where we will store the combination weight matrices of each layer
    # first layer
    combination_weights.append(model.lifting_conv.combination_weights.permute(1,0)) #shape (group order, num filter banks)
    # gconv layers
    for gconv in model.gconvs:
        combination_weights.append(gconv.combination_weights)
    # final layer
    combination_weights.append(model.final_layer.combination_weights)
    #should all be the same shape (group order, num filter banks), so we can stack them
    combination_weights = torch.stack(combination_weights, dim=0) #shape (num layers, group order, num filter banks)
    #expand dims to do pairwise differences
    cw_exp1 = combination_weights.unsqueeze(2) #shape (num layers, group order, 1, num filter banks)
    cw_exp2 = combination_weights.unsqueeze(1) #shape (num layers, 1, group order, num filter banks)
    diffs = cw_exp1 - cw_exp2 #shape (num layers, group order, group order, num filter banks) - broadcasts the subtraction
    abs_diffs = torch.abs(diffs)
    reg_loss = torch.sum(abs_diffs) #L1 norm of all pairwise differences
    reg_loss = alpha * reg_loss
    return reg_loss