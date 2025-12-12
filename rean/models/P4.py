"""
Contains a model class for P4.
First defines a lift conv block, then a full group block for p4
finally defines a full model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  .. utils import  rot_img, count_parameters



class LiftingConvolution(nn.Module):
    """Lifting Convolution Layer for finite rotation group

    Attributes:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: kernel size
        group_order: the order of rotation groups (e.g p4 has order 4)
        activation: whether to use relu.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 group_order,
                 activation = True
                 ):
        super(LiftingConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation

        # Initialize an unconstrained kernel.
        self.weight = torch.nn.Parameter(torch.zeros(self.out_channels,
                                                     self.in_channels,
                                                     self.kernel_size,
                                                     self.kernel_size))

        # Initialize weights
        stdv = np.sqrt(1/(self.in_channels*self.kernel_size*self.kernel_size))
        self.weight.data.uniform_(-stdv, stdv)

    def generate_filter_bank(self):
        # Obtain a stack of rotated filters
        # Rotate kernels by 0, 90, 180, and 270 degrees
        # ==============================
        filter_bank = torch.stack([rot_img(self.weight, -np.pi*2/self.group_order*i)
                                   for i in range(self.group_order)])
        # ==============================

        # [#out, group_order, #in, k, k]
        filter_bank = filter_bank.transpose(0,1)
        return filter_bank


    def forward(self, x):
        # input shape: [bz, #in, h, w]
        # output shape: [bz, #out, group order, h, w]

        # generate filter bank given input group order
        filter_bank = self.generate_filter_bank()

        # concatenate the first two dims before convolution.
        # ==============================
        x = F.conv2d(
            input=x,
            weight=filter_bank.reshape(
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




class GroupConvolution(nn.Module):
    """Group Convolution Layer for finite rotation group

    Attributes:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: kernel size
        group_order: the order of rotation groups
        activation: whether to use relu.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 group_order,
                 activation = True,
                 ):
        super(GroupConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_order = group_order
        self.activation = activation

        # Initialize an unconstrained kernel.
        # the weights have an additional group order dimension.
        self.weight = torch.nn.Parameter(torch.zeros(self.out_channels,
                                                     self.in_channels,
                                                     self.group_order, # this is different from the lifting convolution
                                                     self.kernel_size,
                                                     self.kernel_size))

        stdv = np.sqrt(1/(self.in_channels*self.kernel_size*self.kernel_size))
        self.weight.data.uniform_(-stdv, stdv)

    def generate_filter_bank(self):
        # Obtain a stack of rotated and cyclic shifted filters
        filter_bank = []
        weights = self.weight.reshape(self.out_channels*self.in_channels, #each index on collapsed dimension tracks a unique color channel pair, which is not rotated, and the indices have no structure with one another
                                     self.group_order,
                                     self.kernel_size,
                                     self.kernel_size)

        for i in range(self.group_order): #loop over each (nontranslational) group element, here rotations
        #here, i is the index of the g out order
            # planar rotation
            rotated_filter = rot_img(weights, -np.pi*2/self.group_order*i) #angle of rotation is determined by group elemt, i.e r_2 is 180 degrees

            # cyclic shift - weights[:, h, :, :] is the kernel for relative element h.
            #to get kernels that bring g_in to g_out properly, we need to make sure that our full filter bank has W(g_out^-1 g_in) has W(h) - that is, has the right relative transformation
            shifted_indices = torch.roll(torch.arange(0, self.group_order, 1), shifts = i)
            shifted_rotated_filter = rotated_filter[:,shifted_indices]


            filter_bank.append(shifted_rotated_filter.reshape(self.out_channels, #list along g out
                                                            self.in_channels,
                                                            self.group_order,
                                                            self.kernel_size,
                                                            self.kernel_size))
        # stack
        # reshape output signal to shape [#out, g_order, #in, g_order, k, k].
        filter_bank = torch.stack(filter_bank).transpose(0,1)  #stack turns list of g outs into a tensor dimension
        return filter_bank

    def forward(self, x):
        # input shape: [bz, in, group order, x, y]
        # output shape: [bz, out, group order, x, y]

        # Generate filter bank with shape [#out, g_order, #in, g_order, h, w]
        filter_bank = self.generate_filter_bank()

        # Reshape filter_bank to use F.conv2d. basically, for this step, we act like a channel is a combination of color and group. we have already done the work te ensure this doesnt break equivariance, by weight tying when we built filter bank
        # [#out, g_order, #in, g_order, h, w] -> [#out*g_order, #in*g_order, h, w]
        # ==============================
        x = torch.nn.functional.conv2d(
            input=x.reshape(
                x.shape[0],
                x.shape[1] * x.shape[2],
                x.shape[3],
                x.shape[4]
                ),
            weight=filter_bank.reshape(
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


class P4CNN(torch.nn.Module):
    """A full P4-CNN model. No pooling layers, with final invariant classifier.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 hidden_dim,
                 group_order,
                 num_gconvs, # number of group convolution layers.
                 classes = 10,#number of classes for classification task
                 ):
        super().__init__()
        hidden_dim = int(hidden_dim // np.sqrt(group_order)) #adjust number of trainable params
        out_channels = int(out_channels//np.sqrt(group_order))

        # First Layer
        self.lifting_conv = LiftingConvolution(in_channels = in_channels,
                                               out_channels = hidden_dim,
                                               kernel_size = kernel_size,
                                               group_order = group_order,
                                               activation = True)
        # Middle Layers
        self.gconvs = []
        for i in range(num_gconvs):
            self.gconvs.append(GroupConvolution(in_channels = hidden_dim,
                                                out_channels = hidden_dim,
                                                kernel_size = kernel_size,
                                                group_order = group_order,
                                                activation = True))
        self.gconvs = nn.Sequential(*self.gconvs)


        # Final Layer
        # To generate equivariant outputs
        self.final_layer = GroupConvolution(in_channels = hidden_dim,
                                            out_channels = out_channels,
                                            kernel_size = kernel_size,
                                            group_order = group_order,
                                            activation = False)
        self.linear = nn.Linear(out_channels, classes)

        print(count_parameters(self), "trainable parameters in P4CNN model")


    def forward(self, x):
        out = self.lifting_conv(x)

        out = self.gconvs(out)

        # functions on (g,x,y) -> functions on (x,y)
        #out = torch.mean(self.final_layer(out), dim = 2)

        # If we want to have a invariant classifer, we can average over the last  three dimensions- all group dimensions
        out = torch.mean(self.final_layer(out), dim = (2,3,4)) #left with just batch and color channels
        logits = self.linear(out)

        return logits