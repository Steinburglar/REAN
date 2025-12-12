import numpy as np
import torch
from torchvision import transforms
from   .. utils import rot_img
import math
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class RandomGroupRotation:
    """Rotate the image by a random angle within a specified range.
    Like most transforms, acts on a single image of shape [C, H, W]."""

    def __init__(self, group_order = 4, interpolation=InterpolationMode.BILINEAR, expand=False, center=None, fill=None,):
        self.group_order = group_order
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill
        self.angles = [i * (2 * np.pi / group_order) for i in range(group_order)]

    def __call__(self, x):
        angle = math.degrees(np.random.choice(self.angles))
        return TF.rotate(x, angle, interpolation=self.interpolation, expand=self.expand, center=self.center, fill=self.fill)

class IsoNoise:
    """Add isotropic Gaussian noise to the image.
    Acts on a single image of shape [C, H, W]."""

    def __init__(self, mean=0.0, std=0.1, gamma=0.0):
        self.mean = mean
        self.std = std


    def __call__(self, x):
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise

class AnisoNoise:
    """Add anisotropic Gaussian noise to the image, -no correlations
    uses a bifurcated horizontal sine pattern
    Acts on a single image of shape [C, H, W].
    """

    def __init__(self, mean=0.0, std=0.1, gamma=1):
        self.mean = mean
        self.std = std
        self.gamma = gamma #controls the size of the sinusoidal perturbations, must be between 0 and 1 (cant be > 1, or there will be negative alphas)
    def __call__(self, x):
        _, H, W = x.shape
        #make a horizontal band - pattern for the left half of the image:
        left_angles =  torch.linspace(0, np.pi, H) #half sine wave from 0 to pi
        right_angles = torch.linspace(0, 4 * np.pi, H) #full sine wave from 0 to 2pi
        alpha = torch.zeros((H, W))
        for i in range(W):
            if i < W // 2:
                alpha[:, i] = 0.5 + self.gamma* torch.sin(left_angles) #the
            else:
                alpha[:, i] = self.gamma + self.gamma *torch.cos(right_angles)
        #normalize alpha so that the average alpha is 1
        alpha = alpha / torch.mean(alpha)

        self.alpha = alpha
        #assign noise to each pixel based on alpha
        noise = torch.randn_like(x) * self.std * np.sqrt(alpha) + self.mean
        return x + noise

    def alpha(self):
        return self.alpha









