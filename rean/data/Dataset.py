import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from rean.data.DataTransforms import RandomGroupRotation, IsoNoise, AnisoNoise
import numpy as np

def make_datasets(
        dataname = "mnist", #can be "mnist" or "cifar10"
        group_order =4, #order of the rotation group for data augmentation
        train_noise = None, #can be None, "none", "iso", or "aniso"
        test_noise = None, #can be None, "iso", or "aniso"
        noise_params = {}, #dictionary of parameters for the noise transforms
        root = './data', #root directory for the dataset
        rotate_train =  True,
        rotate_test = True #whether to apply random rotations during training
    ):
    """Create training and test datasets with specified transformations"""
    if dataname.lower() == "mnist":
        base_dataset = datasets.MNIST #assigning a function to a variable, not calling it yet
        in_channels = 1
    elif dataname.lower() == "cifar10":
        base_dataset = datasets.CIFAR10
        in_channels = 3
    else:
        raise ValueError("Unsupported dataset. Choose either 'mnist' or 'cifar10'.")

    if train_noise == "none":
        train_noise = None
    if test_noise == "none":
        test_noise = None
    train_transform = build_transforms(rotate_train, group_order, train_noise, noise_params)
    test_transform = build_transforms(rotate_test, group_order, test_noise, noise_params)


    full_train_dataset = base_dataset(root=root, train=True, download=True, transform=train_transform)
    test_dataset = base_dataset(root=root, train=False, download=True, transform=test_transform)

    # Split full_train_dataset into training and validation sets (e.g., 80-20 split)
    num_train = int(0.8 * len(full_train_dataset))
    num_val = len(full_train_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [num_train, num_val],
                                                     generator=torch.Generator().manual_seed(42))  # for reproducibility
    return train_dataset, val_dataset, test_dataset, in_channels


def build_transforms(rotate, group_order, noise_type, noise_params):
        transform_list = []

        if rotate:
            transform_list.append(RandomGroupRotation(group_order=group_order)) #expects PIL image?

        transform_list.append(transforms.ToTensor())
        if noise_type == "iso":
            transform_list.append(IsoNoise(**noise_params))
        elif noise_type == "aniso":
            transform_list.append(AnisoNoise(**noise_params))
        return transforms.Compose(transform_list)