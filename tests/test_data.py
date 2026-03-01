"""
Tests functions for the dataset and data transforms. Tests behavior from REAN/data/Dataset.py and REAN/data/DataTransforms.py.
"""

import torch
from rean.data.Dataset import make_datasets
from rean.data.DataTransforms import RandomGroupRotation, IsoNoise, AnisoNoise
import numpy as np


def test_make_datasets():
    # Test that datasets are created with correct transformations
    train_dataset, val_dataset, test_dataset, in_channels = make_datasets(
        dataname="mnist",
        group_order=4,
        train_noise="iso",
        test_noise="aniso",
        noise_params={"std": 0.1, "gamma": 0.5},
        rotate_train=True,
        rotate_test=False
    )
    assert in_channels == 1, "MNIST should have 1 input channel"

    train_t = getattr(train_dataset.dataset.transform, "transforms", [])
    val_t = getattr(val_dataset.dataset.transform, "transforms", [])
    test_t  = getattr(test_dataset.transform, "transforms", [])

    assert (train_t == val_t)
    assert any(isinstance(t, RandomGroupRotation) for t in train_t)
    assert any(isinstance(t, IsoNoise) for t in train_t)
    assert any(isinstance(t, AnisoNoise) for t in test_t)


def test_random_group_rotation():
    # Test that RandomGroupRotation rotates images by valid angles
    transform = RandomGroupRotation(group_order=4)
    image = torch.zeros((1, 28, 28))  # Dummy image
    rotated_image = transform(image)
    # Since the rotation is random, we can't test the exact output, but we can check that the output shape is correct
    assert rotated_image.shape == image.shape, "Rotated image should have the same shape as input image"

def test_iso_noise():
    # Test that IsoNoise adds noise with correct mean and std
    torch.manual_seed(42)  # Set seed for deterministic behavior

    transform = IsoNoise(mean=0.0, std=0.1)
    image = torch.zeros((1, 28, 28))  # Dummy, empty image
    noise = transform(image)
    assert torch.isclose(noise.mean(), torch.tensor(0.0), atol=0.01), "Noise mean should be close to 0"
    assert torch.isclose(noise.std(), torch.tensor(0.1), atol=0.01), "Noise std should be close to 0.1"

def test_aniso_noise():
    # Test that AnisoNoise adds noise with correct mean and std
    torch.manual_seed(42)  # Set seed for deterministic behavior

    transform = AnisoNoise(mean=0.0, std=0.1, gamma=0.5)
    image = torch.zeros((1, 28, 28))  # Dummy, empty image
    noise = transform(image)
    assert torch.isclose(noise.mean(), torch.tensor(0.0), atol=0.01), "Noise mean should be close to 0"
    assert torch.isclose(noise.std(), torch.tensor(0.1), atol=0.01), "Noise std should be close to 0.1"
