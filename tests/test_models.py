"""
Unit Tests for REAN Models. Tests behavior from REAN/models and any related utility functions in REAN/utils/utils.py.
"""
import torch
from rean.models.CNN import PlainCNN
from rean.models.P4 import P4CNN
from rean.models.RelaxedP4 import RelaxedP4CNN, regularization_loss
import numpy as np


def test_plain_cnn():
    model = PlainCNN(in_channels=1,
                     out_channels=16,
                     kernel_size=3,
                     hidden_dim=32,
                     num_gconvs=2,
                     classes=10)
    input = torch.randn(1, 1, 28, 28)  #batch size 1, 1 channel, 28x28 image
    output = model(input)
    assert output.shape == (1, 10), "Output shape should be (1, 10) for batch size 1 and 10 classes"

def test_p4_cnn():
    torch.manual_seed(42)  # Set seed for deterministic behavior
    model = P4CNN(in_channels=1,
                  out_channels=16,
                  kernel_size=3,
                  hidden_dim=32,
                  num_gconvs=2,
                  classes=10,
                  group_order=4)
    input = torch.randn(1, 1, 28, 28)  #batch size 1, 1 channel, 28x28 image
    output = model(input)
    rotated_input = torch.rot90(input, k=1, dims=[2, 3])  # Rotate the input by 90 degrees
    rotated_output = model(rotated_input)
    assert output.shape == (1, 10), "Output shape should be (1, 10) for batch size 1 and 10 classes"
    assert torch.allclose(output, rotated_output, atol=1e-5), "P4CNN should produce the same output for rotated inputs"


def test_relaxed_p4_cnn():
    torch.manual_seed(42)  # Set seed for deterministic behavior
    model = RelaxedP4CNN(in_channels=1,
                         out_channels=16,
                         kernel_size=3,
                         hidden_dim=32,
                         num_gconvs=2,
                         classes=10,
                         group_order=4)
    input = torch.randn(1, 1, 28, 28)  #batch size 1, 1 channel, 28x28 image
    output = model(input)
    assert output.shape == (1, 10), "Output shape should be (1, 10) for batch size 1 and 10 classes"
    reg_loss = regularization_loss(model)
    assert isinstance(reg_loss.item(), float), "Regularization loss should be a scalar float value"
    assert reg_loss.item() >= 0, "Regularization loss should be non-negative"
