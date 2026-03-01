"""
Tests for training loop and related functions in rean/training/train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rean.training.train import evaluate, train_one_epoch, train_full


class TinyNet(nn.Module):
    """Minimal model for isolated training-loop tests."""

    def __init__(self, in_dim=4, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def dummy_loader(num_samples=24, in_dim=4, num_classes=3, batch_size=8):
    torch.manual_seed(42)
    x = torch.randn(num_samples, in_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def dummy_reg_loss(outputs, labels, model_arg):
    pred = F.cross_entropy(outputs, labels)
    reg = 0.01 * sum((p ** 2).sum() for p in model_arg.parameters())
    return pred + reg

def test_train_one_epoch():
    device = torch.device("cpu")
    model = TinyNet().to(device)
    loader = dummy_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_function = dummy_reg_loss

    before = [p.detach().clone() for p in model.parameters()]
    epoch_loss = train_one_epoch(model, loader, loss_function, optimizer, device)
    after = [p.detach().clone() for p in model.parameters()]

    assert isinstance(epoch_loss, float)
    assert epoch_loss > 0
    assert any(not torch.allclose(b, a) for b, a in zip(before, after)), "Parameters should update"
    assert model.training is True


def test_evaluate():
    device = torch.device("cpu")
    model = TinyNet().to(device)
    loader = dummy_loader()
    criterion = nn.CrossEntropyLoss()

    before = [p.detach().clone() for p in model.parameters()]
    val_loss, val_acc = evaluate(model, device, loader, criterion)
    after = [p.detach().clone() for p in model.parameters()]

    assert isinstance(val_loss, float)
    assert val_loss > 0
    assert isinstance(val_acc, float)
    assert 0.0 <= val_acc <= 100.0
    assert all(torch.allclose(b, a) for b, a in zip(before, after)), "evaluate() should not update params"
    assert model.training is False

#Test train_full not here, since it is inherently an integration test.


