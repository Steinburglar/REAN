#train.py
#flexible to accept different datasets and model architectures
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rean.data.Dataset import make_datasets
from rean.models.CNN import PlainCNN
from rean.models.P4 import P4CNN
from rean.models.RelaxedP4 import RelaxedP4CNN
from rean.models.RelaxedP4 import regularization_loss
import argparse
import numpy as np
import random
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    model.train(True) #set model to training mode
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels, model) #pass model for regularization if needed - if no regularization, loss_function ignores it
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, device, dataloader, criterion):
    model.train(False) #set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = 100 * correct / total #%CORRECT
    return epoch_loss, accuracy

def get_model(model_name, **kwargs):
    """
    Handles creation of appropriately sized model of correct architecture, given model_name


    :param model_name: string specifying model architecture can be "PlainCNN", "P4CNN", or "RelaxedP4CNN"
    :return: instantiated model
    """
    if model_name.lower() == "plaincnn":
        model = PlainCNN(**kwargs)
    elif model_name.lower() == "p4cnn":
        model = P4CNN(**kwargs)
    elif model_name.lower() == "relaxedp4cnn":

        model = RelaxedP4CNN(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model




def train_full(model_name, num_epochs, train_ds, val_ds, device, batch_size=64, learning_rate=0.001, prev_runs=None, **kwargs):
    """
    num_epochs is the desired TOTAL number of epochs. If resuming and a matching run is found, training will
    continue from the number of epochs already recorded in that run's run_data.json.
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = get_model(model_name, **kwargs)
    model = model.to(device)
    pred_loss = nn.CrossEntropyLoss()

    if model_name.lower() == "relaxedp4cnn":
        reg_loss = regularization_loss
    else:
        reg_loss = lambda model, **kw: 0

    def loss(outputs, labels, model):
        return pred_loss(outputs, labels) + reg_loss(model, alpha=0.0001)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Default run_data for new runs
    default_run_data = {"train_loss": [],
                        "val_loss": [],
                        "val_acc": [],
                        "epochs": num_epochs,
                        "model_name": model_name,
                        "learning_rate": learning_rate,
                        "kwargs": kwargs
                        }

    run_dir = None
    run_data = None
    start_epoch = 0

    if prev_runs is not None:
        rundatapath = prev_runs / "run_data.json"
        testdatapath = prev_runs / "test_data.json"
        with rundatapath.open("r", encoding="utf-8") as f:
            run_data = json.load(f)
        with testdatapath.open("r", encoding="utf-8") as f:
            test_data = json.load(f)
        run_data.update(test_data)

    else:
        run_data = default_run_data

    # If run_data has a different total epochs recorded, update it to reflect requested TOTAL num_epochs
    start_epoch = run_data["epochs"]

    # If we've already completed the desired total epochs, just return loaded model and run_data
    if start_epoch >= num_epochs:
        # ensure best model is loaded (already attempted), return as-is
        return run_data, model

    best_val_acc = 0.0
    best_state = None
    # If run_data had existing val_acc, set best_val_acc accordingly
    if run_data.get("val_acc"):
        try:
            best_val_acc = max(run_data["val_acc"])
        except Exception:
            best_val_acc = 0.0

    # Train remaining epochs; keep indices consistent with total epochs (append to existing lists)
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss, optimizer, device)
        val_loss, val_acc = evaluate(model, device, val_loader, pred_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
        run_data["train_loss"].append(train_loss)
        run_data["val_loss"].append(val_loss)
        run_data["val_acc"].append(val_acc)
    run_data["epochs"] = num_epochs  # keep total epochs updated


    # If we loaded a best_state (or set during this run), restore it to the returned model
    if best_state is not None:
        model.load_state_dict(best_state)

    return run_data, model

if __name__ == "__main__":
    print("Starting training script...")
    dataset_name = "mnist"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    group_order = 4
    train_noise = None
    test_noise = None
    noise_params = {}
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 2
    hidden_dim = 8
    num_gconvs = 3
    kernel_size = 3
    classes = 10
    model_name = "PlainCNN"  # could be extended to other models later

    train_ds, val_ds, test_ds, in_channels = make_datasets()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model = RelaxedP4CNN(in_channels, out_channels=hidden_dim, group_order=group_order, kernel_size=kernel_size, #make this not hardcoded later
                     hidden_dim=hidden_dim, num_gconvs=num_gconvs, classes=classes, )
    model = model.to(device)
    pred_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, pred_loss, optimizer, device)
        val_loss, val_acc = evaluate(model, device, val_loader, pred_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")


