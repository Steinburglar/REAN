#makes plotters that take in data and make plots.
#generally, assumes all run data comes as a dictionary with keys like "train_loss", "val_loss", "val_acc", etc.
#run_dictionary should also contain data about the model and dataset used, for labeling purposes.

import matplotlib.pyplot as plt
import numpy as np


class LossPlot:
    """
    Class that handles plotting training and validation loss curves. is designed to take as many different run data dictionaries as needed
    for flexibility, so we can plot results from different models and datasets on the same axes.

    One object instance per axes. takes the ax that it should draw on as input.
    """


    def __init__(self, ax, runs, title, labels, train = True, val = True, xlabel="Epochs", ylabel="Loss" ):
        """
        :param ax: matplotlib axes to plot on
        :param runs: list of dictionaries, each containing keys "train_loss" and "val_loss", as well as "label" for legend
        :param title: title of the plot
        :param labels: list of attributes to include in the legend labels for all runs
        :param xlabel: label for x-axis
        :param ylabel: label for y-axis
        """
        self.runs = runs
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        #setup matplotlib figure and axes
        self.ax = ax
        self.labels = labels
        self.plot()

    def plot(self, ):
        for run_data in self.runs:
            # Construct label from specified attributes
            label = " - ".join([str(run_data[attr]) for attr in self.labels])
            epochs = range(1, run_data["epochs"] + 1)
            self.ax.plot(epochs, run_data["train_loss"], label=label + " Train Loss")
            self.ax.plot(epochs, run_data["val_loss"], label= label+ " Val Loss")
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.legend()
        self.ax.grid(True)
    def __call__(self):
        return self.ax