#!/usr/bin/env
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import numpy as np
import torch
import os


def plot_data_with_distribution_seaborn(log1, log2=None, 
                                        save_path=None, 
                                        label1=None, 
                                        label2=None, 
                                        title=None):
    """
    Plots one or two datasets with the average values and distributions over iterations using Seaborn.

    Parameters:
        log1 (list of lists): The first list of scores (each element is a list of scores for an iteration).
        log2 (list of lists, optional): The second list of scores (each element is a list of scores for an iteration). Defaults to None.
        save_path (str): Path to save the plot. Defaults to None.
        label1 (str): Label for the first dataset. Defaults to "Fraction of Valid Peptide SMILES".
        label2 (str, optional): Label for the second dataset. Defaults to None.
        title (str): Title of the plot. Defaults to "Fraction of Valid Peptides Over Iterations".
    """
    # Prepare data for log1
    data1 = pd.DataFrame({
        "Iteration": np.repeat(range(1, len(log1) + 1), [len(scores) for scores in log1]),
        label1: [score for scores in log1 for score in scores],
        "Dataset": label1,
        "Style": "Log1"
    })

    # Prepare data for log2 if provided
    if log2 is not None:
        data2 = pd.DataFrame({
            "Iteration": np.repeat(range(1, len(log2) + 1), [len(scores) for scores in log2]),
            label2: [score for scores in log2 for score in scores],
            "Dataset": label2,
            "Style": "Log2"
        })
        data = pd.concat([data1, data2], ignore_index=True)
    else:
        data = data1
    
    palette = {
        label1: "#8181ED",  # Default color for log1
        label2: "#D577FF"   # Default color for log2 (if provided)
    }

    # Set Seaborn theme
    sns.set_theme()
    sns.set_context("paper")

    # Create the plot
    sns.relplot(
        data=data, 
        kind="line",
        x="Iteration", 
        y=label1, 
        hue="Dataset", 
        style="Style", 
        markers=True, 
        dashes=True,
        ci="sd",  # Show standard deviation
        height=5, 
        aspect=1.5,
        palette=palette
    )

    # Titles and labels
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(label1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()
    
def plot_data(log1, log2=None, 
                    save_path=None, 
                    label1="Log 1", 
                    label2=None, 
                    title="Fraction of Valid Peptides Over Iterations", 
                    palette=None):
    """
    Plots one or two datasets with their mean values over iterations.

    Parameters:
        log1 (list): The first list of mean values for each iteration.
        log2 (list, optional): The second list of mean values for each iteration. Defaults to None.
        save_path (str): Path to save the plot. Defaults to None.
        label1 (str): Label for the first dataset. Defaults to "Log 1".
        label2 (str, optional): Label for the second dataset. Defaults to None.
        title (str): Title of the plot. Defaults to "Mean Values Over Iterations".
        palette (dict, optional): A dictionary defining custom colors for datasets. Defaults to None.
    """
    # Prepare data for log1
    data1 = pd.DataFrame({
        "Iteration": range(1, len(log1) + 1),
        "Fraction of Valid Peptides": log1,
        "Dataset": label1
    })

    # Prepare data for log2 if provided
    if log2 is not None:
        data2 = pd.DataFrame({
            "Iteration": range(1, len(log2) + 1),
            "Fraction of Valid Peptides": log2,
            "Dataset": label2
        })
        data = pd.concat([data1, data2], ignore_index=True)
    else:
        data = data1

    palette = {
        label1: "#8181ED",  # Default color for log1
        label2: "#D577FF"   # Default color for log2 (if provided)
    }

    # Set Seaborn theme
    sns.set_theme()
    sns.set_context("paper")

    # Create the plot
    sns.lineplot(
        data=data, 
        x="Iteration", 
        y="Fraction of Valid Peptides", 
        hue="Dataset", 
        style="Dataset", 
        markers=True, 
        dashes=False, 
        palette=palette
    )

    # Titles and labels
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Fraction of Valid Peptides")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()