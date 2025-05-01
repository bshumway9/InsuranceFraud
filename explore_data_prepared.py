#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import math

feature_names = ["CP", "Weight", "Height"]
label_name = "Score"
column_count = len(feature_names) + 1
filename = "student_data.csv"
data = pd.read_csv(filename)

def pdf_figure(figure_number):
    """Configure figure for portrait orientation paper"""
    # letter paper dimensions
    width = 6.5
    height = 9
    fig = plt.figure(figure_number, figsize=(width, height))
    return fig

def png_figure(figure_number):
    """Configure figure for landscape orientation screen"""
    # 16:9 ratio, on paper dimensions
    width = 9
    height = 5
    fig = plt.figure(figure_number, figsize=(width, height))
    return fig

def histogram_column(fig, series, plot_count, plot_number):
    """
    Add a axes as a subplot, 
    set to log scale on the y-axis,
    histogram the values in the series, with 20 bins,
    create 5 tick marks on the x-axis,
    """
    ax = fig.add_subplot(plot_count, plot_count, plot_number)
    ax.set_yscale("log")
    n, bins, patches = ax.hist(series, bins=20)
    ax.set_xlabel(series.name)
    ax.locator_params(axis='x', tight=True, nbins=5)
    return ax, n

def histogram_all(data, feature_names, label_name):
    """
    For each feature and the label, add a histogram as a subplot.
    Scale each y-axis to the same range for better comparison.
    """
    figure_number = 1
    fig = png_figure(figure_number)
    fig.suptitle("Feature Histograms")

    plot_count = int(math.ceil(math.sqrt(column_count)))
    plot_number = 1
    n_max = 1
    all_ax = []
    for column_name in feature_names + [label_name]:
        ax, n = histogram_column(fig, data[column_name], plot_count, plot_number)
        if max(n) > n_max:
            n_max = max(n)
        all_ax.append(ax)
        plot_number += 1

    for ax in all_ax:
        ax.set_ylim(bottom=1.0, top=n_max)

    fig.tight_layout()
    figure_name = "showcase_prepared_histograms.png"
    fig.savefig(figure_name)
    plt.close(fig)
    return


def scatter_column(fig, feature_series, label_series, plot_count, plot_number):
    """
    Use the feature values as the x-axis, and the label as the y-axis.
    Scatter plot the data in the new axes created here.
    """
    ax = fig.add_subplot(plot_count, plot_count, plot_number)
    ax.scatter(feature_series, label_series, s=1)
    ax.set_xlabel(feature_series.name)
    ax.set_ylabel(label_series.name)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return ax

def scatter_all(data, feature_names, label_name):
    """
    For each feature, scatter plot it vs the label.
    """
    figure_number = 2
    fig = png_figure(figure_number)
    fig.suptitle("Features vs. Label")

    plot_count = int(math.ceil(math.sqrt(column_count)))
    plot_number = 1
    all_ax = []
    for column_name in feature_names + [label_name]:
        ax = scatter_column(fig, data[column_name], data[label_name], plot_count, plot_number)
        all_ax.append(ax)
        plot_number += 1

    fig.tight_layout()
    figure_name = "showcase_prepared_scatters.png"
    fig.savefig(figure_name)
    plt.close(fig)
    return

def main():
    histogram_all(data, feature_names, label_name)
    scatter_all(data, feature_names, label_name)
    return

if __name__ == "__main__":
    main()