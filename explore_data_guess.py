#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import math

feature_names = ["CP", "Weight", "Height"]
label_name = "Score"
column_count = len(feature_names) + 1
filename = "student_data.csv"
data = pd.read_csv(filename)
guesses = {
    "CP": [550, 0],
    "Weight": [300, 18],
    "Height": [-60, 1200]
}

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

def scatter_all(data, feature_names, label_name, guesses):
    """
    For each feature, scatter plot it vs the label.
    If there is a guess, plot it too.
    """
    figure_number = 2
    fig = png_figure(figure_number)
    fig.suptitle("Features vs. Label")

    plot_count = int(math.ceil(math.sqrt(column_count)))
    plot_number = 1
    all_ax = []
    for column_name in feature_names + [label_name]:
        ax = scatter_column(fig, data[column_name], data[label_name], plot_count, plot_number)
        if column_name in guesses:
            xmax = data[column_name].max()
            xmin = data[column_name].min()
            ymin = guesses[column_name][0] + guesses[column_name][1]*xmin
            ymax = guesses[column_name][0] + guesses[column_name][1]*xmax
            ax.plot([xmin, xmax], [ymin, ymax], color='magenta')
            s = "{:.1f} + {:.1f}*{}".format(guesses[column_name][0], guesses[column_name][1], column_name)
            ax.text(0.1, 0.8, s, 
                    horizontalalignment='left', 
                    verticalalignment='center', 
                    transform=ax.transAxes)
        all_ax.append(ax)
        plot_number += 1

    fig.tight_layout()
    figure_name = "showcase_prepared_scatters_guess.png"
    fig.savefig(figure_name)
    plt.close(fig)
    return

def main():
    scatter_all(data, feature_names, label_name, guesses)
    return

if __name__ == "__main__":
    main()