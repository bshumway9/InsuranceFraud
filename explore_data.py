#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import math

# feature_names = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", 
#                 "Stress Level", "Sleep Duration", "Daily Steps"]
# label_name = "Heart Rate"
# filename = "Sleep_health_and_lifestyle_dataset.csv"

# feature_names = ["Age", "Weight (kg)", "Height (m)", "Max_BPM", "Avg_BPM", 
#                 "Session_Duration (hours)", "Calories_Burned", "Fat_Percentage", 
#                 "Water_Intake (liters)", "Workout_Frequency (days/week)", "BMI", "Calories_Burned"]
# label_name = "Resting_BPM"
# filename = "gym_members_exercise_tracking.csv"

# feature_names = ["age", "education", "cigsPerDay", "totChol", "sysBP", 
#                 "diaBP", "BMI", "glucose"]
# label_name = "heartRate"  # Target variable
# filename = "Heart_Disease (1).csv"
feature_names = [
            # "Gender", 
            # "Transaction_Location", 
            # "Account_Type", 
            # "Transaction_Type", 
            # "Merchant_Category", 
            # "Device_Type",
            # "Transaction_Currency",
            # "Transaction_Device",
            "Age",
            "Account_Balance", 
            # "Transaction_Date",
            # "Transaction_Time",
            "Transaction_Amount",
        ]
feature_names = ["Age", "Deductible", "DriverRating", "Days_Policy_Accident", 
    "Days_Policy_Claim", "PastNumberOfClaims", "AgeOfVehicle", 
    "AgeOfPolicyHolder", "NumberOfSuppliments", "NumberOfCars", "Year"
]
c_feature_names = [            "Gender", 
            "Transaction_Location", 
            "Account_Type", 
            "Transaction_Type", 
            "Merchant_Category", 
            "Device_Type",
            "Transaction_Currency",
            "Transaction_Device",]
c_feature_names = [            "type"]
c_feature_names = [
    "Month", "WeekOfMonth", "DayOfWeek", "Make", "AccidentArea", 
    "DayOfWeekClaimed", "MonthClaimed", "WeekOfMonthClaimed", "Sex", 
    "MaritalStatus", "Fault", "PolicyType", "VehicleCategory", 
    "VehiclePrice", "PoliceReportFiled", "WitnessPresent", "AgentType", 
    "AddressChange_Claim", "BasePolicy"
]
label_name = "FraudFound_P"  # Choose your target variable
filename = "FO-train.csv"
column_count = len(feature_names) + 1

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
    figure_name = "showcase_histograms.png"
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
    figure_name = "showcase_scatters.png"
    fig.savefig(figure_name)
    plt.close(fig)
    return

def plot_categorical_features(data, categorical_features, label_name):
    """
    For each categorical feature, create a bar chart grouped by the label.
    """
    for feature in categorical_features:
        grouped_data = data.groupby([feature, label_name]).size().unstack()
        grouped_data.plot(kind='bar', stacked=False, figsize=(8, 6))
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.title(f"Count Plot of {feature} by {label_name}")
        plt.legend(title=label_name)
        plt.tight_layout()
        figure_name = f"categorical_{feature}_vs_{label_name}.png"
        plt.savefig(figure_name)
        plt.close()

def main():
    histogram_all(data, feature_names, label_name)
    scatter_all(data, feature_names, label_name)
    plot_categorical_features(data, c_feature_names, label_name)
    return

if __name__ == "__main__":
    main()