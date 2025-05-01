#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sklearn.metrics
import argparse
import sys
import os.path

def calculate_scores(actual_file, prediction_file, label_column="Is_Fraud", id_column=None):
    """
    Compare predictions in a CSV file with actual values and calculate evaluation metrics.
    
    Parameters:
    -----------
    actual_file : str
        Path to the CSV file containing the actual values
    prediction_file : str
        Path to the CSV file containing the predictions
    label_column : str, default="Is_Fraud"
        Name of the column containing the target values
    id_column : str, default=None
        Name of the column to use as identifier (if None, assumes index alignment)
        
    Returns:
    --------
    dict
        Dictionary containing various evaluation metrics
    """
    # Load the data
    try:
        actual_data = pd.read_csv(actual_file, index_col=0)
        prediction_data = pd.read_csv(prediction_file, index_col=0)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None
    
    # Ensure the label column exists in both files
    if label_column not in actual_data.columns:
        print(f"Error: Label column '{label_column}' not found in actual data file")
        return None
    
    if label_column not in prediction_data.columns:
        print(f"Error: Label column '{label_column}' not found in prediction data file")
        return None
    
    # If ID column is specified, align the data using it
    if id_column is not None:
        if id_column not in actual_data.columns or id_column not in prediction_data.columns:
            print(f"Error: ID column '{id_column}' not found in both files")
            return None
        
        # Merge on ID column
        merged_data = pd.merge(
            actual_data[[id_column, label_column]], 
            prediction_data[[id_column, label_column]], 
            on=id_column, 
            suffixes=('_actual', '_pred')
        )
        
        y_true = merged_data[f"{label_column}_actual"]
        y_pred = merged_data[f"{label_column}_pred"]
    else:
        # Ensure the indices match or use proper alignment
        common_indices = actual_data.index.intersection(prediction_data.index)
        
        if len(common_indices) == 0:
            print("Error: No common indices found between the two files")
            return None
        
        y_true = actual_data.loc[common_indices, label_column]
        y_pred = prediction_data.loc[common_indices, label_column]
    
    # Calculate various metrics
    scores = {}
    
    # Determine if classification or regression based on unique values
    unique_values = len(np.unique(y_true))
    is_binary = unique_values == 2
    
    if is_binary:
        # Binary classification metrics
        scores['accuracy'] = sklearn.metrics.accuracy_score(y_true, y_pred.round())
        scores['precision'] = sklearn.metrics.precision_score(y_true, y_pred.round())
        scores['recall'] = sklearn.metrics.recall_score(y_true, y_pred.round())
        scores['f1'] = sklearn.metrics.f1_score(y_true, y_pred.round())
        scores['roc_auc'] = sklearn.metrics.roc_auc_score(y_true, y_pred)
        
        # Confusion matrix
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred.round())
        scores['confusion_matrix'] = cm.tolist()
    else:
        # Regression metrics
        scores['r2'] = sklearn.metrics.r2_score(y_true, y_pred)
        scores['mse'] = sklearn.metrics.mean_squared_error(y_true, y_pred)
        scores['rmse'] = np.sqrt(scores['mse'])
        scores['mae'] = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    
    # Display the results
    print("\n===== Evaluation Results =====")
    print(f"Total samples compared: {len(y_true)}")
    
    if is_binary:
        print("\nClassification Metrics:")
        print(f"Accuracy: {scores['accuracy']:.4f}")
        print(f"Precision: {scores['precision']:.4f}")
        print(f"Recall: {scores['recall']:.4f}")
        print(f"F1 Score: {scores['f1']:.4f}")
        print(f"ROC AUC: {scores['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"True Negatives: {cm[0][0]}")
        print(f"False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}")
        print(f"True Positives: {cm[1][1]}")
    else:
        print("\nRegression Metrics:")
        print(f"R² Score: {scores['r2']:.4f}")
        print(f"Mean Squared Error: {scores['mse']:.4f}")
        print(f"Root Mean Squared Error: {scores['rmse']:.4f}")
        print(f"Mean Absolute Error: {scores['mae']:.4f}")
    
    return scores


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare prediction and actual CSV files and calculate scores')
    parser.add_argument('--actual-file', '-a', type=str, help='Path to the CSV file with actual values')
    parser.add_argument('--prediction-file', '-p', type=str, help='Path to the CSV file with predictions')
    parser.add_argument('--label', '-l', default="Is_Fraud", type=str, help='Name of the target column (default: Is_Fraud)')
    parser.add_argument('--id-column', '-i', default=None, type=str, help='Name of the ID column for alignment')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.actual_file):
        print(f"Error: Actual file '{args.actual_file}' does not exist")
        return 1
        
    if not os.path.exists(args.prediction_file):
        print(f"Error: Prediction file '{args.prediction_file}' does not exist")
        return 1
    
    # Calculate and display scores
    scores = calculate_scores(args.actual_file, args.prediction_file, args.label, args.id_column)
    
    return 0 if scores is not None else 1

if __name__ == "__main__":
    sys.exit(main())
