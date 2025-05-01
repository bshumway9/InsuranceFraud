#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path

import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.base
import sklearn.metrics
import sklearn.impute
import sklearn.svm
import sklearn.ensemble
import joblib
import pprint
import matplotlib.pyplot as plt
import imblearn as imbl

from pipeline_elements import *
from data_overhead import *
from make_pipeline import *

def get_one_hot_encoder(my_args):
    """
    Get the one-hot-encoder for the categorical features and prepare for SMOTE.
    """
    if my_args.categorical_missing_strategy == "most_frequent":
        missing_strategy = sklearn.impute.SimpleImputer(strategy="most_frequent")
    else:
        missing_strategy = sklearn.impute.SimpleImputer(strategy="constant", fill_value="NULL")

    items = []
    items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))
    items.append(("missing-data", missing_strategy))
    items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))
    categorical_pipeline = sklearn.pipeline.Pipeline(items)

    # Ensure the output is compatible with SMOTE by converting to dense format
    def to_dense(X):
        return X.toarray() if hasattr(X, "toarray") else X

    items.append(("to-dense", sklearn.preprocessing.FunctionTransformer(to_dense, validate=False)))
    
    return categorical_pipeline

def do_fit(my_args):
    """
    fit pipeline to training data
    """
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    X, y = load_data(my_args, train_file)
    print("Columns in the dataset:", X.columns.tolist())
    categorical_features = ["Month", "Make", "AccidentArea", "MonthClaimed", "Sex", 
                            "MaritalStatus", "Fault", "PolicyType", "VehiclePrice", 
                            "PoliceReportFiled", "WitnessPresent", "AgentType", 
                            "AddressChange_Claim", "Days_Policy_Accident", 
                            "Days_Policy_Claim", "PastNumberOfClaims", "AgeOfVehicle", 
                            "AgeOfPolicyHolder", "NumberOfSuppliments", "NumberOfCars"]
    categorical_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
    oversample = imbl.over_sampling.SMOTENC(sampling_strategy='minority', 
                                             categorical_features=categorical_features, 
                                             random_state=my_args.random_seed,
                                             k_neighbors=2)
    undersample = imbl.under_sampling.RandomUnderSampler()

    X, y = oversample.fit_resample(X, y)
    # X, y = undersample.fit_resample(X, y)
    
    pipeline = make_fit_pipeline(my_args)
    print("Pipeline:")
    pipeline.fit(X, y)
    print("Pipeline fit done.")
    model_file = get_model_filename(my_args.model_file, train_file)
    print("saving model")
    joblib.dump(pipeline, model_file)

    return

def do_cross(my_args):
    """
    do cross validation with training data
    """
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_fit_pipeline(my_args)

    cv_results = sklearn.model_selection.cross_validate(pipeline, X, y, cv=my_args.cv_count, n_jobs=-1, verbose=3, scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'),)
    # print(cv_results.keys())
    print("R2:", cv_results['test_r2'], cv_results['test_r2'].mean())
    print("MSE:", cv_results['test_neg_mean_squared_error'], cv_results['test_neg_mean_squared_error'].mean())
    print("MAE:", cv_results['test_neg_mean_absolute_error'], cv_results['test_neg_mean_absolute_error'].mean())

    pipeline.fit(X, y)
    model_file = get_model_filename(my_args.model_file, train_file)
    joblib.dump(pipeline, model_file)
    return

def show_score(my_args):
    """
    shows the already trained  model's score on training data.
    also on the test data, if --show-test 1
    """

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    X_train, y_train = load_data(my_args, train_file)
    X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)
    regressor = pipeline['model']
    
    basename = get_basename(train_file)
    score_train = regressor.score(pipeline['features'].transform(X_train), y_train)
    if my_args.show_test:
        score_test = regressor.score(pipeline['features'].transform(X_test), y_test)
        print("{}: train_score: {} test_score: {}".format(basename, score_train, score_test))
    else:
        print("{}: train_score: {}".format(basename, score_train))
    return

def show_loss(my_args):
    """
    shows the already trained model's loss on training data.
    # commented out for Kaggle data
    # also on the test data if --show-test 1
    """

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    # test_file = get_test_filename(my_args.test_file, train_file)
    # if not os.path.exists(test_file):
    #     raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    X_train, y_train = load_data(my_args, train_file)
    # commented out for Kaggle data
    # X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)

    y_train_predicted = pipeline.predict(X_train)
    # commented out for Kaggle data
    # y_test_predicted = pipeline.predict(X_test)

    basename = get_basename(train_file)
    
    loss_train = sklearn.metrics.mean_squared_error(y_train, y_train_predicted)
    # commented out for Kaggle data
    # if my_args.show_test:
    #     loss_test = sklearn.metrics.mean_squared_error(y_test, y_test_predicted)
    #     print("{}: L2(MSE) train_loss: {} test_loss: {}".format(basename, loss_train, loss_test))
    # else:
    print("{}: L2(MSE) train_loss: {}".format(basename, loss_train))

    loss_train = sklearn.metrics.mean_absolute_error(y_train, y_train_predicted)
    # commented out for Kaggle data
    # if my_args.show_test:
    #     loss_test = sklearn.metrics.mean_absolute_error(y_test, y_test_predicted)
    #     print("{}: L1(MAE) train_loss: {} test_loss: {}".format(basename, loss_train, loss_test))
    # else:
    print("{}: L1(MAE) train_loss: {}".format(basename, loss_train))

    loss_train = sklearn.metrics.r2_score(y_train, y_train_predicted)
    # commented out for Kaggle data
    # if my_args.show_test:
    #     loss_test = sklearn.metrics.r2_score(y_test, y_test_predicted)
    #     print("{}: R2 train_loss: {} test_loss: {}".format(basename, loss_train, loss_test))
    # else:
    print("{}: R2 train_loss: {}".format(basename, loss_train))
    return

def do_predict(my_args):
    """
    Do predictions on the test data using the already trained model.
    Writes the result to file. Designed for use with Kaggle competitions.
    """
    test_file = my_args.test_file
    if not os.path.exists(test_file):
        raise Exception("testing data file: {} does not exist.".format(test_file))

    model_file = get_model_filename(my_args.model_file, test_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)
    
    # Try to predict probabilities if the model supports it
    try:
        # Get probability predictions
        y_test_predicted_proba = pipeline.predict_proba(X_test)
        print(f"Probability prediction shape: {y_test_predicted_proba.shape}")
        print(f"First 5 probability predictions: {y_test_predicted_proba[:5]}")
        
        # Apply threshold for binary classification (default 0.5)
        y_test_binary = (y_test_predicted_proba[:, 1] >= 0.5).astype(int)
        
        # Check distribution of predictions
        unique_values, counts = np.unique(y_test_binary, return_counts=True)
        print(f"Distribution of binary predictions: {dict(zip(unique_values, counts))}")
        
        # Save both probability and binary predictions
        merged_proba = X_test.index.to_frame()
        merged_proba[my_args.label] = y_test_predicted_proba[:, 1]
        merged_proba.to_csv("predictions_proba.csv", index=False)
        print("Saved probability predictions to predictions_proba.csv")
        
        merged_binary = X_test.index.to_frame()
        merged_binary[my_args.label] = y_test_binary
        merged_binary.to_csv("predictions.csv", index=False)
        print("Saved binary predictions to predictions.csv")
        
    except (AttributeError, IndexError) as e:
        # Fallback to standard prediction if predict_proba is not available
        print(f"Warning: Could not use probability prediction: {e}")
        print("Using standard prediction method instead")
        
        # Standard prediction
        y_test_predicted = pipeline.predict(X_test)
        
        # Check if predictions are already binary (0/1)
        unique_values = np.unique(y_test_predicted)
        print(f"Unique prediction values: {unique_values}")
        
        # If predictions are not binary, convert to binary
        if len(unique_values) > 2 or (len(unique_values) <= 2 and not np.all(np.isin(unique_values, [0, 1]))):
            print("Converting predictions to binary values")
            y_test_binary = (y_test_predicted >= 0.5).astype(int)
        else:
            y_test_binary = y_test_predicted
        
        # Check distribution of predictions
        unique_values, counts = np.unique(y_test_binary, return_counts=True)
        print(f"Distribution of binary predictions: {dict(zip(unique_values, counts))}")
        
        merged = X_test.index.to_frame()
        merged[my_args.label] = y_test_binary
        merged.to_csv("predictions.csv", index=False)
        print("Saved binary predictions to predictions.csv")

    return

def do_proba(my_args):
    """
    Do predictions on the test data using the already trained model.
    Writes the result to file. Designed for use with Kaggle competitions.
    """
    test_file = my_args.test_file
    if not os.path.exists(test_file):
        raise Exception("testing data file: {} does not exist.".format(test_file))

    model_file = get_model_filename(my_args.model_file, test_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)

    y_test_predicted = pipeline.predict_proba(X_test)

    merged = X_test.index.to_frame()
    merged[my_args.label] = y_test_predicted[:,1]
    merged.to_csv("predictions_proba.csv", index=False)

    return

def do_grid_search(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_fit_pipeline(my_args)
    fit_params = make_fit_params(my_args)

    search_grid = sklearn.model_selection.GridSearchCV(pipeline, fit_params,
                                                       scoring="f1_micro",
                                                       cv=my_args.cv_count,
                                                       n_jobs=-1, verbose=1)
    search_grid.fit(X, y)
    print("Best parameters used:")
    print(search_grid.best_params_)
    search_grid_file = get_search_grid_filename(my_args.search_grid_file, train_file)
    joblib.dump(search_grid, search_grid_file)

    model_file = get_model_filename(my_args.model_file, train_file)
    joblib.dump(search_grid.best_estimator_, model_file)

    return

def do_random_search(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_fit_pipeline(my_args)
    fit_params = make_fit_params(my_args)

    search_grid = sklearn.model_selection.RandomizedSearchCV(pipeline, fit_params,
                                                             scoring="f1_micro",
                                                             cv=my_args.cv_count,
                                                             n_jobs=-1, verbose=1,
                                                             n_iter=my_args.n_search_iterations)
    search_grid.fit(X, y)
    
    search_grid_file = get_search_grid_filename(my_args.search_grid_file, train_file)
    joblib.dump(search_grid, search_grid_file)

    model_file = get_model_filename(my_args.model_file, train_file)
    joblib.dump(search_grid.best_estimator_, model_file)

    return

def show_best_params(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    search_grid_file = get_search_grid_filename(my_args.search_grid_file, train_file)
    if not os.path.exists(search_grid_file):
        raise Exception("Search grid file, '{}', does not exist.".format(search_grid_file))


    search_grid = joblib.load(search_grid_file)

    pp = pprint.PrettyPrinter(indent=4)
    print("Best Score:", search_grid.best_score_)
    print("Best Params:")
    pp.pprint(search_grid.best_params_)

    return


def do_cross_score(my_args):
    """
    do cross validation scoring with training data
    """
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_fit_pipeline(my_args)

    # scoring="accuracy" is a classification metric.
    # scoring="r2" is a regression metric.
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    score = sklearn.model_selection.cross_val_score(pipeline, X, y, cv=my_args.cv_count, n_jobs=-1, scoring="accuracy")
    print("Cross Validation Score: {:.3f} : {}".format(score.mean(), ["{:.3f}".format(x) for x in score]))

    return

from cm_display import print_cm
def do_confusion_matrix(my_args):
    """
    do cross validation and show confusion matrix
    """
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_fit_pipeline(my_args)

    y_pred = sklearn.model_selection.cross_val_predict(pipeline, X, y, cv=my_args.cv_count, n_jobs=-1)
    cm = sklearn.metrics.confusion_matrix(y, y_pred)
    labels = ["F", "T"]
    print()
    print()
    print_cm(cm, labels)
    print()
    print()

    pscore = sklearn.metrics.precision_score(y, y_pred)
    rscore = sklearn.metrics.recall_score(y, y_pred)
    f1score = sklearn.metrics.f1_score(y, y_pred)

    print("Precision: {:.3f}".format(pscore))
    print("Recall:    {:.3f}".format(rscore))
    print("F1:        {:.3f}".format(f1score))

    return


def do_precision_recall_plot(my_args):
    """
    plot the precision-recall curve
    """
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_fit_pipeline(my_args)

    y_pred = sklearn.model_selection.cross_val_predict(pipeline, X, y, cv=my_args.cv_count, n_jobs=-1, method=my_args.cross_val_predict_method)
    if my_args.cross_val_predict_method == "predict_proba":
        y_pred = y_pred[:, 1]
    precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(y, y_pred)

    # compute maximum f1 score, and its threshold
    numerator = 2 * recalls * precisions
    denom = recalls + precisions
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    #
    threshold = max_f1_thresh

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.vlines(threshold, 0, 1.0, "k", "dotted", label="max f1 {:.3f}".format(max_f1))
    plt.title(my_args.model_type + " Precision+Recall")
    plt.xlabel("Threshold")
    plt.grid(True)
    plt.legend()
    plt.savefig(my_args.image_file)
    plt.clf()


    return

def do_precision_recall_curve(my_args):
    """
    plot the precision-recall curve
    """
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_fit_pipeline(my_args)

    y_pred = sklearn.model_selection.cross_val_predict(pipeline, X, y, cv=my_args.cv_count, n_jobs=-1, method=my_args.cross_val_predict_method)
    if my_args.cross_val_predict_method == "predict_proba":
        y_pred = y_pred[:, 1]
    precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(y, y_pred)


    # compute maximum f1 score, the precision and recall at that point
    numerator = 2 * recalls * precisions
    denom = recalls + precisions
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    max_f1_precision = precisions[np.argmax(f1_scores)]
    max_f1_recall = recalls[np.argmax(f1_scores)]
    #


    plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
    plt.title(my_args.model_type + " Precision/Recall")
    plt.vlines(max_f1_recall, 0, max_f1_precision, "k", "dotted", label="max f1 {:.3f}".format(max_f1))
    plt.hlines(max_f1_precision, 0, max_f1_recall, "k", "dotted")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.grid(True)
    plt.legend()
    plt.savefig(my_args.image_file)
    plt.clf()


    return


def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Fit Data Using Pipeline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('action', default='fit',
                        choices=[ "fit", "score", "loss", "cross", "predict", "grid-search", "show-best-params", "random-search",
                                  "cross-score", "confusion-matrix", "precision-recall-plot", "pr-curve" ], 
                        nargs='?', help="desired action")
    parser.add_argument('--model-type',    '-M', default="SGD", type=str,   choices=["SGD", "linear", "SVM", "boost", "forest", "tree", "vote"], help="Model type")
    parser.add_argument('--train-file',    '-t', default="",    type=str,   help="name of file with training data")
    parser.add_argument('--test-file',     '-T', default="",    type=str,   help="name of file with test data (default is constructed from train file name)")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from train file name when fitting)")
    parser.add_argument('--search-grid-file', '-g', default="", type=str,   help="name of file for the search grid (default is constructed from train file name when fitting)")
    parser.add_argument('--random-seed',   '-R', default=314159265,type=int,help="random number seed (-1 to use OS entropy)")
    parser.add_argument('--features',      '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label',         '-l', default="label",   type=str,   help="column name for label")
    parser.add_argument('--use-polynomial-features', '-p', default=0,         type=int,   help="degree of polynomial features.  0 = don't use (default=0)")
    parser.add_argument('--use-scaler',    '-s', default=0,         type=int,   help="0 = don't use scaler, 1 = do use scaler (default=0)")
    parser.add_argument('--categorical-missing-strategy', default="",   type=str, choices=("", "most_frequent"), help="strategy for missing categorical information")
    parser.add_argument('--numerical-missing-strategy', default="",   type=str,  choices=("", "mean", "median", "most_frequent"), help="strategy for missing numerical information")
    parser.add_argument('--show-test',     '-S', default=0,         type=int,   help="0 = don't show test loss, 1 = do show test loss (default=0)")
    parser.add_argument('--n-search-iterations', default=10,        type=int,   help="number of random iterations in randomized grid search.")
    parser.add_argument('--cv-count',            default=3,         type=int,   help="number of partitions for cross validation.")
    parser.add_argument('--image-file',          default="image.png", type=str,   help="name of file to store output images")
    parser.add_argument('--cross-val-predict-method', default="", type=str,   help="method argument for cross_val_predict, will be determined by model-type")

    my_args = parser.parse_args(argv[1:])

    
    if my_args.model_type in ("SGD", "linear"):
        my_args.cross_val_predict_method = "decision_function"
    elif my_args.model_type in ("SVM", "boost", "forest", "tree", "vote"):
        my_args.cross_val_predict_method = "predict_proba"
    else:
        raise Exception("???")

    return my_args

def main(argv):
    my_args = parse_args(argv)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'fit':
        do_fit(my_args)
    elif my_args.action == "score":
        show_score(my_args)
    elif my_args.action == "loss":
        show_loss(my_args)
    elif my_args.action == "cross":
        do_cross(my_args)
    elif my_args.action == "predict":
        do_predict(my_args)
        # do_proba(my_args)
    elif my_args.action == 'grid-search':
        do_grid_search(my_args)
    elif my_args.action == 'random-search':
        do_random_search(my_args)
    elif my_args.action == "show-best-params":
        show_best_params(my_args)
    elif my_args.action == "cross-score":
        do_cross_score(my_args)
    elif my_args.action == "confusion-matrix":
        do_confusion_matrix(my_args)
    elif my_args.action == "precision-recall-plot":
        do_precision_recall_plot(my_args)
    elif my_args.action == "pr-curve":
        do_precision_recall_curve(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))
        
    return

if __name__ == "__main__":
    main(sys.argv)

