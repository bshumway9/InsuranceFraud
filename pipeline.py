#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path

import pandas as pd
import numpy as np
import sklearn.feature_selection
import sklearn.linear_model
from sklearn.model_selection import GridSearchCV
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.base
import sklearn.metrics
import sklearn.impute
import joblib


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from showcase_common_fine import numerical_features, categorical_features, label_name


class PipelineNoop(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Just a placeholder with no actions on the data.
    """
    
    def __init__(self):
        return

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        return X

def get_test_filename(test_file, filename):
    if test_file == "":
        basename = get_basename(filename)
        test_file = "{}-test.csv".format(basename)
    return test_file

def get_validation_filename(validation_file, filename):
    if validation_file == "":
        basename = get_basename(filename)
        validation_file = "{}-val.csv".format(basename)
    return validation_file

def get_basename(filename):
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.info("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))

    stub = "-train"
    if basename[len(basename)-len(stub):] == stub:
        basename = basename[:len(basename)-len(stub)]

    return basename

def get_model_filename(model_file, filename):
    if model_file == "":
        basename = get_basename(filename)
        model_file = "{}-model.joblib".format(basename)
    return model_file

def get_data(filename, label_column):
    """
    Assumes column 0 is the instance index stored in the
    csv file.  If no such column exists, remove the
    index_col=0 parameter.
    """
    data = pd.read_csv(filename, index_col=0)
    if label_column in data.columns:
        data = data.dropna(subset=[label_column])
    return data

def load_data(my_args, filename):
    data = get_data(filename, my_args.label)
    feature_columns, label_column = get_feature_and_label_names(my_args, data)
    X = data[feature_columns]
    if label_column in data.columns:
        y = data[label_column]
    else:
        y = None
    return X, y

def get_feature_and_label_names(my_args, data):
    label_column = my_args.label
    feature_columns = my_args.features

    if label_column in data.columns:
        label = label_column
    else:
        label = ""

    features = []
    for feature_column in data.columns:
        if feature_column != label:
            features.append(feature_column)
    # if feature_columns is not None:
    #     for feature_column in feature_columns:
    #         if feature_column in data.columns:
    #             features.append(feature_column)

    # no features specified, so add all non-labels
    if len(features) == 0:
        for feature_column in data.columns:
            if feature_column != label:
                features.append(feature_column)

    return features, label

def make_pipeline():
    # Separate numeric and categorical columns
    numeric_features = numerical_features
    categorical_feature_list = categorical_features

    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_feature_list)
        ])

    # Create full pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ("feature_selection", sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.f_regression, k=8)),
        ('regressor', sklearn.linear_model.ARDRegression(max_iter=1000, tol=1e-3))
    ])

    return full_pipeline

def make_pipeline_1():
    # Separate numeric and categorical columns
    numeric_features = numerical_features
    categorical_feature_list = categorical_features

    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_feature_list)
        ])

    # Create full pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ("feature_selection", sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.f_regression, k=50)),
        ('regressor', sklearn.linear_model.SGDRegressor(max_iter=1000, tol=1e-3))
    ])

    return full_pipeline

def make_pipeline_2():
    # Separate numeric and categorical columns
    numeric_features = numerical_features
    categorical_feature_list = categorical_features

    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_feature_list)
        ])
    
    # items = []
    # items.append(("numerical", numeric_transformer))
    # items.append(("categorical", categorical_transformer))
    # preprocessor = sklearn.pipeline.FeatureUnion(transformer_list=items)

    # Create full pipeline
    # full_pipeline = Pipeline([
    #     ('preprocessor', preprocessor),
    #     ("feature_selection", sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.f_regression, k=160)),
    #     ('regressor', sklearn.linear_model.HuberRegressor(max_iter=1000, tol=1e-3))
    # ])

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ("feature_selection", sklearn.feature_selection.SelectKBest(
            score_func=sklearn.feature_selection.f_regression, 
            k=180  # Let cross-validation help determine optimal k
        )),
        ('regressor', sklearn.linear_model.HuberRegressor(
            max_iter=1000, 
            tol=1e-3,
            epsilon=1.35  # Default value, can be tuned
        ))
    ])

    return full_pipeline

def make_numerical_feature_pipeline(my_args):
    items = []

    if my_args.numerical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.numerical_missing_strategy)))
    if my_args.use_polynomial_features:
        items.append(("polynomial-features", sklearn.preprocessing.PolynomialFeatures(degree=my_args.use_polynomial_features)))
    if my_args.use_scaler:
        items.append(("scaler", sklearn.preprocessing.StandardScaler()))
    items.append(("noop", PipelineNoop()))
    
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline

def make_SGD_fit_pipeline(my_args):
    items = []
    items.append(("features", make_numerical_feature_pipeline(my_args)))
    items.append(("model", sklearn.linear_model.SGDRegressor()))
    return sklearn.pipeline.Pipeline(items)

def do_fit(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    # Create the pipeline
    pipeline = make_pipeline()
    
    # Define parameter grid for GridSearchCV
    # param_grid = {
    #     'feature_selection__k': [10, 20, 40, 80, 120, 140, 160, 170, 180, 190, 200, 'all']  # Different k values to try
    # }
    
    # # Create GridSearchCV object
    # grid_search = GridSearchCV(
    #     pipeline,
    #     param_grid,
    #     cv=5,  # 5-fold cross-validation
    #     scoring='r2',  # You can change this to 'neg_mean_squared_error' or other metrics
    #     n_jobs=-1,  # Use all available cores
    #     verbose=2
    # )
    
    # # Fit the grid search
    # grid_search.fit(X, y)
    
    # # Print results
    # print("Best parameters:", grid_search.best_params_)
    # print("Best cross-validation score:", grid_search.best_score_)
    
    # # Save the best model
    # model_file = get_model_filename(my_args.model_file, train_file)
    # joblib.dump(grid_search.best_estimator_, model_file)
    pipeline.fit(X, y)
    model_file = get_model_filename(my_args.model_file, train_file)
    joblib.dump(pipeline, model_file)
    
    return

def get_feature_names(pipeline, X):
    primary_feature_names = list(X.columns[:])
    if 'polynomial-features' in pipeline['features'].named_steps:
        secondary_powers = pipeline['features']['polynomial-features'].powers_
        feature_names = []
        for powers in secondary_powers:
            s = ""
            for i in range(len(powers)):
                for j in range(powers[i]):
                    if len(s) > 0:
                        s += "*"
                    s += primary_feature_names[i]
            feature_names.append(s)
            logging.info("powers: {}  s: {}".format(powers, s))
    else:
        logging.info("polynomial-features not in features: {}".format(pipeline['features'].named_steps))
        feature_names = primary_feature_names
    return feature_names

def get_scale_offset(pipeline, count):
    if 'scaler' in pipeline['features'].named_steps:
        scaler = pipeline['features']['scaler']
        logging.info("scaler: {}".format(scaler))
        logging.info("scale: {}  mean: {}  var: {}".format(scaler.scale_, scaler.mean_, scaler.var_))
        theta_scale = 1.0 / scaler.scale_
        intercept_offset = scaler.mean_ / scaler.scale_
    else:
        theta_scale = np.ones(count)
        intercept_offset = np.zeros(count)
        logging.info("scaler not in features: {}".format(pipeline['features'].named_steps))
    return theta_scale, intercept_offset


def show_score(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if my_args.show_test and not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    validation_file = get_validation_filename(my_args.validation_file, train_file)
    if my_args.validation_file and not os.path.exists(validation_file):
        raise Exception("validation data file, '{}', does not exist.".format(validation_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    X_train, y_train = load_data(my_args, train_file)
    pipeline = joblib.load(model_file)
    
    basename = get_basename(train_file)
    y_train_predicted = pipeline.predict(X_train)
    # Calculate R2 score (regression metric) instead of accuracy score (classification metric)
    score_train = sklearn.metrics.r2_score(y_train, y_train_predicted)
    
    # You could also calculate RMSE or other regression metrics
    rmse_train = np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_predicted))
    
    output = "{}: train_score: {}".format(basename, score_train)
    
    if my_args.validation_file:
        X_val, y_val = load_data(my_args, validation_file)
        y_val_predicted = pipeline.predict(X_val)
        score_val = sklearn.metrics.r2_score(y_val, y_val_predicted)
        output += " validation_score: {}".format(score_val)
    
    if my_args.show_test:
        X_test, y_test = load_data(my_args, test_file)
        y_test_predicted = pipeline.predict(X_test)
        score_test = sklearn.metrics.r2_score(y_test, y_test_predicted)
        output += " test_score: {}".format(score_test)
    
    print(output)
    return

def show_loss(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if my_args.show_test and not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    validation_file = get_validation_filename(my_args.validation_file, train_file)
    if my_args.validation_file and not os.path.exists(validation_file):
        raise Exception("validation data file, '{}', does not exist.".format(validation_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    X_train, y_train = load_data(my_args, train_file)
    pipeline = joblib.load(model_file)

    y_train_predicted = pipeline.predict(X_train)
    
    if my_args.validation_file:
        X_val, y_val = load_data(my_args, validation_file)
        y_val_predicted = pipeline.predict(X_val)
    
    if my_args.show_test:
        X_test, y_test = load_data(my_args, test_file)
        y_test_predicted = pipeline.predict(X_test)

    basename = get_basename(train_file)
    
    # MSE loss calculations
    loss_train = sklearn.metrics.mean_squared_error(y_train, y_train_predicted)
    output = "{}: L2(MSE) train_loss: {}".format(basename, loss_train)
    
    if my_args.validation_file:
        loss_val = sklearn.metrics.mean_squared_error(y_val, y_val_predicted)
        output += " validation_loss: {}".format(loss_val)
    
    if my_args.show_test:
        loss_test = sklearn.metrics.mean_squared_error(y_test, y_test_predicted)
        output += " test_loss: {}".format(loss_test)
    
    print(output)

    # MAE loss calculations
    loss_train = sklearn.metrics.mean_absolute_error(y_train, y_train_predicted)
    output = "{}: L1(MAE) train_loss: {}".format(basename, loss_train)
    
    if my_args.validation_file:
        loss_val = sklearn.metrics.mean_absolute_error(y_val, y_val_predicted)
        output += " validation_loss: {}".format(loss_val)
    
    if my_args.show_test:
        loss_test = sklearn.metrics.mean_absolute_error(y_test, y_test_predicted)
        output += " test_loss: {}".format(loss_test)
    
    print(output)

    # R2 score calculations
    loss_train = sklearn.metrics.r2_score(y_train, y_train_predicted)
    output = "{}: R2 train_loss: {}".format(basename, loss_train)
    
    if my_args.validation_file:
        loss_val = sklearn.metrics.r2_score(y_val, y_val_predicted)
        output += " validation_loss: {}".format(loss_val)
    
    if my_args.show_test:
        loss_test = sklearn.metrics.r2_score(y_test, y_test_predicted)
        output += " test_loss: {}".format(loss_test)
    
    print(output)
    return

def show_model(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    pipeline = joblib.load(model_file)
    regressor = pipeline.named_steps['regressor']
    # regressor = pipeline['model']
    # features = pipeline['features']

    print("Model Information:")
    print("coef_: {}".format(regressor.coef_))
    print("intercept_: {}".format(regressor.intercept_))
    print("n_iter_: {}".format(regressor.n_iter_))
    print("n_features_in_: {}".format(regressor.n_features_in_))


    print("\nFeature Selection Information:")
    feature_selector = pipeline.named_steps['feature_selection']
    selected_features_mask = feature_selector.get_support()
    n_selected = selected_features_mask.sum()
    print(f"Number of selected features: {n_selected}")


    try:
        scaler = pipeline.named_steps['scaler']
        # scaler = features["scaler"]
        print("scaler.mean_: {}".format(scaler.mean_))
        print("scaler.var_: {}".format(scaler.var_))
    except:
        print("No scaler.")
    return



def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Fit Data With Linear Regression Using Pipeline')
    parser.add_argument('action', default='SGD',
                        choices=[ "SGD", "score", "loss", "show-model" ], 
                        nargs='?', help="desired action")
    parser.add_argument('--train-file',    '-t', default="",    type=str,   help="name of file with training data")
    parser.add_argument('--test-file',     '-T', default="",    type=str,   help="name of file with test data (default is constructed from train file name)")
    parser.add_argument('--validation-file', '-v', default="",  type=str,   help="name of file with validation data (default is constructed from train file name)")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from train file name when fitting)")
    parser.add_argument('--random-seed',   '-R', default=314159265,type=int,help="random number seed (-1 to use OS entropy)")
    parser.add_argument('--features',      '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label',         '-l', default="label",   type=str,   help="column name for label")
    parser.add_argument('--use-polynomial-features', '-p', default=0,         type=int,   help="degree of polynomial features.  0 = don't use (default=0)")
    parser.add_argument('--use-scaler',    '-s', default=0,         type=int,   help="0 = don't use scaler, 1 = do use scaler (default=0)")
    parser.add_argument('--numerical-missing-strategy', default="",   type=str,   help="strategy for missing numerical information")
    parser.add_argument('--show-test',     '-S', default=0,         type=int,   help="0 = don't show test loss, 1 = do show test loss (default=0)")

    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #
    allowed_numerical_missing_strategies = ("mean", "median", "most_frequent")
    if my_args.numerical_missing_strategy != "":
        if my_args.numerical_missing_strategy not in allowed_numerical_missing_strategies:
            raise Exception("Missing numerical strategy {} is not in the allowed list {}.".format(my_args.numerical_missing_strategy, allowed_numerical_missing_strategies))

    
    return my_args

def main(argv):
    my_args = parse_args(argv)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'SGD':
        do_fit(my_args)
    elif my_args.action == "score":
        show_score(my_args)
    elif my_args.action == "loss":
        show_loss(my_args)
    elif my_args.action == "show-model":
        show_model(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))
        
    return

if __name__ == "__main__":
    main(sys.argv)

