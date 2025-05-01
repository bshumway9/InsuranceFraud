#!/usr/bin/env python3






########USE PROB A NOT STRAIGHT ACROSS
################################################################
#
# These custom functions help with constructing common pipelines.
# They make use of my_args, and object that has been configured
# by the argparse module to match user requests.
#
import sklearn.ensemble
from pipeline_elements import *
import sklearn.impute
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.tree

def make_numerical_predictor_params(my_args):
    params = { 
        "features__numerical__numerical-features-only__do_predictors" : [ True ],
        "features__numerical__numerical-features-only__do_numerical" : [ True ],
    }
    if my_args.numerical_missing_strategy:
        params["features__numerical__missing-data__strategy"] = [ 'median', 'mean', 'most_frequent' ]
    if my_args.use_polynomial_features:
        params["features__numerical__polynomial-features__degree"] = [ 2 ] # [ 1, 2, 3 ]

    return params

def make_categorical_predictor_params(my_args):
    params = { 
        "features__categorical__categorical-features-only__do_predictors" : [ True ],
        "features__categorical__categorical-features-only__do_numerical" : [ False ],
        "features__categorical__encode-category-bits__categories": [ 'auto' ],
        "features__categorical__encode-category-bits__handle_unknown": [ 'ignore' ],
    }
    if my_args.categorical_missing_strategy:
        params["features__categorical__missing-data__strategy"] = [ 'most_frequent' ]
    return params

def make_predictor_params(my_args):
    p1 = make_numerical_predictor_params(my_args)
    p2 = make_categorical_predictor_params(my_args)
    p1.update(p2)
    return p1

def make_tree_params(my_args):
    tree_params = {
        "model__criterion": [ "entropy" ], # [ "entropy", "gini" ],
        "model__splitter": [ "best" ], # [ "best", "random" ],
        "model__max_depth": [ 1, 2, 3, 4, None ],
        "model__min_samples_split": [ 2 ], # [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64 ],
        "model__min_samples_leaf":  [ 1 ],  # [ 0.01, 0.02, 0.04, 0.1 ],
        "model__max_features":  [ None ], # [ "sqrt", "log2", None ],
        "model__max_leaf_nodes": [ None ], # [ 2, 4, 8, 16, 32, 64, None ],
        "model__min_impurity_decrease": [ 0.0 ], # [ 0.0, 0.01, 0.02, 0.04, 0.1, 0.2 ],
    }
    return tree_params

def make_forest_params(my_args):
    forest_params = {
        "model__n_estimators": [ 25], # [ 10, 50, 100, 200, 400, 800 ], #[100]
        "model__criterion": [ "entropy" ], # [ "entropy", "gini" ],
        "model__max_depth": [ None ], #[3]
        "model__min_samples_split": [ 5 ], #[2]
        "model__min_samples_leaf": [ 1 ], #[1]
        "model__min_weight_fraction_leaf": [ 0.0 ], #[0.0]
        "model__max_features": [ "sqrt" ], # [ "sqrt", "log2", None ],
        "model__max_leaf_nodes": [ None ], # [ 2, 4, 8, 16, 32, 64, None ],
        "model__min_impurity_decrease": [ 0.0 ], # [0.0]
        "model__bootstrap": [ True ], # [ True, False ],
        "model__oob_score": [ False ], # [ True, False ],
        "model__n_jobs": [ -1 ], # [ None, -1, 1, 2, 4, 8 ],
    }
    return forest_params

def make_boost_params(my_args):
    boost_params = {
        "model__loss": [ "log_loss", "exponential" ], # [ "log_loss", "exponential" ],
        "model__learning_rate": [.1, .32], #[ 0.01, 0.1, 0.32 ], #[.32]
        "model__n_estimators": [50, 100, 200], # [ 10, 50, 100, 200, 400, 800 ], #[100]
        "model__subsample": [1.0], #[ 0.1, 0.2, 0.4, 0.8, 1.0 ], # [1.0]
        "model__criterion": [ "friedman_mse" ], # [ "friedman_mse", "mse", "mae" ],
        "model__min_samples_split": [ 2 ], #[2]
        "model__min_samples_leaf": [ 1 ], #[1]
        "model__min_weight_fraction_leaf": [ 0.0 ], #[0.0]
        "model__max_depth": [3, 5], #[ 1, 2, 3, 4, None ], #[3]
        "model__min_impurity_decrease": [ 0.0 ], # [0.0]
        "model__max_features": [ None ], # [ "sqrt", "log2", None ],
        "model__max_leaf_nodes": [ None ], # [ 2, 4, 8, 16, 32, 64, None ],
    }
    return boost_params

def make_SVM_params(my_args):
    # svm_params = {
    #     "model__C": [ 1.0 ], # [ 0.01, 0.1, 1.0, 10.0, 100.0 ],
    #     # "model__kernel": [ "linear", "poly", "rbf", "sigmoid" ], #["rbf"]
    #     "model__degree": [ 2, 3, 4, 5 ], #[3]
    #     # "model__gamma": [ "auto", "scale" ],
    #     # "model__coef0": [ 0.0, 0.1, 0.2, 0.4, 0.8 ],
    #     # "model__shrinking": [ True, False ], # [ True, False ],
    #     # "model__tol": [ 1e-3, 1e-4, 1e-5 ],
    #     # # "model__epsilon": [ 0.1, 0.2, 0.4, 0.8 ],
    #     # "model__cache_size": [ 100, 200, 400, 800 ],
    #     # "model__max_iter": [ -1, 1000, 2000, 4000 ],
    # }
    # return svm_params
    return {}

def make_linear_params(my_args):
    linear_params = {
        "model__fit_intercept": [ True, False ], # [ True, False ],
        "model__positive": [ False, True ], # [ True, False ],
        "model__copy_X": [ True, False ], # [ True, False ],
        # "model__n_jobs": [ -1 ], # [ None, -1, 1, 2, 4, 8 ],
    }
    return linear_params

def make_SGD_params(my_args):
    sgd_params = {
        "model__loss": ['hinge'], #['hinge', 'squared_error', 'squared_epsilon_insensitive', 'log_loss', 'squared_hinge', 'modified_huber', 'huber', 'perceptron', 'epsilon_insensitive'],
        "model__penalty": [ "l2", "l1", "elasticnet", None ],
        "model__alpha": [ 0.0001, 0.001, 0.01, 0.1, 1.0 ],
        "model__l1_ratio": [ 0.15, 0.3, 0.5, 0.7, 0.85 ],
        "model__fit_intercept": [ True ], # [ True, False ],
        # "model__max_iter": [ 1000 ], # [ 1000, 2000, 4000, 8000 ],
        "model__tol":  [ 1e-3, 1e-4, 1e-5 ],
        "model__shuffle": [ True ], # [ True, False ],
        "model__epsilon": [ 0.1, 0.2, 0.4, 0.8 ],
        "model__learning_rate": ['optimal'], #[ "constant", "optimal", "invscaling", "adaptive" ],
        "model__eta0": [ 0.01, 0.1, 0.32 ], #[0.0]
        "model__power_t": [ 0.25, 0.5, 0.75, 1.0 ],
        "model__early_stopping": [ False ], # [ True, False ],
        "model__validation_fraction": [ 0.1 ], # [ 0.1, 0.2, 0.4, 0.8 ],
        "model__n_iter_no_change": [ 5, 10, 20, 40 ],
        "model__warm_start": [ False ], # [ True, False ],
        "model__average": [ False ], # [ True, False ],
        "model__n_jobs": [ -1 ], # [ None, -1, 1, 2, 4, 8 ],
    }
    return sgd_params

def make_vote_params(my_args):
    params = {
        "model__estimator": [sklearn.ensemble.GradientBoostingClassifier()],
        "model__n_estimators": [10],
        "model__n_jobs": [-1]
    }
    return params

def make_fit_params(my_args):
    params = make_predictor_params(my_args)
    if my_args.model_type == "SGD":
        model_params = make_SGD_params(my_args)
    elif my_args.model_type == "linear":
        model_params = make_linear_params(my_args)
    elif my_args.model_type == "SVM":
        model_params = make_SVM_params(my_args)
    elif my_args.model_type == "boost":
        model_params = make_boost_params(my_args)
    elif my_args.model_type == "forest":
        model_params = make_forest_params(my_args)
    elif my_args.model_type == "tree":
        model_params = make_tree_params(my_args)
    elif my_args.model_type == "vote":
        model_params = make_vote_params(my_args)
    else:
        raise Exception("Unknown model type: {} [SGD, linear, SVM, boost, forest]".format(my_args.model_type))

    params.update(model_params)
    return params

def make_numerical_feature_pipeline(my_args):
    print("make_numerical_feature_pipeline")
    items = []

    items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))

    if my_args.numerical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.numerical_missing_strategy)))
    if my_args.use_polynomial_features:
        items.append(("polynomial-features", sklearn.preprocessing.PolynomialFeatures(degree=my_args.use_polynomial_features)))
    if my_args.use_scaler:
        items.append(("scaler", sklearn.preprocessing.StandardScaler()))
    items.append(("noop", PipelineNoop()))
    
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline


def make_categorical_feature_pipeline(my_args):
    print("make_categorical_feature_pipeline")
    items = []
    
    items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))

    if my_args.categorical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.categorical_missing_strategy)))
    items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))

    categorical_pipeline = sklearn.pipeline.Pipeline(items)
    return categorical_pipeline

def make_feature_pipeline(my_args):
    print("make_feature_pipeline")
    """
    Numerical features and categorical features are usually preprocessed
    differently. We split them out here, preprocess them, then merge
    the preprocessed features into one group again.
    """
    items = []

    items.append(("numerical", make_numerical_feature_pipeline(my_args)))
    items.append(("categorical", make_categorical_feature_pipeline(my_args)))
    pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
    return pipeline


def make_fit_pipeline_regression(my_args):
    """
    These are all regression models.
    """
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.model_type == "SGD":
        items.append(("model", sklearn.linear_model.SGDRegressor(max_iter=10000, n_iter_no_change=100, penalty=None))) # verbose=3, 
    elif my_args.model_type == "linear":
        items.append(("model", sklearn.linear_model.LinearRegression()))
    elif my_args.model_type == "SVM":
        items.append(("model", sklearn.svm.SVR()))
    elif my_args.model_type == "boost":
        items.append(("model", sklearn.ensemble.GradientBoostingRegressor()))
    elif my_args.model_type == "forest":
        items.append(("model", sklearn.ensemble.RandomForestRegressor()))
    elif my_args.model_type == "tree":
        items.append(("model", sklearn.tree.DecisionTreeRegressor()))
    else:
        raise Exception("Unknown model type: {} [SGD, linear, SVM, boost, forest]".format(my_args.model_type))

    return sklearn.pipeline.Pipeline(items)

def make_fit_pipeline_classification(my_args):
    print("make_fit_pipeline_classification")
    """
    These are all classification models.
    """
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.model_type == "SGD":
        items.append(("model", sklearn.linear_model.LogisticRegression(max_iter=10000, verbose=3))) # verbose=3, 
    elif my_args.model_type == "linear":
        items.append(("model", sklearn.linear_model.RidgeClassifier()))
    elif my_args.model_type == "SVM":
        items.append(("model", sklearn.svm.SVC(probability=True, max_iter=10000, class_weight="balanced"))) # verbose=3,
    elif my_args.model_type == "boost":
        items.append(("model", sklearn.ensemble.GradientBoostingClassifier()))
    elif my_args.model_type == "vote":
        estimators = [
            ('svm', sklearn.svm.SVC(probability=True)),
            ('gb', sklearn.ensemble.GradientBoostingClassifier())
        ]
        items.append(("model", sklearn.ensemble.VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)))
    elif my_args.model_type == "forest":
        items.append(("model", sklearn.ensemble.RandomForestClassifier(n_estimators=25, n_jobs=-1, min_samples_split=5, criterion="entropy", class_weight="balanced_subsample")))
    elif my_args.model_type == "tree":
        items.append(("model", sklearn.tree.DecisionTreeClassifier()))
    else:
        raise Exception("Unknown model type: {} [SGD, linear, SVM, boost, forest]".format(my_args.model_type))

    return sklearn.pipeline.Pipeline(items)

def make_fit_pipeline(my_args):
    return make_fit_pipeline_classification(my_args)
