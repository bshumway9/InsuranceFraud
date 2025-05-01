#!/usr/bin/env python3

from pipeline_elements import *
import sklearn.impute
import sklearn.preprocessing
import sklearn.pipeline
import pandas as pd
import numpy as np
import joblib
import os
import imblearn as imbl

def make_numerical_feature_pipeline():
    items = []
    items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))
    items.append(("missing-data", sklearn.impute.SimpleImputer(strategy="median")))
    items.append(("scaler", sklearn.preprocessing.StandardScaler()))
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline


def make_categorical_feature_pipeline():
    items = []
    items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))
    items.append(("missing-data", sklearn.impute.SimpleImputer(strategy="constant", fill_value="NULL")))
    items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))
    categorical_pipeline = sklearn.pipeline.Pipeline(items)
    return categorical_pipeline

def make_feature_pipeline():
    items = []
    items.append(("numerical", make_numerical_feature_pipeline()))
    items.append(("categorical", make_categorical_feature_pipeline()))
    pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
    return pipeline

def preprocess_dataframe(pipeline, dataframe, label):
    """
    Preprocess a dataframe with the given pipeline.
    Assumes the pipeline has been fit.
    Assumes dataframe has an index column, and preserves it.
    If dataframe has a series identified by label, it is preserved.
    Assumes all other columns are features, and transforms them.
    """
    
    # list of features to transform
    feature_names = list(dataframe.columns)
    have_label = label in feature_names
    if have_label:
        feature_names.remove(label)
    cols = ["Month", "Make", "AccidentArea", "MonthClaimed","Sex", 
    "MaritalStatus", "Fault", "PolicyType",
    "VehiclePrice", "PoliceReportFiled", "WitnessPresent", "AgentType", 
    "AddressChange_Claim", 
        "Days_Policy_Accident", "Days_Policy_Claim", 
    "PastNumberOfClaims", "AgeOfVehicle", "AgeOfPolicyHolder", "NumberOfSuppliments", "NumberOfCars","Age", "Deductible","Year"]
    # separate features and label
    X = dataframe[cols]
    if have_label:
        y = dataframe[label]
    
    # Print class distribution before SMOTE
    if have_label:
        print(f"Class distribution before SMOTE: {y.value_counts().to_dict()}")
    
    categorical_features = ["Month", "Make", "AccidentArea", "MonthClaimed", "Sex", 
                            "MaritalStatus", "Fault", "PolicyType", "VehiclePrice", 
                            "PoliceReportFiled", "WitnessPresent", "AgentType", 
                            "AddressChange_Claim", "Days_Policy_Accident", 
                            "Days_Policy_Claim", "PastNumberOfClaims", "AgeOfVehicle", 
                            "AgeOfPolicyHolder", "NumberOfSuppliments", "NumberOfCars"]
    oversample = imbl.over_sampling.SMOTENC(sampling_strategy='minority', 
                                             categorical_features=categorical_features, 
                                             random_state=42,
                                             k_neighbors=2)
    X, y = oversample.fit_resample(X, y)
    
    # Print class distribution after SMOTE
    if have_label:
        print(f"Class distribution after SMOTE: {pd.Series(y).value_counts().to_dict()}")
        print(f"Data shape after SMOTE: {X.shape}")
    
    # transform features
    X_transformed = pipeline.transform(X)
    
    # if the transform became sparse, densify it.
    # this usually happens because of one-hot-encoding
    if not isinstance(X_transformed, np.ndarray):
        X_transformed = X_transformed.todense()

    # reconstruct a dataframe, we've lost the labels of features
    df1 = pd.DataFrame(X_transformed)
    
    if have_label:
        # add labels
        df1[label] = y

    # Create a new index since SMOTE has changed the number of samples
    # We can't use the original index anymore
    df1.index = pd.RangeIndex(start=0, stop=len(df1), step=1)
    print(f"Final preprocessed data shape: {df1.shape}")

    return df1

def fit_pipeline_to_dataframe(dataframe):
    pipeline = make_feature_pipeline()
    pipeline.fit(dataframe)
    return pipeline

def save_pipeline(pipeline, filename):
    joblib.dump(pipeline, filename)
    return

def load_pipeline(filename):
    pipeline = joblib.load(filename)
    return pipeline

def preprocess_file(input_filename, output_filename, pipeline_filename, label):
    dataframe = pd.read_csv(input_filename, index_col=0)
    if not os.path.exists(pipeline_filename):
        pipeline = fit_pipeline_to_dataframe(dataframe)
        save_pipeline(pipeline, pipeline_filename)
    else:
        pipeline = load_pipeline(pipeline_filename)

    processed_dataframe = preprocess_dataframe(pipeline, dataframe, label)
    processed_dataframe.to_csv(output_filename, index=True)
    
    return

def main_train():
    data_filename = "FO-train-train.csv"
    out_filename = "preprocessed-train.csv"
    pipeline_filename = "preprocessor.joblib"
    label = "FraudFound_P"
    preprocess_file(data_filename, out_filename, pipeline_filename, label)
    return

def main_test():
    data_filename = "FO-train-val.csv"
    out_filename = "preprocessed-test.csv"
    pipeline_filename = "preprocessor.joblib"
    label = "FraudFound_P"
    preprocess_file(data_filename, out_filename, pipeline_filename, label)
    return

def main():
    # main_train()
    main_test()

if __name__ == "__main__":
    main()
