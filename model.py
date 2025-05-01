#!/usr/bin/env python3

import sklearn
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.linear_model
import joblib

# read data, define fields, etc.
from showcase_common_fine import *



import sklearn.pipeline
import sklearn.preprocessing
import sklearn.base
from sklearn.impute import SimpleImputer

import pandas as pd


class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, do_predictors=True, do_numerical=True):
        self.mCategoricalPredictors = categorical_features
        self.mNumericalPredictors = numerical_features
        self.mLabels = [label_name]
        self.do_numerical = do_numerical
        self.do_predictors = do_predictors
        
        if do_predictors:
            if do_numerical:
                self.mAttributes = self.mNumericalPredictors
            else:
                self.mAttributes = self.mCategoricalPredictors                
        else:
            self.mAttributes = self.mLabels
            
        return

    def fit( self, X, y=None ):
        # no fit necessary
        self.is_fitted_ = True
        return self

    def transform( self, X, y=None ):
        # only keep columns selected
        values = X[self.mAttributes]
        return values


filename = train_filename
data = pd.read_csv(filename, index_col=0)

items = []
items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))
num_pipeline = sklearn.pipeline.Pipeline(items)


items = []
items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))
items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))

cat_pipeline = sklearn.pipeline.Pipeline(items)


#
# Merge data back together
#
items = []
items.append(("numerical", num_pipeline))
items.append(("categorical", cat_pipeline))
pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
#
#
#


# pipeline.fit(data)
# data_transform = pipeline.transform(data)






# # After pipeline.fit(data) and data_transform = pipeline.transform(data)

# # scale data with x' = (x - u) / s
# scaler = sklearn.preprocessing.StandardScaler()

# # Since data_transform may be a sparse matrix from FeatureUnion, 
# # convert to dense array if needed
# if scipy.sparse.issparse(data_transform):
#     data_transform = data_transform.toarray()

# # find u and s
# scaler.fit(data_transform)

# # transform data 
# X_train = scaler.transform(data_transform)




full_pipeline = sklearn.pipeline.Pipeline([
    ("features", pipeline),  # Your existing FeatureUnion
    ("imputer", SimpleImputer(strategy='mean')),  # Add imputer step
    ("scaler", sklearn.preprocessing.StandardScaler(with_mean=False)),
])



# # Alternative approach - integrate scaler into pipeline
# full_pipeline = sklearn.pipeline.Pipeline([
#     ("features", pipeline),  # Your existing FeatureUnion
#     ("scaler", sklearn.preprocessing.StandardScaler(with_mean=False)),
# ])

# Then just do
full_pipeline.fit(data)
X_train = full_pipeline.transform(data)



# # scale data with x' = (x - u) / s
# scaler = sklearn.preprocessing.StandardScaler()
# # find u and s
# scaler.fit(data_transform) 
# # transform data
# X_train = scaler.transform(data_transform) 

# peek at scaled data
print("Scaled Features")
# print(feature_names)
print(X_train[:5,:])

# do the fit/training
regressor = sklearn.linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
regressor.fit(X_train, y_train)









# Replace the last line with:
joblib.dump((regressor, full_pipeline), model_filename)
# regressor, full_pipeline = joblib.load(model_filename)


# # save the trained model
# joblib.dump((regressor,scaler), model_filename)