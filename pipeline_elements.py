#!/usr/bin/env python3


################################################################
#
# These custom classes help with pipeline building and debugging
#
import sklearn.base
import pandas as pd

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

class Printer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Pipeline member to display the data at this stage of the transformation.
    """
    
    def __init__(self, title):
        self.title = title
        return

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        print("{}::type(X)".format(self.title), type(X))
        print("{}::X.shape".format(self.title), X.shape)
        if not isinstance(X, pd.DataFrame):
            print("{}::X[0]".format(self.title), X[0])
        print("{}::X".format(self.title), X)
        return X

class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, do_predictors=True, do_numerical=True):
        # skip "name", "ticket", "cabin", "boat", "body", "home.dest"
        # self.mCategoricalPredictors = ["type"]
        # self.mNumericalPredictors = ["step","amount","oldbalanceOrg","newbalanceOrig",
        #                              "oldbalanceDest","newbalanceDest",
        #                              "isFlaggedFraud"]
        # self.mLabels = ["isFraud"]
        self.mCategoricalPredictors = ["Month", 
                                    #    "WeekOfMonth", "DayOfWeek", 
                                       "Make", "AccidentArea", 
    # "DayOfWeekClaimed",
      "MonthClaimed",
        # "WeekOfMonthClaimed", 
        "Sex", 
    "MaritalStatus", "Fault", "PolicyType",
    #   "VehicleCategory", 
    "VehiclePrice", "PoliceReportFiled", "WitnessPresent", "AgentType", 
    "AddressChange_Claim", 
    # "BasePolicy",
        "Days_Policy_Accident", "Days_Policy_Claim", 
    "PastNumberOfClaims", "AgeOfVehicle", "AgeOfPolicyHolder", "NumberOfSuppliments", "NumberOfCars",]
        self.mNumericalPredictors = ["Age", "Deductible",
                                    #   "DriverRating", 
    # "Days_Policy_Accident", "Days_Policy_Claim", 
    # "PastNumberOfClaims",
    #   "AgeOfVehicle", 
    # "AgeOfPolicyHolder",
    #   "NumberOfSuppliments",
        # "NumberOfCars",
          "Year"]
        self.mLabels = ["FraudFound_P"]
        # Categorical features
        # self.mCategoricalPredictors =  [
        #     "Gender", 
        #     # "Transaction_Location", 
        #     "Account_Type", 
        #     "Transaction_Type", 
        #     "Merchant_Category", 
        #     "Device_Type",
        #     # "Transaction_Currency",
        #     "Transaction_Device",
        # ]
        
        
        # # Numerical features
        # self.mNumericalPredictors = [
        #     "Age",
        #     "Account_Balance", 
        #     # "Transaction_Date",
        #     # "Transaction_Time",
        #     "Transaction_Amount",
        # ]
        # self.mLabels = ["Is_Fraud"]
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

#
# These custom classes help with pipeline building and debugging
#
################################################################
