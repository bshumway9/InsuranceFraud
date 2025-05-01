#!/usr/bin/env python3

import pandas as pd

###########################################################
# Set global values for filenames, features, label
# Load data
###########################################################

feature_names =            ["type","step","amount","oldbalanceOrg","newbalanceOrig",
                                     "oldbalanceDest","newbalanceDest",
                                     "isFlaggedFraud"]

numerical_features = ["step","amount","oldbalanceOrg","newbalanceOrig",
                                     "oldbalanceDest","newbalanceDest",
                                     "isFlaggedFraud"]

categorical_features = ["type"]
label_name = "isFraud"
# feature_names =            [
#             "Gender", 
#             # "Transaction_Location", 
#             "Account_Type", 
#             "Transaction_Type", 
#             "Merchant_Category", 
#             "Device_Type",
#             # "Transaction_Currency",
#             "Transaction_Device",
#             "Age",
#             "Account_Balance", 
#             # "Transaction_Date",
#             # "Transaction_Time",
#             "Transaction_Amount",
#         ]

# numerical_features = ["Age",
#             "Account_Balance", 
#             # "Transaction_Date",
#             # "Transaction_Time",
#             "Transaction_Amount",]

# categorical_features = [
#             "Gender", 
#             # "Transaction_Location", 
#             "Account_Type", 
#             "Transaction_Type", 
#             "Merchant_Category", 
#             "Device_Type",
#             # "Transaction_Currency",
#             "Transaction_Device",
#         ]
                         

# label_name = "Is_Fraud"

feature_names = [
    "Age", "Deductible", "DriverRating", "Days_Policy_Accident", 
    "Days_Policy_Claim", "PastNumberOfClaims", "AgeOfVehicle", 
    "AgeOfPolicyHolder", "NumberOfSuppliments", "NumberOfCars", "Year",
    "Month", "WeekOfMonth", "DayOfWeek", "Make", "AccidentArea", 
    "DayOfWeekClaimed", "MonthClaimed", "WeekOfMonthClaimed", "Sex", 
    "MaritalStatus", "Fault", "PolicyType", "VehicleCategory", 
    "VehiclePrice", "PoliceReportFiled", "WitnessPresent", "AgentType", 
    "AddressChange_Claim", "BasePolicy"
]

numerical_features = [
    "Age", "Deductible", 
    # "DriverRating", 
    # "Days_Policy_Accident", "Days_Policy_Claim", 
    # "PastNumberOfClaims",
    #   "AgeOfVehicle", 
    # "AgeOfPolicyHolder",
    #   "NumberOfSuppliments",
        # "NumberOfCars",
          "Year"
]

categorical_features = [
    "Month", 
    # "WeekOfMonth", "DayOfWeek", 
    "Make", "AccidentArea", 
    # "DayOfWeekClaimed",
      "MonthClaimed", 
    #   "WeekOfMonthClaimed",
        "Sex", 
    "MaritalStatus", "Fault", "PolicyType",
    #   "VehicleCategory", 
    "VehiclePrice", "PoliceReportFiled", "WitnessPresent", "AgentType", 
    "AddressChange_Claim",
    # "BasePolicy",
        "Days_Policy_Accident", "Days_Policy_Claim", 
    "PastNumberOfClaims", "AgeOfVehicle", "AgeOfPolicyHolder", "NumberOfSuppliments", "NumberOfCars",
]

label_name = "FraudFound_P"
train_filename = "FO-train-train.csv"
data = pd.read_csv(train_filename, index_col=0)
X_train = data[feature_names]
y_train = data[label_name]

model_filename = "FO-model.joblib"