# Fraud Detection System for Insurance Claims

## Overview

This repository contains the code and documentation for a fraud detection system designed to identify potentially fraudulent insurance claims. The model utilizes machine learning techniques to classify claims as either fraudulent or non-fraudulent. This project was developed using real-world insurance claim data provided by Oracle found on kaggle at https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection.

## Model Overview

The fraud detection model is built using scikit-learn's classification capabilities. Specifically, it employs a Random Forest classifier. To address the inherent class imbalance in fraud detection datasets (where non-fraudulent claims significantly outnumber fraudulent ones), the Synthetic Minority Over-sampling Technique for Nominal and Continuous features (SMOTE-NC) is integrated into the pipeline.

## Dataset and Preprocessing

The dataset used in this project comprises insurance claim information, including both numerical and categorical features.

**Numerical Features:**
- Age
- Deductible
- Year

**Categorical Features:**
- Month
- Policy Type
- Make of Car
- ... (and other relevant categorical variables)

The data underwent the following preprocessing steps:

1.  **Data Splitting:** The dataset was initially split into 80% training data and 20% testing data. The training data was further divided into 80% training and 20% validation sets.
2.  **Feature Selection:** Redundant features, such as a combined feature containing policy type and make of car (where the individual features already existed), were removed.
3.  **Oversampling:** SMOTE-NC was applied to the training data to address the class imbalance by oversampling the minority class (fraudulent claims). This technique is specifically chosen for its ability to handle both numerical and categorical features.

## Training Progress

During the training phase, the model achieved a high training accuracy of 98.5% and an $R^2$ loss score of 0.737.

## Challenges and Solutions

The primary challenge encountered during this project was the significant class imbalance within the dataset. This imbalance initially hindered the model's ability to effectively identify fraudulent claims. The solution implemented was the application of SMOTE-NC for oversampling the minority (fraudulent) class. This technique helped to provide a more balanced dataset for the model to learn from.

## Model Evaluation

The final model was evaluated on the held-out test data, yielding the following results:

**Validation Accuracy:** 85.8%
**Test Accuracy:** 85.5%

While a separate model achieved a higher test accuracy of 93.7%, the current model was chosen due to its more balanced precision and recall scores. The priority of this model is to effectively identify a higher proportion of actual fraudulent claims, even if it results in a slightly higher number of false positives. This trade-off is considered more beneficial for an insurance provider, who would likely prefer to investigate more potential fraud cases than miss actual instances of fraud.

[MLProjectPresentation.pdf](https://github.com/user-attachments/files/19996581/MLProjectPresentation.pdf)

