Churn Prediction System
Overview

The Churn Prediction System is a machine learning–based classification project designed to predict whether a customer or employee is satisfied with a company’s services or is likely to leave (churn).
Churn prediction is a critical business problem, as retaining existing customers or employees is significantly more cost-effective than acquiring new ones.

This project demonstrates a complete end-to-end ML pipeline, including data preprocessing, feature engineering, model training, and evaluation using industry-standard practices.

Problem Statement

The goal of this project is to classify individuals into:

Churn = Yes → Likely to leave the company

Churn = No → Likely to stay with the company

The model learns patterns from historical data to make accurate predictions on unseen data.

Dataset

The dataset used in this project is Churn.csv.

Dataset Description

Each row represents a customer or employee record

The target variable is Churn

The dataset contains:

Categorical features (e.g., contract type, payment method, services)

Numerical features (e.g., tenure, monthly charges)

Missing values handled during preprocessing

Target Variable

Churn

Yes → Individual is unhappy / likely to leave

No → Individual is satisfied / likely to stay

Data Preprocessing

To ensure clean and reliable input for the model, the following preprocessing steps are applied:

Missing Value Handling

SimpleImputer is used to fill missing values

Mean strategy for numerical features

Most frequent strategy for categorical features

Categorical Feature Encoding

OneHotEncoder is used to convert categorical variables into numerical format

Prevents the model from assuming ordinal relationships between categories

ColumnTransformer

Separates numerical and categorical preprocessing

Ensures cleaner, modular, and maintainable code

Pipeline

Combines preprocessing and model training into a single pipeline

Prevents data leakage

Makes training and inference consistent

Model Used

The classification task is performed using:

Random Forest Classifier

An ensemble learning method based on multiple decision trees

Advantages:

Handles both numerical and categorical data effectively

Reduces overfitting compared to single decision trees

Provides strong performance on structured/tabular data

Model Evaluation

The model is evaluated using multiple performance metrics to ensure robustness:

Confusion Matrix

Shows correct and incorrect predictions

Helps analyze false positives and false negatives

Classification Report

Precision

Recall

F1-score

Support

ROC-AUC Score

Measures the model’s ability to distinguish between churn and non-churn classes

ROC Curve

Visual representation of the trade-off between true positive rate and false positive rate

Project Workflow

Load and explore the dataset

Separate features and target variable

Apply preprocessing using ColumnTransformer

Build an end-to-end Pipeline

Train the model using RandomForestClassifier

Evaluate model performance using standard classification metrics

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn (for visualization)

Use Cases

Customer retention analysis

Employee attrition prediction

Business decision support systems

Predictive analytics applications

Future Improvements

Hyperparameter tuning using GridSearchCV or RandomizedSearchCV

Feature importance analysis

Trying advanced models (XGBoost, LightGBM)

Deployment using Flask or FastAPI

Model explainability using SHAP or LIME
