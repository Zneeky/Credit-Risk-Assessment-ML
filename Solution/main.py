import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib


# 1.Importing Dataset
dataset = pd.read_excel('../Data/e_NewApplications_CreditScore_Needed.xlsx')

# shows count of rows and columns
print(dataset.shape)

# shows first few rows of the code
print(dataset.head())

#dropping customer ID column from the dataset
dataset=dataset.drop('ID',axis=1)

# shows count of rows and columns
print(dataset.shape)

# explore missing values
dataset.isna().sum()

# filling missing values with mean
dataset=dataset.fillna(dataset.mean())

# 2.Train Test Split
X_fresh = dataset

# Loading normalisation coefficients - exported from the model code file as f2_Normalisation 
sc = joblib.load('../Data/f2_Normalisation_CreditScoring')
X_fresh = sc.transform(X_fresh)
