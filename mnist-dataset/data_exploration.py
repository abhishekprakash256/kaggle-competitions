"""
explore the data on mnist dataset 
"""

#imports 
import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
#from xgboost.sklearn import XGBClassifier

import torch as th 
from torch import nn

#local cloud
FILE_PATH_train_c = "/home/ubuntu/s3/digit-recognizer/train.csv"
PREDICTION_DATA_c = "/home/ubuntu/s3/digit-recognize/test.csv"


#local ubuntu
FILE_PATH_train_l = "/home/abhi/Datasets/digit-recognizer/train.csv"
PREDICTION_DATA_l = "/home/abhi/Datasets/digit-recognizer/test.csv"


#local mac 
FILE_PATH_train_m = "/Users/abhi/Datasets/digit-recognizer/train.csv"
PREDICTION_DATA_m = "/Users/abhi/Datasets/digit-recognizer/test.csv"


df_train = pd.read_csv(FILE_PATH_train_m) #use this 

df_predtion = pd.read_csv(PREDICTION_DATA_m)


#split into the train and the test dataset 
y = df_train["label"]

X = df_train.drop("label", axis= 1)

#train and test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print(y_train)