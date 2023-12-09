"""
make the model and train the model on the datasets and generate results

"""

#imports 
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import io
from contextlib import redirect_stdout
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


#import the helper classes
from data_handler import *




#File paths 

#cloud
FILE_PATH_train_c = "/home/ubuntu/s3/titanic_dataset/train.csv"
FILE_PATH_test_c = "/home/ubuntu/s3/titanic_dataset/test.csv"
FILE_PATH_SUB_c = "/home/ubuntu/s3/titanic_dataset/sample_submission.csv"


#local linux
L_FILE_PATH_train_l = "/home/abhi/Datasets/titanic_dataset/train.csv"
L_FILE_PATH_test_l = "/home/abhi/Datasets/titanic_dataset/test.csv"
L_FILE_PATH_SUB_l = "/home/abhi/Datasets/titanic_dataset/sample_submission.csv"


#local mac
M_FILE_PATH_train_l = "/Users/abhi/Datasets/titanic_dataset/train.csv"
M_FILE_PATH_test_l = "/Users/abhi/Datasets/titanic_dataset/test.csv"
M_FILE_PATH_SUB_l = "/Users/abhi/Datasets/titanic_dataset/sample_submission.csv"


INFO_FILE_NAME = "data_info.txt"
DESCRIBE_FILE_NAME = "data_describe.txt"
NULL_COUNTS_FILE = "null_count.txt"
HEAD_FILE = "head_info.txt"



