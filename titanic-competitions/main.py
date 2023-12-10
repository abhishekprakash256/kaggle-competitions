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

"""
INFO_FILE_NAME = "data_info.txt"
DESCRIBE_FILE_NAME = "data_describe.txt"
NULL_COUNTS_FILE = "null_count.txt"
HEAD_FILE = "head_info.txt"
"""




class Models():

    def __init__(self):
        """
        make the model instances as objects 
        """
        self.dst = None
        self.rf = None
        self.gbc = None
        self.lg = None
    
    def make_models(self):
        """
        The function to make the models
        """
        #dst model
        self.dst = DecisionTreeClassifier()

        #rf model
        self.rf = RandomForestClassifier()

        #gbc model
        self.gbc = GradientBoostingClassifier(n_estimators=50, learning_rate= 0.1,random_state=0)

        #logistic regression 

        self.lg = LogisticRegression()

    

def train_test():
    """
    The function to train and test the model performance 
    """

    #load the datasets
    data = Data()
    data.read_data(M_FILE_PATH_train_l,M_FILE_PATH_test_l)
    data.data_prep()
    data.train_test_divider()

    #make the models
    models = Models()
    models.make_models()


    models.dst = DecisionTreeClassifier()

    models.dst.fit(data.train_X,data.train_y)
    y_pred_dst = models.dst.predict(data.test_X)

    cm_dst = confusion_matrix(data.test_y, y_pred_dst) 
    # Accuracy 
    accuracy_dst = accuracy_score(data.test_y, y_pred_dst) 
    # Precision 
    precision_dst = precision_score(data.test_y, y_pred_dst) 
    # Recall 
    recall_dst = recall_score(data.test_y, y_pred_dst) 
    # F1-Score 
    f1_dst = f1_score(data.test_y, y_pred_dst) 

    print("The CM score decison tree", cm_dst )

    print("the accuracy score decison tree", accuracy_dst)

    print("the precision score decsion tree", precision_dst)

    print("the recall score descision tree", recall_dst)

    print("f1 score decison tree", f1_dst)



    models.rf.fit(data.train_X,data.train_y)

    y_pred_rf = models.rf.predict(data.test_X)

    cm_rf = confusion_matrix(data.test_y, y_pred_rf) 
    # Accuracy 
    accuracy_rf = accuracy_score(data.test_y, y_pred_rf) 
    # Precision 
    precision_rf = precision_score(data.test_y, y_pred_rf) 
    # Recall 
    recall_rf = recall_score(data.test_y, y_pred_rf) 
    # F1-Score 
    f1_rf = f1_score(data.test_y, y_pred_rf) 

    print("The CM score random forest", cm_rf )

    print("the accuracy score random forest", accuracy_rf)

    print("the precision score random forest", precision_rf)

    print("the recall score random forest", recall_rf)

    print("f1 score random forest", f1_rf)


    models.gbc.fit(data.train_X,data.train_y)

    y_pred_gbc = models.gbc.predict(data.test_X)


    cm_gbc = confusion_matrix(data.test_y, y_pred_gbc) 
    # Accuracy 
    accuracy_gbc = accuracy_score(data.test_y, y_pred_gbc) 
    # Precision 
    precision_gbc = precision_score(data.test_y, y_pred_gbc) 
    # Recall 
    recall_gbc = recall_score(data.test_y, y_pred_gbc) 
    # F1-Score 
    f1_gbc = f1_score(data.test_y, y_pred_gbc) 

    print("The CM score gbc", cm_gbc )

    print("the accuracy score gbc", accuracy_gbc)

    print("the precision score gbc", precision_gbc)

    print("the recall score gbc", recall_gbc)

    print("f1 score gbc", f1_gbc)



    models.lg.fit(data.train_X,data.train_y)

    y_pred_lg = models.lg.predict(data.test_X)


    cm_lg = confusion_matrix(data.test_y, y_pred_lg) 
    # Accuracy 
    accuracy_lg = accuracy_score(data.test_y, y_pred_lg) 
    # Precision 
    precision_lg = precision_score(data.test_y, y_pred_lg) 
    # Recall 
    recall_lg = recall_score(data.test_y, y_pred_lg) 
    # F1-Score 
    f1_lg = f1_score(data.test_y, y_pred_lg) 

    print("The CM score gbc", cm_lg )

    print("the accuracy score gbc", accuracy_lg)

    print("the precision score gbc", precision_lg)

    print("the recall score gbc", recall_lg)

    print("f1 score gbc", f1_lg)




if __name__== "__main__":

    train_test()
