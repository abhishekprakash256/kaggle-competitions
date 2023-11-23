"""
To habdle the data and prepare the data for the model to learn 
"""

#imports 
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns



#File paths 

#cloud
FILE_PATH_train_c = "/home/ubuntu/s3/titanic_dataset/train.csv"
FILE_PATH_test_c = "/home/ubuntu/s3/titanic_dataset/test.csv"
FILE_PATH_SUB_c = "/home/ubuntu/s3/titanic_dataset/sample_submission.csv"


#local
FILE_PATH_train_l = "/home/abhi/Datasets/titanic_dataset/train.csv"
FILE_PATH_test_l = "/home/abhi/Datasets/titanic_dataset/test.csv"
FILE_PATH_SUB_l = "/home/abhi/Datasets/titanic_dataset/sample_submission.csv"




class Data():
    def __init__(self):
        self.train = None
        self.test = None

    def read_data(self,train_data_path,test_data_path):
        """
        The function to read the data 
        """

        train = pd.read_csv(train_data_path)
        test = pd.read_csv(test_data_path)






if __name__ == "__main__":
    data = Data()

