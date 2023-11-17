"""
explore the titanic dataset
make the feture 

"""
#imports 
import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
#FILE PATH 
#cloud 
FILE_PATH_train_c = "/home/ubuntu/s3/titanic_dataset/train.csv"
FILE_PATH_test_c = "/home/ubuntu/s3/titanic_dataset/test.csv"


#local
FILE_PATH_train_l = "/home/abhi/Datasets/titanic_dataset/train.csv"
FILE_PATH_test_l = "/home/abhi/Datasets/titanic_dataset/test.csv"


df_train = pd.read_csv(FILE_PATH_train_c)
df_test = pd.read_csv(FILE_PATH_test_c)


print(df_train.info())

#print(df_train.corr())


print(df_train.describe())

"""
print(df_train["HomePlanet"].unique())
print(df_train["CryoSleep"].unique())
print(df_train["Cabin"].unique())
print(df_train["Destination"].unique())
print(df_train["VIP"].unique())

"""

#to drop the column 
#homeplanet , Cryosleep, Destination


"""
n = df_train["RoomService"].unique()
print(len(n))


num_null_values = df_train.isnull().sum()
print(num_null_values)
"""


columns = ["HomePlanet","CryoSleep", "Destination","Name", "Cabin", "PassengerId"]

df_train = df_train.drop(columns, axis = 1)

df_train = pd.get_dummies(df_train, columns = ["VIP"])

df_train["Transported"] = df_train['Transported'].astype(float)  
df_train["VIP_False"] = df_train['VIP_False'].astype(float)
df_train["VIP_True"] = df_train['VIP_True'].astype(float)  

df_train = df_train.dropna()

num_null_values = df_train.isnull().sum()
print(num_null_values)


