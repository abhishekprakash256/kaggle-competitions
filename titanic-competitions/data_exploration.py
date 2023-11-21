"""
explore the titanic dataset
make the feture 

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
from xgboost.sklearn import XGBClassifier


encoder = OneHotEncoder(sparse=False)
#FILE PATH 
#cloud 
FILE_PATH_train_c = "/home/ubuntu/s3/titanic_dataset/train.csv"
PREDICTION_DATA_c = "/home/ubuntu/s3/titanic_dataset/test.csv"


#local
FILE_PATH_train_l = "/home/abhi/Datasets/titanic_dataset/train.csv"
PREDICTION_DATA_l = "/home/abhi/Datasets/titanic_dataset/test.csv"


#df_train = pd.read_csv(FILE_PATH_train_c)
#df_test = pd.read_csv(FILE_PATH_test_c)



df_train = pd.read_csv(FILE_PATH_train_c)
df_test = pd.read_csv(PREDICTION_DATA_c)


#print(df_train.info())

#print(df_train.corr())


#print(df_train.describe())

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

#columns = ["HomePlanet","CryoSleep", "Destination","Name", "Cabin", "PassengerId"]

columns = ["HomePlanet","CryoSleep", "Destination","Name", "Cabin", "PassengerId"]

#--------------------train data---------------------------
df_train = df_train.drop(columns, axis = 1)

df_train = pd.get_dummies(df_train, columns = ["VIP"])

df_train["Transported"] = df_train['Transported'].astype(float)  
df_train["VIP_False"] = df_train['VIP_False'].astype(float)
df_train["VIP_True"] = df_train['VIP_True'].astype(float)  

df_train = df_train.dropna()

X = df_train.drop("Transported", axis= 1)

y = df_train["Transported"]


#train and test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#make the model and fit the data 

#-------------decision tree model------------------------
dst = DecisionTreeClassifier()


dst.fit(X_train,y_train)

y_pred_dst = dst.predict(X_test)


cm_dst = confusion_matrix(y_test, y_pred_dst) 
# Accuracy 
accuracy_dst = accuracy_score(y_test, y_pred_dst) 
# Precision 
precision_dst = precision_score(y_test, y_pred_dst) 
# Recall 
recall_dst = recall_score(y_test, y_pred_dst) 
# F1-Score 
f1_dst = f1_score(y_test, y_pred_dst) 

print("The CM score decison tree", cm_dst )

print("the accuracy score decison tree", accuracy_dst)

print("the precision score decsion tree", precision_dst)

print("the recall score descision tree", recall_dst)

print("f1 score decison tree", f1_dst)


#-----------------------------random forest --------------------------

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)


cm_rf = confusion_matrix(y_test, y_pred_rf) 
# Accuracy 
accuracy_rf = accuracy_score(y_test, y_pred_rf) 
# Precision 
precision_rf = precision_score(y_test, y_pred_rf) 
# Recall 
recall_rf = recall_score(y_test, y_pred_rf) 
# F1-Score 
f1_rf = f1_score(y_test, y_pred_rf) 

print("The CM score random forest", cm_rf )

print("the accuracy score random forest", accuracy_rf)

print("the precision score random forest", precision_rf)

print("the recall score random forest", recall_rf)

print("f1 score random forest", f1_rf)


#------------------------ GradientBoostingClassifier----------------------------

gbc = GradientBoostingClassifier()

gbc.fit(X_train,y_train)

y_pred_gbc = gbc.predict(X_test)


cm_gbc = confusion_matrix(y_test, y_pred_gbc) 
# Accuracy 
accuracy_gbc = accuracy_score(y_test, y_pred_gbc) 
# Precision 
precision_gbc = precision_score(y_test, y_pred_gbc) 
# Recall 
recall_gbc = recall_score(y_test, y_pred_gbc) 
# F1-Score 
f1_gbc = f1_score(y_test, y_pred_gbc) 

print("The CM score gbc", cm_gbc )

print("the accuracy score gbc", accuracy_gbc)

print("the precision score gbc", precision_gbc)

print("the recall score gbc", recall_gbc)

print("f1 score gbc", f1_gbc)


#---------------------------LogisticRegression-------------------------------

lg = LogisticRegression()

lg.fit(X_train,y_train)

y_pred_lg = lg.predict(X_test)


cm_lg = confusion_matrix(y_test, y_pred_lg) 
# Accuracy 
accuracy_lg = accuracy_score(y_test, y_pred_lg) 
# Precision 
precision_lg = precision_score(y_test, y_pred_lg) 
# Recall 
recall_lg = recall_score(y_test, y_pred_lg) 
# F1-Score 
f1_lg = f1_score(y_test, y_pred_lg) 

print("The CM score gbc", cm_lg )

print("the accuracy score gbc", accuracy_lg)

print("the precision score gbc", precision_lg)

print("the recall score gbc", recall_lg)

print("f1 score gbc", f1_lg)

#----------------------using the xgboost ----------------

xg = XGBClassifier()

xg.fit(X_train,y_train)

y_pred_xg = xg.predict(X_test)


cm_xg = confusion_matrix(y_test, y_pred_xg) 
# Accuracy 
accuracy_xg = accuracy_score(y_test, y_pred_xg) 
# Precision 
precision_xg = precision_score(y_test, y_pred_xg) 
# Recall 
recall_xg = recall_score(y_test, y_pred_xg) 
# F1-Score 
f1_xg = f1_score(y_test, y_pred_xg) 

print("The CM score xg", cm_xg )

print("the accuracy score xg", accuracy_xg)

print("the precision score xg", precision_xg)

print("the recall score xg", recall_xg)

print("f1 score xg", f1_xg)


#----------------------------
