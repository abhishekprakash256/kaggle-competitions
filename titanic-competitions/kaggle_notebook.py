"""
using the kaggle notebook 
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc 


#cloud
FILE_PATH_train_c = "/home/ubuntu/s3/titanic_dataset/train.csv"
PREDICTION_DATA_c = "/home/ubuntu/s3/titanic_dataset/test.csv"

#local
FILE_PATH_train_l = "/home/abhi/Datasets/titanic_dataset/train.csv"
PREDICTION_DATA_l = "/home/abhi/Datasets/titanic_dataset/test.csv"


train = pd.read_csv(FILE_PATH_train_l)
test = pd.read_csv(PREDICTION_DATA_l)


print("train data",train.shape)
print(test.shape)

#drop the passangerID column 
train.drop('PassengerId',axis = 1,inplace = True)
test.drop('PassengerId',axis = 1 , inplace = True)


#combined trained and test
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.Transported.values
all_data = pd.concat([train.drop('Transported',axis = 1),test],axis=0).reset_index(drop = True)

print("y_train",y_train.shape)

print(all_data.shape)

numeric_cols = all_data.select_dtypes(include='number')

corrmat = numeric_cols.corr()

plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.9,square=True)

#filing the missing values 

#VIP and non-VIP customers spend differently, so choose to group them with VIP columns and average them to fill in the missing values
all_data['RoomService'] = all_data.groupby('VIP')['RoomService'].transform(lambda x:x.fillna(x.mean()))

#VRDeck,FoodCourt,ShoppngMall,Spa are the same with above
cols = ['VRDeck','FoodCourt','ShoppingMall','Spa']
for c in cols:
    all_data[c] = all_data.groupby('VIP')[c].transform(lambda x:x.fillna(x.mean()))


#Destination, HomePlanet, Cabin, CryoSleep,VIP missing value filled with mode()
cols = ['Destination','HomePlanet','Cabin','CryoSleep','VIP']
for c in cols:
    all_data[c] = all_data[c].fillna(all_data[c].mode()[0])


#Age missing value fill with mean
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].mean())

all_data['RoomService'] = all_data.groupby('VIP')['RoomService'].transform(lambda x:x.fillna(x.mean()))

cols = ['VRDeck','FoodCourt','ShoppingMall','Spa']
for c in cols:
    all_data[c] = all_data.groupby('VIP')[c].transform(lambda x:x.fillna(x.mean()))



#LabelEncoder for not number data column
from sklearn.preprocessing import LabelEncoder
cols = ('HomePlanet','CryoSleep','Cabin','Destination','VIP')
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values))   
    all_data[c] = lbl.transform(list(all_data[c].values))



train = all_data[:ntrain].drop('Name',axis=1)
test = all_data[ntrain:].drop('Name',axis=1)

print(train.shape)
print(test.shape)

print(test.info())


"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#RandomForestClassifier()
rf_classifier = RandomForestClassifier()

rf_classifier.fit(train, y_train)

predictions = rf_classifier.predict(test)

print(type(y_train))
print(type(predictions))

print(y_train.shape)

print(predictions.shape)



cm_rf = confusion_matrix(y_train, predictions) 
# Accuracy 
accuracy_rf = accuracy_score(y_train, predictions) 
# Precision 
precision_rf = precision_score(y_train, predictions) 
# Recall 
recall_rf = recall_score(y_train, predictions) 
# F1-Score 
f1_rf = f1_score(y_train, predictions) 

print("The CM score decison tree", cm_rf )

print("the accuracy score decison tree", accuracy_rf)

print("the precision score decsion tree", precision_rf)

print("the recall score descision tree", recall_rf)

print("f1 score decison tree", f1_rf)

"""