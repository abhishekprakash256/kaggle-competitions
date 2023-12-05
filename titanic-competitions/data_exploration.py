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

import torch as th 
from torch import nn


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
df_predtion = pd.read_csv(PREDICTION_DATA_c)


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

#---------data manipulation ---------

"""

print(df_train.info())


print(df_train["CryoSleep"].unique())
print(df_train["Destination"].unique())

"""

"""

 0   PassengerId   8693 non-null   object
 1   HomePlanet    8492 non-null   object
 2   CryoSleep     8476 non-null   object
 3   Cabin         8494 non-null   object
 4   Destination   8511 non-null   object
 5   Age           8514 non-null   float64
 6   VIP           8490 non-null   object
 7   RoomService   8512 non-null   float64
 8   FoodCourt     8510 non-null   float64
 9   ShoppingMall  8485 non-null   float64
 10  Spa           8510 non-null   float64
 11  VRDeck        8505 non-null   float64
 12  Name          8493 non-null   object
 13  Transported   8693 non-null   bool

"""



#columns = ["HomePlanet","CryoSleep", "Destination","Name", "Cabin", "PassengerId"]

columns = ["Name","Cabin","PassengerId"]

#--------------------train data---------------------------
df_train = df_train.drop(columns, axis = 1)

df_train = pd.get_dummies(df_train, columns = ["Destination"])

df_train = pd.get_dummies(df_train, columns = ["HomePlanet"])

df_train = pd.get_dummies(df_train, columns = ["VIP"])

df_train = pd.get_dummies(df_train, columns = ["CryoSleep"])

#df_train = pd.get_dummies(df_train,columns = ["Destination"])

df_train["Transported"] = df_train['Transported']


df_train = df_train.dropna()

X = df_train.drop("Transported", axis= 1)

y = df_train["Transported"]

#train and test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.info())
#make the model and fit the data 

"""
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

gbc = GradientBoostingClassifier(n_estimators=50, learning_rate= 0.1,random_state=0)

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

"""
#------------------------------------------------


#shaping the input data 

X_train, X_test, y_train, y_test = X_train.values ,X_test.values,y_train.values,y_test.values

X_train, X_test, y_train, y_test = X_train.astype(float) ,X_test.astype(float) ,y_train.astype(float) ,y_test.astype(float)


print(X_train[0])
print(y_train[0])


class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


input_size = 16  # Replace with your input feature size
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 1  # 1 for binary classification (0 or 1)

# Create an instance of the SimpleClassifier
model = SimpleClassifier(input_size, hidden_size, output_size)

# Print the model architecture


X_train, X_test, y_train, y_test = th.from_numpy(X_train) , th.from_numpy(X_test), th.from_numpy(y_train) , th.from_numpy(y_test) 

X_train, X_test, y_train, y_test = X_train.float(), X_test.float(), y_train.float(), y_test.float()


EPOCHS = 1000

def train_and_test():

    """
    The funcition to train and test the model 
    """

    loss_fn = nn.BCEWithLogitsLoss()

    # Define the optimizer
    optimizer = th.optim.SGD(params=model.parameters(), lr=0.1)


    #the loop for trainer
    for epoch in range(EPOCHS):
        model.train()

        y_pred = model(X_train)

        loss = loss_fn(y_pred.view(-1), y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with th.inference_mode():
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred.view(-1), y_test)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Training Loss: {loss}, Test Loss: {test_loss}")


train_and_test()