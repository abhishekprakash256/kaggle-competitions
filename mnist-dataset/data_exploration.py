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


#print(X_train.head())
#print(y_train.head())


#image size is 28 X 28 as square root of 784 pixels 


#take one row and stack one another 

test = X_train.iloc[0][0:28].to_numpy().reshape(1,28)
test2 = X_train.iloc[0][28:56].to_numpy().reshape(1,28)

#print(test.shape)
#print(test2.shape)

#combine = np.concatenate((test, test2), axis = 0)

#zeros_array = np.zeros((1, 28))
#zeros_array = test.values
#zeros_array = zeros_array.reshape(1,28)

#print(combine.shape)


"""
make a array and assign the value in the array and then we can use that array to train the model 
"""




#get the one row in array 


one_row = X_train.iloc[0]
#print(one_row)

#make the loop for the combine the aarry 


#make an whole array of the numpy

"""
for i in range(0,757,28):

    segment = one_row[i:i+28].to_numpy().reshape(1,28)

    #print(segment.shape)

    image_arr = np.append(image_arr, segment)

    #image_arr.append(segment)

#image_arr = np.array(image_arr)

#print(image_arr.shape)
"""



X_train_data = np.empty((len(X_train), 28, 28))

# Iterate through each row in X_train
for index, row in X_train.iterrows():
    
    if index >= len(X_train_data):
        break
    
    # Reshape the row into a 28x28 array
    segment = row.values.reshape(28, 28)
    
    # Assign segment to full_data
    X_train_data[index] = segment


print(X_train_data[0].shape)

#print(y_train.shape)

#make model 


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel, 32 output channels, 3x3 kernel, padding=1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32 input channels, 64 output channels, 3x3 kernel, padding=1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer with kernel size 2 and stride 2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Input size after pooling: 64 channels, 7x7 output size, output size: 128
        self.fc2 = nn.Linear(128, 10)  # Output size: 10 (for classification)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))  # Apply convolution, ReLU, and pooling
        x = self.pool(nn.functional.relu(self.conv2(x)))  # Apply convolution, ReLU, and pooling
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output
        x = nn.functional.relu(self.fc1(x))  # Apply fully connected layer and ReLU
        x = self.fc2(x)  # Apply fully connected layer
        return x


model = SimpleModel()


# Assuming X_train_data[0] is a 28x28 NumPy array
X_train_data_batch = np.expand_dims(X_train_data[0], axis=0)  # Add a batch dimension

y_pred = model(th.tensor(X_train_data_batch, dtype = th.float32))  # Convert to torch tensor and pass to the model

print(y_pred)