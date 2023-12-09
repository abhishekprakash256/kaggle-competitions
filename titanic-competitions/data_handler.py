"""
To habdle the data and prepare the data for the model to learn 
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




#File paths 

#cloud
FILE_PATH_train_c = "/home/ubuntu/s3/titanic_dataset/train.csv"
FILE_PATH_test_c = "/home/ubuntu/s3/titanic_dataset/test.csv"
FILE_PATH_SUB_c = "/home/ubuntu/s3/titanic_dataset/sample_submission.csv"


#local
FILE_PATH_train_l = "/home/abhi/Datasets/titanic_dataset/train.csv"
FILE_PATH_test_l = "/home/abhi/Datasets/titanic_dataset/test.csv"
FILE_PATH_SUB_l = "/home/abhi/Datasets/titanic_dataset/sample_submission.csv"

INFO_FILE_NAME = "data_info.txt"
DESCRIBE_FILE_NAME = "data_describe.txt"
NULL_COUNTS_FILE = "null_count.txt"
HEAD_FILE = "head_info.txt"


class Data():

	def __init__(self):
		self.train = None
		self.test = None
		#self.combine_data = None
		self.X = None 
		self.y = None
		self.train_X = None
		self.train_y = None
		self.test_X = None
		self.test_y = None

	def info_to_text(self,info,file_name):
		"""
		Function to help write in the file
		"""
		with open(file_name, "w") as file:

			file.write(info)


	def read_data(self,train_data_path,test_data_path):
		"""
		The function to read the data 
		"""

		self.train = pd.read_csv(train_data_path)
		self.test = pd.read_csv(test_data_path)

	def data_visualization(self):
		"""
		The function to visualize the data 
		"""
		col_lst = []

		#find the values that are int 

		result = self.train.select_dtypes(include=[float])

		for col in result.columns:
			col_lst.append(col)

		#make the correlation matrix

		#plot 1 

		corrmat = self.train[col_lst].corr()

		plt.figure(figsize=(12, 8))

		sns.heatmap(corrmat,vmax=0.9,square=True)

		plt.savefig('corr_mat.png')

		plt.clf()

		#get the info of the dataset

		buffer = io.StringIO()
		with redirect_stdout(buffer):
			self.train.info()

		info_string = buffer.getvalue()

		self.info_to_text(info_string,INFO_FILE_NAME)

		#get the decribe of the dataset 
		describe_string = self.train.describe().to_string()

		self.info_to_text(describe_string,DESCRIBE_FILE_NAME)

		#get the head of the data
		head_string = self.train.head().to_string()

		self.info_to_text(head_string,HEAD_FILE)
		
		#get the null count of the file 
		nan_count = self.train.isnull().sum()
		nan_count = str(nan_count)
		self.info_to_text(nan_count,NULL_COUNTS_FILE)

		#plot the data in graphs and save 

		#plot 2 

		sns.scatterplot(x="Age", y="Transported", hue="VIP", data=self.train, marker="o", color="b")

		plt.savefig('data_plotting.png')

		plt.clf()

		#plot 3 

		sns.pairplot(data = self.train, hue='Transported', palette='viridis')

		plt.savefig('pair_plotting.png')

		plt.clf()


	def data_prep(self):
		"""
		the function to prep the data
		handle missing values and fill the data 
		"""

		#list the columns to drop 

		columns_drop = ["Name","PassengerId"]
		test_col = ["Transported"]

		#make the train and test data 
		self.train_X = self.train.drop("Transported", axis = 1)
		self.train_y = self.train["Transported"]


		#drop the columns not with non necessary features
		self.train_X = self.train_X.drop(columns= columns_drop, axis=1)

		#filing the missing values 
		self.train_X['RoomService'] = self.train_X.groupby('VIP')['RoomService'].transform(lambda x:x.fillna(x.mean()))


		#Destination, HomePlanet, Cabin, CryoSleep,VIP missing value filled with mode()
		cols = ['Destination','HomePlanet','Cabin','CryoSleep','VIP']
		for c in cols:
			self.train_X[c] = self.train_X[c].fillna(self.train_X[c].mode()[0])


		#Age missing value fill with mean
		self.train_X['Age'] = self.train_X['Age'].fillna(self.train_X['Age'].mean())

		self.train_X['RoomService'] = self.train_X.groupby('VIP')['RoomService'].transform(lambda x:x.fillna(x.mean()))

		cols = ['VRDeck','FoodCourt','ShoppingMall','Spa']
		for c in cols:
			self.train_X[c] = self.train_X.groupby('VIP')[c].transform(lambda x:x.fillna(x.mean()))


		#filling the data with one hot encoder		
		cols = ('HomePlanet','CryoSleep','Cabin','Destination','VIP')
		for c in cols:
			lbl = LabelEncoder() 
			lbl.fit(list(self.train_X[c].values))   
			self.train_X[c] = lbl.transform(list(self.train_X[c].values))


		print(self.train_X.info())

		print(self.train_y.info())








if __name__ == "__main__":
	data = Data()
	data.read_data(FILE_PATH_train_l,FILE_PATH_test_l)
	#data.data_visualization()
	data.data_prep()


