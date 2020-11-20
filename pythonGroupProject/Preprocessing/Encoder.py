import pandas as pd
import random as rnd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Preprocessing.DataObject import DataObject
from sklearn import preprocessing

class Encoder:
	def __init__(self, dataObject):
		self.trainingData = dataObject.trainingData
		self.testingData = dataObject.testingData
		self.combinedData = dataObject.combinedData

	def encodeData(self):
		X = self.trainingData;
		# Remove rows with missing target, separate target from predictors
		X.dropna(axis=0, subset=['SalePrice'], inplace=True)
		#sale price 
		y = X.SalePrice;
		X.drop(['SalePrice'], axis=1, inplace=True)
		
		object_cols = ['MSSubClass', 'MSZoning', 'Alley', 'LandContour', 'LotConfig',
	   'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
	   'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
	   'Foundation', 'Heating', 'Electrical', 'GarageType', 'MiscFeature',
	   'MoSold', 'SaleType', 'SaleCondition'];
		
		XtrainDummies = pd.get_dummies(X[object_cols]);
		XtrainFinal = pd.concat([X,XtrainDummies],axis='columns');
		
		for i in object_cols:
		 XtrainFinal= XtrainFinal.drop([i],axis = 'columns');


	
		
		# X_train, X_valid, y_train, y_valid = train_test_split(X, y,
		#                                               train_size=0.8, test_size=0.2,
		#                                               random_state=0)
		# # All categorical columns
		# object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"];
		# # Get number of unique entries in each column with categorical data
		# object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
		# d = dict(zip(object_cols, object_nunique));
		# # Print number of unique entries by column, in ascending order
		# print(sorted(d.items(), key=lambda x: x[1]))
			  
		# # Columns that will be one-hot encode

		# low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]
		# # Columns that will be dropped from the dataset
		
		# high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))
		
		# from sklearn.preprocessing import OneHotEncoder
		# # Use as many lines of code as you need!
		# # Apply one-hot encoder to each column with categorical data
		# OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
		# OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
		# OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))
		
		
		# # One-hot encoding removed index; put it back
		# OH_cols_train.index = X_train.index
		# OH_cols_valid.index = X_valid.index
		
		# # # Remove categorical columns (will replace with one-hot encoding)
		# num_X_train = X_train.drop(object_cols, axis=1)
		# num_X_valid = X_valid.drop(object_cols, axis=1)
		
		# # # Add one-hot encoded columns to numerical features
		# OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1) # Your code here
		# OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1) # Your code here
		return DataObject(self.trainingData, self.testingData, self.combinedData)

